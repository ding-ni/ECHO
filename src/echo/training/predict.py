"""预测入口只读取目标上下文及来源权重，不读取评分标签。"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

from echo.data.episodes import target_input
from echo.data.permissions import assert_source_permissions, support_scaling
from echo.hydro.hbv import RegionalHBV
from echo.models.echo import DirectStage, EchoStage
from echo.training.pipeline import load_station, local_prediction, simulate_q, stage_forward
from echo.utils import digest, read_json, run_metadata, write_json


def local_spline(q, support_q, support_h, area):
    mask = np.isfinite(support_h)
    center, scale = support_scaling(support_h, mask)

    def basis(values):
        u = np.log1p(np.maximum(values, 0) / (area / 86.4))
        b = np.clip(u[:, None] - np.arange(32)[None] * 0.2, 0, 0.2)
        b[:, -1] += np.maximum(u - 6.4, 0)
        return np.column_stack([np.ones(len(u)), b])

    fit = lsq_linear(
        basis(support_q)[mask],
        (support_h[mask] - center) / scale,
        bounds=(np.r_[-np.inf, np.full(32, 1e-4)], np.full(33, np.inf)),
    )
    return center + scale * (basis(q) @ fit.x)


def predict(
    index_path,
    hydro_run,
    output,
    model_run=None,
    baseline=None,
    split="development_test",
    budget_days=None,
    members=64,
    device="cpu",
    seed=20261019,
    query_hours=None,
):
    if split == "blind_holdout":
        raise ValueError("第一阶段不解封正式盲测")
    if (model_run is None) == (baseline is None):
        raise ValueError("模型与基线必须且只能指定一个")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device = torch.device(device)
    index = read_json(index_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, config, checkpoint = None, {}, None
    if model_run:
        checkpoint = torch.load(Path(model_run) / "model.pt", map_location=device, weights_only=False)
        if checkpoint["index_hash"] != digest(index_path):
            raise ValueError("预测数据清单与模型训练清单不一致")
        config = checkpoint["config"]
    hc, hydro = None, None
    needs_hydro = (model_run is not None and config["model"] != "direct_lstm") or baseline == "local_spline"
    if needs_hydro:
        if hydro_run is None:
            raise ValueError("该模型需要已训练的水文前端")
        root = Path(hydro_run)
        hc = torch.load(root / "hydro_full.pt", map_location=device, weights_only=False)
        if hc.get("index_hash") != digest(index_path):
            raise ValueError("水文前端与当前预测数据清单不一致")
        hydro = RegionalHBV(hc["geo_dim"]).to(device)
        hydro.load_state_dict(hc["state"])
        hydro.eval().requires_grad_(False)
    if model_run:
        if hc is not None and bool(config.get("no_geo")) != bool(hc["config"].get("no_geo")):
            raise ValueError("后端与水文前端的地理消融设置不一致")
        if config["model"] == "direct_lstm":
            model = DirectStage(checkpoint["geo_dim"], config.get("hidden", 64))
        else:
            model = EchoStage(
                checkpoint["geo_dim"],
                config.get("dynamic", False),
                config.get("probabilistic", False),
                config.get("hidden", 64),
                config.get("posterior_rank", 0),
                config.get("learn_hydro_adapter", False),
            )
        model.load_state_dict(checkpoint["state"])
        model.to(device).eval()
    elif baseline not in {"persistence", "local_spline"}:
        raise ValueError("指定模型或基线")
    records, failures = [], []
    for row in index["stations"]:
        if row["split"] != split:
            continue
        try:
            if hc is not None:
                assert_source_permissions(hc["trained_systems"], {row["terminal_system_id"]})
            if model_run:
                assert_source_permissions(checkpoint["trained_systems"], {row["terminal_system_id"]})
            d = load_station(row)
            if config.get("no_geo"):
                d["geo"] = np.zeros_like(d["geo"])
            inp = target_input(d, budget_days)
            if query_hours is not None:
                if query_hours < 1 or inp.query_start + query_hours > inp.query_end:
                    raise ValueError("指定查询小时数超出目标气象覆盖")
                inp = replace(inp, query_end=inp.query_start + query_hours)
            with torch.no_grad():
                if baseline == "persistence":
                    pred = np.full((1, inp.query_end - inp.query_start), inp.support_h[inp.support_mask][-1])
                elif needs_hydro:
                    qref, _ = simulate_q(hydro, d, device, stop=inp.query_end)
                    q = qref[0].cpu().numpy()
                else:
                    q = None
                if baseline == "local_spline":
                    pred = local_spline(
                        q[inp.query_start :],
                        q[inp.support_start : inp.support_end],
                        inp.support_h,
                        inp.area_km2,
                    )[None]
                elif isinstance(model, EchoStage):
                    pred, _, _ = local_prediction(
                        model, inp, q, device, members if model.probabilistic else 1, hydro
                    )
                    pred = pred.cpu().numpy()
                elif model is not None:
                    pred, _, _ = stage_forward(model, inp, q, device, config, hydro)
                    pred = pred.cpu().numpy()
            path = output / (row["station_id"] + ".npz")
            intervals = {"median": np.median(pred, axis=0)}
            if len(pred) > 1:
                for level in [.5, .9]:
                    low, high = np.quantile(pred, [(1 - level) / 2, (1 + level) / 2], axis=0)
                    intervals[f"lower_{int(level * 100)}"] = low
                    intervals[f"upper_{int(level * 100)}"] = high
            np.savez_compressed(
                path,
                members=pred,
                time_ns=d["time_ns"][inp.query_start : inp.query_end],
                support_h=inp.support_h,
                station_id=np.array(row["station_id"]),
                system=np.array(row["terminal_system_id"]),
                **intervals,
            )
            records.append(
                {
                    "station_id": row["station_id"],
                    "path": str(path.resolve()),
                    "terminal_system_id": row["terminal_system_id"],
                }
            )
        except Exception as exc:
            failures.append({"station_id": row["station_id"], "error": str(exc)})
    write_json(
        output / "predictions.json",
        {
            "scope": index["scope"],
            "stations": records,
            "failures": failures,
            "mode": "conditional_simulation_no_H_updates",
            "budget_days": budget_days or "complete_water_year",
            "query_hours_requested": query_hours,
            "model": baseline or config.get("model"),
            "uncertainty_scope": ("conditional_local_parameters_only_no_weather_or_residual_calibration"
                                  if config.get("probabilistic") else "deterministic"),
            "metadata": run_metadata(config, [index_path]),
        },
    )
    return {"predicted": len(records), "failures": failures}
