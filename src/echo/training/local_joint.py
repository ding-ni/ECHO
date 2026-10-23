"""笔记本真实联合逻辑验证：有限训练、后验诊断、短查询及独立评分。"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from echo.data.episodes import target_input, tokens
from echo.evaluation.report import evaluate
from echo.hydro.hbv import RegionalHBV
from echo.models.echo import EchoStage
from echo.training.pipeline import fit_stage, load_station, tensor
from echo.training.predict import predict
from echo.utils import digest, read_json, run_metadata, write_json


def run(index_path, config_path, output):
    index_path, config_path, output = Path(index_path), Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    index = read_json(index_path)
    if not (output / "model.pt").exists():
        fit_stage(index_path, config, output)
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["config"] != config or checkpoint["index_hash"] != digest(index_path):
        raise ValueError("联合检查点与指定配置或数据清单不一致")
    model = EchoStage(checkpoint["geo_dim"], config["dynamic"], True, config["hidden"],
                      config["posterior_rank"], config["learn_hydro_adapter"]).eval()
    model.load_state_dict(checkpoint["state"])
    hydro_root = output.parent / config["hydrology_run"]
    hc = torch.load(hydro_root / "hydro_full.pt", map_location="cpu", weights_only=False)
    hydro = RegionalHBV(hc["geo_dim"]).eval()
    hydro.load_state_dict(hc["state"])
    diagnostics = []
    torch.set_num_threads(2)
    for row in index["stations"]:
        if row["split"] not in {"validation", "development_test"}:
            continue
        # 诊断只读目标历史上下文及许可气象，不读取评分标签。
        data = load_station(row, exclude_q=True)
        inp = target_input(data, budget_days=7)
        qref = np.load(hydro_root / "参考流量" / (row["station_id"] + ".npy"))
        start, end = inp.support_start, inp.support_end
        tok, _, _ = tokens(inp.weather[start:end], qref[start:end], inp.area_km2,
                            inp.support_h, inp.support_mask)
        with torch.no_grad():
            geo = tensor(inp.geography[None], "cpu")
            context, _ = model.encoder(tensor(tok[None], "cpu"), geo)
            mean, logsd, factor = model.posterior.statistics(context, geo)
            covariance = torch.diag_embed((2 * logsd).exp()) + factor @ factor.transpose(-1, -2)
            adapted = hydro.parameters_for(geo, parameter_delta=model.hydro_adapter(mean[:, :8]))
            original = hydro.parameters_for(geo)
        diagnostics.append({
            "station_id": row["station_id"], "split": row["split"],
            "posterior_dimension": 32, "covariance_rank_update": config["posterior_rank"],
            "min_covariance_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
            "hydro_curve_cross_covariance_norm": float(covariance[:, :8, 8:24].norm()),
            "hydro_dynamic_cross_covariance_norm": float(covariance[:, :8, 24:].norm()),
            "curve_dynamic_cross_covariance_norm": float(covariance[:, 8:24, 24:].norm()),
            "regional_HBV_parameters": original[0].tolist(),
            "adapted_HBV_parameters_at_posterior_mean": adapted[0].tolist(),
            "future_H_labels_read": False,
        })
    adapter_change = float((model.hydro_adapter.weight.detach() - hc["state"]["local_offset.weight"]).norm())
    if not np.isfinite(adapter_change) or adapter_change <= 0:
        raise ValueError("地方HBV适配映射没有产生有效训练变化")
    if any(not np.isfinite(r["min_covariance_eigenvalue"]) or r["min_covariance_eigenvalue"] <= 0
           for r in diagnostics):
        raise ValueError("地方后验协方差未通过正定性检查")
    write_json(output / "地方后验与HBV适配诊断.json", {
        "stations": diagnostics, "adapter_change_from_pretrained_norm": adapter_change,
        "scope": "real_joint_logic_smoke_not_full_posterior_validation",
        "posterior": "diagonal_plus_rank4_gaussian_not_conditional_spline_flow",
        "regional_HBV_frozen": True, "learned_shared_hydro_adapter": True,
        "future_H_labels_read": False,
        **run_metadata(config, [index_path, config_path, output / "model.pt"]),
    })
    for split in ["validation", "development_test"]:
        prediction = output / "短查询预测" / split
        score = output / "短查询评分" / split
        if prediction.exists() or score.exists():
            raise ValueError("短查询结果已存在，不覆盖既有预测和评分")
        result = predict(index_path, hydro_root, prediction, model_run=output, split=split,
                         budget_days=7, members=16, device="cpu", query_hours=24)
        if result["failures"]:
            raise ValueError(f"联合短查询预测失败: {result['failures']}")
        evaluate(prediction, index_path.parent / "评分标签", score)
    result = {"scope": "real_joint_logic_smoke_not_yearly_skill_comparison",
              "training_source_sites": len(config["training_station_ids"]),
              "completed_target_sites": len(diagnostics), "support_days": 7, "query_hours": 24,
              "prediction_members": 16, "adapter_change_from_pretrained_norm": adapter_change,
              "complete": True, "regional_HBV_frozen": True}
    write_json(output / "联合逻辑验证结果.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.index, args.config, args.output)
