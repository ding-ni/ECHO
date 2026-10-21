"""缩短日历的合成集，实际调用区域预训练/交叉拟合/训练/预测/评分接口。"""

import numpy as np
import pandas as pd
import torch

from echo.evaluation.report import evaluate
from echo.hydro.hbv import RegionalHBV
from echo.training.pipeline import fit_stage
from echo.training.predict import predict
from echo.utils import digest, write_json


def test_real_training_interfaces_on_short_synthetic_calendar(tmp_path):
    torch.manual_seed(1)
    torch.set_num_threads(2)
    rng = np.random.default_rng(1)
    times = pd.date_range("2013-10-01", periods=240, freq="h", tz="UTC")
    rows = []
    teacher = RegionalHBV(2)
    labels = tmp_path / "labels"
    labels.mkdir()
    for i in range(4):
        geo = np.array([i / 10, 0.5], dtype="float32")
        weather = np.column_stack(
            [rng.gamma(0.5, 2.0, len(times)), np.full(len(times), 15.0), np.full(len(times), 0.1)]
        ).astype("float32")
        with torch.no_grad():
            q, _, _ = teacher(torch.tensor(weather[None]), torch.tensor(geo[None]), torch.tensor([100.0]))
        q = q[0].numpy()
        h = (1 + np.log1p(q)).astype("float32")
        path = tmp_path / f"{i:08}.npz"
        payload = {
            "weather": weather,
            "geo": geo,
            "area": np.float32(100),
            "time_ns": times.asi8,
            "station_id": np.array(f"{i:08}"),
            "system": np.array(f"R{i}"),
            "support_start": np.int64(144),
            "support_end": np.int64(192),
        }
        if i < 3:
            payload.update(h=h, q=q)
        else:
            payload["support_h"] = h[144:192]
            pd.DataFrame({"time_ns": times.asi8[192:], "H": h[192:]}).to_parquet(
                labels / f"{i:08}.parquet", index=False
            )
        np.savez_compressed(path, **payload)
        rows.append(
            {
                "station_id": f"{i:08}",
                "terminal_system_id": f"R{i}",
                "split": "train" if i < 3 else "development_test",
                "crossfit_fold": i if i < 3 else -1,
                "path": str(path),
                "sha256": digest(path),
            }
        )
    index = tmp_path / "index.json"
    write_json(index, {"scope": "synthetic_demo", "stations": rows})
    periods = {
        "train_start": "2013-10-01",
        "train_end": "2013-10-07",
        "validation_end": "2013-10-11",
        "stage_validation_origin": "2013-10-08",
    }
    common = {
        "seed": 20261019,
        "device": "cpu",
        "periods": periods,
        "query_hours": 24,
        "hydro_epochs": 1,
        "hydro_blocks_per_station": 1,
        "epochs": 1,
        "episodes_per_station": 1,
        "support_days": [1],
        "validation_support_hours": 24,
        "hidden": 8,
        "train_members": 1,
        "hydrology_run": "hydro",
    }
    hydro = tmp_path / "runs/hydro"
    fit_stage(index, {**common, "model": "hydrology"}, hydro)
    for kind in ["echo_static", "direct_lstm", "echo_dynamic", "echo_gaussian"]:
        run = tmp_path / "runs" / kind
        config = {
            **common,
            "model": kind,
            "dynamic": kind in {"echo_dynamic", "echo_gaussian"},
            "probabilistic": kind == "echo_gaussian",
        }
        if kind in {"echo_dynamic", "echo_gaussian"}:
            previous = "echo_static" if kind == "echo_dynamic" else "echo_dynamic"
            config["initialize_from"] = str(tmp_path / "runs" / previous / "model.pt")
            config["freeze_static"] = kind == "echo_dynamic"
        fit_stage(index, config, run)
        forecasts = tmp_path / (kind + "_predictions")
        result = predict(index, hydro, forecasts, model_run=run, members=4)
        assert result["predicted"] == 1, result
        result = evaluate(forecasts, labels, tmp_path / (kind + "_scores"))
        assert result["sites"] == 1
        assert result["scope"] == "synthetic_demo"
