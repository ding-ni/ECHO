"""短时合成样例，验证真实核心模块可训练、可预测、可评分；不能作实测证据。"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from echo.data.permissions import ForecastInput
from echo.evaluation.report import evaluate, render_report
from echo.hydro.hbv import RegionalHBV
from echo.models.echo import EchoStage
from echo.training.pipeline import local_prediction
from echo.utils import write_json


def run_demo(output):
    torch.manual_seed(20261019)
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261019)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    geo = np.array([0.2, 0.4, 0.6, 0.0, 0.0, 0.0], dtype="float32")
    weather = np.column_stack([rng.gamma(0.5, 2.0, 336), np.full(336, 15.0), np.full(336, 0.1)]).astype(
        "float32"
    )
    hydro = RegionalHBV(len(geo))
    with torch.no_grad():
        q, _, balance = hydro(torch.tensor(weather[None]), torch.tensor(geo[None]), torch.tensor([100.0]))
    q = q[0].numpy()
    truth = 1.5 + 0.8 * np.log1p(q / (100 / 86.4))
    h = truth[72:168].copy()
    inp = ForecastInput(
        "SYN_TARGET", "SYN_HELD_SYSTEM", weather, geo, 100.0, h, np.isfinite(h), 72, 168, 168, 336
    )
    model = EchoStage(len(geo), dynamic=True, probabilistic=True)
    hydro.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    source_geo = geo + 0.1
    source_weather = np.column_stack(
        [rng.gamma(0.6, 1.5, 336), np.full(336, 12.0), np.full(336, 0.08)]
    ).astype("float32")
    with torch.no_grad():
        source_q, _, _ = hydro(
            torch.tensor(source_weather[None]), torch.tensor(source_geo[None]), torch.tensor([100.0])
        )
    source_q = source_q[0].numpy()
    source_truth = 1.5 + 0.8 * np.log1p(source_q / (100 / 86.4))
    source_h = source_truth[72:168]
    # 来源天气与水位独立生成；目标查询H完全不进入优化。
    source = ForecastInput(
        "SYN_SOURCE",
        "SYN_SOURCE_SYSTEM",
        source_weather,
        source_geo,
        100.0,
        source_h,
        np.isfinite(source_h),
        72,
        168,
        168,
        336,
    )
    losses = []
    for _ in range(4):
        pred, kl, scale = local_prediction(model, source, source_q, "cpu", 2, hydro, gradients=True)
        loss = (pred - torch.tensor(source_truth[168:])[None]).square().mean() + 0.001 * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        pred, _, _ = local_prediction(model, inp, q, "cpu", 8, hydro)
    predictions, labels = root / "预测", root / "评分标签"
    predictions.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)
    times = pd.date_range("2020-10-01", periods=168, freq="h", tz="UTC").asi8
    np.savez_compressed(
        predictions / "SYN_TARGET.npz",
        members=pred.numpy(),
        time_ns=times,
        support_h=h,
        station_id=np.array("SYN_TARGET"),
        system=np.array("SYN_HELD_SYSTEM"),
    )
    pd.DataFrame({"time_ns": times, "H": truth[168:]}).to_parquet(labels / "SYN_TARGET.parquet", index=False)
    write_json(
        predictions / "predictions.json",
        {
            "scope": "synthetic_demo",
            "model": "echo_gaussian_demo",
            "mode": "synthetic_conditional_simulation",
            "budget_days": "synthetic_96_hours",
            "stations": [
                {
                    "station_id": "SYN_TARGET",
                    "terminal_system_id": "SYN_HELD_SYSTEM",
                    "path": str((predictions / "SYN_TARGET.npz").resolve()),
                }
            ],
        },
    )
    summary = evaluate(predictions, labels, root / "评估")
    report = render_report(root / "评估", predictions, labels)
    result = {
        "scope": "synthetic_demo_not_real_performance",
        "losses": losses,
        "max_balance_error_float32": float(balance.abs().max()),
        "report": str(report),
        "successfully_scored": summary["sites"],
    }
    write_json(root / "demo_results.json", result)
    return result
