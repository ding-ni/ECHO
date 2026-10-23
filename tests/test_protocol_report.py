import numpy as np
import pandas as pd
import pytest

from echo.evaluation.report import evaluate, render_report
from echo.utils import digest, write_json


def test_protocol_member_and_total_intervals_share_labels_and_scorer(tmp_path):
    times = np.arange(168, dtype="int64") * 3600 * 10**9
    truth = 1 + np.sin(np.arange(168) / 24)
    path = tmp_path / "001.npz"
    np.savez_compressed(path, members=np.stack([truth - 1, truth + 1]),
                        members_without_residual=np.stack([truth - .1, truth + .1]),
                        support_h=np.linspace(0, 2, 168), time_ns=times)
    labels = tmp_path / "labels"
    labels.mkdir()
    pd.DataFrame({"time_ns": times, "H": truth}).to_parquet(labels / "001.parquet")
    row = {"station_id": "001", "terminal_system_id": "synthetic_system",
           "path": str(path), "sha256": digest(path)}
    for name, member_field in [("total", "members"), ("members", "members_without_residual")]:
        write_json(tmp_path / name / "predictions.json", {
            "scope": "synthetic_demo", "mode": "conditional_simulation", "model": "protocol_echo",
            "budget_days": 7, "stations": [row], "failures": [], "member_field": member_field,
            "architecture_status": "ProtocolECHO_schema_test_only", "uncertainty_scope": name})
        summary = evaluate(tmp_path / name, labels, tmp_path / (name + "_eval"))
        assert summary["sites"] == 1
        expected = .5 if name == "total" else .05
        assert summary["all_period"]["CRPS_m"] == pytest.approx(expected)
    report = render_report(tmp_path / "total_eval", tmp_path / "total", labels)
    assert "完整模型执行以该运行" in report.read_text(encoding="utf-8")
    # Corrupted prediction files are failures, never silently rescored.
    path.write_bytes(b"changed")
    summary = evaluate(tmp_path / "total", labels, tmp_path / "changed_eval")
    assert summary["sites"] == 0 and len(summary["failures"]) == 1
