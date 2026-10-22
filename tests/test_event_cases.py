import numpy as np
import pandas as pd
import pytest

from echo.evaluation.cases import freeze_cases, select_windows
from echo.utils import digest, write_json


def test_case_selection_is_chronological_and_uses_same_coverage_gate():
    observed = np.zeros(1000)
    observed[[40, 300, 800]] = [20, 50, 100]
    observed[1:25] = np.nan  # First scoring block fails the 90% gate.
    threshold, windows = select_windows(observed, np.arange(10))
    assert threshold == pytest.approx(8.55)
    assert windows == [(276, 349), (776, 849)]
    _, first = select_windows(observed, np.arange(10), count=1)
    assert first == [(276, 349)]  # Do not choose the larger peak at hour 800.


def test_frozen_cases_reject_target_q_and_preserve_existing_manifest(tmp_path):
    path = tmp_path / "target.npz"
    labels = tmp_path / "labels"
    labels.mkdir()
    truth = np.zeros(1000)
    truth[[200, 500]] = 200
    pd.DataFrame({"time_ns": np.arange(100, 1100), "H": truth}).to_parquet(labels / "00000001.parquet")
    arrays = {"time_ns": np.arange(1100), "support_h": np.arange(100), "support_end": 100}
    np.savez(path, **arrays, q=np.ones(1100))
    index = tmp_path / "index.json"

    def save_index():
        write_json(index, {"stations": [{"station_id": "00000001", "split": "validation",
                                          "path": str(path), "sha256": digest(path)}]})

    save_index()
    output = tmp_path / "cases.json"
    with pytest.raises(ValueError, match="禁止字段"):
        freeze_cases(index, labels, output)
    np.savez(path, **arrays)
    save_index()
    record = freeze_cases(index, labels, output)
    assert len(record["cases"]) == 2
    assert record["selection_uses_predictions"] is False
    before = digest(output)
    with pytest.raises(ValueError, match="已经存在"):
        freeze_cases(index, labels, output)
    assert digest(output) == before
