import copy

import numpy as np
import pytest
import torch

from echo.data.permissions import ForecastInput
from echo.hydro.hbv import RegionalHBV
from echo.hydro.reference import audit_references, seal_references, verify_references
from echo.models.echo import EchoStage
from echo.training.pipeline import local_prediction
from echo.training.predict import predict
from echo.utils import digest, write_json


def test_reference_manifest_rejects_mutation_and_wrong_fold(tmp_path):
    (tmp_path / "参考流量").mkdir()
    rows = []
    for i in range(4):
        path = tmp_path / f"{i}.npz"
        np.savez(path, time_ns=np.arange(10))
        rows.append({"station_id": f"{i:08}", "terminal_system_id": f"R{i}",
                     "split": "train" if i < 3 else "development_test",
                     "crossfit_fold": i if i < 3 else -1, "path": str(path), "sha256": digest(path)})
        np.save(tmp_path / "参考流量" / f"{i:08}.npy", np.ones(10))
    for fold in [None, 0, 1, 2]:
        name = "hydro_full.pt" if fold is None else f"hydro_fold{fold}.pt"
        torch.save({"index_hash": "fixture", "trained_systems": [f"R{i}" for i in range(3) if i != fold],
                    "forbidden_systems": ["R3"] + ([] if fold is None else [f"R{fold}"])}, tmp_path / name)
    index = {"stations": rows, "forbidden_systems": ["R3"]}
    seal_references(index, "fixture", tmp_path)
    verify_references(index, "fixture", tmp_path)
    wrong = copy.deepcopy(index)
    wrong["stations"][0]["crossfit_fold"] = 1
    with pytest.raises(ValueError, match="折分"):
        verify_references(wrong, "fixture", tmp_path)
    np.save(tmp_path / "参考流量/00000000.npy", np.full(10, 200.))
    with pytest.raises(ValueError, match="哈希"):
        verify_references(index, "fixture", tmp_path)
    with pytest.raises(ValueError, match="不能原位"):
        seal_references(index, "fixture", tmp_path)


def test_reference_replay_respects_no_geo_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "参考流量").mkdir()
    rows = []
    for i in range(3):
        path = tmp_path / f"{i}.npz"
        np.savez(path, time_ns=np.arange(10), weather=np.ones((10, 3)),
                 geo=np.ones(2), area=np.float32(100))
        rows.append({"station_id": str(i), "terminal_system_id": f"R{i}", "split": "train",
                     "crossfit_fold": i, "path": str(path), "sha256": digest(path)})
        np.save(tmp_path / "参考流量" / f"{i}.npy", np.ones(10))
    index = tmp_path / "index.json"
    write_json(index, {"stations": rows})
    for fold in [None, 0, 1, 2]:
        name = "hydro_full.pt" if fold is None else f"hydro_fold{fold}.pt"
        torch.save({"state": RegionalHBV(2).state_dict(), "geo_dim": 2,
                    "config": {"no_geo": True}, "index_hash": digest(index),
                    "trained_systems": [f"R{i}" for i in range(3) if i != fold],
                    "forbidden_systems": [] if fold is None else [f"R{fold}"]}, tmp_path / name)
    calls = []

    def replay(model, inputs, device):
        np.testing.assert_array_equal(inputs["geo"], np.zeros(2))
        calls.append(True)
        return torch.ones(1, 10), None

    monkeypatch.setattr("echo.training.pipeline.simulate_q", replay)
    assert audit_references(index, tmp_path, replay=True)["replayed_stations"] == 3
    assert len(calls) == 3


def test_real_prediction_entry_ignores_scoring_labels(tmp_path, monkeypatch):
    import pandas as pd

    torch.manual_seed(73)
    path, index = tmp_path / "target.npz", tmp_path / "index.json"
    weather = np.tile([1., 15., .1], (80, 1)).astype("float32")
    np.savez(path, station_id=np.array("00000001"), system=np.array("TARGET"), weather=weather,
             geo=np.zeros(2, dtype="float32"), area=np.float32(100), time_ns=np.arange(80),
             support_h=np.linspace(1., 2., 24).astype("float32"), support_start=16, support_end=40)
    write_json(index, {"scope": "synthetic_demo", "stations": [
        {"station_id": "00000001", "terminal_system_id": "TARGET", "split": "development_test",
         "crossfit_fold": -1, "path": str(path), "sha256": digest(path)}]})
    hydro, stage = tmp_path / "hydro", tmp_path / "stage"
    hydro.mkdir()
    stage.mkdir()
    torch.save({"state": RegionalHBV(2).state_dict(), "geo_dim": 2, "trained_systems": ["SOURCE"],
                "config": {}, "index_hash": digest(index)}, hydro / "hydro_full.pt")
    torch.save({"state": EchoStage(2, dynamic=True, hidden=8).state_dict(), "geo_dim": 2,
                "trained_systems": ["SOURCE"], "index_hash": digest(index),
                "config": {"model": "echo_dynamic", "dynamic": True, "hidden": 8}}, stage / "model.pt")
    labels = tmp_path / "评分标签"
    labels.mkdir()
    label_path = labels / "00000001.parquet"
    pd.DataFrame({"time_ns": np.arange(40, 80), "H": np.ones(40)}).to_parquet(label_path)
    first = predict(index, hydro, tmp_path / "first", model_run=stage)
    pd.DataFrame({"time_ns": np.arange(40, 80), "H": np.full(40, 99999.)}).to_parquet(label_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("预测入口不得读取评分标签")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    second = predict(index, hydro, tmp_path / "second", model_run=stage)
    assert first == second == {"predicted": 1, "failures": []}
    with np.load(tmp_path / "first/00000001.npz") as a, np.load(tmp_path / "second/00000001.npz") as b:
        np.testing.assert_array_equal(a["members"], b["members"])


@pytest.mark.parametrize("dynamic", [False, True])
def test_future_weather_has_no_shortcut_when_reference_q_fixed(dynamic):
    torch.manual_seed(2)
    model = EchoStage(2, dynamic=dynamic, hidden=8).eval()
    weather = np.tile([1., 15., .1], (80, 1)).astype("float32")
    changed = weather.copy()
    changed[40:, 0] += 100
    h = np.linspace(1., 2., 24).astype("float32")
    q = np.linspace(.1, 5., 80).astype("float32")
    common = (np.zeros(2, dtype="float32"), 100., h, np.isfinite(h), 16, 40, 40, 80)
    first = ForecastInput("S", "R", weather, *common)
    second = ForecastInput("S", "R", changed, *common)
    with torch.no_grad():
        a = local_prediction(model, first, q, "cpu")[0]
        b = local_prediction(model, second, q, "cpu")[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
