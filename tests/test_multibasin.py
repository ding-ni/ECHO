import numpy as np
import pandas as pd
import pytest
import torch

from echo.data.scale import source_pool
from echo.training.multibasin import load_sources, station_logq_loss
from echo.utils import digest, write_json


def test_station_loss_masks_missing_and_weights_stations_equally():
    predicted = torch.tensor([[1., 1., 1., 1.], [2., 2., 2., 2.]], requires_grad=True)
    observed = torch.tensor([[2., 2., 2., 2.], [100., float("nan"), 100., 100.]])
    loss, points = station_logq_loss(predicted, observed)
    torch.testing.assert_close(loss, (torch.log(torch.tensor(2.)) - torch.log(torch.tensor(3.))).square())
    assert points == 4
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert torch.equal(predicted.grad[1], torch.zeros(4))
    assert station_logq_loss(predicted.detach(), torch.full_like(observed, float("nan"))) == (None, 0)


def test_scale_pool_keeps_all_source_sites_without_reading_target_q(tmp_path):
    manifest, roles = tmp_path / "manifest.csv", tmp_path / "roles.json"
    pd.DataFrame({"station_id": ["001", "002", "003"], "terminal_system_id": ["S", "S", "T"],
                  "weather_available": True, "topology_verified": True, "area_km2": 100}).to_csv(manifest, index=False)
    write_json(roles, {"manifest_sha256": digest(manifest), "systems": [
        {"terminal_system_id": "S", "system_split": "train", "crossfit_fold": 0},
        {"terminal_system_id": "T", "system_split": "validation", "crossfit_fold": -1}]})
    pool, forbidden = source_pool(manifest, roles)
    assert pool.station_id.tolist() == ["001", "002"]
    assert forbidden == {"T"}


def test_pretrain_loader_rejects_h_and_target_system(tmp_path):
    path = tmp_path / "source.npz"
    np.savez(path, h=np.zeros(4))
    index = tmp_path / "index.json"
    row = {"station_id": "001", "terminal_system_id": "S", "split": "train", "path": str(path),
           "crossfit_fold": 0, "sha256": digest(path)}
    write_json(index, {"scope": "real_source_Q_pretraining", "forbidden_systems": ["T"], "stations": [row]})
    with pytest.raises(ValueError, match="携带H"):
        load_sources(index)
    row["terminal_system_id"] = "T"
    write_json(index, {"scope": "real_source_Q_pretraining", "forbidden_systems": ["T"], "stations": [row]})
    with pytest.raises(ValueError, match="非训练水系"):
        load_sources(index)


def test_interrupted_training_resumes_optimizer_and_next_epoch(tmp_path, monkeypatch):
    import echo.training.multibasin as module

    class TinyHydro(torch.nn.Module):
        def __init__(self, dimension):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, weather, geo, area, state=None):
            predicted = weather[:, :, 0] * self.weight
            return predicted, weather.new_zeros((len(weather), 5)), torch.zeros_like(predicted)

    times = pd.date_range("2011-10-01", "2019-10-01", freq="h", inclusive="left", tz="UTC").asi8
    rows = [{"terminal_system_id": f"R{i}", "station_id": str(i)} for i in range(3)]
    arrays = [{"weather": np.ones((len(times), 3), dtype="float32"),
               "q": np.full(len(times), 2., dtype="float32"), "geo": np.zeros(2), "area": np.float32(100)}
              for _ in rows]
    config = {"seed": 17, "epochs": 2, "basin_batch_size": 2, "query_hours": 4,
              "blocks_per_epoch": 1, "validate_every": 1}
    monkeypatch.setattr(module, "RegionalHBV", TinyHydro)
    monkeypatch.setattr(module, "advance", lambda *a: None)
    calls = []

    def interrupt(*args):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return 1.

    monkeypatch.setattr(module, "validate", interrupt)
    resumed = tmp_path / "resumed.pt"
    with pytest.raises(RuntimeError, match="interruption"):
        module.fit_one(rows, arrays, times, {"T"}, config, resumed, "fixture")
    monkeypatch.setattr(module, "validate", lambda *a: 1.)
    module.fit_one(rows, arrays, times, {"T"}, config, resumed, "fixture")
    fresh = tmp_path / "fresh.pt"
    module.fit_one(rows, arrays, times, {"T"}, config, fresh, "fixture")
    a = torch.load(resumed.with_suffix(".latest.pt"), weights_only=False)
    b = torch.load(fresh.with_suffix(".latest.pt"), weights_only=False)
    assert a["next_epoch"] == b["next_epoch"] == 2
    assert a["history"] == b["history"]
    torch.testing.assert_close(a["state"]["weight"], b["state"]["weight"], rtol=0, atol=0)
