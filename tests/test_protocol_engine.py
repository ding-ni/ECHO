"""Training-control tests use a labelled tiny frontend, never an official-Mamba claim."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from echo.models.protocol import ProtocolCurve, ProtocolECHO, ProtocolHydrology
from echo.training.protocol_engine import (
    ProtocolEpisode,
    SourceLabels,
    cache_references,
    calibrate_residuals,
    fit_hydrology_bank,
    fit_response_stages,
    load_stage_a_frontend,
    validation_score,
)
from echo.training.protocol_steps import MemberTrajectory


def episode(station="source0", role="train", fold=0):
    start = "2013-10-01" if role == "train" else "2018-10-01"
    origin = pd.Timestamp(start)
    times = pd.date_range(origin - pd.DateOffset(years=2), origin + pd.Timedelta(hours=192),
                          freq="h", inclusive="left").asi8
    a, b, c = len(times) - 192, len(times) - 24, len(times)
    data = {
        "time_ns": times, "weather_nodes": np.tile([1., 5., .1], (2, len(times), 1)).astype("float32"),
        "node_features": np.ones((2, 8), dtype="float32"), "node_areas": np.array([10., 20.]),
        "downstream_parent": np.array([1, -1]), "edge_index": np.array([[0], [1]]),
        "edge_attr": np.ones((1, 3)), "edge_attr_mask": np.ones((1, 3), dtype=bool),
        "terrain_tokens": np.ones((16, 14)), "terrain_padding_mask": np.zeros(16, dtype=bool),
        "climate_features": np.ones(64), "support_h": np.linspace(1, 2, b - a),
    }
    provenance = {"source_snapshots": ["synthetic_geometry"],
                  "weather_kind": "incremental_area_weighted_original_grid",
                  "terrain_kind": "DEM_local_valley_multiscale",
                  "graph_kind": "connected_child_parent_edge_contraction",
                  "climate_cutoff_ns": int(times[a]), "node_feature_names": [], "terrain_feature_names": []}
    return ProtocolEpisode(station, "system_" + station, role, fold, data, provenance, a, b, c)


class TinyHydrology(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_offset = nn.Linear(8, 1, bias=False)
        self.regionalizer = nn.Linear(8, 1)

    def forward(self, weather, node_features, node_areas, parents, environment=None, **kwargs):
        gain = torch.nn.functional.softplus(self.regionalizer(node_features).mean() + environment.mean())
        q = weather[:, :, :, 0].mean(1) * gain
        return {"q_outlet": q, "state": None, "routing_state": None}


class TinyFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.environment_encoder = nn.Module()
        self.environment_encoder.net = nn.Linear(64, 256)
        self.environment_encoder.forward = lambda geo: self.environment_encoder.net(geo["climate_features"])
        self.hydrology = TinyHydrology()


def test_A_can_validate_without_circular_reference_and_future_labels_are_separate():
    item = episode()
    assert item.validate(require_reference=False)["support_hours"] == 168
    with pytest.raises(ValueError):
        item.validate()
    item.inputs["q"] = np.zeros(1)
    with pytest.raises(ValueError, match="监督Q"):
        item.validate(require_reference=False)


def test_A_threefold_cache_excludes_own_system_and_loads_real_checkpoint(tmp_path):
    pairs = [(episode("source" + str(i), fold=i), SourceLabels(np.ones(24), np.ones(192) * (i + 1)))
             for i in range(3)]
    bank = fit_hydrology_bank(TinyFrontend, pairs, tmp_path / "A", steps=1)
    refs, setup = cache_references(TinyFrontend, bank, [p[0] for p in pairs], tmp_path / "references")
    for ref, (item, _) in zip(refs, pairs, strict=True):
        assert item.system_id not in ref["reference_trained_systems"]
        assert np.load(ref["path"]).shape == item.inputs["time_ns"].shape
    assert setup["knot_width"] > 0
    actual = TinyFrontend()
    load_stage_a_frontend(actual, bank[0])
    with pytest.raises(ValueError):
        load_stage_a_frontend(actual, bank[1])
    # Changing Q of fold 0 cannot change its excluded frontend weights.
    changed = [(item, SourceLabels(labels.query_h, labels.q_supervision * 100
                                  if item.fold == 0 else labels.q_supervision)) for item, labels in pairs]
    second_bank = fit_hydrology_bank(TinyFrontend, changed, tmp_path / "A_changed", steps=1)
    first = torch.load(bank[1]["path"], weights_only=False)["frontend_state"]
    second = torch.load(second_bank[1]["path"], weights_only=False)["frontend_state"]
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_A_disallows_target_system_and_residual_validation_cannot_use_development(tmp_path):
    pairs = [(episode("source" + str(i), fold=i), SourceLabels(np.ones(24), np.ones(192))) for i in range(3)]
    with pytest.raises(ValueError, match="禁用水系"):
        fit_hydrology_bank(TinyFrontend, pairs, tmp_path / "bad_A", forbidden_systems={"system_source0"})
    item = episode("development", role="development_test", fold=-1)
    with pytest.raises(ValueError, match="不能使用开发"):
        validation_score(TinyFrontend(), [(item, SourceLabels(np.ones(24)))])


def test_source_AR1_calibration_does_not_bridge_missing_hours_or_station_boundaries():
    calibrated = calibrate_residuals([np.array([-2., -1., np.nan, 2., 1.]), np.array([2., 1., -1., -2.])])
    assert calibrated["observations"] == 8
    assert calibrated["source_validation_only"]
    assert abs(calibrated["rho"]) < 1
    assert calibrated["innovation_sd_normalized_H"] > 0


def test_A_subtree_matches_full_model_capacity_without_loading_official_history(monkeypatch):
    monkeypatch.setattr("echo.models.protocol.OfficialHistoryEncoder", lambda: nn.Linear(8, 256))
    frontend, model = ProtocolHydrology(8, 14), ProtocolECHO(8, 14, .1)
    for name in ["environment_encoder", "hydrology"]:
        subtree, complete = getattr(frontend, name).state_dict(), getattr(model, name).state_dict()
        assert subtree.keys() == complete.keys()
        assert all(subtree[key].shape == complete[key].shape for key in subtree)


def test_B_to_E_optimizer_checkpoints_calibration_and_frozen_exit(tmp_path, monkeypatch):
    # Isolate training control from hydrological runtime / official CUDA kernels.
    # Real HBV and posterior properties are exercised in their dedicated tests.
    class ResponseController(TinyFrontend):
        configure_stage = ProtocolECHO.configure_stage

        def __init__(self):
            super().__init__()
            self.history_encoder = nn.Linear(8, 256)
            self.posterior = nn.Linear(256, 32)
            self.curve = ProtocolCurve(.2)
            self.dynamic = nn.Linear(1, 1)
            self.stage = None

    def trajectory(model, item, members=4, stochastic=True, **kwargs):
        env = model.environment_encoder({"climate_features": torch.ones(1, 64)})
        code = model.posterior(model.history_encoder(torch.ones(1, 8))).expand(members, -1)
        count = item.query_end - item.support_start
        q = torch.nn.functional.softplus(model.hydrology.local_offset(code[:, :8])).expand(-1, count)
        u = torch.log1p(q / (30 / 86.4))
        h = model.curve(u, env.expand(members, -1), code[:, 8:24], torch.ones(members) * .5)
        if model.stage in {"C", "D", "E"}:
            h = h + model.dynamic(u[..., None])[..., 0]
        length = item.support_end - item.support_start
        result = MemberTrajectory(h[:, :length], h[:, length:], q[:, :length], q[:, length:], None, None, None)
        return result, code.square().mean(), env, code, torch.zeros(1), torch.ones(1), torch.ones(1) * .5

    monkeypatch.setattr("echo.training.protocol_engine.episode_trajectory", trajectory)
    pairs = [(episode("source" + str(i), fold=i), SourceLabels(np.linspace(1, 2, 24), np.ones(192)))
             for i in range(3)]
    bank = fit_hydrology_bank(TinyFrontend, pairs, tmp_path / "A", steps=1)
    validation = [(episode("validation", role="validation", fold=-1), SourceLabels(np.linspace(1, 2, 24)))]
    model = ResponseController()
    result = fit_response_stages(model, pairs, validation, tmp_path / "response", steps_per_stage=2,
                                 stage_a_receipt=bank[0])
    assert result["stages"] == list("BCDE") and result["optimizer_steps"] == 6
    assert all(not p.requires_grad for p in model.parameters()) and model.stage == "E"
    assert result["residual_model"]["source_validation_only"]
    for stage in "BCD":
        assert (tmp_path / "response" / f"stage_{stage}.pt").is_file()
