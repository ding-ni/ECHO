import copy

import numpy as np
import pandas as pd
import pytest
import torch

from echo.data.prepare import prepare
from echo.data.splits import build_splits, freeze_system_roles
from echo.data.usgs_q_evidence import _assert_response_station, _hourly_q, audit_source_q
from echo.hydro.hbv import RegionalHBV, _hbv_sequence, inference_sequence
from echo.utils import write_json


def _manifest():
    return pd.DataFrame([
        {"station_id": f"{i:08}", "terminal_system_id": f"R{i // 2}",
         "topology_verified": True, "topology_evidence": "fixture",
         "weather_available": True, "area_km2": 100., "support_coverage": 1.,
         "query_coverage": 1., "time_support_verified": True, "datum_verified": True}
        for i in range(40)
    ])


def test_frozen_roles_survive_quality_changes_and_row_order(tmp_path):
    manifest, output = tmp_path / "manifest.csv", tmp_path / "roles.json"
    frame = _manifest()
    frame.to_csv(manifest, index=False)
    roles = freeze_system_roles(manifest, output)
    assert b"\r\n" not in output.read_bytes()  # 固定角色的字节哈希不能随Git/Linux检出换行变化。
    first = build_splits(frame, max_sites=10, system_roles=roles)
    changed = frame.sample(frac=1, random_state=3).copy()
    changed.loc[changed.index[:8], "datum_verified"] = False
    second = build_splits(changed, max_sites=10, system_roles=roles)
    for field in ["system_split", "crossfit_fold", "target_Q_forbidden"]:
        assert first.set_index("station_id")[field].to_dict() == second.set_index("station_id")[field].to_dict()
    assert len(second[second.split.isin(["train", "validation", "development_test"])]) <= 10
    assert freeze_system_roles(manifest, output) == roles
    with pytest.raises(ValueError, match="冻结"):
        freeze_system_roles(manifest, output, seed=9)


def test_forged_fold_rejected_before_any_data_read(tmp_path):
    frame = _manifest()
    roles = {"systems": [{"terminal_system_id": f"R{i}", "system_split": "train",
                           "crossfit_fold": i % 3} for i in range(20)]}
    split = build_splits(frame, max_sites=10, system_roles=roles)
    split.loc[split.split == "train", "crossfit_fold"] = 9
    manifest, mapping = tmp_path / "manifest.csv", tmp_path / "mapping.json"
    roles_path = tmp_path / "roles.json"
    split.to_csv(manifest, index=False)
    write_json(roles_path, roles)
    write_json(mapping, {"verified": True, "evidence": "fixture", "timezone": "UTC",
                         "time_support": "step_end"})
    with pytest.raises(ValueError, match="折号"):
        prepare(manifest, mapping, tmp_path / "out", datum_registry="unused",
                system_roles=roles_path, source_evidence="unused", height_evidence="unused",
                height_snapshots="unused")


def test_q_audit_cannot_request_target_or_failed_topology(tmp_path, monkeypatch):
    import echo.data.usgs_q_evidence as module

    frame = _manifest()
    frame.loc[frame.terminal_system_id == "R0", "topology_verified"] = False
    manifest, roles_path = tmp_path / "manifest.csv", tmp_path / "roles.json"
    frame.to_csv(manifest, index=False)
    write_json(roles_path, {"systems": [
        {"terminal_system_id": "R0", "system_split": "train"},
        {"terminal_system_id": "R1", "system_split": "train"},
        {"terminal_system_id": "R2", "system_split": "development_test"},
        {"terminal_system_id": "R3", "system_split": "blind_holdout"},
    ]})
    requested = []

    def capture(row, snapshots):
        requested.append(row.terminal_system_id)
        return {"station_id": row.station_id, "time_support_verified": True,
                "source_h_time_support_verified": True}

    monkeypatch.setattr(module, "_station", capture)
    audit_source_q(manifest, roles_path, tmp_path / "audit.json", tmp_path / "snapshots")
    assert requested == ["R1"]


def test_official_q_requires_correct_parameter_station_and_exact_hour():
    payload = {"value": {"timeSeries": [{
        "sourceInfo": {"siteCode": [{"value": "00000001"}]},
        "variable": {"variableCode": [{"value": "00060"}], "unit": {"unitCode": "ft3/s"},
                     "noDataValue": -999999.},
        "values": [{"value": [
            {"dateTime": "2018-10-01T00:00:00-04:00", "value": "10"},
            {"dateTime": "2018-10-01T00:15:00-04:00", "value": "200"},
            {"dateTime": "2018-10-01T01:00:00-04:00", "value": "50"},
        ]}]
    }]}}
    _assert_response_station(payload, "https://example.org/?sites=00000001")
    with pytest.raises(ValueError, match="站号"):
        _assert_response_station(payload, "https://example.org/?sites=00000002")
    result = _hourly_q(payload, "2018-10-01 04:00", "2018-10-01 05:00")
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(10 * 0.028316846592)
    invalid = copy.deepcopy(payload)
    invalid["value"]["timeSeries"][0]["variable"]["variableCode"][0]["value"] = "00065"
    with pytest.raises(ValueError, match="00060"):
        _hourly_q(invalid, "2018-10-01", "2018-10-02")


def test_scripted_warmup_matches_equations_and_chunked_state():
    torch.manual_seed(91)
    model = RegionalHBV(2)
    weather = torch.rand(2, 500, 3)
    weather[:, :, 1] = weather[:, :, 1] * 30 - 10
    geo, area = torch.rand(2, 2), torch.tensor([100., 400.])
    with torch.no_grad():
        params = model.parameters_for(geo)
        state = torch.zeros(2, 5)
        reference = _hbv_sequence(weather, state, params, area)
        actual = inference_sequence()(weather, state, params, area)
        first = model(weather[:, :211], geo, area)
        last = model(weather[:, 211:], geo, area, state=first[1])
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(torch.cat([first[0], last[0]], 1), actual[0])
    torch.testing.assert_close(last[1], actual[1])
    assert np.isfinite(actual[0].numpy()).all()
