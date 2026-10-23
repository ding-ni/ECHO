import numpy as np
import pytest
import torch

from echo.hydro.hbv import RegionalHBV
from echo.hydro.import_multibasin import import_and_generate
from echo.hydro.reference import verify_references
from echo.utils import digest, read_json, write_json


def test_import_provenance_preserves_extra_source_systems_and_rejects_weight_mutation(tmp_path):
    h_rows = []
    for i in range(4):
        path = tmp_path / f"{i}.npz"
        np.savez(path, station_id=str(i), system=f"R{i}", geo=np.zeros(2, dtype="float32"),
                 weather=np.tile([1., 15., .1], (10, 1)).astype("float32"),
                 area=np.float32(100), time_ns=np.arange(10))
        h_rows.append({"station_id": str(i), "terminal_system_id": f"R{i}",
                       "path": str(path), "sha256": digest(path),
                       "split": "train" if i < 3 else "development_test",
                       "crossfit_fold": i if i < 3 else -1})
    # R4是没有H视图、但合格的额外来源Q水系，应保留真实预训练成员而非伪装成H名单。
    q_rows = h_rows[:3] + [{"station_id": "4", "terminal_system_id": "R4",
                           "split": "train", "crossfit_fold": 0}]
    features = {"features": ["x"], "geo_center": [0.], "geo_scale": [1.]}
    q_path, h_path = tmp_path / "q.json", tmp_path / "h.json"
    write_json(q_path, {"scope": "real_source_Q_pretraining", "stations": q_rows,
                        "forbidden_systems": ["R3"], **features})
    h_index = {"scope": "real_development", "stations": h_rows, **features,
               "forbidden_systems": ["R3"], "geography_source_Q_index_sha256": digest(q_path)}
    write_json(h_path, h_index)
    source = tmp_path / "source"
    source.mkdir()
    for fold in [None, 0, 1, 2]:
        name = "hydro_full.pt" if fold is None else f"hydro_fold{fold}.pt"
        kept = {r["terminal_system_id"] for r in q_rows if fold is None or r["crossfit_fold"] != fold}
        held = {r["terminal_system_id"] for r in q_rows} - kept
        torch.save({"state": RegionalHBV(2).state_dict(), "geo_dim": 2, "config": {},
                    "trained_systems": sorted(kept), "forbidden_systems": sorted(held | {"R3"}),
                    "index_hash": digest(q_path)}, source / name)
    out = tmp_path / "import"
    result = import_and_generate(q_path, h_path, source, out)
    assert result["generated_stations"] == 4
    record = verify_references(h_index, digest(h_path), out)
    assert "R4" in record["checkpoints"]["hydro_full.pt"]["trained_systems"]
    assert "R4" not in record["checkpoints"]["hydro_fold0.pt"]["trained_systems"]
    checkpoint = torch.load(out / "hydro_full.pt", weights_only=False)
    checkpoint["state"]["local_offset.weight"] += 1
    torch.save(checkpoint, out / "hydro_full.pt")
    with pytest.raises(ValueError, match="权重"):
        verify_references(h_index, digest(h_path), out)
    changed = read_json(h_path)
    changed["geo_scale"] = [2.]
    write_json(h_path, changed)
    with pytest.raises(ValueError, match="地理统计"):
        import_and_generate(q_path, h_path, source, tmp_path / "bad_geo")
