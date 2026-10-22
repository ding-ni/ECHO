"""参考流量与来源水系、检查点及数据版本的绑定。"""

from pathlib import Path

import numpy as np
import torch

from echo.utils import digest, read_json, utc_now, write_json


def inspect_references(index, index_hash, root):
    root = Path(root)
    sources = [row for row in index["stations"] if row["split"] == "train"]
    source_systems = {row["terminal_system_id"] for row in sources}
    forbidden = set(index.get("forbidden_systems", [])) | {
        row["terminal_system_id"] for row in index["stations"] if row["split"] != "train"
    }
    if source_systems & forbidden:
        raise ValueError("目标水系进入来源训练清单")
    checkpoints = {}
    for fold in [None, 0, 1, 2]:
        name = "hydro_full.pt" if fold is None else f"hydro_fold{fold}.pt"
        checkpoint = torch.load(root / name, map_location="cpu", weights_only=False)
        held = {row["terminal_system_id"] for row in sources if row["crossfit_fold"] == fold}
        expected = source_systems - held
        trained = set(checkpoint["trained_systems"])
        if checkpoint.get("index_hash") != index_hash or trained != expected:
            raise ValueError(f"{name} 与冻结数据版本或来源折分不一致")
        if trained & (forbidden | held) or not (forbidden | held).issubset(checkpoint["forbidden_systems"]):
            raise ValueError(f"{name} 未正确排除目标及本折来源水系")
        checkpoints[name] = {"sha256": digest(root / name), "trained_systems": sorted(trained),
                             "forbidden_systems": sorted(checkpoint["forbidden_systems"])}
    records = []
    for row in index["stations"]:
        if row["split"] not in {"train", "validation", "development_test"}:
            continue
        name = f"hydro_fold{row['crossfit_fold']}.pt" if row["split"] == "train" else "hydro_full.pt"
        if row["terminal_system_id"] in checkpoints[name]["trained_systems"]:
            raise ValueError("站点参考Q由包含自身水系Q的前端生成")
        path = root / "参考流量" / (row["station_id"] + ".npy")
        q = np.load(path, allow_pickle=False)
        with np.load(row["path"], allow_pickle=False) as station:
            if digest(row["path"]) != row["sha256"]:
                raise ValueError("模型视图数据哈希变更")
            if row["split"] != "train" and ({"q", "h", "future_h"} & set(station.files)):
                raise ValueError("目标模型视图含Q或未来H")
            if q.shape != (len(station["time_ns"]),) or not np.isfinite(q).all():
                raise ValueError("参考Q长度不匹配或含非有限值")
        records.append({"station_id": row["station_id"], "terminal_system_id": row["terminal_system_id"],
                        "split": row["split"], "checkpoint": name, "sha256": digest(path),
                        "hours": len(q)})
    return {"index_sha256": index_hash, "checkpoints": checkpoints, "stations": records}


def seal_references(index, index_hash, root):
    record = inspect_references(index, index_hash, root)
    path = Path(root) / "reference_index.json"
    if path.is_file():
        previous = read_json(path)
        if any(previous.get(key) != value for key, value in record.items()):
            raise ValueError("已封存的参考Q或检查点发生变化，不能原位重新放行")
        return record
    write_json(path, {**record, "created_utc": utc_now()})
    return record


def verify_references(index, index_hash, root):
    path = Path(root) / "reference_index.json"
    if not path.is_file():
        raise ValueError("参考Q缺少检查点/来源水系绑定，先运行reference-audit")
    recorded = read_json(path)
    actual = inspect_references(index, index_hash, root)
    if any(recorded.get(key) != value for key, value in actual.items()):
        raise ValueError("参考Q或水文检查点哈希与封存记录不一致")
    return actual


def audit_references(index_path, root, replay=False):
    index = read_json(index_path)
    record = inspect_references(index, digest(index_path), root)
    replay_records = []
    if replay:
        from echo.hydro.hbv import RegionalHBV
        from echo.training.pipeline import simulate_q

        torch.set_num_threads(2)
        for row in record["stations"]:
            original = next(r for r in index["stations"] if r["station_id"] == row["station_id"])
            checkpoint = torch.load(Path(root) / row["checkpoint"], map_location="cpu", weights_only=False)
            model = RegionalHBV(checkpoint["geo_dim"])
            model.load_state_dict(checkpoint["state"])
            model.eval()
            with np.load(original["path"], allow_pickle=False) as data:
                inputs = {key: data[key] for key in ["weather", "geo", "area"]}
            if checkpoint.get("config", {}).get("no_geo"):
                inputs["geo"] = np.zeros_like(inputs["geo"])
            actual, _ = simulate_q(model, inputs, "cpu")
            cached = np.load(Path(root) / "参考流量" / (row["station_id"] + ".npy"), allow_pickle=False)
            error = float(np.max(np.abs(actual[0].numpy() - cached)))
            if not np.allclose(actual[0].numpy(), cached, rtol=1e-6, atol=1e-6):
                raise ValueError(f"{row['station_id']}参考Q重算不一致")
            replay_records.append({"station_id": row["station_id"], "max_abs_difference_m3s": error})
    seal_references(index, digest(index_path), root)
    report = {**record, "replay": replay_records, "future_H_labels_read": False,
              "checked_utc": utc_now(), "scope": "engineering_integrity_not_predictive_skill"}
    write_json(Path(root) / "reference_audit.json", report)
    return {"verified_stations": len(record["stations"]), "replayed_stations": len(replay_records),
            "future_H_labels_read": False}
