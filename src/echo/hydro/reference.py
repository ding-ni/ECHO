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
    imported = None
    import_path = root / "pretrained_import.json"
    if import_path.is_file():
        imported = read_json(import_path)
        if imported.get("target_index_sha256") != index_hash:
            raise ValueError("多流域预训练导入记录与水位数据清单不一致")
        q_path = Path(imported["source_index_path"])
        if not q_path.is_file() or digest(q_path) != imported["source_index_sha256"]:
            raise ValueError("多流域预训练来源Q数据清单缺失或哈希改变")
        q_index = read_json(q_path)
        if q_index.get("scope") != "real_source_Q_pretraining":
            raise ValueError("导入记录没有绑定来源Q专用数据清单")
    for fold in [None, 0, 1, 2]:
        name = "hydro_full.pt" if fold is None else f"hydro_fold{fold}.pt"
        checkpoint = torch.load(root / name, map_location="cpu", weights_only=False)
        held = {row["terminal_system_id"] for row in sources if row["crossfit_fold"] == fold}
        expected = source_systems - held
        trained = set(checkpoint["trained_systems"])
        provenance = checkpoint.get("pretrained_import")
        if provenance:
            if imported is None or provenance != imported["checkpoints"][name]:
                raise ValueError(f"{name} 的多流域导入来源记录不匹配")
            source_path = Path(provenance["source_path"])
            if not source_path.is_file() or digest(source_path) != provenance["source_sha256"]:
                raise ValueError(f"{name} 的原始多流域检查点缺失或哈希改变")
            source_trained = set(provenance["trained_systems"])
            source_forbidden = set(provenance["forbidden_systems"])
            source_held = set(provenance["source_fold_systems"].get(str(fold), [])) if fold is not None else set()
            q_sources = q_index["stations"]
            expected_source = {r["terminal_system_id"] for r in q_sources
                               if fold is None or r["crossfit_fold"] != fold}
            original = torch.load(source_path, map_location="cpu", weights_only=False)
            if (trained != expected_source or any(r["split"] != "train" for r in q_sources)
                    or trained & set(q_index["forbidden_systems"])
                    or original.get("index_hash") != imported["source_index_sha256"]
                    or set(original["state"]) != set(checkpoint["state"])
                    or any(not torch.equal(value, checkpoint["state"][key])
                           for key, value in original["state"].items())):
                raise ValueError("导入检查点权重或训练水系与原始来源Q预训练不一致")
            if (checkpoint.get("index_hash") != index_hash or trained != source_trained
                    or source_trained & (source_forbidden | source_held)
                    or not (forbidden | held).issubset(checkpoint["forbidden_systems"])):
                raise ValueError(f"{name} 未通过来源Q水系和目标水系双重隔离核验")
            if fold is not None and not held.issubset(source_held):
                raise ValueError(f"{name} 的来源Q折分未排除当前水位训练水系")
        elif checkpoint.get("index_hash") != index_hash or trained != expected:
            raise ValueError(f"{name} 与冻结数据版本或来源折分不一致")
        elif trained & (forbidden | held) or not (forbidden | held).issubset(checkpoint["forbidden_systems"]):
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
    result = {"index_sha256": index_hash, "checkpoints": checkpoints, "stations": records}
    if imported is not None:
        result["pretrained_import_sha256"] = digest(import_path)
    return result


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
