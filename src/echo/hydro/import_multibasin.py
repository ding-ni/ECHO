"""将已训练的来源Q多流域HBV接入水位视图并生成留水系参考Q。"""

from pathlib import Path

import numpy as np
import torch

from echo.data.permissions import assert_source_permissions
from echo.hydro.hbv import RegionalHBV
from echo.utils import digest, read_json, utc_now, write_json


def import_and_generate(q_index_path, h_index_path, source_run, output):
    q_index_path, h_index_path = Path(q_index_path), Path(h_index_path)
    source_run, output = Path(source_run), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("多流域HBV接入目录非空，请使用新的运行编号")
    q_index, h_index = read_json(q_index_path), read_json(h_index_path)
    if q_index.get("scope") != "real_source_Q_pretraining" or h_index.get("scope") != "real_development":
        raise ValueError("输入必须分别是来源Q预训练视图和真实水位开发视图")
    q_sources = [r for r in q_index["stations"] if r["split"] == "train"]
    h_sources = [r for r in h_index["stations"] if r["split"] == "train"]
    q_system_fold = {}
    for row in q_sources:
        system, fold = row["terminal_system_id"], int(row["crossfit_fold"])
        if system in q_system_fold and q_system_fold[system] != fold:
            raise ValueError("来源Q同一水系被分配到多个折")
        q_system_fold[system] = fold
    h_system_fold = {}
    for row in h_sources:
        system, fold = row["terminal_system_id"], int(row["crossfit_fold"])
        if system not in q_system_fold or q_system_fold[system] != fold:
            raise ValueError(f"水位来源水系 {system} 未与来源Q训练水系及折号对齐")
        h_system_fold[system] = fold

    q_forbidden = set(q_index.get("forbidden_systems", []))
    h_forbidden = set(h_index.get("forbidden_systems", [])) | {
        r["terminal_system_id"] for r in h_index["stations"] if r["split"] != "train"
    }
    assert_source_permissions(h_system_fold, h_forbidden)
    q_fold_systems = {
        str(fold): sorted(system for system, assigned in q_system_fold.items() if assigned == fold)
        for fold in range(3)
    }
    q_train_systems = set(q_system_fold)
    expected = {
        "hydro_full.pt": q_train_systems,
        **{f"hydro_fold{fold}.pt": q_train_systems - set(q_fold_systems[str(fold)]) for fold in range(3)},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "参考流量").mkdir()
    imported = {}
    for name, expected_systems in expected.items():
        source_path = source_run / name
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        trained = set(checkpoint.get("trained_systems", []))
        source_forbidden = set(checkpoint.get("forbidden_systems", []))
        if (checkpoint.get("index_hash") != digest(q_index_path)
                or trained != expected_systems
                or trained & (q_forbidden | source_forbidden)
                or not (q_forbidden | (q_train_systems - trained)).issubset(source_forbidden)):
            raise ValueError(f"{name} 与来源Q清单、水系折分或训练排除记录不一致")
        source_fold_systems = q_fold_systems if name == "hydro_full.pt" else {
            str(fold): q_fold_systems[str(fold)] for fold in range(3)
        }
        provenance = {
            "source_path": str(source_path.resolve()),
            "source_sha256": digest(source_path),
            "source_index_path": str(q_index_path.resolve()),
            "source_index_sha256": digest(q_index_path),
            "trained_systems": sorted(trained),
            "forbidden_systems": sorted(source_forbidden),
            "source_fold_systems": source_fold_systems,
        }
        imported[name] = provenance
        torch.save({**checkpoint, "index_hash": digest(h_index_path), "pretrained_import": provenance}, output / name)

    import_record = {
        "source_index_path": str(q_index_path.resolve()),
        "source_index_sha256": digest(q_index_path),
        "target_index_path": str(h_index_path.resolve()),
        "target_index_sha256": digest(h_index_path),
        "checkpoints": imported,
        "q_source_stations": len(q_sources),
        "q_source_systems": len(q_train_systems),
        "h_source_stations": len(h_sources),
        "h_source_systems": len(h_system_fold),
        "created_utc": utc_now(),
        "future_H_labels_read": False,
    }
    write_json(output / "pretrained_import.json", import_record)

    from echo.training.pipeline import load_station, simulate_q

    torch.set_num_threads(2)
    generated = []
    model_cache = {}
    for row in h_index["stations"]:
        if row["split"] not in {"train", "validation", "development_test"}:
            continue
        name = f"hydro_fold{row['crossfit_fold']}.pt" if row["split"] == "train" else "hydro_full.pt"
        if name not in model_cache:
            checkpoint = torch.load(output / name, map_location="cpu", weights_only=False)
            model = RegionalHBV(checkpoint["geo_dim"])
            model.load_state_dict(checkpoint["state"])
            model.eval().requires_grad_(False)
            model_cache[name] = model
        if row["split"] == "train" and row["terminal_system_id"] in set(imported[name]["trained_systems"]):
            raise ValueError("留折模型包含当前站点所属水系")
        data = load_station(row)
        q, _ = simulate_q(model_cache[name], data, "cpu")
        path = output / "参考流量" / f"{row['station_id']}.npy"
        np.save(path, q[0].numpy())
        generated.append({"station_id": row["station_id"], "system": row["terminal_system_id"],
                          "split": row["split"], "checkpoint": name, "hours": int(q.shape[1]),
                          "sha256": digest(path)})
        print(f"参考Q {len(generated)}/{len(h_index['stations'])}: {row['station_id']} {name}", flush=True)

    from echo.hydro.reference import seal_references, verify_references

    seal_references(h_index, digest(h_index_path), output)
    verify_references(h_index, digest(h_index_path), output)
    write_json(output / "generation.json", {
        **import_record,
        "generated_stations": generated,
        "reference_index_sha256": digest(output / "reference_index.json"),
        "complete": True,
    })
    return {"generated_stations": len(generated), "h_source_systems": len(h_system_fold),
            "q_source_systems": len(q_train_systems), "output": str(output)}
