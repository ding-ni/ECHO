"""依据固定水系角色和逐站证据生成试运行清单，不手写站号/折号。"""

from pathlib import Path

import pandas as pd

from echo.data.datum import validate_registry_station
from echo.data.splits import build_splits
from echo.utils import digest, read_json, write_json


def build_pilot(manifest, roles_path, datum_registry, height_report, source_report, output, max_sites=10):
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    roles = read_json(roles_path)
    if roles["manifest_sha256"] != digest(manifest):
        raise ValueError("候选清单与冻结角色的输入哈希不同")
    by_system = {r["terminal_system_id"]: r["system_split"] for r in roles["systems"]}
    height = {r["station_id"]: r for r in read_json(height_report)["stations"]}
    source_document = read_json(source_report)
    if source_document["roles_sha256"] != digest(roles_path):
        raise ValueError("来源观测核验水系角色不一致")
    sources = {r["station_id"]: r for r in source_document["stations"]}
    frame["time_support_verified"] = False
    frame["source_time_support_verified"] = False
    frame["datum_verified"] = False
    frame["datum_id"] = "unknown"
    frame["datum_evidence"] = ""
    frame["release_reason"] = "not_in_fixed_H_review"
    for i, row in frame[frame.station_id.isin(height)].iterrows():
        sid, role = row.station_id, by_system.get(row.terminal_system_id, "excluded")
        h_ok = height[sid].get("time_support_verified") is True
        source = sources.get(sid, {})
        qh_ok = (source.get("time_support_verified") is True
                 and source.get("source_h_time_support_verified") is True)
        frame.loc[i, "source_time_support_verified"] = qh_ok
        frame.loc[i, "time_support_verified"] = qh_ok if role == "train" else h_ok
        start, end = ("2013-10-01", "2019-10-01") if role == "train" else (
            "2019-10-01", "2021-10-01"
        )
        try:
            frame.loc[i, "datum_id"] = validate_registry_station(datum_registry, sid, start, end)
            frame.loc[i, "datum_verified"] = True
            frame.loc[i, "datum_evidence"] = str(Path(datum_registry).resolve()) + "#sha256=" + digest(datum_registry)
            frame.loc[i, "release_reason"] = "metadata_passed" if frame.loc[i, "time_support_verified"] else (
                "source_QH_time_unknown" if role == "train" else "support_H_time_unknown"
            )
        except ValueError as exc:
            frame.loc[i, "release_reason"] = str(exc)
    result = build_splits(frame, max_sites=max_sites, system_roles=roles)
    selected = result[result.split.isin(["train", "validation", "development_test"])]
    train = selected[selected.split == "train"]
    if len(train.terminal_system_id.unique()) < 3 or set(train.crossfit_fold) != {0, 1, 2}:
        raise ValueError("合格来源未覆盖冻结三折；应补证而非重新随机分折")
    if not {"validation", "development_test"}.issubset(set(selected.split)):
        raise ValueError("固定角色中缺少合格的验证或开发评估站，应继续补证")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "scope": "real_pilot_metadata_release_not_performance",
        "max_sites": max_sites,
        "split_counts": selected.split.value_counts().to_dict(),
        "selected_stations": selected[["station_id", "terminal_system_id", "split", "crossfit_fold"]].to_dict("records"),
        "inputs": {str(Path(p).resolve()): digest(p) for p in [
            manifest, roles_path, datum_registry, height_report, source_report
        ]},
        "manifest_sha256": digest(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
