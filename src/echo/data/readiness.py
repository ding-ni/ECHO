"""按冻结水系角色汇总扩容待办与迁移清单，不读取目标Q或未来H数值。"""

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from echo.data.datum import validate_registry_station
from echo.utils import ROOT, digest, read_json, utc_now, write_json


def build(output):
    output = Path(output)
    if output.exists():
        raise ValueError("准备清单目录已存在，请另建版本，保留旧快照")
    output.mkdir(parents=True)
    manifest = ROOT / "本地数据/审计/候选站点_边界复核.csv"
    roles_path = ROOT / "01_数据准备/固定水系角色_v1.json"
    q_path = ROOT / "本地数据/模型视图/多流域Q预训练_v1/index.json"
    h_path = ROOT / "本地数据/模型视图/扩展水位训练_v2/index.json"
    registry = ROOT / "01_数据准备/历史水尺基准登记_扩展来源_v2.json"
    source_evidence = ROOT / "本地数据/审计/USGS扩展来源QH核验_合并_v2.json"
    height_evidence = ROOT / "本地数据/审计/USGS支持期水位核验.json"
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    roles, q_index, h_index = read_json(roles_path), read_json(q_path), read_json(h_path)
    if digest(manifest) != roles["manifest_sha256"]:
        raise ValueError("冻结水系清单来源不一致")
    role_by_system = {r["terminal_system_id"]: r for r in roles["systems"]}
    q_by_id = {r["station_id"]: r for r in q_index["stations"]}
    h_by_id = {r["station_id"]: r for r in h_index["stations"]}
    for row in q_index["stations"]:
        if row["split"] != "train" or role_by_system[row["terminal_system_id"]]["system_split"] != "train":
            raise ValueError("迁移来源Q视图含禁用水系")
    for row in h_index["stations"]:
        if role_by_system[row["terminal_system_id"]]["system_split"] != row["split"]:
            raise ValueError("水位视图与冻结水系角色不一致")
        if row["split"] != "train":
            with np.load(row["path"], allow_pickle=False) as data:
                if {"q", "h", "future_h"} & set(data.files):
                    raise ValueError("迁移目标上下文含Q或未来H")
    sources = {r["station_id"]: r for r in read_json(source_evidence)["stations"]}
    heights = {r["station_id"]: r for r in read_json(height_evidence)["stations"]}
    rows = []
    for row in frame.itertuples():
        role = role_by_system.get(row.terminal_system_id)
        if role is None or not row.weather_available or not row.topology_verified or not 20 <= row.area_km2 <= 5000:
            continue
        split, sid = role["system_split"], row.station_id
        h_coverage = row.support_coverage >= .8 and row.query_coverage >= .9
        if split == "blind_holdout":
            # 只保留冻结角色与原有覆盖元数据，不读取盲测时序或核验测流。
            time_ok, datum_ok, datum_reason = False, False, "blind_holdout_not_opened"
        else:
            source = sources.get(sid, {})
            time_ok = ((source.get("time_support_verified") is True
                        and source.get("source_h_time_support_verified") is True) if split == "train"
                       else heights.get(sid, {}).get("time_support_verified") is True)
            try:
                start, stop = ("2013-10-01", "2019-10-01") if split == "train" else ("2019-10-01", "2021-10-01")
                validate_registry_station(registry, sid, start, stop)
                datum_ok, datum_reason = True, ""
            except ValueError as exc:
                datum_ok, datum_reason = False, str(exc)
        tasks = []
        if split == "blind_holdout":
            tasks.append("keep_blind_frozen")
        else:
            if not time_ok:
                tasks.append("source_QH_time_reconciliation" if split == "train" else "target_support_H_time_reconciliation")
            if not datum_ok:
                tasks.append("historical_gage_epoch_review")
            if sid not in h_by_id:
                tasks.append("source_H_episode_coverage_and_prepare" if split == "train" else "target_H_coverage_and_prepare")
        rows.append({"station_id": sid, "terminal_system_id": row.terminal_system_id,
                     "fixed_role": split, "crossfit_fold": role["crossfit_fold"],
                     "Q_view_ready": sid in q_by_id, "H_view_ready": sid in h_by_id,
                     "original_target_H_coverage_gate": h_coverage,
                     "time_evidence_ready": time_ok, "datum_evidence_ready": datum_ok,
                     "datum_remaining_reason": datum_reason, "next_tasks": ";".join(tasks)})
    table = pd.DataFrame(rows).sort_values(["fixed_role", "terminal_system_id", "station_id"])
    table.to_csv(output / "逐站扩容待办.csv", index=False, encoding="utf-8-sig")

    # 仅列入当前可运行模型视图及参考权重；本清单不上传数据，也不打包26GB原档案。
    files = {}
    def add(path, use, expected=None):
        path = Path(path).resolve()
        relative = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"迁移清单数据哈希变更: {relative}")
        files[relative] = {"path": relative, "sha256": actual, "bytes": path.stat().st_size, "use": use}
    inputs = [manifest, roles_path, q_path, h_path, registry, source_evidence, height_evidence]
    for path in inputs:
        add(path, "metadata")
    for record in read_json(registry)["stations"]:
        snapshot = (registry.parent / record["snapshot"]).resolve()
        add(snapshot, "historical_gage_evidence", record["snapshot_sha256"])
        add(snapshot.with_suffix(".provenance.json"), "historical_gage_evidence_receipt")
    for index, use in [(q_index, "source_Q_only_training"), (h_index, "H_training_or_target_context")]:
        for row in index["stations"]:
            add(row["path"], use, row["sha256"])
    for path in (h_path.parent / "评分标签").glob("*.parquet"):
        add(path, "scoring_only_do_not_load_in_training")
    hydro = ROOT / "本地运行/扩展水位训练_v2/区域水文_多流域"
    for path in hydro.rglob("*"):
        if path.is_file():
            add(path, "sealed_HBV_and_reference_Q")
    original = ROOT / "本地运行/多流域Q预训练_v1/区域水文"
    for name in ["hydro_full.pt", "hydro_fold0.pt", "hydro_fold1.pt", "hydro_fold2.pt"]:
        add(original / name, "original_pretraining_provenance")
    runs = ROOT / "本地运行/扩展水位训练_v2"
    for directory in ["直接水位LSTM", "ECHO静态_多流域", "ECHO动态_多流域", "ECHO联合推断_本地小样例_v1"]:
        for name in ["model.pt", "metadata.json", "history.json", "start_metadata.json"]:
            path = runs / directory / name
            if path.is_file():
                add(path, "completed_local_model_and_training_metadata")
    result = {"created_utc": utc_now(), "scope": "data_preparation_metadata_not_model_selection",
              "candidate_rows": len(table),
              "target_H_coverage_candidate_rows": int(table.original_target_H_coverage_gate.sum()),
              "fixed_roles": dict(Counter(table.fixed_role)),
              "Q_ready_sites": len(q_index["stations"]),
              "Q_ready_systems": len({r["terminal_system_id"] for r in q_index["stations"]}),
              "H_ready_sites": len(h_index["stations"]),
              "H_ready_sources": len([r for r in h_index["stations"] if r["split"] == "train"]),
              "target_Q_read": False, "future_H_values_read": False,
              "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
              "next_tasks": dict(Counter(task for r in rows for task in r["next_tasks"].split(";") if task)),
              "transfer_files": len(files), "transfer_bytes": sum(f["bytes"] for f in files.values())}
    write_json(output / "数据准备汇总.json", result)
    write_json(output / "服务器迁移文件清单.json", {
        "schema_version": 1, "root": ".", "files": sorted(files.values(), key=lambda r: r["path"]),
        "absolute_paths_inside_indexes_require_explicit_rebinding": True,
        "note": "仅为数据资产与哈希清单；原索引和权重包含Windows路径，迁移需显式重绑定并重验，不能直接宣称Linux可运行",
        "blind_timeseries_included": False,
    })
    return result


def verify_transfer(manifest, root):
    root = Path(root).resolve()
    records = read_json(manifest)
    files = records["files"]
    seen, total = set(), 0
    for row in files:
        name = row["path"]
        relative = PurePosixPath(name)
        if (relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts
                or "\\" in name or name in seen):
            raise ValueError("迁移清单含绝对路径、目录跳转或重复文件")
        seen.add(name)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"迁移文件缺失或越出根目录: {name}")
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"迁移文件大小或哈希不一致: {name}")
        total += row["bytes"]
    return {"verified_files": len(files), "verified_bytes": total,
            "absolute_path_rebinding_done": False,
            "note": "数据资产完整性通过不代表Windows索引已迁移或Linux环境已验证"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(verify_transfer(args.verify, args.root) if args.verify else build(args.output))
