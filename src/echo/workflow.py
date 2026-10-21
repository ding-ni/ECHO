"""下载完成后的本地数据接续任务；未核验语义时停在审计，不自动训练。"""

import argparse
import time

from echo.data.archive import extract_verified
from echo.data.audit import audit_tree
from echo.data.inventory import build_inventory
from echo.utils import ROOT, utc_now, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait", action="store_true")
    p.add_argument("--max-wait-hours", type=float, default=24.0)
    a = p.parse_args()
    raw = ROOT / "本地数据/原始"
    tasks = [(raw / "QH独立下载/Hourly2.zip", raw / "qh"), (raw / "下载/timeseries.7z", raw / "weather")]
    state = ROOT / "本地运行/阶段状态.json"
    done = set()
    started = time.monotonic()
    while len(done) < len(tasks):
        for archive, target in tasks:
            if target in done:
                continue
            if archive.exists() and archive.with_name(archive.name + ".audit.json").exists():
                write_json(state, {"status": "extracting", "asset": archive.name, "updated_utc": utc_now()})
                extract_verified(archive, target)
                write_json(ROOT / "本地运行" / (target.name + "_schema.json"), audit_tree(target, 20))
                done.add(target)
                if target.name == "qh":
                    early_inventory = ROOT / "本地数据/审计/QH候选站点.csv"
                    if not early_inventory.exists():
                        write_json(
                            state,
                            {
                                "status": "auditing_complete_QH_archive",
                                "updated_utc": utc_now(),
                                "real_training_complete": False,
                            },
                        )
                        build_inventory(target, raw / "weather", raw / "attributes", early_inventory)
                        from echo.data.coverage_report import render_coverage

                        render_coverage(early_inventory, ROOT / "本地运行/完整QH数据覆盖")
        if len(done) < len(tasks):
            write_json(
                state,
                {
                    "status": "waiting_for_verified_archives",
                    "completed": [p.name for p in done],
                    "pending": [p.name for _, p in tasks if p not in done],
                    "updated_utc": utc_now(),
                    "real_training_complete": False,
                },
            )
            if not a.wait:
                return
            if time.monotonic() - started > a.max_wait_hours * 3600:
                write_json(
                    state,
                    {
                        "status": "download_wait_timeout",
                        "real_training_complete": False,
                        "updated_utc": utc_now(),
                        "action": "检查下载状态后重新运行接续审计",
                    },
                )
                return
            time.sleep(30)
    result = build_inventory(
        raw / "qh", raw / "weather", raw / "attributes", ROOT / "本地数据/审计/候选站点.csv"
    )
    topo_root = raw / "HydroBASINS/hybas_na_lev12_v1c"
    if (topo_root / "complete.json").exists():
        import pandas as pd

        from echo.geo.hydrobasins import map_terminals

        candidate_path = ROOT / "本地数据/审计/候选站点.csv"
        topology_path = ROOT / "本地数据/审计/终端水系.csv"
        result["topology"] = map_terminals(next(topo_root.rglob("*.shp")), candidate_path, topology_path)
        candidates = pd.read_csv(candidate_path, dtype={"station_id": str})
        topology = pd.read_csv(topology_path, dtype={"station_id": str})
        candidates = candidates.drop(columns=["terminal_system_id", "topology_evidence"]).merge(
            topology, on="station_id", how="left"
        )
        candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    write_json(
        state,
        {
            "status": "awaiting_semantic_and_topology_verification",
            "inventory": result,
            "real_training_complete": False,
            "updated_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
