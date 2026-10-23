"""Live evidence inventory: raw assets, derived inputs and full readiness are distinct."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from echo.utils import read_json, utc_now, write_json


def _read(path, default):
    return read_json(path) if path.exists() else default


def inventory(root, output):
    root, output = Path(root), Path(output)
    data = root / "本地数据"
    hindex = _read(data / "模型视图/扩展水位训练_v2/index.json", {"stations": []})
    graphs = _read(data / "模型视图/真实分支河网_v1/index.json", {"stations": []})
    env = _read(data / "模型视图/独立静态与历史气候_v1/index.json", {"stations": []})
    weather = _read(data / "模型视图/真实原网格气象_首日_v1/index.json", {"stations": []})
    dem_root = data / "原始/3DEP_本地八站_v1"
    dem_selected = _read(dem_root / "selected_assets.json", {"assets": [], "station_ids": []})
    dem_receipts = list(dem_root.glob("*.receipt.json"))
    nasa_root = data / "原始/NLDAS_FORA0125_H_v2"
    nasa_receipts = list(nasa_root.rglob("*.nc.receipt.json"))
    progress = _read(nasa_root / "current_progress.json", {})
    by_graph = {r["station_id"]: r for r in graphs["stations"]}
    by_env = {r["station_id"]: r for r in env["stations"]}
    by_weather = {r["station_id"]: r for r in weather["stations"]}
    records = []
    for row in hindex["stations"]:
        sid = row["station_id"]
        graph, environmental, forcing = by_graph.get(sid, {}), by_env.get(sid, {}), by_weather.get(sid, {})
        with np.load(row["path"], allow_pickle=False) as archive:
            times = archive["time_ns"]
            keys = set(archive.files)
            if row["split"] != "train" and (keys & {"q", "h", "future_h", "target_q"}):
                raise ValueError("非来源模型视图出现完整H或实测Q字段")
            start, end = pd.Timestamp(times[0], tz="UTC"), pd.Timestamp(times[-1], tz="UTC")
        missing = ["verified_incremental_soil_geology_landcover", "mapped_river_patch_centres",
                   "DEM_valley_tokens_and_vertical_reference", "edge_DEM_drop_and_slope",
                   "full_calendar_original_grid_weather", "staged_training_and_official_backbone_execution"]
        if not graph:
            missing.append("true_incremental_graph")
        if not environmental:
            missing.append("independent_attributes_and_preorigin_climate")
        if graph.get("unmapped_area_fraction", 0) > .02:
            missing.append("catchment_geometry_reconciliation")
        records.append({"station_id": sid, "split": row["split"],
                        "terminal_system_id": row["terminal_system_id"],
                        "basin_mean_weather_start_utc": start.isoformat(),
                        "basin_mean_weather_last_label_utc": end.isoformat(),
                        "basin_mean_weather_hours": len(times),
                        "H_role_view_available": True, "true_graph_available": bool(graph),
                        "nodes": graph.get("nodes"), "catchment_unmapped_fraction": graph.get("unmapped_area_fraction"),
                        "basin_attribute_snapshot_available": bool(environmental),
                        "climate_code_dimension": environmental.get("climate_dimension", 0),
                        "glacier_attribute_status": environmental.get("glacier_attribute_status", "unknown"),
                        "real_node_weather_hours": forcing.get("hours", 0),
                        "effective_weather_grid_cells": forcing.get("effective_original_grid_cells", 0),
                        "protocol_training_ready": False, "missing": missing})
    result = {"inventory_created_utc": utc_now(), "inventory_is_observation_not_dataset_version": True,
              "stations": records, "H_view_sites": len(records),
              "terminal_river_systems": len({r["terminal_system_id"] for r in records}),
              "true_graph_sites": len(graphs["stations"]), "independent_environment_sites": len(env["stations"]),
              "real_original_grid_node_weather_sites": len(weather["stations"]),
              "NLDAS_verified_hour_files": len(nasa_receipts), "NLDAS_progress": progress,
              "DEM_required_tiles": len(dem_selected["assets"]), "DEM_receipted_complete_tiles": len(dem_receipts),
              "DEM_incomplete_prefix_bytes": sum(p.stat().st_size for p in dem_root.glob("*.part")),
              "glacier_positive_archive_sites": env.get("positive_glacier_attribute_stations"),
              "full_model_ready_sites": 0,
              "target_Q_read": False, "H_values_read": False, "credentials_read": False}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "资料状态.json", result)
    pd.DataFrame([{**r, "missing": ";".join(r["missing"])} for r in records]).to_csv(
        output / "逐站完整资料缺项.csv", index=False, encoding="utf-8-sig")
    return {k: v for k, v in result.items() if k not in {"stations", "NLDAS_progress"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("本地数据/审计/完整资料总账_v1"))
    args = parser.parse_args()
    print(inventory(args.root, args.output))
