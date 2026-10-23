"""Aggregate verified original NLDAS cells onto real incremental basin polygons."""

import argparse
import hashlib
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from echo.geo.grid_weather import aggregate_weather, intersect_grid
from echo.utils import digest, read_json, utc_now, write_json


def grid_identity(lat, lon):
    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise ValueError("NLDAS需要一维规则经纬度中心网格")
    if not np.allclose(np.diff(lat), .125) or not np.allclose(np.diff(lon), .125):
        raise ValueError("原产品网格间距不是1/8度，需重新核验")
    return hashlib.sha256(lat.tobytes() + lon.tobytes()).hexdigest()


def build_weights(nodes, lat, lon):
    """Subset original cells by bounds, then perform genuine equal-area overlaps."""
    grid_identity(lat, lon)
    bounds = nodes.to_crs(4326).total_bounds
    iy = np.flatnonzero((lat + .0625 > bounds[1]) & (lat - .0625 < bounds[3]))
    ix = np.flatnonzero((lon + .0625 > bounds[0]) & (lon - .0625 < bounds[2]))
    if not len(iy) or not len(ix):
        raise ValueError("站点位于NLDAS原网格范围之外")
    cell_ids, geometries = [], []
    for i in iy:
        for j in ix:
            cell_ids.append(int(i * len(lon) + j))
            geometries.append(box(lon[j] - .0625, lat[i] - .0625, lon[j] + .0625, lat[i] + .0625))
    cells = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=geometries, crs=4326)
    weights, coverages = intersect_grid(nodes, cells)
    return weights, np.asarray(cell_ids, dtype="int64"), coverages


def read_hour(path, cell_ids, expected_grid_hash):
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        if grid_identity(ds.lat.values, ds.lon.values) != expected_grid_hash:
            raise ValueError("原网格发生改变，不能复用权重")
        if ds.sizes.get("time") != 1:
            raise ValueError("原产品每文件应只有一个小时")
        end = ds.time.values.astype("datetime64[ns]").reshape(-1)[0]
        bounds = ds.time_bnds.values.astype("datetime64[ns]").reshape(-1)
        if len(bounds) != 2 or bounds[1] != end or bounds[1] - bounds[0] != np.timedelta64(1, "h"):
            raise ValueError("NLDAS不是已核验的向后1小时累计时间支持")
        arrays = {}
        for name, unit, method in [("Rainf", "kg m-2", "time: sum"),
                                   ("Tair", "K", "time: point"),
                                   ("PotEvap", "kg m-2", "time: sum")]:
            var = ds[name]
            if var.attrs.get("units") != unit or var.attrs.get("cell_methods") != method:
                raise ValueError(f"{name}单位或时间支持与已核验映射不符")
            if var.dims != ("time", "lat", "lon"):
                raise ValueError("强迫维度顺序不符，不能静默转置网格")
            arrays[name] = var.values[0].reshape(-1)[cell_ids].astype("float64")
        if any(not np.isfinite(a).all() for a in arrays.values()) or (arrays["Rainf"] < 0).any():
            raise ValueError("所用气象格点存在缺测或负降水")
        signed_pet = arrays["PotEvap"]
        # 1 kg m-2 water = 1 mm; one-hour sums are mm per one-hour step.
        forcing = np.stack([arrays["Rainf"], arrays["Tair"] - 273.15, np.maximum(signed_pet, 0)], -1)
        return end.astype("int64"), forcing, signed_pet


def prepare(graph_index, raw_root, output, station_ids=None):
    graph_index, output = Path(graph_index), Path(output)
    if output.exists():
        raise ValueError("输出版本已存在，请新建输出目录")
    records = []
    for receipt_path in sorted(Path(raw_root).rglob("*.nc.receipt.json")):
        receipt = read_json(receipt_path)
        if receipt.get("status") != "downloaded_netcdf_verified":
            continue
        path = receipt_path.with_name(receipt_path.name.removesuffix(".receipt.json"))
        if not path.exists() or path.stat().st_size != receipt["bytes"] or digest(path) != receipt["sha256"]:
            raise ValueError("原网格文件缺失或与下载收据不一致")
        records.append((receipt["time_values"][0], path, receipt["sha256"]))
    records.sort()
    if not records:
        raise ValueError("没有经过完整文件核验的原网格气象")
    source_times = np.array([r[0] for r in records], dtype="datetime64[ns]").astype("int64")
    if len(source_times) > 1 and not np.all(np.diff(source_times) == 3600 * 10**9):
        raise ValueError("原始小时文件有重复或间断；请先补齐，不静默填零")
    with xr.open_dataset(records[0][1], engine="h5netcdf") as first:
        lat, lon = first.lat.values, first.lon.values
        grid_hash = grid_identity(lat, lon)
    graph = read_json(graph_index)
    selected = [r for r in graph["stations"] if station_ids is None or r["station_id"] in station_ids]
    if station_ids is not None and {r["station_id"] for r in selected} != set(station_ids):
        raise ValueError("选定站点在真实河网清单中不完整")
    output.mkdir(parents=True)
    prepared = []
    for row in selected:
        sid = row["station_id"]
        if digest(row["geometry_path"]) != row["geometry_sha256"]:
            raise ValueError("真实流域几何与版本收据不一致")
        nodes = gpd.read_file(row["geometry_path"]).sort_values("node_id").reset_index(drop=True)
        weights, cells, coverage = build_weights(nodes, lat, lon)
        all_forcing, all_signed_pet, times = [], [], []
        for _, path, _ in records:
            stamp, forcing, signed_pet = read_hour(path, cells, grid_hash)
            all_forcing.append(aggregate_weather(weights, forcing[None])[:, 0])
            all_signed_pet.append(np.asarray(weights @ signed_pet))
            times.append(stamp)
        if not np.array_equal(times, source_times):
            raise ValueError("NetCDF文件时间与下载收据不符")
        target = output / f"{sid}.npz"
        np.savez_compressed(
            target, weather_nodes=np.stack(all_forcing, 1),
            signed_PET_nodes=np.stack(all_signed_pet, 1).astype("float32"),
            time_ns=np.asarray(times, dtype="int64"),
            interval_start_ns=np.asarray(times, dtype="int64") - 3600 * 10**9,
            node_areas=nodes.area_km2.to_numpy(), grid_cell_ids=cells,
            weight_data=weights.data, weight_indices=weights.indices, weight_indptr=weights.indptr,
            weight_shape=np.asarray(weights.shape), node_grid_coverage=coverage,
        )
        prepared.append({"station_id": sid, "split": row["split"],
                         "terminal_system_id": row["terminal_system_id"],
                         "path": str(target.resolve()), "sha256": digest(target),
                         "nodes": len(nodes), "hours": len(times),
                         "effective_original_grid_cells": len(np.unique(weights.indices)),
                         "catchment_unmapped_fraction": row["unmapped_area_fraction"],
                         "minimum_node_weather_coverage": float(coverage.min()),
                         "training_ready": False})
        print(f"grid_aggregate station={sid} nodes={len(nodes)} hours={len(times)}", flush=True)
    result = {"created_utc": utc_now(), "graph_index_sha256": digest(graph_index),
              "source_files": [{"path": str(p.resolve()), "sha256": h} for _, p, h in records],
              "source_grid_sha256": grid_hash, "stations": prepared,
              "weather_kind": "incremental_area_weighted_original_grid",
              "channels": ["P_mm_per_hour", "T_C_at_interval_end", "PET_demand_mm_per_hour"],
              "time_label": "UTC_interval_end", "precipitation_and_PET": "backward_hour_sum",
              "signed_PET_preserved": True,
              "PET_demand_policy": "max_signed_original_cell_PET_with_zero_before_area_aggregation",
              "source_reference": "NASA NLDAS2 README Table3 and verified NetCDF time_bnds/cell_methods",
              "complete_calendar_warmup_available": False, "protocol_training_ready": False,
              "target_Q_read": False, "H_values_read": False}
    write_json(output / "index.json", result)
    return {"stations": len(prepared), "hours": len(source_times), "protocol_training_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-index", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--station-ids", nargs="+")
    args = parser.parse_args()
    print(prepare(args.graph_index, args.raw_root, args.output, args.station_ids))
