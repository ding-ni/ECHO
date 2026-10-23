"""Build auditable small-sample spatial inputs from HydroBASINS and CAMELS metadata."""

from collections import deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from echo.utils import digest, read_json, utc_now, write_json

NODE_FEATURES = ["log_incremental_area", "log_upstream_area", "flow_distance_band",
                 "relative_centroid_x_10km", "relative_centroid_y_10km", "mean_elevation_m",
                 "mean_slope_degrees", "basin_relief_m"]
TERRAIN_FEATURES = ["relative_x_10km", "relative_y_10km", "log_incremental_area",
                    "flow_distance_band", "neighborhood_1km", "neighborhood_5km",
                    "neighborhood_10km", "basin_mean_elevation_m", "basin_mean_slope_degrees",
                    "basin_relief_m"]


def climate_vector(weather, times_ns, split, support_end):
    times = pd.to_datetime(times_ns, utc=True)
    limit = pd.Timestamp("2018-10-01", tz="UTC") if split == "train" else pd.Timestamp(support_end, tz="UTC")
    valid = times < limit
    if not valid.any():
        raise ValueError("没有可用历史气象构造气候环境代码")
    values, months = np.asarray(weather)[valid], times.month.to_numpy()[valid]
    pieces = []
    for month in range(1, 13):
        selected = values[months == month]
        pieces.extend(np.nanmean(selected, axis=0).tolist() if len(selected) else [0., 0., 0.])
    annual = np.concatenate([np.nanmean(values, 0), np.nanstd(values, 0), np.nanquantile(values, .9, axis=0)])
    vector = np.nan_to_num(np.concatenate([pieces, annual]), nan=0., posinf=0., neginf=0.)
    return np.pad(vector, (0, max(0, 64 - len(vector))))[:64].astype("float32")


def upstream_depth(local_id, downstream):
    upstream = {}
    for node, parent in downstream.items():
        if parent:
            upstream.setdefault(parent, []).append(node)
    depth, queue = {int(local_id): 0}, deque([int(local_id)])
    while queue:
        current = queue.popleft()
        for child in upstream.get(current, []):
            if child not in depth:
                depth[child] = depth[current] + 1
                queue.append(child)
    return depth


def coarsen_nodes(depth, basin, outlet_xy, topo, node_count=8):
    maximum = max(depth.values())
    assigned = {}
    for hybas_id, level in depth.items():
        band = min(node_count - 1, int(level * node_count / (maximum + 1)))
        assigned.setdefault(band, []).append(hybas_id)
    bands = sorted(assigned, reverse=True)
    areas, features = [], []
    for band in bands:
        nodes = basin.loc[assigned[band]]
        weights = nodes.SUB_AREA.clip(lower=0).to_numpy(float)
        area = float(weights.sum())
        if not np.isfinite(area) or area <= 0:
            raise ValueError("HydroBASINS增量面积无效")
        centroid_x = float(np.average(nodes.geometry.centroid.x, weights=weights))
        centroid_y = float(np.average(nodes.geometry.centroid.y, weights=weights))
        upstream_area = float(nodes.UP_AREA.clip(lower=0).max())
        features.append([np.log1p(area), np.log1p(upstream_area), band / max(node_count - 1, 1),
                         np.clip((centroid_x - outlet_xy[0]) / 10000., -500, 500),
                         np.clip((centroid_y - outlet_xy[1]) / 10000., -500, 500),
                         topo["elevation_m"] / 1000., topo["slope_degrees"], topo["relief_m"] / 1000.])
        areas.append(area)
    parents = list(range(1, len(bands))) + [-1]
    return np.asarray(features, np.float32), np.asarray(areas, np.float32), np.asarray(parents, np.int64), bands


def terrain_tokens(depth, basin, outlet_xy, topo, token_count=16):
    ids = list(depth)
    points = basin.loc[ids].geometry.centroid
    dx = points.x.to_numpy() - outlet_xy[0]
    dy = points.y.to_numpy() - outlet_xy[1]
    distance = np.hypot(dx, dy)
    chosen = np.argsort(distance)[:token_count]
    tokens = []
    for index in chosen:
        radius = distance[index] / 1000.
        scale = 0 if radius <= 1 else 1 if radius <= 5 else 2
        tokens.append([dx[index] / 10., dy[index] / 10.,
                       np.log1p(float(basin.loc[ids[index], "SUB_AREA"])),
                       depth[ids[index]] / max(max(depth.values()), 1),
                       float(scale == 0), float(scale == 1), float(scale == 2),
                       topo["elevation_m"] / 1000., topo["slope_degrees"], topo["relief_m"] / 1000.])
    if not tokens:
        raise ValueError("HydroBASINS未找到局地token")
    padding = [False] * len(tokens)
    while len(tokens) < token_count:
        tokens.append([0.] * len(TERRAIN_FEATURES))
        padding.append(True)
    return np.asarray(tokens, np.float32), np.asarray(padding, bool)


def build_formal_inputs(q_index_path, h_index_path, shapefile, catchment_shapefile,
                        hydroatlas_path, output, station_ids=None):
    q_index, h_index = read_json(q_index_path), read_json(h_index_path)
    q_by_id = {row["station_id"]: row for row in q_index["stations"]}
    h_by_id = {row["station_id"]: row for row in h_index["stations"]}
    wanted = set(station_ids or (set(q_by_id) | set(h_by_id)))
    manifest_path = Path(q_index_path).parents[2] / "审计/候选站点_边界复核.csv"
    manifest = pd.read_csv(manifest_path, dtype={"station_id": str, "terminal_system_id": str}).set_index("station_id", drop=False)
    selected = sorted(wanted & set(manifest.index))
    output = Path(output)
    if output.exists():
        raise ValueError("正式空间输入输出目录已存在，请生成新版本")
    output.mkdir(parents=True)
    basins = gpd.read_file(shapefile, columns=["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "SUB_AREA", "UP_AREA"])
    basins["HYBAS_ID"] = basins.HYBAS_ID.astype("int64")
    basins = basins.set_index("HYBAS_ID", drop=False)
    catchments = gpd.read_file(catchment_shapefile)
    catchments = catchments.set_crs("EPSG:4326") if catchments.crs is None else catchments.to_crs("EPSG:4326")
    catchments["GAGE_ID"] = catchments.GAGE_ID.astype(str).str.zfill(8)
    catchments = catchments.set_index("GAGE_ID", drop=False)
    downstream = {int(row.HYBAS_ID): int(row.NEXT_DOWN) for row in basins.itertuples()}
    usecols = ["STAID", "ele_mt_sav", "ele_mt_smn", "ele_mt_smx", "slp_dg_uav"]
    attributes = pd.read_csv(hydroatlas_path, sep="\t", usecols=usecols, dtype={"STAID": str})
    attributes["STAID"] = attributes.STAID.str.replace(r"\.0$", "", regex=True).str.zfill(8)
    attributes = attributes.set_index("STAID")
    records, failures = [], []
    for number, sid in enumerate(selected, 1):
        row = manifest.loc[sid]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"候选清单站号重复: {sid}")
        if sid not in attributes.index or pd.isna(row.local_hybas_id):
            continue
        topo_row = attributes.loc[sid]
        topo = {"elevation_m": float(topo_row.ele_mt_sav),
                "slope_degrees": float(topo_row.slp_dg_uav),
                "relief_m": float(topo_row.ele_mt_smx - topo_row.ele_mt_smn)}
        local_id = int(row.local_hybas_id)
        if local_id not in basins.index:
            continue
        depth = upstream_depth(local_id, downstream)
        if sid not in catchments.index:
            continue
        catchment = catchments.loc[sid].geometry
        source_graph = basins.loc[list(depth)].copy()
        projected = source_graph.to_crs(gpd.GeoSeries([catchment], crs="EPSG:4326").estimate_utm_crs())
        catchment_projected = gpd.GeoSeries([catchment], crs="EPSG:4326").to_crs(projected.crs).iloc[0]
        geometric_catchment_area = float(catchment_projected.area / 1e6)
        projected.geometry = projected.geometry.intersection(catchment_projected)
        projected = projected[~projected.geometry.is_empty].copy()
        projected["SUB_AREA"] = projected.geometry.area / 1e6
        mapped_area = float(projected.SUB_AREA.sum())
        if not len(projected) or mapped_area <= 0 or geometric_catchment_area <= 0:
            continue
        # HydroBASINS-to-CAMELS unmatched geometry remains explicit, assigned to the gauged outlet increment.
        scale = float(row.area_km2) / geometric_catchment_area
        projected["SUB_AREA"] *= scale
        mapped_area *= scale
        unmapped_area = max(0.0, float(row.area_km2) - mapped_area)
        unmapped_fraction = unmapped_area / float(row.area_km2)
        if unmapped_fraction > 0.25:
            failures.append({"station_id": sid, "reason": "topology_catchment_area_unmapped_over_25pct",
                             "unmapped_area_km2": unmapped_area, "unmapped_fraction": unmapped_fraction})
            continue
        clipped_depth = {int(node): depth[int(node)] for node in projected.index}
        if local_id not in clipped_depth:
            raise ValueError(f"{sid}站点所在拓扑单元不与流域边界相交")
        projected.loc[local_id, "SUB_AREA"] += unmapped_area
        outlet = projected.loc[local_id].geometry.centroid
        node_features, node_areas, parents, bands = coarsen_nodes(
            clipped_depth, projected, (outlet.x, outlet.y), topo, 8)
        terrain, terrain_mask = terrain_tokens(clipped_depth, projected, (outlet.x, outlet.y), topo, 16)
        data_row = q_by_id.get(sid, h_by_id.get(sid))
        with np.load(data_row["path"], allow_pickle=False) as z:
            climate = climate_vector(z["weather"], z["time_ns"], data_row["split"],
                                     "2020-10-01" if data_row["split"] != "train" else "2018-10-01")
        path = output / f"{sid}.npz"
        np.savez_compressed(path, node_features=node_features, node_areas=node_areas,
                            downstream_parent=parents, terrain_tokens=terrain,
                            terrain_padding_mask=terrain_mask, climate_features=climate,
                            source_hybas_count=np.int32(len(depth)),
                            source_hybas_ids=np.asarray(sorted(depth), dtype="int64"))
        records.append({"station_id": sid, "terminal_system_id": row.terminal_system_id,
                        "split": data_row["split"], "crossfit_fold": data_row["crossfit_fold"],
                        "path": str(path.resolve()), "sha256": digest(path),
                        "source_hybas_count": len(clipped_depth), "coarsened_nodes": len(node_areas),
                        "incremental_area_km2": float(node_areas.sum()),
                        "reported_station_area_km2": float(row.area_km2),
                        "area_ratio_to_station": float(node_areas.sum() / row.area_km2),
                        "topology_unmapped_area_km2": unmapped_area,
                        "topology_unmapped_fraction": unmapped_fraction,
                        "geometry_to_reported_area_scale": scale,
                        "catchment_boundary_source": str(Path(catchment_shapefile).resolve()),
                        "terrain_resolution": "HydroBASINS polygons plus CAMELS catchment summary; local DEM unavailable",
                        "terrain_tokens": 16, "coarsened_bands": bands})
        if number % 10 == 0 or number == len(selected):
            print(f"正式空间输入 {number}/{len(selected)} 有效={len(records)}", flush=True)
    if not records:
        raise ValueError("没有站点可以生成正式空间输入")
    write_json(output / "index.json", {
        "schema_version": 1, "scope": "formal_echo_spatial_static_inputs",
        "q_index_sha256": digest(q_index_path), "h_index_sha256": digest(h_index_path),
        "shapefile_sha256": digest(shapefile), "hydroatlas_sha256": digest(hydroatlas_path),
        "terrain_resolution": "basin_summary_not_local_dem", "stations": records, "failures": failures,
        "future_H_read": False, "target_Q_read": False, "created_utc": utc_now(),
        "node_features": NODE_FEATURES, "terrain_features": TERRAIN_FEATURES})
    return {"stations": len(records), "output": str(output)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-index", type=Path, required=True)
    parser.add_argument("--h-index", type=Path, required=True)
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--catchment-shapefile", type=Path, required=True)
    parser.add_argument("--hydroatlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--station-ids", nargs="*")
    args = parser.parse_args()
    print(build_formal_inputs(args.q_index, args.h_index, args.shapefile,
                              args.catchment_shapefile, args.hydroatlas, args.output, args.station_ids))
