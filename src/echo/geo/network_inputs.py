"""Build real branching incremental graphs; missing DEM/weather remain explicit."""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from echo.geo.formal_inputs import upstream_depth
from echo.geo.incremental import contract_incremental_graph, induced_parents
from echo.utils import digest, read_json, utc_now, write_json


def build(h_index_path, shapefile, catchment_shapefile, output, maximum_nodes=8):
    h_index_path, output = Path(h_index_path), Path(output)
    if output.exists():
        raise ValueError("输出目录已存在，使用新版本目录保留历史证据")
    index = read_json(h_index_path)
    manifest_path = h_index_path.parents[2] / "审计/候选站点_边界复核.csv"
    manifest = pd.read_csv(manifest_path, dtype={"station_id": str}).set_index("station_id")
    basin = gpd.read_file(shapefile).set_index("HYBAS_ID", drop=False)
    catchment = gpd.read_file(catchment_shapefile)
    # CAMELSH dataset documentation explicitly states WGS84; not inferred from numbers.
    catchment = catchment.set_crs(4326) if catchment.crs is None else catchment.to_crs(4326)
    catchment["GAGE_ID"] = catchment.GAGE_ID.astype(str).str.zfill(8)
    catchment = catchment.set_index("GAGE_ID")
    downstream = {int(n): int(p) for n, p in zip(basin.HYBAS_ID, basin.NEXT_DOWN, strict=True)}
    output.mkdir(parents=True)
    records, failures = [], []
    for row in index["stations"]:
        sid = row["station_id"]
        if row["split"] not in {"train", "validation", "development_test"}:
            continue
        try:
            meta = manifest.loc[sid]
            if str(meta.terminal_system_id) != row["terminal_system_id"]:
                raise ValueError("水位清单和拓扑水系不一致")
            outlet_id = int(meta.local_hybas_id)
            polygon = catchment.loc[sid].geometry
            station = Point(float(meta.longitude), float(meta.latitude))
            polygon_series = gpd.GeoSeries([polygon], crs=4326)
            crs = polygon_series.estimate_utm_crs()
            projected_polygon = polygon_series.to_crs(crs).iloc[0]
            point = gpd.GeoSeries([station], crs=4326).to_crs(crs).iloc[0]
            distance_m = point.distance(projected_polygon)
            if distance_m > 2000:
                raise ValueError("站点距自身流域边界超过2km")
            depth = upstream_depth(outlet_id, downstream)
            cells = basin.loc[sorted(depth)].to_crs(crs).copy()
            cells.geometry = cells.geometry.intersection(projected_polygon)
            cells = cells[~cells.geometry.is_empty & (cells.geometry.area > 1)].copy()
            geometric_area = projected_polygon.area / 1e6
            areas = (cells.geometry.area / 1e6).to_dict()
            parent_map = induced_parents(cells.index, downstream, outlet_id)
            graph = contract_incremental_graph(parent_map, areas, maximum_nodes)
            n = len(graph["groups"])
            features, distance, geometries = [], [], []
            for group, representative in zip(graph["groups"], graph["representative_ids"], strict=True):
                part = cells.loc[group]
                geometry = part.geometry.union_all()
                geometries.append(geometry)
                area = float(sum(areas[node] for node in group))
                centroid = geometry.centroid
                flow_distance = float(basin.loc[representative].DIST_SINK - basin.loc[outlet_id].DIST_SINK)
                distance.append(flow_distance)
                features.append([np.log1p(area), np.log1p(float(part.UP_AREA.max())),
                                 flow_distance / 100, (centroid.x - point.x) / 10000,
                                 (centroid.y - point.y) / 10000])
            parents = graph["parents"]
            edge = np.asarray([[i, int(p)] for i, p in enumerate(parents) if p >= 0], dtype="int64")
            edge = edge.reshape(-1, 2).T
            # Only documented path distance is known. DEM drop/slope remain masked.
            edge_attr = np.zeros((edge.shape[1], 3), dtype="float32")
            edge_mask = np.zeros_like(edge_attr, dtype=bool)
            for j, (source, target) in enumerate(edge.T):
                length = distance[source] - distance[target]
                if length < -1e-5:
                    raise ValueError("下游流路距离反向，需核查源拓扑")
                edge_attr[j, 0], edge_mask[j, 0] = max(length, 0), True
            area_total = float(graph["areas_km2"].sum())
            uncovered = max(0., 1 - area_total / geometric_area)
            path = output / f"{sid}.npz"
            all_cells = np.concatenate([np.asarray(g, dtype="int64") for g in graph["groups"]])
            offsets = np.r_[0, np.cumsum([len(g) for g in graph["groups"]])]
            np.savez_compressed(path, node_geometry_features=np.asarray(features, np.float32),
                                node_areas=graph["areas_km2"], downstream_parent=parents,
                                edge_index=edge, edge_attr=edge_attr, edge_attr_mask=edge_mask,
                                member_cell_ids=all_cells, member_offsets=offsets,
                                source_parent_ids=np.asarray([parent_map[int(cell)] for cell in all_cells]),
                                source_cell_areas=np.asarray([areas[int(cell)] for cell in all_cells]))
            shapes = gpd.GeoDataFrame({"node_id": range(n), "parent": parents,
                                      "area_km2": graph["areas_km2"]}, geometry=geometries, crs=crs)
            shapes.to_crs(4326).to_file(output / f"{sid}.geojson", driver="GeoJSON")
            records.append({**row, "path": str(path.resolve()), "sha256": digest(path),
                            "nodes": n, "original_cells": len(cells),
                            "branch_nodes": int(sum(np.sum(parents == i) > 1 for i in range(n))),
                            "mapped_area_km2": area_total, "catchment_geometry_area_km2": geometric_area,
                            "reported_area_km2": float(meta.area_km2),
                            "unmapped_area_fraction": uncovered,
                            "station_distance_to_catchment_m": distance_m,
                            "geometry_path": str((output / f"{sid}.geojson").resolve()),
                            "geometry_sha256": digest(output / f"{sid}.geojson"),
                            "protocol_training_ready": False,
                            "missing": ["node_environment_attributes", "DEM_valley_tokens",
                                        "incremental_grid_weather", "edge_DEM_drop_and_slope"]})
            print(f"branching_graph station={sid} nodes={n} branches={records[-1]['branch_nodes']}", flush=True)
        except (ValueError, KeyError) as exc:
            failures.append({"station_id": sid, "reason": str(exc)})
    write_json(output / "index.json", {
        "schema_version": 2, "scope": "real_branching_graph_geometry_not_complete_training_input",
        "h_index_sha256": digest(h_index_path), "shapefile_sha256": digest(shapefile),
        "catchment_sha256": digest(catchment_shapefile),
        "manifest_sha256": digest(manifest_path), "created_utc": utc_now(),
        "aggregation": "connected_child_parent_edge_contraction",
        "area_policy": "geometric_clipped_areas_no_missing_area_assignment_or_rescaling",
        "edge_features": ["downstream_path_km", "DEM_drop_m", "DEM_slope"],
        "stations": records, "failures": failures,
        "target_Q_read": False, "H_values_read": False})
    return {"stations": len(records), "failures": len(failures)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-index", type=Path, required=True)
    parser.add_argument("--shapefile", type=Path, required=True)
    parser.add_argument("--catchment-shapefile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.h_index, args.shapefile, args.catchment_shapefile, args.output))
