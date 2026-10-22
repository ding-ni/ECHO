"""用独立流域边界复核终端水系，冲突水系整体隔离。"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from echo.utils import digest, utc_now, write_json


def crosscheck_boundaries(stations, basins, min_fraction=0.05, min_area_km2=0.01):
    """按较小流域的重叠比例发现跨终端冲突；不使用Q、H或模型误差。"""
    if not 0 < min_fraction <= 1 or min_area_km2 <= 0:
        raise ValueError("重叠阈值必须为正")
    if stations.station_id.duplicated().any() or basins.station_id.duplicated().any():
        raise ValueError("站点或流域边界重复")
    if basins.crs is None:
        raise ValueError("边界缺少坐标系")
    table = stations.copy()
    table["station_id"] = table.station_id.astype(str)
    shapes = basins[["station_id", "geometry"]].copy()
    shapes["station_id"] = shapes.station_id.astype(str)
    shapes = shapes[shapes.station_id.isin(table.station_id)].to_crs("EPSG:5070").reset_index(drop=True)
    metadata = table.set_index("station_id")
    invalid = set(shapes.loc[~shapes.is_valid | shapes.is_empty | shapes.geometry.isna(), "station_id"])
    shapes = shapes[~shapes.station_id.isin(invalid)].reset_index(drop=True)
    shapes["terminal"] = shapes.station_id.map(metadata.terminal_system_id)
    areas = shapes.area.to_numpy()
    invalid |= set(shapes.loc[~np.isfinite(areas) | (areas <= 0), "station_id"])
    shapes = shapes[~shapes.station_id.isin(invalid)].reset_index(drop=True)
    areas = shapes.area.to_numpy()
    pairs = shapes.sindex.query(shapes.geometry, predicate="intersects")
    conflicts, shared_pairs = [], 0
    bad_systems = set()
    # 同终端内的上下游本就同组，无需对巨大嵌套多边形反复求交。
    terminals = shapes.terminal.fillna("").to_numpy()
    left, right = pairs
    different = (left < right) & (terminals[left] != terminals[right])
    different &= (terminals[left] != "") & (terminals[right] != "")
    for i, j in zip(left[different], right[different]):
        a, b = shapes.iloc[i], shapes.iloc[j]
        area = a.geometry.intersection(b.geometry).area
        fraction = area / min(areas[i], areas[j])
        if area < min_area_km2 * 1e6 or fraction < min_fraction:
            continue
        shared_pairs += 1
        if pd.isna(a.terminal) or pd.isna(b.terminal) or a.terminal == b.terminal:
            continue
        bad_systems.update([a.terminal, b.terminal])
        conflicts.append(
            {
                "station_a": a.station_id,
                "station_b": b.station_id,
                "terminal_a": a.terminal,
                "terminal_b": b.terminal,
                "intersection_km2": area / 1e6,
                "fraction_of_smaller": fraction,
            }
        )
    known = set(shapes.station_id)
    points = gpd.GeoDataFrame(
        table[["station_id"]], geometry=gpd.points_from_xy(table.longitude, table.latitude), crs="EPSG:4326"
    )
    points = points.to_crs("EPSG:5070").set_index("station_id")
    shapes = shapes.set_index("station_id")
    # 出口可能位于边界旁；记录距离，2km仅为开发期人工复核门槛，不做最近点改配。
    distances = {sid: points.loc[sid].geometry.distance(shape.geometry) for sid, shape in shapes.iterrows()}
    table["boundary_distance_m"] = table.station_id.map(distances)
    table["topology_verified"] = False
    table["topology_review_reason"] = ""
    for idx, row in table.iterrows():
        reasons = []
        if row.station_id not in known:
            reasons.append("missing_or_invalid_boundary")
        if pd.isna(row.terminal_system_id) or not str(row.terminal_system_id).strip():
            reasons.append("unknown_terminal")
        if pd.isna(row.topology_evidence) or not str(row.topology_evidence).strip():
            reasons.append("unknown_topology_evidence")
        if row.terminal_system_id in bad_systems:
            reasons.append("cross_terminal_overlap_system_quarantined")
        if pd.isna(row.boundary_distance_m) or row.boundary_distance_m > 2000:
            reasons.append("gage_boundary_distance")
        table.at[idx, "topology_review_reason"] = ";".join(reasons)
        table.at[idx, "topology_verified"] = not reasons
    result = {
        "scope": "topology_boundary_crosscheck_not_full_station_approval",
        "created_utc": utc_now(),
        "stations": len(table),
        "boundary_matches": len(known),
        "invalid_boundaries": sorted(invalid),
        "substantial_cross_terminal_overlap_pairs": shared_pairs,
        "cross_terminal_conflicts": conflicts,
        "quarantined_terminal_systems": sorted(bad_systems),
        "verified_stations": int(table.topology_verified.sum()),
        "thresholds": {
            "fraction_of_smaller": min_fraction,
            "minimum_overlap_km2": min_area_km2,
            "maximum_gage_boundary_distance_m": 2000,
        },
    }
    return table, result


def audit_boundaries(manifest, shapefile, output, source_crs=None, crs_evidence=None):
    stations = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    basins = gpd.read_file(shapefile, columns=["GAGE_ID"]).rename(columns={"GAGE_ID": "station_id"})
    if basins.crs is None:
        if not source_crs or not crs_evidence:
            raise ValueError("边界缺少坐标系，必须显式提供官方坐标系依据")
        basins = basins.set_crs(source_crs)
    table, result = crosscheck_boundaries(stations, basins)
    result["boundary_crs"] = str(basins.crs)
    result["crs_evidence"] = crs_evidence or "shapefile_projection_metadata"
    result["input_sha256"] = {str(manifest): digest(manifest)}
    shape = Path(shapefile)
    result["boundary_component_sha256"] = {
        p.name: digest(p) for p in sorted(shape.parent.glob(shape.stem + ".*"))
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    write_json(output.with_suffix(".summary.json"), result)
    return result
