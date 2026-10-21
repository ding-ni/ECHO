"""组合真实站点候选清单，未知基准/时间/拓扑留待核验，不把未知写成已通过。"""

from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.netcdf import open_netcdf
from echo.utils import write_json


def build_inventory(qh_root, weather_root, attribute_root, output, topology_path=None):
    attribute_root, output = Path(attribute_root), Path(output)
    basin = next(attribute_root.rglob("attributes_gageii_BasinID.csv"))
    topo = next(attribute_root.rglob("attributes_gageii_Topo.csv"))
    basin = pd.read_csv(basin, dtype={"STAID": str}).set_index("STAID")
    topo = pd.read_csv(topo, dtype={"STAID": str}).set_index("STAID")
    weather_paths = {}
    for path in Path(weather_root).rglob("*.nc"):
        sid = path.stem.removesuffix("_hourly")
        if sid in weather_paths:
            raise ValueError(f"同站有多个气象文件，需明确版本: {sid}")
        weather_paths[sid] = path
    linkage = {}
    if topology_path:
        table = pd.read_csv(topology_path, dtype=str)
        if not {"station_id", "terminal_system_id", "topology_evidence"} <= set(table.columns):
            raise ValueError("拓扑表缺少必要字段")
        if table.station_id.duplicated().any():
            raise ValueError("拓扑表站点重复")
        linkage = table.set_index("station_id").to_dict("index")
    rows, failures = [], []
    for path in sorted(Path(qh_root).rglob("*.nc")):
        sid = path.stem.removesuffix("_hourly")
        try:
            if sid not in basin.index:
                raise ValueError("缺少独立站点属性")
            with open_netcdf(path) as ds:
                if "water_level" not in ds or str(ds.water_level.attrs.get("units")) != "m":
                    raise ValueError("Q/H字段模式与已核验样本不一致")
                h = ds.water_level.to_series()
                if h.index.has_duplicates or not h.index.is_monotonic_increasing:
                    raise ValueError("重复或逆序时间")
                support = h.loc["2019-10-01":"2020-09-30"]
                query = h.loc["2020-10-01":"2021-09-30"]
                support_n = int(np.isfinite(support).sum())
                query_n = int(np.isfinite(query).sum())
            b = basin.loc[sid]
            t = topo.loc[sid] if sid in topo.index else {}

            def get_topo(key):
                return float(t.get(key, np.nan))

            slope = get_topo("SLOPE_PCT")
            rows.append(
                {
                    "station_id": sid,
                    "terminal_system_id": linkage.get(sid, {}).get("terminal_system_id", ""),
                    "topology_evidence": linkage.get(sid, {}).get("topology_evidence", ""),
                    "area_km2": float(b.DRAIN_SQKM),
                    "latitude": float(b.LAT_GAGE),
                    "longitude": float(b.LNG_GAGE),
                    "slope_degrees": float(np.degrees(np.arctan(slope / 100))),
                    "relief_m": get_topo("ELEV_MAX_M_BASIN") - get_topo("ELEV_MIN_M_BASIN"),
                    "support_coverage": min(support_n / 8784, 1.0),
                    "query_coverage": min(query_n / 8760, 1.0),
                    "time_support_verified": False,
                    "datum_verified": False,
                    "datum_id": "unknown",
                    "Q_provenance_class": "archive_rating_derived_or_unknown",
                    "qh_path": str(path.resolve()),
                    "weather_path": str(weather_paths.get(sid, "")),
                    "weather_available": sid in weather_paths,
                }
            )
        except Exception as exc:
            failures.append({"station_id": sid, "error": str(exc)})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    summary = {
        "scope": "candidate_inventory_not_training_approval",
        "candidate_stations": len(rows),
        "weather_intersection": int(frame.weather_available.sum()) if len(frame) else 0,
        "support_80pct": int((frame.support_coverage >= 0.8).sum()) if len(frame) else 0,
        "failures": failures,
        "required_verification": ["time_support", "datum", "terminal_topology"],
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary
