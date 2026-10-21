"""官方HydroBASINS北美拓扑：点落区匹配并核验NEXT_DOWN与MAIN_BAS。"""

import json
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from echo.data.archive import safe_member
from echo.data.download import request
from echo.geo.topology import terminal_of
from echo.utils import digest, utc_now, write_json

URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_na_lev12_v1c.zip"
EXPECTED_BYTES = 65261324


def download_topology(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "hybas_na_lev12_v1c.zip"
    part = target.with_suffix(".zip.part")
    audit = target.with_suffix(".zip.audit.json")
    if not target.exists():
        for attempt in range(20):
            offset = part.stat().st_size if part.exists() else 0
            if offset == EXPECTED_BYTES:
                break
            if offset > EXPECTED_BYTES:
                raise ValueError("拓扑下载体量异常")
            try:
                with request(URL, {"Range": f"bytes={offset}-"} if offset else {}) as response:
                    if offset and response.status != 206:
                        raise ValueError("拓扑服务端不支持续传，请保留片段并诊断")
                    with part.open("ab") as out:
                        while True:
                            block = response.read1(1024**2)
                            if not block:
                                break
                            out.write(block)
                if part.stat().st_size != EXPECTED_BYTES:
                    raise OSError("拓扑连接提前结束")
                break
            except OSError:
                if attempt == 19:
                    raise
                time.sleep(min(2**attempt, 20))
        if part.stat().st_size != EXPECTED_BYTES:
            raise ValueError("拓扑文件体量不匹配")
        with zipfile.ZipFile(part) as z:
            if z.testzip() is not None:
                raise ValueError("拓扑ZIP CRC失败")
        part.replace(target)
    sha = digest(target)
    if audit.exists() and json.loads(audit.read_text(encoding="utf-8"))["sha256"] != sha:
        raise ValueError("已下载拓扑内容发生改变")
    write_json(
        audit,
        {
            "url": URL,
            "bytes": target.stat().st_size,
            "sha256": sha,
            "official_checksum_available": False,
            "validation": "official_HTTPS_size_ZIP_CRC_local_SHA256",
            "source_product": "HydroBASINS v1c North America level12",
            "retrieved_utc": utc_now(),
            "license_reference": "https://www.hydrosheds.org/page/license",
        },
    )
    extracted = root / "hybas_na_lev12_v1c"
    if not (extracted / "complete.json").exists():
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as z:
            for member in z.infolist():
                safe_member(member.filename)
            z.extractall(extracted)
        write_json(extracted / "complete.json", {"archive_sha256": sha})
    return next(extracted.rglob("*.shp"))


def map_terminals(shapefile, stations, output):
    basins = gpd.read_file(shapefile, columns=["HYBAS_ID", "NEXT_DOWN", "MAIN_BAS", "ENDO", "COAST"])
    stations = pd.read_csv(stations, dtype={"station_id": str})
    if stations.station_id.duplicated().any():
        raise ValueError("站点重复")
    nodes = set(basins.HYBAS_ID.astype("int64"))
    downstream = {int(a): int(b) for a, b in zip(basins.HYBAS_ID, basins.NEXT_DOWN)}
    outlets = {node for node, nxt in downstream.items() if nxt == 0}
    points = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations.longitude, stations.latitude), crs="EPSG:4326"
    ).to_crs(basins.crs)
    joined = gpd.sjoin(points, basins, how="left", predicate="intersects")
    rows, failures = [], []
    for station, matches in joined.groupby("station_id", sort=True):
        try:
            matches = matches.dropna(subset=["HYBAS_ID", "MAIN_BAS"])
            if not len(matches) or matches.MAIN_BAS.nunique() != 1:
                raise ValueError("站点落区缺失或跨终端边界；不自动按最近点猜水系")
            terminals = {terminal_of(int(row.HYBAS_ID), downstream, outlets) for _, row in matches.iterrows()}
            if len(terminals) != 1 or int(next(iter(terminals))) != int(matches.MAIN_BAS.iloc[0]):
                raise ValueError("NEXT_DOWN追踪与MAIN_BAS不一致")
            rows.append(
                {
                    "station_id": station,
                    "terminal_system_id": "HYBAS_" + next(iter(terminals)),
                    "topology_evidence": "HydroBASINS_v1c_NA_L12_point_intersection_NEXT_DOWN_MAIN_BAS",
                    "local_hybas_id": int(matches.HYBAS_ID.iloc[0]),
                }
            )
        except ValueError as exc:
            failures.append({"station_id": station, "error": str(exc)})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["station_id", "terminal_system_id", "topology_evidence", "local_hybas_id"]
    ).to_csv(output, index=False, encoding="utf-8-sig")
    result = {
        "matched_stations": len(rows),
        "terminal_systems": len({r["terminal_system_id"] for r in rows}),
        "graph_nodes": len(nodes),
        "failures": failures,
        "shapefile_sha256": digest(shapefile),
        "limitation": "公共水系尺度分组；跨边界与落区歧义保持未知，需结合流域重叠复核",
    }
    write_json(output.with_suffix(".summary.json"), result)
    return result
