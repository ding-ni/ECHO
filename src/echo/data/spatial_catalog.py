"""Query official spatial data catalogues for the fixed local pool, without Q/H."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from echo.utils import digest, read_json, utc_now, write_json


def _json(url, params=None):
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                         status_forcelist=[429, 500, 502, 503, 504])))
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json(), response.url


def build(index_path, graph_index_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("空间资料目录已存在，请新建版本")
    index_path, graph_index_path = Path(index_path), Path(graph_index_path)
    index, graph = read_json(index_path), read_json(graph_index_path)
    metadata = pd.read_csv(index_path.parents[2] / "审计/候选站点_边界复核.csv",
                           dtype={"station_id": str}).set_index("station_id")
    graphs = {row["station_id"]: row for row in graph["stations"]}
    output.mkdir(parents=True)

    def station_query(row):
        sid = row["station_id"]
        meta = metadata.loc[sid]
        import math
        dy = 10 / 111.32
        dx = dy / math.cos(math.radians(float(meta.latitude)))
        lon, lat = float(meta.longitude), float(meta.latitude)
        # 10 km enclosing square, preserving a single maximum local DEM footprint.
        bbox = [lon - dx, lat - dy, lon + dx, lat + dy]
        try:
            catalogue, url = _json("https://tnmaccess.nationalmap.gov/api/v1/products", {
                "datasets": "National Elevation Dataset (NED) 1 arc-second", "bbox": ",".join(map(str, bbox)),
                "max": 100, "outputFormat": "JSON"})
            write_json(output / "USGS目录快照" / f"{sid}.json", {
                "url": url, "retrieved_utc": utc_now(), "response": catalogue})
            items = catalogue.get("items", [])
            assets = [{key: item.get(key) for key in ["sourceId", "title", "downloadURL", "sizeInBytes",
                                                      "boundingBox", "publicationDate", "format"]} for item in items]
            if len(items) < catalogue.get("total", len(items)):
                raise ValueError("USGS目录分页未完整取得")
            status = "catalogued_not_downloaded" if assets else "no_USGS_DEM_coverage"
            error = None
        except (requests.RequestException, ValueError) as exc:
            assets, status, error = [], "catalogue_request_failed", str(exc)
        g = graphs.get(sid)
        result = {"station_id": sid, "terminal_system_id": row["terminal_system_id"],
                  "split": row["split"], "crossfit_fold": row["crossfit_fold"],
                  "local_bbox_wgs84": bbox, "H_context_available": True,
                  "branching_geometry_available": g is not None,
                  "unmapped_catchment_fraction": None if g is None else g["unmapped_area_fraction"],
                  "DEM_status": status, "DEM_assets": assets, "error": error,
                  "grid_weather_status": "not_acquired",
                  "node_attributes_status": "not_extracted",
                  "local_terrain_tokens_status": "not_extracted",
                  "protocol_ready": False}
        print(f"spatial_catalog station={sid} DEM={status} assets={len(assets)}", flush=True)
        return result

    permitted = [r for r in index["stations"] if r["split"] in {"train", "validation", "development_test"}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(station_query, permitted))
    nasa = None
    nasa_usable = False
    try:
        nasa, url = _json("https://cmr.earthdata.nasa.gov/search/collections.json", {
            "short_name": "NLDAS_FORA0125_H", "version": "2.0", "page_size": 10})
        nasa_usable = bool(nasa.get("feed", {}).get("entry", []))
        write_json(output / "NLDAS集合目录.json", {"url": url, "response": nasa, "retrieved_utc": utc_now()})
    except requests.RequestException as exc:
        write_json(output / "NLDAS集合目录失败.json", {"error": str(exc), "retrieved_utc": utc_now()})
    unique = {a["downloadURL"]: a for row in records for a in row["DEM_assets"] if a["downloadURL"]}
    result = {"schema_version": 1, "created_utc": utc_now(),
              "h_index_sha256": digest(index_path), "graph_index_sha256": digest(graph_index_path),
              "stations": records, "unique_DEM_assets": list(unique.values()),
              "NLDAS_collection_metadata_available": nasa_usable,
              "target_Q_read": False, "H_values_read": False,
              "completed_training_dataset": False}
    write_json(output / "index.json", result)
    table = pd.DataFrame([{k: r[k] for k in r if k not in {"DEM_assets", "local_bbox_wgs84"}}
                          for r in records])
    table.to_csv(output / "逐站空间资料状态.csv", index=False, encoding="utf-8-sig")
    return {"stations": len(records), "unique_DEM_assets": len(unique),
            "catalogued_sites": sum(r["DEM_status"] == "catalogued_not_downloaded" for r in records)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--graph-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.index, args.graph_index, args.output))
