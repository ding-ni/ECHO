"""Resume and verify fixed public USGS GeoTIFF assets selected from source catalogues."""

import argparse
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from echo.utils import digest, read_json, utc_now, write_json


def select_assets(catalogue_root, station_ids):
    selected = {}
    for sid in station_ids:
        doc = read_json(Path(catalogue_root) / "USGS目录快照" / f"{sid}.json")
        for item in doc["response"].get("items", []):
            url = item.get("downloadURL", "")
            match = re.search(r"USGS_1_(n\d+w\d+)_(\d{8})\.tif$", url)
            if not match or urlparse(url).hostname != "prd-tnm.s3.amazonaws.com":
                continue
            tile, date = match.groups()
            if tile not in selected or date > selected[tile]["date"]:
                selected[tile] = {"url": url, "date": date, "source_id": item.get("sourceId"),
                                  "title": item.get("title"), "catalogue_bytes": item.get("sizeInBytes")}
    return list(selected.values())


def download_asset(spec, root):
    import rasterio
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(spec["url"]).path).name
    path, part = root / name, root / (name + ".part")
    receipt_path = root / (name + ".receipt.json")
    if path.exists():
        receipt = read_json(receipt_path)
        if receipt["sha256"] != digest(path) or receipt["url"] != spec["url"]:
            raise ValueError("已完成DEM资产与来源收据不一致")
        return receipt
    session = requests.Session()
    head = session.head(spec["url"], timeout=45)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    etag = head.headers.get("ETag", "").strip('"')
    if expected != int(spec["catalogue_bytes"]):
        raise ValueError(f"USGS目录大小与实际对象不一致: {name}")
    source_path = root / (name + ".source.json")
    source = {"url": spec["url"], "bytes": expected, "etag": etag}
    if source_path.exists() and read_json(source_path) != source:
        raise ValueError("未完成DEM资产远端版本改变，不拼接不同版本")
    write_json(source_path, source)
    offset = part.stat().st_size if part.exists() else 0
    if offset > expected:
        raise ValueError("DEM部分文件大于远端对象")
    if shutil.disk_usage(root).free < expected - offset + 1024**3:
        raise OSError("DEM下载需要额外保留1GiB工作空间")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise OSError("范围下载需要curl；不回退到无总时限的整瓦片连接")
    chunk_path, header_path = root / (name + ".chunk"), root / (name + ".headers")
    failures_at_offset = 0
    while offset < expected:
        # Small bounded ranges also make progress on slow institutional proxies.
        stop = min(expected, offset + 256 * 1024) - 1
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run([
            curl, "--location", "--fail", "--silent", "--show-error",
            "--connect-timeout", "10", "--max-time", "60",
            "--range", f"{offset}-{stop}", "--header", f'If-Match: "{etag}"',
            "--dump-header", str(header_path), "--output", str(chunk_path), spec["url"],
        ], capture_output=True, text=True, timeout=70, creationflags=flags)
        headers = header_path.read_text(encoding="latin-1") if header_path.exists() else ""
        ranges = re.findall(r"Content-Range:\s*bytes (\d+)-(\d+)/(\d+)", headers, re.I)
        received = chunk_path.stat().st_size if chunk_path.exists() else 0
        # curl timeout can leave a valid contiguous prefix. Keep it only after
        # the exact Range response and pinned If-Match identity were checked.
        valid_prefix = (bool(ranges) and tuple(map(int, ranges[-1])) == (offset, stop, expected)
                        and 0 < received <= stop - offset + 1 and result.returncode in {0, 18, 28})
        if not valid_prefix:
            failures_at_offset += 1
            if failures_at_offset < 4:
                print(f"DEM retry {name} offset={offset} attempt={failures_at_offset}", flush=True)
                continue
            raise OSError(f"DEM范围响应或长度核验失败；curl状态={result.returncode}")
        if result.returncode == 0 and received != stop - offset + 1:
            raise ValueError("DEM完成响应的分块长度不匹配")
        with part.open("ab") as stream, chunk_path.open("rb") as block:
            shutil.copyfileobj(block, stream)
        offset += received
        failures_at_offset = 0
        write_json(root / (name + ".status.json"), {
            "status": "downloading", "verified_contiguous_bytes": offset,
            "expected_bytes": expected, "updated_utc": utc_now()})
        if offset % (4 * 1024**2) < 256 * 1024 or offset == expected:
            print(f"DEM range {name} {offset}/{expected}", flush=True)
    if part.stat().st_size != expected:
        raise ValueError("DEM文件长度与官方对象不一致")
    if re.fullmatch(r"[a-fA-F0-9]{32}", etag) and digest(part, "md5") != etag.lower():
        raise ValueError("DEM单对象S3 ETag与MD5核验不一致")
    with rasterio.open(part) as raster:
        # Read every block, detecting incomplete/corrupt compression and recording CRS/units.
        for _, window in raster.block_windows(1):
            raster.read(1, window=window)
        metadata = {"crs": str(raster.crs), "height": raster.height, "width": raster.width,
                    "bounds": list(raster.bounds), "pixel_size": list(raster.res),
                    "vertical_band_units": list(raster.units), "tags": raster.tags()}
    receipt = {**spec, "bytes": expected, "etag": etag, "sha256": digest(part),
               "validation": "official_https_catalogue_size_etag_when_singlepart_full_geotiff_block_read_local_SHA256",
               "official_checksum_published": False, "retrieved_utc": utc_now(),
               "raster": metadata, "vertical_datum_verified": False,
               "path": str(path.resolve())}
    part.replace(path)
    write_json(receipt_path, receipt)
    print(f"DEM verified {name} bytes={expected}", flush=True)
    return receipt


def _run(catalogue_root, output, station_ids):
    output = Path(output)
    specs = select_assets(catalogue_root, station_ids)
    write_json(output / "selected_assets.json", {"station_ids": station_ids, "assets": specs,
                                               "selected_utc": utc_now()})
    def safe_download(spec):
        try:
            return {"complete": True, **download_asset(spec, output)}
        except (ValueError, OSError, requests.RequestException, subprocess.TimeoutExpired) as exc:
            print(f"DEM failed {spec['url']}: {exc}", flush=True)
            return {"complete": False, **spec, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(safe_download, specs))
    write_json(output / "index.json", {"stations": station_ids, "assets": results,
                                      "completed": sum(r["complete"] for r in results),
                                      "failed": sum(not r["complete"] for r in results)})
    return {"completed": sum(r["complete"] for r in results), "assets": len(results)}


def run(catalogue_root, output, station_ids):
    from filelock import FileLock
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".download.lock", timeout=0):
        return _run(catalogue_root, output, station_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--station-ids", nargs="+", required=True)
    args = parser.parse_args()
    print(run(args.catalogue_root, args.output, args.station_ids))
