"""Fixed-product hourly NLDAS acquisition with verified-file restart and safe logs."""

import argparse
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from filelock import FileLock

from echo.data.nldas_access import verify_access
from echo.utils import digest, read_json, utc_now, write_json

BASE_URL = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0"


def hour_index(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h") or not start < end:
        raise ValueError("需要UTC整点且结束时间严格晚于开始时间，范围为左闭右开")
    return pd.date_range(start, end, freq="h", inclusive="left")


def granule(hour):
    hour = pd.Timestamp(hour)
    name = f"NLDAS_FORA0125_H.A{hour:%Y%m%d}.{hour:%H}00.020.nc"
    subdir = f"{hour:%Y}/{hour.dayofyear:03d}"
    return {"time_utc": hour.isoformat(), "filename": name, "subdir": subdir,
            "url": f"{BASE_URL}/{subdir}/{name}"}


def acquisition_plan(start, end):
    hours = hour_index(start, end)
    return {"product": "NLDAS_FORA0125_H", "version": "2.0",
            "start_utc_inclusive": hours[0].isoformat(),
            "end_utc_exclusive": (hours[-1] + pd.Timedelta(hours=1)).isoformat(),
            "hours": len(hours), "estimated_compressed_bytes": len(hours) * 1_849_787,
            "size_is_estimate_from_one_observed_granule": True,
            "time_label": "UTC_interval_end", "forcing_kind": "reanalysis_conditional_simulation",
            "grid_resolution_degrees": .125, "published_grid_not_downscaled": True}


def download_hours(start, end, output, token, workers=2, attempts=3):
    if not 1 <= workers <= 4 or not 1 <= attempts <= 5:
        raise ValueError("并发需为1—4，重试次数需为1—5")
    if not token.strip():
        raise ValueError("缺少本机认证；请使用中文脚本读取加密凭据")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan = acquisition_plan(start, end)
    # Large full manifests are represented by a calendar interval, not duplicated URL lists.
    work_key = f"{hour_index(start, end)[0]:%Y%m%d%H}_{hour_index(start, end)[-1]:%Y%m%d%H}"
    with FileLock(output / ".download.lock", timeout=0):
        write_json(output / f"plan_{work_key}.json", {**plan, "created_utc": utc_now()})
        if shutil.disk_usage(output).free < 1024**3 + min(plan["estimated_compressed_bytes"], 256 * 1024**2):
            raise OSError("气象下载工作空间不足，至少保留1GiB")

        def fetch(hour):
            spec = granule(hour)
            folder = output / spec["subdir"]
            target = folder / spec["filename"]
            receipt_name = spec["filename"] + ".receipt.json"
            receipt_path = folder / receipt_name
            if target.exists() and receipt_path.exists():
                old = read_json(receipt_path)
                if (old.get("status") == "downloaded_netcdf_verified" and old.get("url") == spec["url"]
                        and old.get("bytes") == target.stat().st_size and old.get("sha256") == digest(target)):
                    return {**spec, "status": "verified_existing", "path": str(target.resolve()),
                            "sha256": old["sha256"], "bytes": old["bytes"]}
                return {**spec, "status": "failed_existing_file_integrity", "attempts": 0}
            for attempt in range(1, attempts + 1):
                try:
                    receipt = verify_access(token, folder, granule_url=spec["url"], receipt_name=receipt_name)
                    if receipt["time_values"] != [hour.tz_localize(None).isoformat(timespec="nanoseconds")]:
                        raise ValueError("原文件时间坐标与请求小时不一致")
                    print(f"NLDAS verified {spec['time_utc']} bytes={receipt['bytes']}", flush=True)
                    return {**spec, "status": "downloaded_netcdf_verified", "path": str(target.resolve()),
                            "sha256": receipt["sha256"], "bytes": receipt["bytes"]}
                except (ValueError, RuntimeError, OSError):
                    # Neither HTTP exception text nor signed URLs are logged.
                    print(f"NLDAS retry {spec['time_utc']} attempt={attempt}/{attempts}", flush=True)
                    if attempt < attempts:
                        time.sleep(min(attempt * 2, 6))
            return {**spec, "status": "failed", "attempts": attempts}

        progress_lock = threading.Lock()
        progress = {"plan": plan, "finished_hours_this_run": 0, "completed": 0, "failed": 0}

        def fetch_with_progress(hour):
            result = fetch(hour)
            with progress_lock:
                progress["finished_hours_this_run"] += 1
                key = "completed" if result["status"] in {"verified_existing", "downloaded_netcdf_verified"} else "failed"
                progress[key] += 1
                if progress["finished_hours_this_run"] % 10 == 0 or progress["finished_hours_this_run"] == plan["hours"]:
                    write_json(output / "current_progress.json", {**progress, "updated_utc": utc_now()})
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fetch_with_progress, hour_index(start, end)))
        completed = sum(r["status"] in {"verified_existing", "downloaded_netcdf_verified"} for r in results)
        result = {"plan": plan, "updated_utc": utc_now(), "completed": completed,
                  "failed": len(results) - completed, "files": results,
                  "dataset_training_ready": False}
        write_json(output / f"index_{work_key}.json", result)
        return {"completed": completed, "failed": len(results) - completed, "hours": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("本地数据/原始/NLDAS_FORA0125_H_v2"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.download:
        print(acquisition_plan(args.start, args.end))
        return
    result = download_hours(args.start, args.end, args.output,
                            os.environ.pop("ECHO_EARTHDATA_TOKEN", ""), args.workers)
    print(result)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
