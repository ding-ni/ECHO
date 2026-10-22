"""独立USGS水位核验；仅查询00065和支持期，不查询目标Q或未来H。"""

import http.client
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.download import request
from echo.data.netcdf import open_netcdf
from echo.data.splits import group_key
from echo.utils import digest, utc_now, write_json

SUPPORT_WINDOWS = [("2019-10-01", "2019-10-03"), ("2020-03-07", "2020-03-10")]


def hourly_height(payload, start, end):
    series = payload["value"]["timeSeries"]
    if len(series) != 1:
        raise ValueError("USGS水位序列缺失或存在多种方法，不能自动择优合并")
    series = series[0]
    codes = {str(item["value"]) for item in series["variable"]["variableCode"]}
    if codes != {"00065"} or series["variable"]["unit"]["unitCode"] != "ft":
        raise ValueError("只接受USGS参数00065、英尺单位")
    values = series["values"]
    if len(values) != 1:
        raise ValueError("多套USGS方法记录尚未核验")
    values = values[0]["value"]
    times = pd.to_datetime([v["dateTime"] for v in values], utc=True)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("USGS水位时间重复或逆序")
    raw = pd.to_numeric(pd.Series([v["value"] for v in values]), errors="coerce").to_numpy(float)
    raw[raw == series["variable"]["noDataValue"]] = np.nan
    raw = pd.Series(raw, index=times)
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    raw = raw[(raw.index >= start) & (raw.index < end)]
    # 最新Hourly2的H与UTC整点瞬时值吻合；不使用旧版Q处理脚本的小时均值。
    # 缺少整点就保留缺测，不以最近/首条/均值替代。
    return raw[(raw.index.minute == 0) & (raw.index.second == 0) & (raw.index.microsecond == 0)] * 0.3048


def snapshot_json(url, path):
    path = Path(path)
    if path.exists():
        provenance = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        if provenance["url"] != url or provenance["sha256"] != digest(path):
            raise ValueError("USGS证据缓存与来源校验记录不一致")
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            with request(url) as response:
                chunks = []
                while block := response.read1(64 * 1024):
                    chunks.append(block)
                data = json.loads(b"".join(chunks))
            break
        except (OSError, ValueError, http.client.HTTPException):
            if attempt == 2:
                raise
    write_json(path, data)
    write_json(
        path.with_suffix(".provenance.json"), {"url": url, "retrieved_utc": utc_now(), "sha256": digest(path)}
    )
    return data


def reconcile_station(row, snapshots):
    sid = str(row.station_id)
    result = {
        "station_id": sid,
        "windows": [],
        "time_support_verified": False,
        "datum_verified": False,
        "datum_id": "unknown",
        "datum_evidence": "",
    }
    try:
        meta_path = snapshots / f"{sid}_site.json"
        meta = snapshot_json(
            f"https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items/USGS-{sid}?f=json",
            meta_path,
        )["properties"]
        if str(meta["monitoring_location_number"]) != sid:
            raise ValueError("USGS站号不一致")
        result["site_name"] = meta["monitoring_location_name"]
        result["site_metadata_vertical_datum"] = meta.get("vertical_datum")
        # 站点海拔/地面高程元数据不是2019—2021年水尺零点历史，不据此放行。
        result["datum_status"] = "historical_gage_zero_evidence_still_required"
        with open_netcdf(row.qh_path) as ds:
            h = ds.water_level.to_series()
        h.index = pd.DatetimeIndex(h.index).tz_localize("UTC")
        for start, end in SUPPORT_WINDOWS:
            path = snapshots / f"{sid}_H_{start}.json"
            query = urllib.parse.urlencode(
                {
                    "format": "json",
                    "sites": sid,
                    "startDT": start + "T00:00:00Z",
                    "endDT": end + "T00:00:00Z",
                    "parameterCd": "00065",
                    "siteStatus": "all",
                }
            )
            original = hourly_height(
                snapshot_json("https://waterservices.usgs.gov/nwis/iv/?" + query, path), start, end
            )
            archive = h.reindex(original.index)
            valid = original.notna() & archive.notna()
            error = np.abs(original[valid] - archive[valid])
            maximum = float(error.max()) if len(error) else None
            passed = len(error) >= 24 and maximum <= 1e-4
            result["windows"].append(
                {
                    "start": start,
                    "end_exclusive": end,
                    "paired_hours": len(error),
                    "max_abs_difference_m": maximum,
                    "passed": bool(passed),
                    "snapshot_sha256": digest(path),
                }
            )
        result["time_support_verified"] = all(w["passed"] for w in result["windows"])
        result["comparison_method"] = "exact_UTC_hour_instantaneous_no_nearest_or_averaging"
        result["time_support"] = "instantaneous" if result["time_support_verified"] else "unknown"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_station_evidence(manifest, output, snapshots, limit=20, seed=20261019):
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    pool = frame[
        frame.weather_available.eq(True)
        & frame.support_coverage.ge(0.8)
        & frame.query_coverage.ge(0.9)
        & frame.area_km2.between(20, 5000)
    ].copy()
    if "topology_verified" in pool:
        pool = pool[pool.topology_verified.eq(True)]
    pool["order"] = pool.station_id.map(lambda x: group_key(x, seed))
    pool = pool.sort_values("order").drop_duplicates("terminal_system_id").head(limit).copy()
    output, snapshots = Path(output), Path(snapshots)
    snapshots.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 先固定审核清单；不能依据对照误差换站或以此宣称已完成正式分层。
    pool.drop(columns="order").to_csv(
        output.with_suffix(".candidates.csv"), index=False, encoding="utf-8-sig"
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(
            executor.map(lambda row: reconcile_station(row, snapshots), [row for _, row in pool.iterrows()])
        )
    write_json(
        output,
        {
            "scope": "support_H_independent_processing_evidence_not_datum_approval",
            "created_utc": utc_now(),
            "manifest_sha256": digest(manifest),
            "seed": seed,
            "observed_Q_requested": False,
            "future_H_requested": False,
            "stations": records,
        },
    )
    evidence = {record["station_id"]: record for record in records}
    pool["time_support_verified"] = pool.station_id.map(lambda x: evidence[x]["time_support_verified"])
    pool["time_support_evidence"] = str(output.resolve()) + "#sha256=" + digest(output)
    # 当前程序不具备自动判定基准历史的能力，所有站明确保留未核验。
    pool["datum_verified"] = False
    pool.drop(columns="order").to_csv(output.with_suffix(".review.csv"), index=False, encoding="utf-8-sig")
    return {
        "reviewed": len(records),
        "time_support_verified": int(pool.time_support_verified.sum()),
        "historical_datum_verified": 0,
        "errors": [r for r in records if "error" in r],
    }
