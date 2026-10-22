"""固定来源训练水系的USGS 00060流量时间支持核验。"""

import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.download import request
from echo.data.netcdf import open_netcdf
from echo.data.splits import group_key
from echo.data.usgs_evidence import hourly_height
from echo.utils import digest, utc_now, write_json

SOURCE_WINDOWS = [("2018-10-01", "2018-10-03"), ("2019-03-07", "2019-03-10")]
FT3_TO_M3 = 0.028316846592


def _snapshot(url, path):
    path = Path(path)
    if path.exists():
        receipt = json.loads(path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        if receipt.get("url") != url or receipt.get("sha256") != digest(path):
            raise ValueError("USGS流量证据缓存与收据不一致")
        data = json.loads(path.read_text(encoding="utf-8"))
        _assert_response_station(data, url)
        return data
    with request(url) as response:
        data = json.loads(response.read())
    _assert_response_station(data, url)
    write_json(path, data)
    write_json(path.with_suffix(".provenance.json"), {"url": url, "retrieved_utc": utc_now(), "sha256": digest(path)})
    return data


def _assert_response_station(payload, url):
    requested = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["sites"][0]
    for series in payload["value"]["timeSeries"]:
        codes = {str(code["value"]) for code in series["sourceInfo"]["siteCode"]}
        if codes != {requested}:
            raise ValueError("USGS响应站号与请求不一致")


def _hourly_q(payload, start, end):
    series = payload["value"]["timeSeries"]
    if len(series) != 1:
        raise ValueError("USGS流量序列缺失或存在多种方法，不能自动择优")
    item = series[0]
    codes = {str(code["value"]) for code in item["variable"]["variableCode"]}
    if codes != {"00060"} or item["variable"]["unit"]["unitCode"] != "ft3/s":
        raise ValueError("只接受USGS参数00060、ft3/s单位")
    values = item["values"]
    if len(values) != 1:
        raise ValueError("多套USGS流量方法尚未核验")
    records = values[0]["value"]
    times = pd.to_datetime([record["dateTime"] for record in records], utc=True)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("USGS流量时间重复或逆序")
    values = pd.to_numeric(pd.Series([record["value"] for record in records]), errors="coerce").to_numpy(float)
    values[values == item["variable"]["noDataValue"]] = np.nan
    series = pd.Series(values * FT3_TO_M3, index=times)
    start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    series = series[(series.index >= start) & (series.index < end)]
    return series[(series.index.minute == 0) & (series.index.second == 0) & (series.index.microsecond == 0)]


def _station(row, snapshots):
    sid = str(row.station_id)
    result = {"station_id": sid, "windows": [], "time_support_verified": False,
              "source_h_time_support_verified": False}
    try:
        result["qh_sha256"] = digest(row.qh_path)
        with open_netcdf(row.qh_path) as ds:
            archive = ds.streamflow.to_series()
            height = ds.water_level.to_series()
        archive.index = pd.DatetimeIndex(archive.index).tz_localize("UTC")
        height.index = pd.DatetimeIndex(height.index).tz_localize("UTC")
        for start, end in SOURCE_WINDOWS:
            query = urllib.parse.urlencode(
                {
                    "format": "json",
                    "sites": sid,
                    "startDT": start + "T00:00:00Z",
                    "endDT": end + "T00:00:00Z",
                    "parameterCd": "00060",
                    "siteStatus": "all",
                }
            )
            path = Path(snapshots) / f"{sid}_Q_{start}.json"
            official = _hourly_q(_snapshot("https://waterservices.usgs.gov/nwis/iv/?" + query, path), start, end)
            archived = archive.reindex(official.index)
            valid = official.notna() & archived.notna()
            errors = np.abs(official[valid] - archived[valid])
            hquery = urllib.parse.urlencode(
                {"format": "json", "sites": sid, "startDT": start + "T00:00:00Z",
                 "endDT": end + "T00:00:00Z", "parameterCd": "00065", "siteStatus": "all"}
            )
            hpath = Path(snapshots) / f"{sid}_H_{start}.json"
            h_original = hourly_height(
                _snapshot("https://waterservices.usgs.gov/nwis/iv/?" + hquery, hpath), start, end
            )
            h_archived = height.reindex(h_original.index)
            h_valid = h_original.notna() & h_archived.notna()
            h_errors = np.abs(h_original[h_valid] - h_archived[h_valid])
            result["windows"].append(
                {
                    "start": start,
                    "end_exclusive": end,
                    "official_hourly_points": int(official.notna().sum()),
                    "paired_hours": int(valid.sum()),
                    "archive_valid_fraction_at_official_hours": float(valid.mean()) if len(valid) else 0.0,
                    "max_abs_difference_m3_s": float(errors.max()) if len(errors) else None,
                    "passed": bool(len(errors) >= 24 and float(errors.max()) <= 1e-3) if len(errors) else False,
                    "snapshot_sha256": digest(path),
                    "height_paired_hours": int(h_valid.sum()),
                    "height_max_abs_difference_m": float(h_errors.max()) if len(h_errors) else None,
                    "height_passed": bool(len(h_errors) >= 24 and float(h_errors.max()) <= 1e-4)
                    if len(h_errors) else False,
                    "height_snapshot_sha256": digest(hpath),
                }
            )
        result["time_support_verified"] = all(window["passed"] for window in result["windows"])
        result["source_h_time_support_verified"] = all(
            window["height_passed"] for window in result["windows"]
        )
        result["time_support"] = "instantaneous_UTC_hour" if result["time_support_verified"] else "unknown"
        result["comparison_method"] = "exact_UTC_hour_instantaneous_no_nearest_or_averaging"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def validate_source_evidence(report_path, row, roles_path):
    """只从既有快照复算许可；不让清单布尔值替代来源Q/H证据。"""
    from echo.utils import read_json

    report = read_json(report_path)
    if report.get("roles_sha256") != digest(roles_path):
        raise ValueError("来源观测核验与冻结水系角色不一致")
    records = [record for record in report["stations"] if record["station_id"] == str(row.station_id)]
    if len(records) != 1:
        raise ValueError("来源观测核验站号缺失或重复")
    record = records[0]
    if record.get("qh_sha256") != digest(row.qh_path):
        raise ValueError("来源观测原始文件哈希不一致")
    snapshots = Path(report["snapshots_path"])
    for start, _ in SOURCE_WINDOWS:
        for kind in ["Q", "H"]:
            path = snapshots / f"{row.station_id}_{kind}_{start}.json"
            if not path.is_file() or not path.with_suffix(".provenance.json").is_file():
                raise ValueError("来源观测快照不完整；准备流程不得临时联网补证")
    actual = _station(row, snapshots)
    if not (actual["time_support_verified"] and actual["source_h_time_support_verified"]):
        raise ValueError("来源训练期Q/H时间支持核验不通过")
    if actual != record:
        raise ValueError("来源观测快照复算结果与登记不一致")
    return True


def audit_source_q(manifest, roles_path, output, snapshots, limit=None, seed=20261019):
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    roles = json.loads(Path(roles_path).read_text(encoding="utf-8"))
    train_systems = {
        row["terminal_system_id"] for row in roles["systems"] if row["system_split"] == "train"
    }
    pool = frame[
        frame.terminal_system_id.isin(train_systems)
        & frame.weather_available.eq(True)
        & frame.topology_verified.eq(True)
        & frame.support_coverage.ge(0.8)
        & frame.area_km2.between(20, 5000)
    ].copy()
    pool["order"] = pool.station_id.map(lambda value: group_key(value, seed))
    pool = pool.sort_values("order").drop_duplicates("terminal_system_id")
    if limit is not None:
        pool = pool.head(limit)
    snapshots = Path(snapshots)
    snapshots.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda row: _station(row, snapshots), [row for _, row in pool.iterrows()]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        output,
        {
            "scope": "source_train_systems_only",
            "created_utc": utc_now(),
            "manifest_sha256": digest(manifest),
            "roles_sha256": digest(roles_path),
            "roles_path": str(Path(roles_path).resolve()),
            "snapshots_path": str(snapshots.resolve()),
            "observed_Q_requested": True,
            "target_and_blind_Q_requested": False,
            "source_windows": SOURCE_WINDOWS,
            "stations": records,
        },
    )
    review = pool.drop(columns="order").copy()
    evidence = {item["station_id"]: item for item in records}
    review["source_time_support_verified"] = review.station_id.map(
        lambda value: evidence[value]["time_support_verified"]
        and evidence[value]["source_h_time_support_verified"]
    )
    review["source_time_support_evidence"] = str(output.resolve()) + "#sha256=" + digest(output)
    review.to_csv(output.with_suffix(".review.csv"), index=False, encoding="utf-8-sig")
    return {
        "reviewed": len(records),
        "time_support_verified": int(review.source_time_support_verified.sum()),
        "errors": [item for item in records if "error" in item],
    }
