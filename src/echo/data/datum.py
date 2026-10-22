"""仅依据独立基准元数据分段，不通过拟合误差或洪峰形状自动改正。"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from echo.utils import digest, read_json


def validate_registry_station(registry_path, station_id, start, end):
    """核对快照、站号、官方原文与实际任务窗口；布尔许可不能代替证据。"""
    registry_path = Path(registry_path)
    registry = read_json(registry_path)
    records = [r for r in registry["stations"] if r["station_id"] == str(station_id)]
    if len(records) != 1:
        raise ValueError("水尺基准登记站号缺失或重复")
    record = records[0]
    if record.get("review_status") != "historical_epoch_reviewed":
        raise ValueError("历史水尺基准尚未核验")
    snapshot = (registry_path.parent / record["snapshot"]).resolve()
    if not snapshot.is_file():
        raise ValueError("基准证据快照不存在")
    if digest(snapshot) != record["snapshot_sha256"]:
        raise ValueError("基准证据快照哈希不一致")
    tree = ET.parse(snapshot)
    if tree.findtext(".//site_no") != str(station_id):
        raise ValueError("基准证据快照站号不一致")
    receipt = read_json(snapshot.with_suffix(".provenance.json"))
    expected_url = (
        "https://sims-manuscripts-bucket.s3.us-west-2.amazonaws.com/"
        f"sims_USGS_{station_id}.xml"
    )
    if receipt.get("url") != expected_url or receipt.get("sha256") != record["snapshot_sha256"]:
        raise ValueError("基准证据来源收据不一致")
    gage = [e.findtext("info") for e in tree.findall(".//Element") if e.findtext("name") == "GAGE"]
    if len(gage) != 1 or gage[0] != record["gage_quote"]:
        raise ValueError("水尺基准登记原文与快照不一致")
    return assert_comparable_datum(start, end, record["epochs"])


def datum_epoch_at(time, epochs):
    time = pd.Timestamp(time)
    time = time.tz_localize("UTC") if time.tz is None else time.tz_convert("UTC")
    matches = []
    for epoch in epochs:
        if not epoch.get("evidence") or not epoch.get("datum_id"):
            raise ValueError("基准时期缺少独立证据")
        start = pd.Timestamp(epoch["start"])
        end = pd.Timestamp(epoch["end"])
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
        if start <= time < end:
            matches.append(epoch)
    if len(matches) != 1:
        raise ValueError("基准时期缺失或重叠")
    return matches[0]


def assert_comparable_datum(start, end, epochs):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("基准核验区间必须非空且递增")
    first = datum_epoch_at(start, epochs)
    last = datum_epoch_at(end - pd.Timedelta(nanoseconds=1), epochs)
    relevant = []
    for epoch in epochs:
        a, b = pd.Timestamp(epoch["start"]), pd.Timestamp(epoch["end"])
        a = a.tz_localize("UTC") if a.tz is None else a.tz_convert("UTC")
        b = b.tz_localize("UTC") if b.tz is None else b.tz_convert("UTC")
        if b <= a:
            raise ValueError("基准时期必须非空且递增")
        if a < end and b > start:
            relevant.append(epoch)
    # 仅检查两端会漏掉窗口中间嵌套的基准变更/重叠时期。
    if first is not last or len(relevant) != 1:
        raise ValueError("支持/未来跨基准时期；需有独立改正依据后另建标准化版本")
    return first["datum_id"]
