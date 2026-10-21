"""仅依据独立基准元数据分段，不通过拟合误差或洪峰形状自动改正。"""

import pandas as pd


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
    first = datum_epoch_at(start, epochs)
    last = datum_epoch_at(pd.Timestamp(end) - pd.Timedelta(nanoseconds=1), epochs)
    if first is not last:
        raise ValueError("支持/未来跨基准时期；需有独立改正依据后另建标准化版本")
    return first["datum_id"]
