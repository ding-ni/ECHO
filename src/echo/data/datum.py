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
