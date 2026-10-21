"""支持/查询任务构建；选择器只读取时间、支持掩码，不读取未来标签。"""

import numpy as np

from echo.data.permissions import ForecastInput, support_scaling


def hours_since(mask):
    age = np.zeros(len(mask), dtype="float32")
    last = -1
    for i, valid in enumerate(mask):
        if valid:
            last = i
        age[i] = min(i - last, 8760) / 8760.0
    return age


def tokens(weather, q, area, h, mask):
    center, scale = support_scaling(h, mask)
    normalized = np.where(mask, (h - center) / scale, 0.0)
    # 固定物理量尺度，不通过目标Q或未来H拟合统计量。
    w = weather / np.array([10.0, 20.0, 1.0], dtype="float32")
    u = np.log1p(np.maximum(q, 0) / (area / 86.4))
    phase = np.sin(np.arange(len(h)) / 8766.0 * 2 * np.pi)
    return (
        np.column_stack([w, u, normalized, mask, hours_since(mask), phase]).astype("float32"),
        center,
        scale,
    )


def source_episode(data, support_start, support_hours, query_hours=168):
    end = support_start + support_hours
    stop = end + query_hours
    h = data["h"][support_start:end]
    mask = np.isfinite(h)
    if mask.mean() < 0.8:
        raise ValueError("episode支持覆盖不足")
    inputs = ForecastInput(
        str(data["station_id"]),
        str(data["system"]),
        data["weather"][:stop],
        data["geo"],
        float(data["area"]),
        h,
        mask,
        support_start,
        end,
        end,
        stop,
    )
    # 查询H返回独立标签对象；绝不合并到ForecastInput。
    return inputs, data["h"][end:stop].copy()


def target_input(data, budget_days=None):
    end = int(data["support_end"])
    h = data["support_h"]
    if budget_days is not None:
        h = h[-budget_days * 24 :]
    start = end - len(h)
    return ForecastInput(
        str(data["station_id"]),
        str(data["system"]),
        data["weather"],
        data["geo"],
        float(data["area"]),
        h,
        np.isfinite(h),
        start,
        end,
        end,
        len(data["weather"]),
    )
