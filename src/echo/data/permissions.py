"""预测对象不含未来标签或目标Q。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastInput:
    station_id: str
    terminal_system_id: str
    weather: np.ndarray
    geography: np.ndarray
    area_km2: float
    support_h: np.ndarray
    support_mask: np.ndarray
    support_start: int
    support_end: int
    query_start: int
    query_end: int

    def __post_init__(self):
        if not (
            0
            <= self.support_start
            < self.support_end
            <= self.query_start
            < self.query_end
            <= len(self.weather)
        ):
            raise ValueError("支持与查询时间区间必须过去在前、未来在后")
        if len(self.support_h) != self.support_end - self.support_start:
            raise ValueError("支持H长度不匹配")
        if self.support_mask.shape != self.support_h.shape:
            raise ValueError("缺测掩码形状不一致")
        if not self.terminal_system_id or self.area_km2 <= 0:
            raise ValueError("缺少水系或有效面积")


def assert_source_permissions(source_systems, forbidden_systems):
    overlap = set(source_systems) & set(forbidden_systems)
    if overlap:
        raise ValueError(f"禁用水系Q进入来源训练: {sorted(overlap)}")


def support_scaling(h, mask):
    vals = np.asarray(h)[np.asarray(mask, dtype=bool) & np.isfinite(h)]
    if not len(vals):
        raise ValueError("没有许可历史H")
    return float(np.median(vals)), float(max(np.subtract(*np.percentile(vals, [75, 25])), 0.1))
