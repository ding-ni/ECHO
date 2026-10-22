"""单节点可微HBV开发前端，储量单位mm，步长小时，逐步水量闭合。"""

from functools import lru_cache

import torch
from torch import nn

from echo.hydro.routing import reservoir_step

LOWER = (20.0, 0.3, 0.2, 0.01, -2.0, 0.001, 1.0, 24.0, 0.25)
UPPER = (1000.0, 6.0, 1.0, 0.5, 2.0, 1.0, 120.0, 5000.0, 240.0)


def bounded_parameters(raw):
    low = raw.new_tensor(LOWER)
    high = raw.new_tensor(UPPER)
    return low + (high - low) * raw.sigmoid()


def hbv_step(weather, state, params):
    """weather[...,3]=P(mm/h),T(C),PET(mm/h)。返回径流深、水量余差与新储量。"""
    p, temp, pet = weather.unbind(-1)
    if torch.any(p < 0) or torch.any(pet < 0):
        raise ValueError("P/PET必须非负，不能静默修正原始资料")
    snow, soil, upper, lower, routed = state.unbind(-1)
    fc, beta, lp, ddf, threshold, perc_rate, kfast, kslow, tau = params.unbind(-1)
    old_total = state.sum(-1)
    snow_fraction = torch.sigmoid((threshold - temp) * 3.0)
    snow = snow + p * snow_fraction
    melt = torch.minimum(snow, ddf * torch.relu(temp - threshold))
    snow = snow - melt
    liquid = p * (1 - snow_fraction) + melt
    relative_soil = (soil / fc).clamp(1e-6, 1)
    recharge = liquid * torch.where(soil > 0, relative_soil.pow(beta), torch.zeros_like(soil))
    soil = soil + liquid - recharge
    excess = torch.relu(soil - fc)
    soil = soil - excess
    aet = torch.minimum(soil, pet * (soil / (lp * fc)).clamp(0, 1))
    soil = soil - aet
    upper = upper + recharge + excess
    perc = torch.minimum(upper, perc_rate)
    upper = upper - perc
    fast = upper * (-torch.expm1(-1.0 / kfast))
    upper = upper - fast
    lower = lower + perc
    slow = lower * (-torch.expm1(-1.0 / kslow))
    lower = lower - slow
    routed, runoff = reservoir_step(routed, fast + slow, tau)
    nxt = torch.stack([snow, soil, upper, lower, routed], -1)
    balance = nxt.sum(-1) - old_total - p + aet + runoff
    return runoff, nxt, balance


def _hbv_sequence(weather, state, params, area_km2):
    runoff: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []
    for x in weather.unbind(1):
        r, state, error = hbv_step(x, state, params)
        runoff.append(r)
        errors.append(error)
    q = torch.stack(runoff, 1) * area_km2[:, None] / 3.6
    return q, state, torch.stack(errors, 1)


@lru_cache(maxsize=1)
def inference_sequence():
    # 相同方程与小时顺序，仅消除无梯度长预热的Python逐步调度开销。
    return torch.jit.script(_hbv_sequence)


class RegionalHBV(nn.Module):
    def __init__(self, geo_dim, hidden=32):
        super().__init__()
        self.regionalizer = nn.Sequential(nn.Linear(geo_dim, hidden), nn.Tanh(), nn.Linear(hidden, 9))
        self.local_offset = nn.Linear(8, 9, bias=False)
        nn.init.normal_(self.local_offset.weight, std=0.01)

    def parameters_for(self, geo, code=None):
        raw = self.regionalizer(geo)
        if code is not None:
            raw = raw + 0.5 * torch.tanh(self.local_offset(code))
        return bounded_parameters(raw)

    def forward(self, weather, geo, area_km2, code=None, state=None):
        if weather.ndim != 3 or weather.shape[-1] != 3:
            raise ValueError("weather形状应为[batch,time,3]；空间版另行实现")
        params = self.parameters_for(geo, code)
        if state is None:
            state = weather.new_zeros((weather.shape[0], 5))
        if not torch.is_grad_enabled():
            return inference_sequence()(weather, state, params, area_km2)
        return _hbv_sequence(weather, state, params, area_km2)
