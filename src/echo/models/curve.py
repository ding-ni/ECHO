"""固定节点的正斜率积分曲线，支持正斜率线性外推。"""

import torch
from torch import nn
from torch.nn import functional as F


def monotone_integral(u, raw_slopes, width=0.2):
    slopes = F.softplus(raw_slopes) + 1e-4
    starts = torch.arange(slopes.shape[-1], device=u.device, dtype=u.dtype) * width
    pieces = (u.unsqueeze(-1) - starts).clamp(0, width)
    mid = (pieces * slopes.unsqueeze(-2)).sum(-1)
    return (
        mid
        + slopes[..., :1] * u.clamp(max=0)
        + slopes[..., -1:] * (u - slopes.shape[-1] * width).clamp(min=0)
    )


class CurveGenerator(nn.Module):
    def __init__(self, geo_dim, segments=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_dim + 16, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, segments + 1),
        )

    def forward(self, u, geo, static_code, uref):
        params = self.net(torch.cat([geo, static_code], -1))
        return (
            params[:, :1]
            + monotone_integral(u, params[:, 1:])
            - monotone_integral(uref[:, None], params[:, 1:])
        )
