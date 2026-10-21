"""统一地方响应生成器；未来天气只能经HBV进入水位。"""

import torch
from torch import nn

from echo.models.context import HistoryEncoder, JointGaussian
from echo.models.curve import CurveGenerator
from echo.models.dynamics import LocalDynamics


class EchoStage(nn.Module):
    def __init__(self, geo_dim, dynamic=False, probabilistic=False, hidden=64):
        super().__init__()
        self.encoder = HistoryEncoder(geo_dim, hidden)
        self.posterior = JointGaussian(geo_dim)
        self.curve = CurveGenerator(geo_dim)
        self.dynamic = LocalDynamics(geo_dim) if dynamic else None
        self.probabilistic = probabilistic

    def infer(self, tokens, geo, members=1):
        context, _ = self.encoder(tokens, geo)
        return self.posterior(context, geo, members, self.probabilistic)

    def generate(self, q, area, geo, code, center, scale, uref, state=None):
        u = torch.log1p(q.clamp_min(0) / (area[:, None] / 86.4))
        h = self.curve(u, geo, code[:, 8:24], uref)
        if self.dynamic is not None:
            residual, state = self.dynamic(u, geo, code[:, 24:], state)
            h = h + residual
        return center[:, None] + scale[:, None] * h, state


class DirectStage(nn.Module):
    def __init__(self, geo_dim, hidden=64):
        super().__init__()
        self.encoder = nn.LSTM(7 + geo_dim, hidden, 2, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, weather, geo, h_input, mask, age, phase, state=None):
        g = geo[:, None].expand(-1, weather.shape[1], -1)
        x = torch.cat(
            [weather, g, h_input[:, :, None], mask[:, :, None], age[:, :, None], phase[:, :, None]], -1
        )
        out, state = self.encoder(x, state)
        return self.head(out).squeeze(-1), state
