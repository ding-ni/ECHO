"""输入选择阻尼旋转：门控不读取旧状态，零态读出严格为零。"""

import torch
from torch import nn


class LocalDynamics(nn.Module):
    def __init__(self, geo_dim, blocks=8):
        super().__init__()
        self.blocks = blocks
        self.gates = nn.Sequential(nn.Linear(geo_dim + 8 + 1, 64), nn.Tanh(), nn.Linear(64, blocks * 6))
        nn.init.zeros_(self.gates[-1].weight)
        nn.init.zeros_(self.gates[-1].bias)
        initial = torch.tensor([1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 719.0])[:blocks]
        fraction = ((initial - 0.25) / (720.0 - 0.25)).clamp(0.0001, 0.9999)
        with torch.no_grad():
            self.gates[-1].bias.view(blocks, 6)[:, 0].copy_(torch.logit(fraction))
        self.readout = nn.Sequential(nn.Linear(blocks * 2, 64), nn.Tanh(), nn.Linear(64, 1))

    def step(self, u, delta, delta2, geo, code, z):
        gates = self.gates(torch.cat([u[:, None], geo, code], -1)).reshape(-1, self.blocks, 6)
        tau = 0.25 + 719.75 * gates[:, :, 0].sigmoid()
        phase = 0.5 * gates[:, :, 1].tanh()
        rho = torch.exp(-1.0 / tau)
        drive = torch.tanh(
            gates[:, :, 2:].reshape(-1, self.blocks, 2, 2)
            @ torch.stack([delta, delta2], -1)[:, None, :, None]
        ).squeeze(-1)
        x, y = z.unbind(-1)
        rotated = torch.stack([phase.cos() * x - phase.sin() * y, phase.sin() * x + phase.cos() * y], -1)
        nxt = rho[:, :, None] * rotated + (1 - rho[:, :, None]) * drive
        flat = nxt.flatten(1)
        # 显式中心化，普通MLP的bias不能产生恒定流量下的非零残差。
        r = self.readout(flat) - self.readout(torch.zeros_like(flat))
        return r.squeeze(-1), nxt

    def forward(self, u, geo, code, state=None):
        if state is None:
            z = u.new_zeros((len(u), self.blocks, 2))
            previous, last_delta = u[:, 0], torch.zeros_like(u[:, 0])
        else:
            z, previous, last_delta = state
        values = []
        for value in u.unbind(1):
            delta = value - previous
            r, z = self.step(value, delta, delta - last_delta, geo, code, z)
            values.append(r)
            previous, last_delta = value, delta
        return torch.stack(values, 1), (z, previous, last_delta)
