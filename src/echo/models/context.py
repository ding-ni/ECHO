"""开发版LSTM与32维对角Gaussian地方推断。"""

import torch
from torch import nn


class HistoryEncoder(nn.Module):
    def __init__(self, geo_dim, hidden=64, layers=2, backbone="lstm"):
        super().__init__()
        if backbone != "lstm":
            raise NotImplementedError("正式骨干须经Linux GPU验证；这里不以LSTM冒充Mamba-3")
        self.lstm = nn.LSTM(8, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden + geo_dim, 64)

    def forward(self, tokens, geo, state=None):
        _, state = self.lstm(tokens, state)
        return torch.tanh(self.head(torch.cat([state[0][-1], geo], -1))), state


class JointGaussian(nn.Module):
    def __init__(self, geo_dim):
        super().__init__()
        self.posterior = nn.Linear(64 + geo_dim, 64)
        self.prior = nn.Linear(geo_dim, 64)

    def forward(self, context, geo, members=1, stochastic=True):
        mean, logsd = self.posterior(torch.cat([context, geo], -1)).chunk(2, -1)
        pm, pls = self.prior(geo).chunk(2, -1)
        logsd, pls = logsd.clamp(-5, 2), pls.clamp(-5, 2)
        eps = torch.randn((members,) + mean.shape, device=mean.device, dtype=mean.dtype)
        codes = mean[None] + eps * logsd.exp()[None] if stochastic else mean[None].expand(members, -1, -1)
        kl = (pls - logsd + ((2 * logsd).exp() + (mean - pm).square()) / (2 * (2 * pls).exp()) - 0.5).mean()
        return codes, kl
