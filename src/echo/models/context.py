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
    def __init__(self, geo_dim, rank=0):
        super().__init__()
        self.rank = rank
        self.posterior = nn.Linear(64 + geo_dim, 64)
        self.prior = nn.Linear(geo_dim, 64)
        if rank:
            self.factor = nn.Linear(64 + geo_dim, 32 * rank)
            nn.init.zeros_(self.factor.weight)
            nn.init.normal_(self.factor.bias, std=0.01)

    def statistics(self, context, geo):
        condition = torch.cat([context, geo], -1)
        mean, logsd = self.posterior(condition).chunk(2, -1)
        logsd = logsd.clamp(-5, 2)
        factor = (0.25 * self.factor(condition).tanh().reshape(-1, 32, self.rank)
                  if self.rank else None)
        return mean, logsd, factor

    def forward(self, context, geo, members=1, stochastic=True):
        mean, logsd, factor = self.statistics(context, geo)
        pm, pls = self.prior(geo).chunk(2, -1)
        logsd, pls = logsd.clamp(-5, 2), pls.clamp(-5, 2)
        eps = torch.randn((members,) + mean.shape, device=mean.device, dtype=mean.dtype)
        codes = mean[None] + eps * logsd.exp()[None] if stochastic else mean[None].expand(members, -1, -1)
        kl = (pls - logsd + ((2 * logsd).exp() + (mean - pm).square()) / (2 * (2 * pls).exp()) - 0.5).mean()
        if factor is not None:
            # 对角加低秩协方差跨水文/曲线/动态三组；一次采样用于整条未来轨迹。
            if stochastic:
                shared = torch.randn((members, len(mean), self.rank, 1), device=mean.device)
                codes = codes + (factor[None] @ shared).squeeze(-1)
            scaled = factor / logsd.exp()[:, :, None]
            eye = torch.eye(self.rank, device=mean.device, dtype=mean.dtype)
            chol = torch.linalg.cholesky(eye + scaled.transpose(-1, -2) @ scaled)
            logdet_ratio = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            extra_trace = (factor.square().sum(-1) / (2 * pls).exp()).sum(-1)
            kl = kl + ((extra_trace - logdet_ratio) / (2 * mean.shape[-1])).mean()
        return codes, kl
