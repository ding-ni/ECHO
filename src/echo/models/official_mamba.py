"""Thin adapter to a pinned official Mamba-3; never substitutes its SSM core."""

import importlib

import torch
from torch import nn

OFFICIAL_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
OFFICIAL_MODULE_SHA256 = "930c3dfa04dea8444b1c9ee8b6ac9cbbc7ef492dc5ae1f4c83051ef953eca33c"


class OfficialHistoryEncoder(nn.Module):
    def __init__(self, token_dim=8):
        super().__init__()
        try:
            module = importlib.import_module("mamba_ssm.modules.mamba3")
        except (ImportError, OSError) as exc:
            raise RuntimeError("正式主模型需要官方Mamba-3及Linux CUDA/TileLang环境，不能回退到自写SSM") from exc
        from echo.utils import digest
        if not module.__file__ or digest(module.__file__) != OFFICIAL_MODULE_SHA256:
            raise RuntimeError("官方Mamba-3源码与项目锁定提交不一致，请先安装固定提交")
        self.input = nn.Linear(token_dim, 256)
        self.norms = nn.ModuleList(nn.LayerNorm(256) for _ in range(6))
        self.blocks = nn.ModuleList(module.Mamba3(
            d_model=256, d_state=128, headdim=64, is_mimo=True, mimo_rank=4,
            chunk_size=16, is_outproj_norm=False, layer_idx=i, dtype=torch.bfloat16,
        ) for i in range(6))
        # No generic reinitialization: preserve official dt_bias and MIMO initialization.
        self.film = nn.ModuleList(nn.Linear(256, 512) for _ in range(6))
        self.queries = nn.Parameter(torch.randn(1, 16, 256) * .02)
        self.query_condition = nn.Linear(256, 16 * 256)
        self.pool = nn.MultiheadAttention(256, 4, batch_first=True)
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(16 * 256, 256), nn.GELU(), nn.LayerNorm(256))

    def forward(self, tokens, environment):
        if tokens.device.type != "cuda":
            raise RuntimeError("官方Mamba-3主模型前向需要CUDA；CPU不得自动替换骨干")
        x = self.input(tokens)
        for norm, film, block in zip(self.norms, self.film, self.blocks, strict=True):
            scale, shift = film(environment).chunk(2, -1)
            conditioned = norm(x) * (1 + scale[:, None]) + shift[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                update = block(conditioned.to(torch.bfloat16))
            x = x + update.to(x.dtype)
        query = self.queries + self.query_condition(environment).reshape(-1, 16, 256)
        pooled = self.pool(query, x, x, need_weights=False)[0]
        return self.output(pooled)
