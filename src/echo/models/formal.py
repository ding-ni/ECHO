"""Protocol-sized ECHO modules with a reduced CPU/GPU reference configuration."""

from __future__ import annotations

import math

import torch
from nflows.distributions.normal import StandardNormal
from nflows.nn.nets import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.permutations import RandomPermutation
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from echo.models.curve import CurveGenerator
from echo.models.dynamics import LocalDynamics


class SetAttentionBlock(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x, padding_mask=None):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, key_padding_mask=padding_mask, need_weights=False)[0]
        return x + self.feedforward(self.norm2(x))


class LocalTerrainSetEncoder(nn.Module):
    """Permutation invariant encoder for 16 relative-position terrain samples."""

    def __init__(self, feature_dim=10, width=128, heads=4, layers=2):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(SetAttentionBlock(width, heads) for _ in range(layers))
        self.seed = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens, padding_mask=None):
        if tokens.ndim != 3 or tokens.shape[1] != 16:
            raise ValueError("局地地形输入必须为[B,16,F]")
        x = self.input(tokens)
        for block in self.blocks:
            x = block(x, padding_mask)
        seed = self.seed.expand(len(x), -1, -1)
        pooled = self.pool(seed, x, x, key_padding_mask=padding_mask, need_weights=False)[0]
        return self.norm(pooled[:, 0])


class BasinGATv2Encoder(nn.Module):
    """Three-layer area-aware GATv2 encoder over incremental subbasin graphs."""

    def __init__(self, node_feature_dim=8, width=128, layers=3, heads=4, edge_feature_dim=None):
        super().__init__()
        self.input = nn.Linear(node_feature_dim, width)
        self.convs = nn.ModuleList(
            GATv2Conv(width, width, heads=heads, concat=False, add_self_loops=True,
                      edge_dim=edge_feature_dim) for _ in range(layers)
        )
        self.edge_feature_dim = edge_feature_dim
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.output = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))

    @staticmethod
    def _batch_edges(edge_index, batch, nodes, device):
        if edge_index.ndim == 2:
            base = edge_index.to(device)
            offsets = torch.arange(batch, device=device)[:, None, None] * nodes
            return (base[None] + offsets).permute(1, 0, 2).reshape(2, -1)
        if edge_index.ndim != 3 or edge_index.shape[:2] != (batch, 2):
            raise ValueError("edge_index必须为[2,E]或[B,2,E]")
        edges, offsets = edge_index.to(device), torch.arange(batch, device=device)[:, None, None] * nodes
        valid = (edges >= 0).all(1)
        source = edges[:, 0] + offsets[:, 0]
        target = edges[:, 1] + offsets[:, 0]
        return torch.stack([source[valid], target[valid]], 0)

    def forward(self, node_features, node_areas, edge_index, node_mask=None, edge_attr=None):
        if node_features.ndim != 3:
            raise ValueError("流域节点特征必须为[B,N,F]")
        batch, nodes, _ = node_features.shape
        mask = torch.ones((batch, nodes), dtype=torch.bool, device=node_features.device)
        if node_mask is not None:
            mask = node_mask.bool()
        if not mask.any(1).all() or (node_areas[mask] <= 0).any():
            raise ValueError("每个河网至少有一个正面积增量节点")
        edge = self._batch_edges(edge_index, batch, nodes, node_features.device)
        attributes = None
        if self.edge_feature_dim is not None:
            if edge_attr is None or edge_attr.shape[-1] != self.edge_feature_dim:
                raise ValueError("边条件化GATv2需要匹配的真实边属性与缺测标记")
            if edge_index.ndim == 2:
                attributes = edge_attr[None].expand(batch, -1, -1).reshape(-1, self.edge_feature_dim)
            else:
                attributes = edge_attr[(edge_index >= 0).all(1)]
            attributes = attributes.to(node_features.device)
        x = self.input(node_features.reshape(batch * nodes, -1))
        for conv, norm in zip(self.convs, self.norms, strict=True):
            update = F.elu(conv(x, edge, edge_attr=attributes))
            x = norm(x + update)
        x = x.reshape(batch, nodes, -1)
        weight = torch.where(mask, node_areas.clamp_min(0), torch.zeros_like(node_areas))
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
        mean = (x * weight[..., None]).sum(1)
        maximum = x.masked_fill(~mask[..., None], torch.finfo(x.dtype).min).amax(1)
        return self.output(torch.cat([mean, maximum], -1))


class SpatialEnvironmentEncoder(nn.Module):
    def __init__(self, node_feature_dim=8, terrain_feature_dim=10, climate_dim=64,
                 width=128, heads=4, terrain_layers=2, edge_feature_dim=None):
        super().__init__()
        self.basin = BasinGATv2Encoder(node_feature_dim, width, layers=3, heads=heads,
                                      edge_feature_dim=edge_feature_dim)
        self.local = LocalTerrainSetEncoder(terrain_feature_dim, width, heads, terrain_layers)
        self.climate = nn.Sequential(nn.Linear(climate_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.LayerNorm(64))
        self.fusion = nn.Sequential(nn.Linear(width * 2 + 64, 256), nn.GELU(), nn.Linear(256, 256), nn.LayerNorm(256))

    def forward(self, geography):
        basin = self.basin(geography["node_features"], geography["node_areas"],
                           geography["edge_index"], geography.get("node_mask"), geography.get("edge_attr"))
        local = self.local(geography["terrain_tokens"], geography.get("terrain_padding_mask"))
        climate = self.climate(geography["climate_features"])
        return self.fusion(torch.cat([basin, local, climate], -1))


class Mamba3ReferenceBlock(nn.Module):
    """CPU-capable selective SSM reference scan; not the official fused Mamba-3 CUDA kernel."""

    def __init__(self, width=64, state_dim=16, head_dim=16, mimo_rank=4, chunk_size=16):
        super().__init__()
        if width % head_dim:
            raise ValueError("Mamba head_dim必须整除d_model")
        self.width, self.state_dim, self.head_dim = width, state_dim, head_dim
        self.heads, self.rank, self.chunk_size = width // head_dim, mimo_rank, chunk_size
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, width * 2)
        self.conv = nn.Conv1d(width, width, kernel_size=4, groups=width)
        self.selective = nn.Linear(width, self.heads * (state_dim * 2 + 1))
        self.log_a = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float())[None].repeat(self.heads, 1))
        self.in_mix = nn.Parameter(torch.randn(self.heads, head_dim, mimo_rank) / math.sqrt(head_dim))
        self.out_mix = nn.Parameter(torch.randn(self.heads, mimo_rank, head_dim) / math.sqrt(mimo_rank))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        drive, gate = self.in_proj(x).chunk(2, -1)
        drive = self.conv(F.pad(drive.transpose(1, 2), (3, 0))).transpose(1, 2)
        drive = F.silu(drive).reshape(len(x), x.shape[1], self.heads, self.head_dim)
        params = self.selective(x)
        b, c, dt = torch.split(params, [self.heads * self.state_dim,
                                        self.heads * self.state_dim, self.heads], -1)
        b = b.reshape(len(x), x.shape[1], self.heads, self.state_dim).tanh()
        c = c.reshape(len(x), x.shape[1], self.heads, self.state_dim).tanh()
        dt = F.softplus(dt.reshape(len(x), x.shape[1], self.heads) - 3.0).clamp(1e-4, 1.0)
        low_rank = torch.einsum("blhd,hdr->blhr", drive, self.in_mix)
        a = -self.log_a.exp()[None]
        state = x.new_zeros((len(x), self.heads, self.rank, self.state_dim), dtype=torch.float32)
        outputs = []
        for chunk_start in range(0, x.shape[1], self.chunk_size):
            for t in range(chunk_start, min(chunk_start + self.chunk_size, x.shape[1])):
                step = dt[:, t, :, None].float()
                denominator = 1.0 - 0.5 * step * a
                a_bar = (1.0 + 0.5 * step * a) / denominator
                b_bar = step / denominator
                state = (a_bar[:, :, None] * state
                         + b_bar[:, :, None] * b[:, t, :, None] * low_rank[:, t, :, :, None].float())
                y_rank = (state * c[:, t, :, None].float()).sum(-1)
                outputs.append(torch.einsum("bhr,hrd->bhd", y_rank, self.out_mix))
        y = torch.stack(outputs, 1).reshape_as(x)
        return residual + self.out_proj(y * F.silu(gate))


class Mamba3HistoryEncoder(nn.Module):
    """Six-block MIMO selective SSM reference plus 16 geography-conditioned queries."""

    def __init__(self, token_dim=8, width=256, state_dim=128, head_dim=64, rank=4,
                 blocks=6, chunk_size=16, geo_dim=256, pooling_queries=16, pooling_heads=4,
                 backend="torch_reference"):
        super().__init__()
        if width % pooling_heads:
            raise ValueError("pooling heads must divide d_model")
        if backend not in {"torch_reference", "official_cuda"}:
            raise ValueError("未知Mamba-3后端")
        if backend == "official_cuda":
            try:
                from mamba_ssm import Mamba3  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("官方Mamba-3 CUDA内核未安装；本机参考扫描不会冒充官方后端") from exc
            raise RuntimeError("检测到mamba_ssm但未找到与协议锁定的Mamba-3接口版本")
        self.backend = backend
        self.input = nn.Linear(token_dim, width)
        self.blocks = nn.ModuleList(
            Mamba3ReferenceBlock(width, state_dim, head_dim, rank, chunk_size) for _ in range(blocks)
        )
        self.film = nn.ModuleList(nn.Linear(geo_dim, width * 2) for _ in range(blocks))
        self.queries = nn.Parameter(torch.randn(1, pooling_queries, width) * 0.02)
        self.query_condition = nn.Linear(geo_dim, pooling_queries * width)
        self.pool = nn.MultiheadAttention(width, pooling_heads, batch_first=True)
        self.output = nn.Sequential(nn.Flatten(), nn.Linear(pooling_queries * width, 256), nn.GELU(), nn.LayerNorm(256))

    def forward(self, tokens, environment):
        x = self.input(tokens)
        for block, film in zip(self.blocks, self.film, strict=True):
            scale, shift = film(environment).chunk(2, -1)
            x = block(x * (1 + scale[:, None]) + shift[:, None])
        query = self.queries + self.query_condition(environment).reshape(len(x), -1, x.shape[-1])
        pooled = self.pool(query, x, x, need_weights=False)[0]
        return self.output(pooled)


class ConditionalJointSplinePosterior(nn.Module):
    """32-D conditional RQ-spline flow posterior with mixed inter-group permutations."""

    def __init__(self, context_dim=256, environment_dim=256, code_dim=32,
                 layers=8, bins=8, hidden=128, tail_bound=5.0, conditioner_factory=ResidualNet):
        super().__init__()
        self.code_dim = code_dim
        condition_dim = context_dim + environment_dim
        self.base = nn.Linear(condition_dim, code_dim * 2)
        self.prior = nn.Linear(environment_dim, code_dim * 2)
        transforms = []
        for layer in range(layers):
            mask = (torch.arange(code_dim) + layer) % 2
            transforms.append(PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=lambda in_features, out_features: conditioner_factory(
                    in_features=in_features, out_features=out_features, hidden_features=hidden,
                    context_features=condition_dim, num_blocks=2, use_batch_norm=False),
                num_bins=bins, tails="linear", tail_bound=tail_bound))
            if layer < layers - 1:
                transforms.append(RandomPermutation(features=code_dim))
        self.flow = CompositeTransform(transforms)
        self.unit = StandardNormal([code_dim])

    def forward(self, context, environment, members=1, stochastic=True):
        condition = torch.cat([context, environment], -1)
        base_mean, base_logsd = self.base(condition).chunk(2, -1)
        base_logsd = base_logsd.clamp(-5, 2)
        epsilon = torch.randn((members,) + base_mean.shape, device=context.device, dtype=context.dtype)
        base = base_mean[None] + epsilon * base_logsd.exp()[None] if stochastic else base_mean[None].expand(members, -1, -1)
        flat_base = base.reshape(-1, self.code_dim)
        flat_context = condition[None].expand(members, -1, -1).reshape(-1, condition.shape[-1])
        codes, logdet = self.flow(flat_base, context=flat_context)
        log_q = (self.unit.log_prob(epsilon.reshape(-1, self.code_dim))
                 - base_logsd[None].expand(members, -1, -1).reshape(-1, self.code_dim).sum(-1)
                 - logdet)
        prior_mean, prior_logsd = self.prior(environment).chunk(2, -1)
        prior_logsd = prior_logsd.clamp(-5, 2)[None].expand(members, -1, -1).reshape(-1, self.code_dim)
        prior_mean = prior_mean[None].expand(members, -1, -1).reshape(-1, self.code_dim)
        log_p = torch.distributions.Normal(prior_mean, prior_logsd.exp()).log_prob(codes).sum(-1)
        kl = (log_q - log_p).mean()
        return codes.reshape(members, len(context), self.code_dim), kl


class FormalECHO(nn.Module):
    """Full structural ECHO graph with a local selective-SSM reference backend."""

    def __init__(self, node_feature_dim=8, terrain_feature_dim=10, climate_dim=64,
                 token_dim=8, width=256, state_dim=128, head_dim=64, rank=4,
                 mamba_blocks=6, pooling_queries=16, pooling_heads=4,
                 posterior_layers=8, spline_bins=8, posterior_hidden=128,
                 curve_segments=32, dynamic_blocks=8, mamba_backend="torch_reference"):
        super().__init__()
        self.environment_encoder = SpatialEnvironmentEncoder(
            node_feature_dim, terrain_feature_dim, climate_dim, width=128, heads=pooling_heads,
            terrain_layers=2)
        self.history_encoder = Mamba3HistoryEncoder(
            token_dim, width, state_dim, head_dim, rank, mamba_blocks, 16, 256,
            pooling_queries, pooling_heads, backend=mamba_backend)
        self.mamba_backend = mamba_backend
        self.posterior = ConditionalJointSplinePosterior(
            256, 256, 32, posterior_layers, spline_bins, posterior_hidden)
        from echo.hydro.spatial import SemiDistributedHBV
        self.hydrology = SemiDistributedHBV(node_feature_dim, hidden=32, local_code_dim=8)
        self.hydro_adapter = nn.Linear(8, 9, bias=False)
        nn.init.zeros_(self.hydro_adapter.weight)
        self.curve = CurveGenerator(256, curve_segments, hidden=128)
        self.dynamic = LocalDynamics(256, blocks=dynamic_blocks)

    def infer(self, history_tokens, geography, members=4, stochastic=True):
        environment = self.environment_encoder(geography)
        context = self.history_encoder(history_tokens, environment)
        codes, kl = self.posterior(context, environment, members, stochastic)
        return codes, kl, environment

    def generate(self, q, area, environment, code, center, scale, uref, state=None):
        u = torch.log1p(q.clamp_min(0) / (area[:, None] / 86.4))
        h = self.curve(u, environment, code[:, 8:24], uref)
        residual, state = self.dynamic(u, environment, code[:, 24:], state)
        return center[:, None] + scale[:, None] * (h + residual), state

    def hydro_parameter_delta(self, code):
        return 0.5 * torch.tanh(self.hydro_adapter(code[..., :8]))

    def simulate_hydrology(self, weather, node_features, node_areas, downstream_parent, code,
                           state=None, routing_state=None):
        """Run the same basin mean forcing in each incremental node when no spatial grid is available."""
        members = code.shape[0]
        weather = weather.expand(members, -1, -1, -1)
        node_features = node_features.expand(members, -1, -1)
        node_areas = node_areas.expand(members, -1)
        local_code = code[:, :8]
        delta = self.hydro_parameter_delta(code)
        return self.hydrology(weather, node_features, node_areas, downstream_parent,
                              code=local_code, parameter_delta=delta,
                              state=state, routing_state=routing_state)
