"""Protocol modules at fixed scientific capacity; stage-specific freezing is explicit."""

import torch
from torch import nn

from echo.hydro.spatial import SemiDistributedHBV
from echo.models.curve import monotone_integral
from echo.models.dynamics import LocalDynamics
from echo.models.formal import ConditionalJointSplinePosterior, SpatialEnvironmentEncoder
from echo.models.official_mamba import OfficialHistoryEncoder


class ProtocolCurve(nn.Module):
    def __init__(self, knot_width):
        super().__init__()
        if knot_width <= 0:
            raise ValueError("曲线节点范围须由来源训练参考Q统计确定")
        self.register_buffer("knot_width", torch.tensor(float(knot_width)))
        self.net = nn.Sequential(nn.Linear(272, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 33))

    def forward(self, u, environment, code, reference):
        parameters = self.net(torch.cat([environment, code], -1))
        return (parameters[:, :1] + monotone_integral(u, parameters[:, 1:], self.knot_width)
                - monotone_integral(reference[:, None], parameters[:, 1:], self.knot_width))


class TwoLayerConditioner(nn.Module):
    """Exactly two hidden layers of width 128, with the flow context concatenated."""

    def __init__(self, in_features, out_features, hidden_features=128, context_features=512,
                 **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features + context_features, hidden_features),
                                 nn.ReLU(), nn.Linear(hidden_features, hidden_features),
                                 nn.ReLU(), nn.Linear(hidden_features, out_features))
        self.hidden_features = hidden_features

    def forward(self, inputs, context=None):
        return self.net(torch.cat([inputs, context], -1))


class ProtocolECHO(nn.Module):
    def __init__(self, node_feature_dim, terrain_feature_dim, knot_width, edge_feature_dim=6):
        super().__init__()
        self.environment_encoder = SpatialEnvironmentEncoder(
            node_feature_dim, terrain_feature_dim, 64, width=128, heads=4, terrain_layers=2,
            edge_feature_dim=edge_feature_dim)
        self.history_encoder = OfficialHistoryEncoder()
        self.posterior = ConditionalJointSplinePosterior(
            256, 256, 32, 8, 8, 128, conditioner_factory=TwoLayerConditioner)
        self.hydrology = SemiDistributedHBV(node_feature_dim, hidden=128, environment_dim=256)
        self.curve = ProtocolCurve(knot_width)
        self.dynamic = LocalDynamics(256, blocks=8)
        self.stage = None

    def infer(self, tokens, geography, members=4, stochastic=True):
        environment = self.environment_encoder(geography)
        context = self.history_encoder(tokens, environment)
        code, kl = self.posterior(context, environment, members, stochastic)
        return code, kl, environment

    def configure_stage(self, stage):
        if stage not in "ABCDE" or len(stage) != 1:
            raise ValueError("训练阶段必须为A/B/C/D/E")
        self.requires_grad_(False)
        if stage == "A":
            self.environment_encoder.requires_grad_(True)
            self.hydrology.regionalizer.requires_grad_(True)
        elif stage == "B":
            for module in [self.history_encoder, self.posterior, self.curve]:
                module.requires_grad_(True)
        elif stage == "C":
            self.dynamic.requires_grad_(True)
        elif stage == "D":
            for module in [self.history_encoder, self.posterior, self.curve, self.dynamic,
                           self.hydrology.local_offset]:
                module.requires_grad_(True)
        self.stage = stage
        return [p for p in self.parameters() if p.requires_grad]

    def simulate(self, weather, geography, environment, code, hydro_state=None, routing_state=None):
        # A single 8-D local map, never two simultaneous offsets.
        hydro_code = code[:, :8] if self.stage in {"D", "E"} else None
        batch = len(code)
        return self.hydrology(
            weather.expand(batch, -1, -1, -1),
            geography["node_features"].expand(batch, -1, -1),
            geography["node_areas"].expand(batch, -1), geography["downstream_parent"],
            code=hydro_code, environment=environment.expand(batch, -1),
            state=hydro_state, routing_state=routing_state)

    def generate(self, q, area, environment, code, center, scale, reference, state=None):
        if self.stage is None:
            raise ValueError("生成前须显式设定A/B/C/D/E阶段")
        if not torch.isfinite(q).all() or (q < 0).any():
            raise ValueError("物理前端输出Q必须为有限非负量，不静默截断")
        u = torch.log1p(q / (area[:, None] / 86.4))
        curve = self.curve(u, environment, code[:, 8:24], reference)
        if self.stage in {"C", "D", "E"}:
            residual, state = self.dynamic(u, environment, code[:, 24:], state)
        else:
            residual = torch.zeros_like(curve)
        return center[:, None] + scale[:, None] * (curve + residual), state
