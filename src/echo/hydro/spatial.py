"""Differentiable HBV over incremental subbasins and conservative downstream routing."""

import torch
from torch import nn

from echo.hydro.hbv import LOWER, UPPER, hbv_step
from echo.hydro.routing import route_network


class SemiDistributedHBV(nn.Module):
    """Shared regional HBV parameters, node-specific geography, local latent offsets."""

    def __init__(self, node_feature_dim=8, hidden=32, local_code_dim=8):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.regionalizer = nn.Sequential(nn.Linear(node_feature_dim, hidden), nn.Tanh(), nn.Linear(hidden, 9))
        self.local_offset = nn.Linear(local_code_dim, 9, bias=False)
        nn.init.normal_(self.local_offset.weight, std=0.01)
        self.register_buffer("lower", torch.tensor(LOWER))
        self.register_buffer("upper", torch.tensor(UPPER))

    def parameter_values(self, node_features, code=None, parameter_delta=None):
        raw = self.regionalizer(node_features)
        if code is not None:
            raw = raw + 0.5 * torch.tanh(self.local_offset(code[:, None].expand(-1, node_features.shape[1], -1)))
        if parameter_delta is not None:
            raw = raw + parameter_delta[:, None]
        return self.lower + (self.upper - self.lower) * raw.sigmoid()

    def forward(self, weather, node_features, node_areas, downstream_parent, code=None,
                parameter_delta=None, state=None, routing_state=None):
        """weather[B,N,T,3], incremental areas[km2], parent index points downstream or -1."""
        if weather.ndim != 4 or weather.shape[-1] != 3:
            raise ValueError("空间HBV气象必须为[B,N,T,3]")
        batch, nodes, steps, _ = weather.shape
        if node_features.shape[:2] != (batch, nodes) or node_areas.shape != (batch, nodes):
            raise ValueError("增量节点特征/面积形状错误")
        if len(downstream_parent) != nodes:
            raise ValueError("下游路由关系与节点数不一致")
        outlet_ids = [i for i, parent in enumerate(downstream_parent) if parent == -1]
        if not outlet_ids or any(parent != -1 and parent <= i for i, parent in enumerate(downstream_parent)):
            raise ValueError("节点须按上游到下游拓扑排序，并至少含一个出口")
        params = self.parameter_values(node_features, code, parameter_delta)
        hydro_state = weather.new_zeros((batch, nodes, 5)) if state is None else state
        route_storage = weather.new_zeros((batch, nodes)) if routing_state is None else routing_state
        tau_seconds = params[..., 8] * 3600.0
        q_steps, runoff_errors, routing_errors = [], [], []
        for t in range(steps):
            runoff, next_state, balance = hbv_step(
                weather[:, :, t].reshape(batch * nodes, 3), hydro_state.reshape(batch * nodes, 5),
                params.reshape(batch * nodes, 9))
            hydro_state = next_state.reshape(batch, nodes, 5)
            lateral = runoff.reshape(batch, nodes) * node_areas / 3.6
            old_storage = route_storage.sum(-1)
            route_storage, routed = route_network(route_storage, lateral, tau_seconds,
                                                  downstream_parent, dt=3600.0)
            outlet_flow = routed[:, outlet_ids].sum(-1)
            route_residual = (route_storage.sum(-1) - old_storage
                              - 3600.0 * lateral.sum(-1) + 3600.0 * outlet_flow)
            q_steps.append(outlet_flow)
            runoff_errors.append(balance.reshape(batch, nodes))
            routing_errors.append(route_residual)
        result = {
            "q_outlet": torch.stack(q_steps, 1),
            "state": hydro_state,
            "routing_state": route_storage,
            "runoff_balance": torch.stack(runoff_errors, 1),
            "routing_balance": torch.stack(routing_errors, 1),
            "parameters": params,
        }
        return result
