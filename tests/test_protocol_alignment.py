import numpy as np
import pytest
import torch
from torch import nn

from echo.geo.incremental import contract_incremental_graph, induced_parents
from echo.hydro.spatial import SemiDistributedHBV
from echo.models.formal import BasinGATv2Encoder, ConditionalJointSplinePosterior
from echo.models.protocol import ProtocolCurve, ProtocolECHO, TwoLayerConditioner
from echo.training.protocol_steps import simulate_members


def test_contraction_preserves_fork_and_areas_and_records_all_cells():
    # Two long tributaries meet at 7, then flow through 8 to the gauge.
    parents = {1: 2, 2: 3, 3: 7, 4: 5, 5: 6, 6: 7, 7: 8, 8: -1}
    areas = {node: float(node) for node in parents}
    result = contract_incremental_graph(parents, areas, 4)
    assert len(result["groups"]) == 4
    assert sorted(sum(result["groups"], [])) == list(range(1, 9))
    assert max(np.sum(result["parents"] == p) for p in range(4)) == 2
    assert result["areas_km2"].sum() == sum(areas.values())
    assert sum(p == -1 for p in result["parents"]) == 1


def test_induced_graph_traces_removed_cells_and_rejects_cycle():
    assert induced_parents([1, 4, 7], {1: 2, 2: 7, 4: 7, 7: 0}, 7) == {1: 7, 4: 7, 7: -1}
    with pytest.raises(ValueError):
        induced_parents([1, 7], {1: 2, 2: 1, 7: 0}, 7)


def test_edge_conditioned_gat_uses_real_edge_attributes():
    torch.manual_seed(4)
    model = BasinGATv2Encoder(8, width=16, heads=4, edge_feature_dim=6)
    nodes, areas = torch.randn(1, 4, 8), torch.ones(1, 4)
    edges = torch.tensor([[0, 1, 2], [2, 2, 3]])
    attrs = torch.randn(3, 6, requires_grad=True)
    model(nodes, areas, edges, edge_attr=attrs).square().sum().backward()
    assert attrs.grad is not None and attrs.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        model(nodes, areas, edges)


def test_hbv_environment_and_one_local_map_change_parameters():
    torch.manual_seed(3)
    model = SemiDistributedHBV(8, environment_dim=256)
    nodes, env, code = torch.randn(1, 3, 8), torch.randn(1, 256), torch.randn(1, 8)
    prior = model.parameter_values(nodes, environment=env)
    assert not torch.allclose(prior, model.parameter_values(nodes, environment=env + 1))
    assert not torch.allclose(prior, model.parameter_values(nodes, code, environment=env))
    with pytest.raises(ValueError):
        model.parameter_values(nodes)


def test_protocol_curve_has_three_hidden_layers_and_monotone_tails():
    model = ProtocolCurve(.17)
    assert len([m for m in model.net if isinstance(m, nn.Linear)]) == 4
    u = torch.linspace(-2, 12, 500)[None]
    h = model(u, torch.randn(1, 256), torch.randn(1, 16), torch.ones(1))
    assert (h[:, 1:] - h[:, :-1] > 0).all()


def test_protocol_flow_conditioner_has_two_hidden_layers_and_gradients():
    model = ConditionalJointSplinePosterior(layers=8, conditioner_factory=TwoLayerConditioner)
    code, kl = model(torch.randn(1, 256), torch.randn(1, 256), members=4)
    (code.square().mean() + kl * 1e-3).backward()
    conditioner = model.flow._transforms[0].transform_net
    assert len([m for m in conditioner.net if isinstance(m, nn.Linear)]) == 3
    assert code.shape == (4, 1, 32) and torch.isfinite(kl)


def test_stage_d_freezes_regional_prior_and_stage_c_freezes_static(monkeypatch):
    # Only test our stage controller; this test never claims to run official Mamba.
    monkeypatch.setattr("echo.models.protocol.OfficialHistoryEncoder", lambda: nn.Linear(8, 256))
    model = ProtocolECHO(8, 10, .2)
    model.configure_stage("D")
    assert not any(p.requires_grad for p in model.hydrology.regionalizer.parameters())
    assert not any(p.requires_grad for p in model.environment_encoder.parameters())
    assert all(p.requires_grad for p in model.hydrology.local_offset.parameters())
    assert not hasattr(model, "hydro_adapter")
    model.configure_stage("C")
    assert not any(p.requires_grad for p in model.curve.parameters())
    assert all(p.requires_grad for p in model.dynamic.parameters())
    assert not model.configure_stage("E")


def test_member_warmup_uses_same_codes_and_carries_dynamic_state():
    class CountingModel:
        def __init__(self):
            self.codes = []
            self.states = []

        def simulate(self, weather, geography, env, code, hydro_state, routing_state):
            self.codes.append(code.clone())
            n = weather.shape[2]
            return {"q_outlet": code[:, :1].expand(-1, n).abs(),
                    "state": torch.ones(len(code), 2, 5), "routing_state": torch.ones(len(code), 2)}

        def generate(self, q, area, env, code, center, scale, reference, state=None):
            self.states.append(state)
            count = 0 if state is None else state[0]
            return q, (count + q.shape[1],)

    model = CountingModel()
    code = torch.randn(4, 32)
    result = simulate_members(model, torch.zeros(1, 2, 17520 + 12, 3), {}, torch.ones(1, 256),
                              code, torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1),
                              17520, 8, 4)
    assert all(torch.equal(c, code) for c in model.codes)
    assert model.states[-1][0] == 2160
    assert result.support.shape == (4, 8) and result.query.shape == (4, 4)
