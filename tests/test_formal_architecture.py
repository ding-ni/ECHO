import numpy as np
import pytest
import torch

from echo.geo.formal_inputs import coarsen_nodes, terrain_tokens
from echo.hydro.spatial import SemiDistributedHBV
from echo.models.formal import FormalECHO


def tiny_geography(batch=2):
    edge = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
    return {
        "node_features": torch.randn(batch, 8, 8),
        "node_areas": torch.full((batch, 8), 12.5),
        "edge_index": edge,
        "terrain_tokens": torch.randn(batch, 16, 10),
        "terrain_padding_mask": torch.zeros(batch, 16, dtype=torch.bool),
        "climate_features": torch.randn(batch, 64),
    }


def test_formal_spatial_mamba_reference_flow_curve_dynamics_forward_backward():
    torch.manual_seed(41)
    torch.set_num_threads(2)
    model = FormalECHO(width=64, state_dim=8, head_dim=16, rank=4,
                       posterior_layers=4, posterior_hidden=32)
    geography = tiny_geography()
    tokens = torch.randn(2, 24, 8)
    codes, kl, environment = model.infer(tokens, geography, members=2, stochastic=True)
    q = torch.rand(2, 24) * 20
    generated = []
    for member in range(2):
        h, _ = model.generate(q, torch.tensor([100., 150.]), environment,
                              codes[member], torch.tensor([1., 2.]), torch.tensor([.4, .5]),
                              torch.tensor([1.1, .9]))
        generated.append(h)
    prediction = torch.stack(generated)
    loss = prediction.square().mean() + kl
    loss.backward()
    assert prediction.shape == (2, 2, 24)
    assert torch.isfinite(prediction).all() and torch.isfinite(kl)
    assert model.environment_encoder.basin.convs[0].lin_l.weight.grad is not None
    assert model.history_encoder.blocks[0].selective.weight.grad.abs().max() > 0
    assert model.posterior.flow._transforms[0].transform_net.final_layer.weight.grad.abs().max() > 0


def test_semidistributed_hbv_area_route_mass_and_local_code_gradient():
    torch.manual_seed(8)
    model = SemiDistributedHBV(node_feature_dim=8)
    weather = torch.rand(2, 8, 12, 3)
    weather[..., 1] = 12
    weather[..., 2] *= .1
    node_features = torch.randn(2, 8, 8)
    node_areas = torch.full((2, 8), 10.)
    parents = [1, 2, 3, 4, 5, 6, 7, -1]
    code = torch.randn(2, 8, requires_grad=True)
    delta = torch.randn(2, 9, requires_grad=True) * .01
    result = model(weather, node_features, node_areas, parents, code, delta)
    loss = result["q_outlet"].mean() + result["routing_state"].mean() * 1e-8
    loss.backward()
    assert result["q_outlet"].shape == (2, 12)
    assert result["runoff_balance"].abs().max() < 1e-3
    assert result["routing_balance"].abs().max() < 2e-2
    assert code.grad is not None and torch.isfinite(code.grad).all()


def test_hydrobasin_coarsening_keeps_incremental_areas_and_eight_nodes():
    import geopandas as gpd
    from shapely.geometry import box

    ids = np.arange(1, 65)
    frame = gpd.GeoDataFrame({"HYBAS_ID": ids, "SUB_AREA": np.ones(64),
                              "UP_AREA": np.arange(1., 65.)},
                             geometry=[box(i, 0, i + .9, .9) for i in ids],
                             crs="EPSG:32631").set_index("HYBAS_ID", drop=False)
    depth = {int(i): int(i - 1) for i in ids}
    features, areas, parents, _ = coarsen_nodes(depth, frame, (0., 0.),
                                                {"elevation_m": 400., "slope_degrees": 2., "relief_m": 100.})
    tokens, mask = terrain_tokens(depth, frame, (0., 0.),
                                  {"elevation_m": 400., "slope_degrees": 2., "relief_m": 100.})
    assert features.shape == (8, 8) and areas.shape == (8,)
    np.testing.assert_allclose(areas.sum(), 64.)
    np.testing.assert_array_equal(parents, np.arange(1, 8).tolist() + [-1])
    assert tokens.shape == (16, 10) and mask.shape == (16,)


def test_official_mamba_backend_fails_instead_of_masquerading():
    with pytest.raises(RuntimeError, match="官方Mamba-3"):
        from echo.models.formal import Mamba3HistoryEncoder
        Mamba3HistoryEncoder(width=64, state_dim=8, head_dim=16, geo_dim=256,
                             backend="official_cuda")
