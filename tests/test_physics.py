import pytest
import torch

from echo.hydro.hbv import RegionalHBV, bounded_parameters, hbv_step
from echo.hydro.routing import route_network
from echo.models.curve import monotone_integral
from echo.models.dynamics import LocalDynamics


def test_hbv_water_balance_snow_drought_and_storm():
    torch.manual_seed(11)
    state = torch.zeros(4, 5, dtype=torch.float64)
    params = bounded_parameters(torch.randn(4, 9, dtype=torch.float64))
    for i in range(300):
        p = torch.rand(4, dtype=torch.float64) * (50 if i % 10 == 0 else 0.1)
        weather = torch.stack([p, torch.full_like(p, -5 if i < 50 else 20), torch.full_like(p, 0.2)], -1)
        q, state, balance = hbv_step(weather, state, params)
        assert balance.abs().max() < 1e-10
        assert state.min() >= -1e-12
        assert q.min() >= -1e-12


def test_hbv_batch_gradient_and_chunk_equivalence():
    torch.manual_seed(1)
    m = RegionalHBV(3).double()
    w = torch.rand(2, 30, 3, dtype=torch.float64)
    w[:, :, 1] *= 20
    geo = torch.rand(2, 3, dtype=torch.float64)
    area = torch.tensor([86.4, 100.0], dtype=torch.float64)
    full, last, _ = m(w, geo, area)
    a, state, _ = m(w[:, :17], geo, area)
    b, state, _ = m(w[:, 17:], geo, area, state=state)
    torch.testing.assert_close(full, torch.cat([a, b], 1))
    torch.testing.assert_close(last, state)
    full.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_dry_initial_soil_low_beta_has_finite_gradient():
    raw = torch.full((1, 9), -8.0, dtype=torch.float64, requires_grad=True)
    weather = torch.tensor([[20.0, 15.0, 0.1]], dtype=torch.float64)
    q, state, _ = hbv_step(weather, torch.zeros(1, 5, dtype=torch.float64), bounded_parameters(raw))
    (q.sum() + state.square().sum()).backward()
    assert torch.isfinite(raw.grad).all()


def test_network_conservation_with_branching():
    volume = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float64)
    lateral = torch.tensor([2.0, 3.0, 1.0, 0.0], dtype=torch.float64)
    tau = torch.tensor([1800.0, 7200.0, 5000.0, 9000.0], dtype=torch.float64)
    nxt, out = route_network(volume, lateral, tau, [2, 2, 3, -1])
    error = nxt.sum() - volume.sum() - 3600 * (lateral.sum() - out[-1])
    assert abs(error) < 1e-9
    assert out.min() >= 0
    with pytest.raises(ValueError):
        route_network(volume, lateral, tau, [2, 0, 3, -1])


def test_curve_strict_monotonicity_extrapolation_and_batch_independence():
    grid = torch.linspace(-2, 10, 1000)[None]
    slopes = torch.randn(1, 32)
    y = monotone_integral(grid, slopes)
    assert torch.diff(y).min() > 0
    torch.testing.assert_close(monotone_integral(grid[:, ::3], slopes), y[:, ::3])


def test_dynamic_zero_readout_and_chunk_resume():
    torch.manual_seed(44)
    model = LocalDynamics(3).double()
    with torch.no_grad():
        model.gates[-1].weight.normal_(0, 0.1)
    geo, code = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64)
    constant = torch.ones(2, 50, dtype=torch.float64)
    value, _ = model(constant, geo, code)
    torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
    u = torch.randn(2, 80, dtype=torch.float64)
    full, last = model(u, geo, code)
    a, state = model(u[:, :27], geo, code)
    b, state = model(u[:, 27:], geo, code, state)
    torch.testing.assert_close(full, torch.cat([a, b], 1))
    for x, y in zip(last, state):
        torch.testing.assert_close(x, y)


def test_dynamic_same_input_contraction():
    model = LocalDynamics(2).double()
    geo, code = torch.randn(1, 2, dtype=torch.float64), torch.randn(1, 8, dtype=torch.float64)
    z1, z2 = torch.randn(1, 8, 2, dtype=torch.float64), torch.randn(1, 8, 2, dtype=torch.float64)
    old = torch.linalg.vector_norm(z1 - z2)
    for _ in range(20):
        _, z1 = model.step(
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            geo,
            code,
            z1,
        )
        _, z2 = model.step(
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            geo,
            code,
            z2,
        )
        new = torch.linalg.vector_norm(z1 - z2)
        assert new < old
        old = new


def test_nonzero_initial_state_returns_to_reference_stage():
    model = LocalDynamics(2).double()
    with torch.no_grad():
        model.gates[-1].bias.view(8, 6)[:, 0].fill_(-8.0)
        geo, code = torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 8, dtype=torch.float64)
        state = (
            torch.ones(1, 8, 2, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
        output, state = model(torch.ones(1, 100, dtype=torch.float64), geo, code, state)
        assert torch.linalg.vector_norm(state[0]) < 1e-12
        assert abs(output[0, -1]) < 1e-12
