"""联合地方推断必须保持跨组协方差、配套成员与HBV梯度路径。"""

import numpy as np
import pytest
import torch

from echo.data.permissions import ForecastInput
from echo.hydro.hbv import RegionalHBV
from echo.models.context import JointGaussian
from echo.models.echo import EchoStage
from echo.training.pipeline import context_for, joint_member_prediction, local_prediction


def fixture_inputs():
    rng = np.random.default_rng(2)
    weather = np.column_stack([rng.gamma(.7, 3, 96), np.full(96, 12), np.full(96, .05)]).astype("float32")
    h = np.linspace(1, 2, 48).astype("float32")
    return ForecastInput("S", "R", weather, np.array([.2, .4], dtype="float32"), 100.,
                         h, np.ones(48, dtype=bool), 24, 72, 72, 96)


def test_lowrank_gaussian_kl_matches_multivariate_normal_and_cross_groups():
    torch.manual_seed(10)
    posterior = JointGaussian(2, rank=3)
    with torch.no_grad():
        posterior.factor.bias.fill_(.8)
    context, geo = torch.randn(2, 64), torch.randn(2, 2)
    mean, logsd, factor = posterior.statistics(context, geo)
    pm, pls = posterior.prior(geo).chunk(2, -1)
    pls = pls.clamp(-5, 2)
    q = torch.distributions.LowRankMultivariateNormal(mean, factor, (2 * logsd).exp())
    p = torch.distributions.MultivariateNormal(pm, scale_tril=torch.diag_embed(pls.exp()))
    _, kl = posterior(context, geo, members=2)
    expected = torch.distributions.kl_divergence(q, p).mean() / 32
    torch.testing.assert_close(kl, expected, atol=1e-5, rtol=1e-5)
    covariance = factor @ factor.transpose(-1, -2)
    assert covariance[:, :8, 8:24].abs().max() > 0
    assert covariance[:, 8:24, 24:].abs().max() > 0
    kl.backward()
    assert posterior.factor.weight.grad.abs().max() > 0


def test_batched_members_equal_individual_and_flow_gradient_reaches_adapter():
    torch.manual_seed(20)
    torch.set_num_threads(2)
    inp = fixture_inputs()
    model = EchoStage(2, dynamic=True, probabilistic=True, hidden=8,
                      posterior_rank=2, learn_hydro_adapter=True)
    hydro = RegionalHBV(2).eval().requires_grad_(False)
    reference_q = np.linspace(.1, 5, 96).astype("float32")
    codes, kl, center, scale, uref = context_for(model, inp, reference_q, "cpu", members=3)
    batch = joint_member_prediction(model, inp, codes, kl, center, scale, uref,
                                    hydro, "cpu", True, True)
    individuals = [joint_member_prediction(model, inp, code[None], kl, center, scale, uref,
                                           hydro, "cpu", True, True) for code in codes]
    torch.testing.assert_close(batch[0], torch.cat([r[0] for r in individuals]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch[3]["flow"], torch.cat([r[3]["flow"] for r in individuals]),
                               atol=1e-5, rtol=1e-5)
    (batch[0].square().mean() + batch[3]["flow"].square().mean() * .01 + kl).backward()
    assert torch.isfinite(model.hydro_adapter.weight.grad).all()
    assert model.hydro_adapter.weight.grad.abs().max() > 0
    assert model.posterior.posterior.weight.grad[:8].abs().max() > 0
    assert all(p.grad is None for p in hydro.parameters())


def test_probabilistic_future_h_not_read_and_hydrology_code_changes_flow():
    torch.manual_seed(30)
    model = EchoStage(2, probabilistic=True, hidden=8, posterior_rank=2, learn_hydro_adapter=True)
    hydro = RegionalHBV(2).eval().requires_grad_(False)
    inp = fixture_inputs()
    q = np.linspace(.1, 5, 96).astype("float32")
    torch.manual_seed(3)
    with torch.no_grad():
        a = local_prediction(model, inp, q, "cpu", members=4, hydro=hydro, return_support=True)
    torch.manual_seed(3)
    with torch.no_grad():
        b = local_prediction(model, inp, q, "cpu", members=4, hydro=hydro, return_support=True)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert a[3]["flow"].std(0).max() > 0
    assert a[0].shape == (4, 24)


def test_probability_requires_hydrology_not_fixed_q_only():
    inp = fixture_inputs()
    model = EchoStage(2, probabilistic=True, hidden=8)
    with pytest.raises(ValueError, match="水文前端"):
        local_prediction(model, inp, np.ones(96), "cpu", members=2)
