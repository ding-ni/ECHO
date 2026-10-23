"""Shared member/state and loss primitives for protocol stages; no target labels."""

from dataclasses import dataclass

import torch

from echo.evaluation.metrics import ensemble_crps


@dataclass
class MemberTrajectory:
    support: torch.Tensor
    query: torch.Tensor
    support_q: torch.Tensor
    query_q: torch.Tensor
    hydro_state: torch.Tensor
    routing_state: torch.Tensor
    response_state: tuple | None


def simulate_members(model, weather, geography, environment, code, area, center, scale, reference,
                     support_start, support_hours, query_hours, dynamic_warmup_hours=2160, block=720):
    """One fixed member across spinup/support/query. Weather is [1,N,T,3].

    Caller provides at least two calendar years before support and verifies their
    real UTC timestamps. No future H or observed Q is accepted by this function.
    Spinup derivatives are detached, while the member's actual parameters persist.
    """
    if (support_start < 2 * 365 * 24 or dynamic_warmup_hours < 90 * 24
            or dynamic_warmup_hours > support_start):
        raise ValueError("正式流程需要至少两年气象预热和90天支持前Q状态预热")
    end = support_start + support_hours + query_hours
    if weather.ndim != 4 or weather.shape[-1] != 3 or weather.shape[2] < end:
        raise ValueError("子流域气象形状或支持/查询覆盖不满足正式流程")
    if support_hours < 1 or query_hours < 1 or block < 1 or len(code) < 1:
        raise ValueError("支持、查询、成员与分块大小必须为正")
    hydro_state = routing_state = response_state = None
    member_env = environment.expand(len(code), -1)
    spin_end = support_start - dynamic_warmup_hours
    # First reach the beginning of dynamic spinup with the same local code.
    with torch.no_grad():
        for start in range(0, spin_end, block):
            result = model.simulate(weather[:, :, start:min(start + block, spin_end)],
                                    geography, environment, code, hydro_state, routing_state)
            hydro_state, routing_state = result["state"], result["routing_state"]
        for start in range(spin_end, support_start, block):
            result = model.simulate(weather[:, :, start:min(start + block, support_start)],
                                    geography, environment, code, hydro_state, routing_state)
            hydro_state, routing_state = result["state"], result["routing_state"]
            _, response_state = model.generate(result["q_outlet"], area.expand(len(code)),
                                               member_env, code, center.expand(len(code)),
                                               scale.expand(len(code)), reference.expand(len(code)),
                                               response_state)
    outputs, flows = [], []
    for start in range(support_start, end, block):
        result = model.simulate(weather[:, :, start:min(start + block, end)],
                                geography, environment, code, hydro_state, routing_state)
        hydro_state, routing_state = result["state"], result["routing_state"]
        prediction, response_state = model.generate(
            result["q_outlet"], area.expand(len(code)), member_env, code,
            center.expand(len(code)), scale.expand(len(code)), reference.expand(len(code)), response_state)
        outputs.append(prediction)
        flows.append(result["q_outlet"])
    predictions, q = torch.cat(outputs, 1), torch.cat(flows, 1)
    return MemberTrajectory(predictions[:, :support_hours], predictions[:, support_hours:],
                            q[:, :support_hours], q[:, support_hours:], hydro_state,
                            routing_state, response_state)


def probability_objective(trajectory, support_h, query_h, source_q, scale, kl,
                          support_error_sd, progress, regularizer, static_anchor,
                          source_q_weight=.2, regularizer_weight=.001, anchor_weight=1.):
    """Source-training-only likelihood + query CRPS + Q + KL + regularization.

    support_error_sd must be calibrated on source validation data. Query labels
    are kept outside the forecast context; this function is never called at targets.
    """
    if not 0 <= progress <= 1 or not torch.isfinite(support_error_sd).all() or (support_error_sd <= 0).any():
        raise ValueError("KL进度必须在[0,1]且支持误差尺度须为来源校准正值")
    smask, fmask = torch.isfinite(support_h), torch.isfinite(query_h)
    if not smask.any() or not fmask.any():
        raise ValueError("来源episode缺少有效支持或查询标签")
    observed = support_h[smask] / scale
    distribution = torch.distributions.Normal(trajectory.support[:, smask] / scale, support_error_sd)
    log_likelihood = distribution.log_prob(observed).sum(-1) / smask.sum()
    # Monte Carlo expected support negative log likelihood in the variational objective.
    nll = -log_likelihood.mean()
    crps = ensemble_crps(trajectory.query[:, fmask] / scale, query_h[fmask] / scale)
    q = torch.cat([trajectory.support_q, trajectory.query_q], 1)
    qmask = torch.isfinite(source_q) & (source_q >= 0)
    if not qmask.any():
        raise ValueError("来源训练episode缺少有效Q监督")
    qloss = (torch.log1p(q[:, qmask]) - torch.log1p(source_q[qmask])).square().mean()
    normalized_kl = kl / smask.sum()
    total = (nll + crps + source_q_weight * qloss + progress * normalized_kl
             + regularizer_weight * regularizer + anchor_weight * static_anchor)
    return total, {"support_nll": nll, "query_CRPS": crps, "source_Q": qloss,
                   "KL": normalized_kl, "KL_beta": progress,
                   "regularizer": regularizer, "static_anchor": static_anchor}
