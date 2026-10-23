"""Source-only A--E training orchestration for the unchanged protocol modules.

This engine consumes fully prepared, validated episodes. It never falls back to
basin-mean node forcing or a replacement history backbone when data/runtime are
missing. Official Mamba GPU execution remains a separate integration check.
"""

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from echo.data.episodes import tokens
from echo.data.permissions import assert_source_permissions, support_scaling
from echo.data.spatial_contract import validate_protocol_inputs
from echo.evaluation.metrics import ensemble_crps
from echo.training.protocol_steps import MemberTrajectory, probability_objective, simulate_members
from echo.utils import digest, run_metadata, write_json


@dataclass
class ProtocolEpisode:
    station_id: str
    system_id: str
    role: str
    fold: int
    inputs: dict
    provenance: dict
    support_start: int
    support_end: int
    query_end: int

    def validate(self, require_reference=True):
        if self.role not in {"train", "validation", "development_test"}:
            raise ValueError("主训练器不接收盲测episode")
        if self.role == "train" and self.fold not in {0, 1, 2}:
            raise ValueError("来源水系必须有固定的三折编号")
        if any(k.lower() in {"q", "h", "future_h", "target_q", "observed_q"} for k in self.inputs):
            raise ValueError("预测输入对象不允许混入来源监督Q、完整H或查询H")
        result = validate_protocol_inputs(self.inputs, self.provenance, self.support_start,
                                          self.support_end, self.query_end, target=True,
                                          require_reference=require_reference)
        times = self.inputs["time_ns"]
        start = pd.Timestamp(int(times[self.support_start]), tz="UTC")
        end = pd.Timestamp(int(times[self.query_end - 1]), tz="UTC")
        if self.role == "train" and (start < pd.Timestamp("2013-10-01", tz="UTC")
                                     or end >= pd.Timestamp("2018-10-01", tz="UTC")):
            raise ValueError("来源episode超出2013—2018共享训练期")
        if self.role == "validation" and (start < pd.Timestamp("2018-10-01", tz="UTC")
                                          or end >= pd.Timestamp("2019-10-01", tz="UTC")):
            raise ValueError("来源验证episode超出2018—2019验证期")
        if require_reference:
            systems = self.provenance.get("reference_trained_systems")
            if systems is None or self.system_id in systems:
                raise ValueError("参考Q缺少训练水系证据或用过本水系Q")
        return result


@dataclass
class SourceLabels:
    # Deliberately separate from prediction inputs. Validation Q may be None.
    query_h: np.ndarray
    q_supervision: np.ndarray | None = None


def geography_for(episode, device):
    data = episode.inputs
    names = ["node_features", "node_areas", "terrain_tokens", "terrain_padding_mask", "climate_features"]
    geography = {key: torch.as_tensor(data[key], device=device).unsqueeze(0) for key in names}
    for key in ["node_features", "node_areas", "terrain_tokens", "climate_features"]:
        geography[key] = geography[key].float()
    geography["terrain_padding_mask"] = geography["terrain_padding_mask"].bool()
    geography["edge_index"] = torch.as_tensor(data["edge_index"], dtype=torch.long, device=device)
    geography["edge_attr"] = torch.cat([
        torch.as_tensor(data["edge_attr"], dtype=torch.float32, device=device),
        torch.as_tensor(data["edge_attr_mask"], dtype=torch.float32, device=device)], -1)
    geography["downstream_parent"] = np.asarray(data["downstream_parent"]).astype(int).tolist()
    return geography


def _tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def episode_trajectory(model, episode, members=4, stochastic=True, block=720):
    episode.validate()
    device = next(model.parameters()).device
    geography = geography_for(episode, device)
    data = episode.inputs
    a, b, c = episode.support_start, episode.support_end, episode.query_end
    support_h = np.asarray(data["support_h"])
    mask = np.isfinite(support_h)
    center, scale = support_scaling(support_h, mask)
    areas = np.asarray(data["node_areas"])
    basin_weather = np.average(data["weather_nodes"], axis=0, weights=areas)
    history, _, _ = tokens(basin_weather[a:b], data["reference_q"][a:b], areas.sum(), support_h, mask)
    # Actual calendar season, independent of support-budget length.
    dates = pd.to_datetime(data["time_ns"][a:b], utc=True)
    history[:, -1] = np.sin((dates.dayofyear - 1 + dates.hour / 24) / 365.25 * 2 * np.pi)
    code, kl, environment = model.infer(_tensor(history[None], device), geography, members, stochastic)
    code = code[:, 0]
    area = _tensor([areas.sum()], device)
    center_t, scale_t = _tensor([center], device), _tensor([scale], device)
    reference = _tensor([episode.provenance["source_fixed_reference_u"]], device)
    if model.stage in {"B", "C"}:
        # B/C use the genuinely out-of-fold cached reference, rather than a fresh
        # in-sample simulation from the all-source frontend.
        warm = a - 2160
        state = None
        if model.stage == "C":
            with torch.no_grad():
                for start in range(warm, a, block):
                    q = _tensor(data["reference_q"][start:min(start + block, a)], device)[None].expand(members, -1)
                    _, state = model.generate(q, area.expand(members), environment.expand(members, -1),
                                              code, center_t.expand(members), scale_t.expand(members),
                                              reference.expand(members), state)
        q = _tensor(data["reference_q"][a:c], device)[None].expand(members, -1)
        h, state = model.generate(q, area.expand(members), environment.expand(members, -1),
                                  code, center_t.expand(members), scale_t.expand(members),
                                  reference.expand(members), state)
        length = b - a
        trajectory = MemberTrajectory(h[:, :length], h[:, length:], q[:, :length], q[:, length:],
                                      None, None, state)
    else:
        trajectory = simulate_members(model, _tensor(data["weather_nodes"][:, :c][None], device),
                                      geography, environment, code, area, center_t, scale_t, reference,
                                      a, b - a, c - b, block=block)
    return trajectory, kl, environment, code, center_t, scale_t, reference


def _supervision(episode, labels, require_q=False):
    if episode.role not in {"train", "validation"}:
        raise ValueError("训练和校准只允许来源训练或来源验证水系")
    if len(labels.query_h) != episode.query_end - episode.support_end:
        raise ValueError("查询监督H长度与独立输入边界不一致")
    if require_q:
        if episode.role != "train" or labels.q_supervision is None:
            raise ValueError("Q监督仅允许许可来源训练水系")
        if len(labels.q_supervision) != episode.query_end - episode.support_start:
            raise ValueError("来源Q监督长度不符")


def fit_hydrology_bank(factory, pairs, output, steps=2, learning_rate=2e-4, seed=20261019,
                       forbidden_systems=(), block=168):
    """A stage: all-source prior plus three genuinely held-out-system priors.

    Local fixed-step smoke training does not tune against the held-out fold Q.
    The actual full frontend factory can run on Windows without Mamba.
    """
    if steps < 1 or not pairs:
        raise ValueError("阶段A需要来源episode及正数优化步")
    systems_by_fold = {}
    for episode, labels in pairs:
        episode.validate(require_reference=False)
        _supervision(episode, labels, require_q=True)
        assert_source_permissions({episode.system_id}, forbidden_systems)
        if episode.system_id in systems_by_fold and systems_by_fold[episode.system_id] != episode.fold:
            raise ValueError("同一来源水系被分到不同交叉拟合折")
        systems_by_fold[episode.system_id] = episode.fold
    if set(systems_by_fold.values()) != {0, 1, 2}:
        raise ValueError("三折来源水系尚未齐备")
    output = Path(output)
    if output.exists():
        raise ValueError("阶段A输出版本已存在")
    output.mkdir(parents=True)
    receipts = []
    for excluded in [None, 0, 1, 2]:
        torch.manual_seed(seed)
        model = factory()
        # Local offsets are never learned in A.
        model.hydrology.local_offset.requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        training = [(episode, labels) for episode, labels in pairs if excluded is None or episode.fold != excluded]
        losses = []
        for step in range(steps):
            episode, labels = training[step % len(training)]
            device = next(model.parameters()).device
            geo = geography_for(episode, device)
            weather = _tensor(episode.inputs["weather_nodes"][None], device)
            environment = model.environment_encoder(geo)
            state = route = None
            with torch.no_grad():
                for start in range(0, episode.support_start, 720):
                    warm = model.hydrology(weather[:, :, start:min(start + 720, episode.support_start)],
                                           geo["node_features"], geo["node_areas"], geo["downstream_parent"],
                                           environment=environment, state=state, routing_state=route)
                    state, route = warm["state"], warm["routing_state"]
            stop = min(episode.query_end, episode.support_start + block)
            result = model.hydrology(weather[:, :, episode.support_start:stop], geo["node_features"],
                                     geo["node_areas"], geo["downstream_parent"], environment=environment,
                                     state=state, routing_state=route)
            obs = _tensor(labels.q_supervision[:stop - episode.support_start], device)
            valid = torch.isfinite(obs) & (obs >= 0)
            if not valid.any():
                raise ValueError("阶段A来源窗口无有效Q")
            loss = (torch.log1p(result["q_outlet"][0, valid]) - torch.log1p(obs[valid])).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("阶段A损失非有限")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        name = "all_source" if excluded is None else f"exclude_fold_{excluded}"
        path = output / f"{name}.pt"
        trained_systems = sorted({e.system_id for e, _ in training})
        torch.save({"frontend_state": model.state_dict(), "stage": "A", "seed": seed,
                    "excluded_fold": excluded, "trained_systems": trained_systems,
                    "optimizer_state": optimizer.state_dict(), "losses": losses,
                    "local_fixed_step_run": True}, path)
        receipts.append({"path": str(path.resolve()), "sha256": digest(path),
                         "stage": "A",
                         "excluded_fold": excluded, "trained_systems": trained_systems,
                         "optimizer_steps": steps})
    write_json(output / "index.json", {"stage": "A", "checkpoints": receipts,
                                       "target_Q_read": False, "heldout_fold_Q_used_for_selection": False})
    return receipts


def cache_references(factory, bank_receipts, episodes, output, block=720):
    """A-stage cross-fitted references; no observed Q/H labels accepted here."""
    output = Path(output)
    if output.exists():
        raise ValueError("参考Q版本已存在")
    output.mkdir(parents=True)
    by_fold = {r["excluded_fold"]: r for r in bank_receipts}
    models, records, source_u = {}, [], []
    for episode in episodes:
        episode.validate(require_reference=False)
        excluded = episode.fold if episode.role == "train" else None
        receipt = by_fold[excluded]
        if episode.system_id in receipt["trained_systems"]:
            raise ValueError("交叉拟合参考前端使用过当前水系Q")
        if digest(receipt["path"]) != receipt["sha256"]:
            raise ValueError("A阶段参考检查点哈希不一致")
        if excluded not in models:
            model = factory()
            device = next(model.parameters()).device
            checkpoint = torch.load(receipt["path"], map_location=device, weights_only=False)
            if checkpoint["trained_systems"] != receipt["trained_systems"]:
                raise ValueError("检查点与参考训练水系收据不一致")
            model.load_state_dict(checkpoint["frontend_state"])
            models[excluded] = model.eval()
        model = models[excluded]
        device = next(model.parameters()).device
        geo = geography_for(episode, device)
        weather = _tensor(episode.inputs["weather_nodes"][None], device)
        flows, state, route = [], None, None
        with torch.no_grad():
            environment = model.environment_encoder(geo)
            for start in range(0, weather.shape[2], block):
                result = model.hydrology(weather[:, :, start:start + block], geo["node_features"],
                                         geo["node_areas"], geo["downstream_parent"],
                                         environment=environment, state=state, routing_state=route)
                state, route = result["state"], result["routing_state"]
                flows.append(result["q_outlet"][0].cpu().numpy())
        q = np.concatenate(flows)
        path = output / f"{episode.station_id}.npy"
        np.save(path, q)
        records.append({"station_id": episode.station_id, "terminal_system_id": episode.system_id,
                        "role": episode.role, "path": str(path.resolve()), "sha256": digest(path),
                        "reference_checkpoint_sha256": receipt["sha256"],
                        "reference_trained_systems": receipt["trained_systems"]})
        if episode.role == "train":
            times = pd.to_datetime(episode.inputs["time_ns"], utc=True)
            mask = (times >= "2013-10-01") & (times < "2018-10-01")
            source_u.append(np.log1p(q[mask] / (np.sum(episode.inputs["node_areas"]) / 86.4)))
    if not source_u:
        raise ValueError("曲线固定节点缺少来源参考Q分布")
    u = np.concatenate(source_u)
    source_reference = float(np.median(u))
    knot_width = max(float(np.quantile(u, .99)) / 32, 1e-3)
    write_json(output / "index.json", {"references": records, "source_fixed_reference_u": source_reference,
                                       "source_fixed_knot_width": knot_width,
                                       "target_Q_read": False, "H_values_read": False})
    return records, {"source_fixed_reference_u": source_reference, "knot_width": knot_width}


def load_stage_a_frontend(model, receipt):
    if receipt.get("excluded_fold") is not None or digest(receipt["path"]) != receipt["sha256"]:
        raise ValueError("主响应训练必须加载匹配哈希的all-source阶段A先验")
    device = next(model.parameters()).device
    checkpoint = torch.load(receipt["path"], map_location=device, weights_only=False)
    if checkpoint.get("stage") != "A" or checkpoint.get("trained_systems") != receipt["trained_systems"]:
        raise ValueError("阶段A检查点身份与收据不一致")
    states = checkpoint["frontend_state"]
    for name in ["environment_encoder", "hydrology"]:
        selected = {key.removeprefix(name + "."): value for key, value in states.items()
                    if key.startswith(name + ".")}
        getattr(model, name).load_state_dict(selected, strict=True)


def validation_score(model, pairs, members=4):
    values, residuals = [], []
    model.eval()
    with torch.no_grad():
        for episode, labels in pairs:
            if episode.role != "validation":
                raise ValueError("超参数/停止/校准不能使用开发评估或盲测水系")
            _supervision(episode, labels)
            trajectory, _, _, _, _, scale, _ = episode_trajectory(model, episode, members, stochastic=True)
            obs = _tensor(labels.query_h, trajectory.query.device)
            valid = torch.isfinite(obs)
            if valid.float().mean() < .9:
                raise ValueError("来源验证窗口H覆盖不足90%")
            values.append(float(ensemble_crps(trajectory.query[:, valid] / scale, obs[valid] / scale)))
            residual = ((obs - trajectory.query.mean(0)) / scale).cpu().numpy()
            residuals.append(residual)
    if not values:
        raise ValueError("没有许可来源验证episode")
    return float(np.mean(values)), residuals


def calibrate_residuals(sequences):
    """A source-validation AR(1) residual; pairs never cross station/episode gaps."""
    arrays = [np.asarray(s, dtype=float).reshape(-1) for s in sequences]
    values = np.concatenate([s[np.isfinite(s)] for s in arrays])
    if len(values) < 3:
        raise ValueError("来源残差校准样本不足")
    mean = float(values.mean())
    numerator = denominator = 0.
    for sequence in arrays:
        valid = np.isfinite(sequence[:-1]) & np.isfinite(sequence[1:])
        x, y = sequence[:-1][valid] - mean, sequence[1:][valid] - mean
        numerator += float(x @ y)
        denominator += float(x @ x)
    rho = float(np.clip(numerator / max(denominator, 1e-12), -.99, .99))
    sd = max(float(values.std(ddof=1)), 1e-3)
    return {"mean_normalized_H": mean, "stationary_sd_normalized_H": sd, "rho": rho,
            "innovation_sd_normalized_H": sd * float(np.sqrt(1 - rho**2)),
            "observations": len(values), "source_validation_only": True}


def fit_response_stages(model, train_pairs, validation_pairs, output, *, steps_per_stage=2,
                         learning_rate=2e-4, joint_learning_rate=2e-5, members=4, seed=20261019,
                         patience=10, stage_a_receipt=None):
    """B/C/D/E with validation-selected checkpoints and source AR(1) calibration.

    A fully loaded stage-A frontend receipt is mandatory; callers check its hash
    and load its environment/HBV weights before invoking this function.
    """
    if not stage_a_receipt or not stage_a_receipt.get("sha256") or not stage_a_receipt.get("path"):
        raise ValueError("B阶段必须接入可追溯的阶段A权重")
    schedule = ({stage: int(steps_per_stage) for stage in "BCD"} if isinstance(steps_per_stage, int)
                else {stage: int(steps_per_stage[stage]) for stage in "BCD"})
    if min(schedule.values()) < 1 or schedule["D"] < 2 or not train_pairs or members < 2:
        raise ValueError("训练需要来源episode、正数步数及至少两个概率成员")
    train_systems = {e.system_id for e, _ in train_pairs}
    validation_systems = {e.system_id for e, _ in validation_pairs}
    assert_source_permissions(train_systems, validation_systems)
    if not train_systems.issubset(set(stage_a_receipt["trained_systems"])):
        raise ValueError("阶段A先验未覆盖当前许可来源训练水系")
    assert_source_permissions(stage_a_receipt["trained_systems"], validation_systems)
    for episode, labels in train_pairs:
        if episode.role != "train":
            raise ValueError("训练列表含非来源水系")
        _supervision(episode, labels, require_q=True)
    output = Path(output)
    if output.exists():
        raise ValueError("阶段训练输出目录已存在，不能覆盖历史结果")
    output.mkdir(parents=True)
    load_stage_a_frontend(model, stage_a_receipt)
    torch.manual_seed(seed)
    metadata = run_metadata({"steps_per_stage": steps_per_stage, "members": members, "seed": seed,
                             "learning_rate": learning_rate, "joint_learning_rate": joint_learning_rate})
    write_json(output / "run.json", {**metadata, "stage_A": stage_a_receipt,
                                     "trained_systems": sorted(train_systems),
                                     "validation_systems": sorted(validation_systems)})
    records, anchor, residual_model = [], None, None
    for stage in ["B", "C", "D"]:
        stage_steps = schedule[stage]
        kl_warmup = max(min(stage_steps - 1, 1000), 1)
        parameters = model.configure_stage(stage)
        optimizer = torch.optim.AdamW(parameters, lr=joint_learning_rate if stage == "D" else learning_rate)
        best, stale, started = float("inf"), 0, time.perf_counter()
        stage_path = output / f"stage_{stage}.pt"
        for step in range(stage_steps):
            model.train()
            episode, labels = train_pairs[step % len(train_pairs)]
            trajectory, kl, env, code, _, scale, reference = episode_trajectory(
                model, episode, members, stochastic=stage == "D")
            device = trajectory.support.device
            support_h = _tensor(episode.inputs["support_h"], device)
            query_h = _tensor(labels.query_h, device)
            if stage == "D":
                params = model.curve.net(torch.cat([env.expand(members, -1), code[:, 8:24]], -1))
                slopes = torch.nn.functional.softplus(params[:, 1:])
                regularizer = slopes.diff(n=2).square().mean()
                q = torch.cat([trajectory.support_q, trajectory.query_q], 1)
                area = float(np.asarray(episode.inputs["node_areas"]).sum())
                u = torch.log1p(q / (area / 86.4))
                with torch.no_grad():
                    anchored = anchor(u.detach(), env.expand(members, -1), code[:, 8:24].detach(),
                                      reference.expand(members))
                current = model.curve(u.detach(), env.expand(members, -1), code[:, 8:24], reference.expand(members))
                static_anchor = (current - anchored).square().mean()
                loss, components = probability_objective(
                    trajectory, support_h, query_h, _tensor(labels.q_supervision, device), scale, kl,
                    _tensor(residual_model["stationary_sd_normalized_H"], device),
                    min(step / kl_warmup, 1.), regularizer, static_anchor)
            else:
                smask, qmask = torch.isfinite(support_h), torch.isfinite(query_h)
                if not smask.any() or not qmask.any():
                    raise ValueError("来源监督缺少有效支持或查询H")
                loss = ((trajectory.support[:, smask] - support_h[smask]) / scale).square().mean()
                loss = loss + ((trajectory.query[:, qmask] - query_h[qmask]) / scale).square().mean()
                components = {"normalized_H_MSE": loss}
            if not torch.isfinite(loss):
                raise ValueError(f"阶段{stage}损失非有限，保留日志停止")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            # Common random numbers make checkpoint selection comparable.
            with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
                torch.manual_seed(seed + 1000)
                score, residuals = validation_score(model, validation_pairs, members)
            improved = score < best
            record = {"stage": stage, "step": step + 1, "loss": float(loss.detach()),
                      "validation_CRPS_normalized_H": score,
                      "components": {k: float(v.detach()) if torch.is_tensor(v) else v for k, v in components.items()}}
            records.append(record)
            write_json(output / "training_history.json", records)
            if improved:
                best, stale = score, 0
                torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "stage": stage, "step": step + 1, "seed": seed,
                            "torch_rng_state": torch.get_rng_state(), "validation_score": score,
                            "trained_systems": sorted(train_systems)}, stage_path)
            else:
                stale += 1
            if stale >= patience and (stage != "D" or step >= kl_warmup):
                break
        checkpoint = torch.load(stage_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if stage == "B":
            anchor = copy.deepcopy(model.curve).requires_grad_(False).eval()
        if stage == "C":
            _, residuals = validation_score(model, validation_pairs, members)
            residual_model = calibrate_residuals(residuals)
        write_json(output / f"stage_{stage}.json", {"checkpoint_sha256": digest(stage_path),
                                                    "elapsed_seconds": time.perf_counter() - started,
                                                    "selected_validation_CRPS": best})
    model.configure_stage("E")
    score, residuals = validation_score(model, validation_pairs, members)
    residual_model = calibrate_residuals(residuals)
    write_json(output / "stage_E_calibration.json", {**residual_model, "validation_CRPS": score,
                                                    "member_disagreement_and_total_intervals_must_be_separate": True})
    torch.save({"model_state": model.state_dict(), "stage": "E", "seed": seed,
                "trained_systems": sorted(train_systems), "residual_model": residual_model,
                "shared_weights_frozen": True}, output / "model_E.pt")
    return {"stages": ["B", "C", "D", "E"], "optimizer_steps": len(records),
            "checkpoint": str((output / "model_E.pt").resolve()), "residual_model": residual_model}


def predict_forecast(model, episode, trained_systems, members=64, residual_model=None, seed=20261019):
    """Frozen target inference. Future observations cannot enter the signature."""
    if model.stage != "E" or any(p.requires_grad for p in model.parameters()):
        raise ValueError("目标预测前必须完成阶段E并冻结共享权重")
    if episode.role != "development_test":
        raise ValueError("本地目标入口仅接收已留出的开发评估水系")
    assert_source_permissions(trained_systems, {episode.system_id})
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        trajectory, _, _, _, _, scale, _ = episode_trajectory(model, episode, members, stochastic=True)
    members_h = trajectory.query.cpu().numpy()
    result = {"members_without_residual": members_h,
              "time_ns": episode.inputs["time_ns"][episode.support_end:episode.query_end].copy()}
    if residual_model is not None:
        if not residual_model.get("source_validation_only"):
            raise ValueError("残差模型缺少来源验证校准证据")
        rng = np.random.default_rng(seed + 1)
        mean, sd = residual_model["mean_normalized_H"], residual_model["stationary_sd_normalized_H"]
        rho, innovation = residual_model["rho"], residual_model["innovation_sd_normalized_H"]
        error = np.empty_like(members_h)
        error[:, 0] = mean + sd * rng.normal(size=members)
        for hour in range(1, error.shape[1]):
            error[:, hour] = mean + rho * (error[:, hour - 1] - mean) + innovation * rng.normal(size=members)
        result["members_with_source_calibrated_residual"] = members_h + error * float(scale.cpu()[0])
    return result
