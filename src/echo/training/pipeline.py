"""第一阶段实际训练器：区域HBV、跨水系参考Q、地方响应和直接LSTM。"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from echo.data.episodes import hours_since, source_episode, tokens
from echo.data.permissions import assert_source_permissions, support_scaling
from echo.evaluation.metrics import ensemble_crps
from echo.hydro.hbv import RegionalHBV
from echo.models.echo import DirectStage, EchoStage
from echo.utils import digest, read_json, run_metadata, write_json

DEFAULT_PERIODS = {
    "train_start": "2013-10-01",
    "train_end": "2018-10-01",
    "validation_end": "2019-10-01",
    "stage_validation_origin": "2018-11-01",
}


def tensor(x, device):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def load_station(row, no_geo=False):
    if digest(row["path"]) != row["sha256"]:
        raise ValueError(f"数据文件哈希变更: {row['station_id']}")
    with np.load(row["path"], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if row["split"] != "train" and "q" in data:
        raise ValueError("目标预测文件含Q")
    if no_geo:
        data["geo"] = np.zeros_like(data["geo"])
    return data


def detach_state(state):
    if isinstance(state, tuple):
        return tuple(detach_state(v) for v in state)
    return state.detach() if state is not None else None


def simulate_q(model, data, device, code=None, stop=None, gradients=False, block=720):
    weather = data["weather"] if stop is None else data["weather"][:stop]
    geo = tensor(data["geo"][None], device)
    area = tensor([data["area"]], device)
    state, result = None, []
    if len(weather) == 0:
        return geo.new_zeros((1, 0)), geo.new_zeros((1, 5))
    with torch.set_grad_enabled(gradients):
        for start in range(0, len(weather), block):
            q, state, balance = model(
                tensor(weather[None, start : start + block], device), geo, area, code, state
            )
            if not torch.isfinite(q).all():
                raise FloatingPointError("水文模拟出现非有限值")
            result.append(q)
            if not gradients:
                state = detach_state(state)
    return torch.cat(result, 1), state


def fit_hydrology(rows, forbidden, config, device, output, index_hash=None):
    assert_source_permissions([r["terminal_system_id"] for r in rows], forbidden)
    if not rows:
        raise ValueError("来源水系为空")
    first = load_station(rows[0])
    model = RegionalHBV(len(first["geo"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("hydro_lr", 0.002))
    rng = np.random.default_rng(config["seed"])
    epochs = config.get("hydro_epochs", 5)
    history, best_loss, best = [], float("inf"), None
    periods = config.get("periods", DEFAULT_PERIODS)
    for epoch in range(epochs):
        losses = []
        random.shuffle(rows)
        for row in rows:
            d = load_station(row, config.get("no_geo", False))
            times = pd.to_datetime(d["time_ns"], utc=True)
            train_start = int(
                np.searchsorted(times.asi8, pd.Timestamp(periods["train_start"], tz="UTC").value)
            )
            train_end = int(np.searchsorted(times.asi8, pd.Timestamp(periods["train_end"], tz="UTC").value))
            block = config.get("query_hours", 168)
            begin = int(rng.integers(train_start, train_end - block))
            # 至少两年实际气象预热；忽略预热状态参数导数是显式数值近似。
            _, state = simulate_q(model, d, device, stop=begin)
            for i in range(config.get("hydro_blocks_per_station", 8)):
                start = begin + i * block
                if start + block > train_end:
                    break
                obs = tensor(d["q"][start : start + block], device)[None]
                mask = torch.isfinite(obs) & (obs >= 0)
                q, state, balance = model(
                    tensor(d["weather"][None, start : start + block], device),
                    tensor(d["geo"][None], device),
                    tensor([d["area"]], device),
                    state=state,
                )
                if mask.sum() >= block * 0.8:
                    loss = (torch.log1p(q[mask]) - torch.log1p(obs[mask])).square().mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                state = state.detach()
        # 来源时间留出验证：2018-10至2019-10。它不读取目标Q。
        if not losses:
            raise ValueError("该轮没有有效来源Q训练块，不能验收未训练前端")
        val = []
        for row in rows:
            d = load_station(row, config.get("no_geo", False))
            times = pd.to_datetime(d["time_ns"], utc=True)
            mask = ((times >= periods["train_end"]) & (times < periods["validation_end"])
                    & np.isfinite(d["q"]) & (d["q"] >= 0))
            q, _ = simulate_q(
                model,
                d,
                device,
                stop=int(
                    np.searchsorted(times.asi8, pd.Timestamp(periods["validation_end"], tz="UTC").value)
                ),
            )
            m = tensor(mask[: q.shape[1]], device).bool()
            if m.any():
                obs = tensor(d["q"][: q.shape[1]], device)
                val.append(float((torch.log1p(q[0, m]) - torch.log1p(obs[m].clamp_min(0))).square().mean()))
        score = float(np.mean(val)) if val else float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_logQ_mse": float(np.mean(losses)) if losses else None,
                "source_temporal_validation": score,
            }
        )
        write_json(output.with_suffix(".history.json"), history)
        print(f"{output.stem} epoch={epoch + 1}/{epochs} source_validation={score:.6f}", flush=True)
        if score < best_loss:
            best_loss, best = score, copy.deepcopy(model.state_dict())
    if best is None:
        raise ValueError("没有有效来源流量验证，不能生成水文权重")
    model.load_state_dict(best)
    torch.save(
        {
            "state": best,
            "geo_dim": len(first["geo"]),
            "trained_systems": sorted({r["terminal_system_id"] for r in rows}),
            "forbidden_systems": sorted(forbidden),
            "config": config,
            "index_hash": index_hash,
        },
        output,
    )
    return model.eval()


def reference_models(index, config, device, root, index_hash=None):
    sources = [r for r in index["stations"] if r["split"] == "train"]
    forbidden = set(index.get("forbidden_systems", [])) | {
        r["terminal_system_id"] for r in index["stations"] if r["split"] != "train"
    }
    if len({r["terminal_system_id"] for r in sources}) < 3:
        raise ValueError("三折交叉拟合至少需要3个来源水系")
    root.mkdir(parents=True, exist_ok=True)
    if {r["crossfit_fold"] for r in sources} != {0, 1, 2}:
        raise ValueError("来源站未覆盖三折，不能生成交叉拟合前端")
    full = fit_hydrology(list(sources), forbidden, config, device, root / "hydro_full.pt", index_hash)
    folds = {}
    for fold in range(3):
        kept = [r for r in sources if r["crossfit_fold"] != fold]
        excluded = forbidden | {r["terminal_system_id"] for r in sources if r["crossfit_fold"] == fold}
        folds[fold] = fit_hydrology(kept, excluded, config, device, root / f"hydro_fold{fold}.pt", index_hash)
    cache = root / "参考流量"
    cache.mkdir(exist_ok=True)
    for row in index["stations"]:
        if row["split"] not in {"train", "validation", "development_test"}:
            continue
        model = folds[row["crossfit_fold"]] if row["split"] == "train" else full
        assert_source_permissions(model_systems(model, root, row), {row["terminal_system_id"]})
        data = load_station(row, config.get("no_geo", False))
        q, _ = simulate_q(model, data, device)
        np.save(cache / (row["station_id"] + ".npy"), q[0].cpu().numpy())
    return full


def model_systems(model, root, row):
    name = f"hydro_fold{row['crossfit_fold']}.pt" if row["split"] == "train" else "hydro_full.pt"
    return torch.load(root / name, map_location="cpu", weights_only=False)["trained_systems"]


def context_for(model, inputs, reference_q, device, members=1):
    start, end = inputs.support_start, inputs.support_end
    tok, center, scale = tokens(
        inputs.weather[start:end],
        reference_q[start:end],
        inputs.area_km2,
        inputs.support_h,
        inputs.support_mask,
    )
    codes, kl = model.infer(tensor(tok[None], device), tensor(inputs.geography[None], device), members)
    qslice = reference_q[start:end]
    uref = float(np.median(np.log1p(qslice / (inputs.area_km2 / 86.4))))
    return codes, kl, center, scale, uref


def local_prediction(
    model, inputs, reference_q, device, members=1, hydro=None, gradients=False, return_support=False
):
    codes, kl, center, scale, uref = context_for(model, inputs, reference_q, device, members)
    geo, area = tensor(inputs.geography[None], device), tensor([inputs.area_km2], device)
    predictions, support_predictions, support_flows = [], [], []
    warm_start = max(0, inputs.support_start - 90 * 24)
    gradient_start = (
        max(inputs.support_start, inputs.support_end - 720) if gradients else inputs.support_start
    )
    for code in codes:
        if model.probabilistic and hydro is not None:
            d = {"weather": inputs.weather, "geo": inputs.geography, "area": inputs.area_km2}
            # 每条轨迹固定同一个地方成员；预热无梯度，许可支持和未来保持连续状态。
            _, hydro_state = simulate_q(hydro, d, device, code[:, :8].detach(), stop=warm_start)
            history_q = []
            with torch.no_grad():
                for position in range(warm_start, gradient_start, 720):
                    history, hydro_state, _ = hydro(
                        tensor(inputs.weather[None, position : min(position + 720, gradient_start)], device),
                        geo,
                        area,
                        code[:, :8].detach(),
                        hydro_state,
                    )
                    history_q.append(history)
            q, _, _ = hydro(
                tensor(inputs.weather[None, gradient_start : inputs.query_end], device),
                geo,
                area,
                code[:, :8],
                hydro_state,
            )
            if history_q:
                q = torch.cat([*history_q, q], 1)
        else:
            q = tensor(reference_q[None, warm_start : inputs.query_end], device)
        response_state = None
        prefix = gradient_start - warm_start
        if prefix:
            with torch.no_grad():
                prefix_h, response_state = model.generate(
                    q[:, :prefix],
                    area,
                    geo,
                    code.detach(),
                    tensor([center], device),
                    tensor([scale], device),
                    tensor([uref], device),
                )
        h, _ = model.generate(
            q[:, prefix:],
            area,
            geo,
            code,
            tensor([center], device),
            tensor([scale], device),
            tensor([uref], device),
            detach_state(response_state),
        )
        predictions.append(h[0, inputs.query_start - gradient_start :])
        active_support = h[0, : inputs.support_end - gradient_start]
        if gradient_start > inputs.support_start:
            active_support = torch.cat(
                [prefix_h[0, inputs.support_start - warm_start : prefix], active_support]
            )
        support_predictions.append(active_support)
        support_flows.append(q[0, inputs.support_start - warm_start : inputs.support_end - warm_start])
    result = (torch.stack(predictions), kl, scale)
    return (
        (*result, {"stage": torch.stack(support_predictions), "flow": torch.stack(support_flows)})
        if return_support
        else result
    )


def fit_stage(index_path, config, root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("运行目录非空，请使用新的运行编号，保留已有结果")
    root.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 20261019)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 2))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    index = read_json(index_path)
    if index.get("failures"):
        raise ValueError("模型视图含准备失败站点；请解决后另建版本再训练")
    scope = index.get("scope")
    if scope not in {"real_development", "synthetic_demo"}:
        raise ValueError("数据范围未明确")
    periods = config.get("periods", DEFAULT_PERIODS)
    if scope == "real_development" and periods != DEFAULT_PERIODS:
        raise ValueError("真实主协议日期已锁定；短日期只供明确标识的合成测试")
    hydro_root = root.parent / config.get("hydrology_run", "区域水文")
    model_kind = config["model"]
    if model_kind not in {"hydrology", "direct_lstm", "echo_static", "echo_dynamic", "echo_gaussian"}:
        raise ValueError("未实现的模型配置，不能自动替换骨干")
    begin = time.monotonic()
    if model_kind == "hydrology":
        reference_models(index, config, device, root, digest(index_path))
        write_json(
            root / "metadata.json",
            {
                **run_metadata(config, [index_path]),
                "scope": scope,
                "actual_seconds": time.monotonic() - begin,
            },
        )
        return
    if not (hydro_root / "hydro_full.pt").exists():
        raise ValueError("先训练水文前端并生成交叉拟合参考Q")
    hydro_ckpt = torch.load(hydro_root / "hydro_full.pt", map_location=device, weights_only=False)
    if hydro_ckpt.get("index_hash") != digest(index_path):
        raise ValueError("水文前端与当前数据清单不一致")
    if bool(config.get("no_geo")) != bool(hydro_ckpt["config"].get("no_geo")):
        raise ValueError("无地理消融必须同时重训水文前端，不能沿用带地理参考Q")
    hydro = RegionalHBV(hydro_ckpt["geo_dim"]).to(device)
    hydro.load_state_dict(hydro_ckpt["state"])
    hydro.requires_grad_(False)
    rows = [r for r in index["stations"] if r["split"] == "train"]
    dimension = len(load_station(rows[0])["geo"])
    if model_kind == "direct_lstm":
        model = DirectStage(dimension, config.get("hidden", 64)).to(device)
    else:
        model = EchoStage(
            dimension,
            config.get("dynamic", False),
            config.get("probabilistic", False),
            config.get("hidden", 64),
        ).to(device)
    previous = config.get("initialize_from")
    if config.get("initialize_from_run"):
        previous = root.parent / config["initialize_from_run"] / "model.pt"
    if previous:
        checkpoint = torch.load(previous, map_location=device, weights_only=False)
        if checkpoint["index_hash"] != digest(index_path):
            raise ValueError("初始化权重与当前数据划分不一致")
        model.load_state_dict(checkpoint["state"], strict=False)
    if config.get("freeze_static") and isinstance(model, EchoStage):
        model.curve.requires_grad_(False)
        model.encoder.requires_grad_(False)
        model.posterior.requires_grad_(False)
    fold_hydros = {}
    if isinstance(model, EchoStage) and model.probabilistic:
        for fold in range(3):
            ck = torch.load(hydro_root / f"hydro_fold{fold}.pt", map_location=device, weights_only=False)
            fm = RegionalHBV(ck["geo_dim"]).to(device)
            fm.load_state_dict(ck["state"])
            fm.eval().requires_grad_(False)
            fold_hydros[fold] = fm
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("learning_rate", 0.0002),
        weight_decay=0.01,
    )
    rng = np.random.default_rng(seed)
    history, best_score, best, actual_hours = [], float("inf"), None, 0
    for epoch in range(config.get("epochs", 10)):
        model.train()
        train_losses = []
        for row in rows:
            d = load_station(row)
            if config.get("no_geo"):
                d["geo"] = np.zeros_like(d["geo"])
            qref = np.load(hydro_root / "参考流量" / (row["station_id"] + ".npy"))
            t = pd.to_datetime(d["time_ns"], utc=True)
            first = int(np.searchsorted(t.asi8, pd.Timestamp(periods["train_start"], tz="UTC").value))
            stop = int(np.searchsorted(t.asi8, pd.Timestamp(periods["train_end"], tz="UTC").value))
            for _ in range(config.get("episodes_per_station", 4)):
                days = int(rng.choice(config.get("support_days", [7, 30, 90, 366])))
                length, query = days * 24, config.get("query_hours", 168)
                origin = int(rng.integers(first + length, stop - query))
                if days == 366:
                    previous_year = t[origin] - pd.DateOffset(years=1)
                    length = origin - int(np.searchsorted(t.asi8, previous_year.value))
                try:
                    inp, label = source_episode(d, origin - length, length, query)
                except ValueError:
                    continue
                if np.isfinite(label).mean() < 0.9:
                    continue
                episode_hydro = fold_hydros.get(row["crossfit_fold"], hydro)
                if isinstance(model, EchoStage):
                    prediction, kl, scale, support_prediction = local_prediction(
                        model,
                        inp,
                        qref,
                        device,
                        config.get("train_members", 4),
                        episode_hydro,
                        gradients=True,
                        return_support=True,
                    )
                else:
                    prediction, kl, scale = stage_forward(model, inp, qref, device, config, episode_hydro)
                obs = tensor(label, device)
                mask = torch.isfinite(obs)
                loss = ensemble_crps(prediction[:, mask] / scale, obs[mask] / scale)
                if isinstance(model, EchoStage):
                    support_obs = tensor(inp.support_h, device)
                    support_mask = tensor(inp.support_mask, device).bool()
                    loss = (
                        loss
                        + 0.5
                        * ((support_prediction["stage"][:, support_mask] - support_obs[support_mask]) / scale)
                        .square()
                        .mean()
                    )
                    if model.probabilistic:
                        source_q = tensor(d["q"][inp.support_start : inp.support_end], device)
                        qmask = torch.isfinite(source_q) & (source_q >= 0)
                        if qmask.any():
                            loss = (
                                loss
                                + 0.2
                                * (
                                    torch.log1p(support_prediction["flow"][:, qmask])
                                    - torch.log1p(source_q[qmask])
                                )
                                .square()
                                .mean()
                            )
                beta = min(1.0, (epoch + 1) / max(1, config.get("epochs", 10)))
                valid_labels = int(inp.support_mask.sum()) + int(mask.sum())
                loss = loss + beta * 32.0 / max(valid_labels, 1) * kl
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.detach()))
                actual_hours += query + length
        if not train_losses:
            raise ValueError("该轮没有合格的支持/查询训练样本")
        model.eval()
        validation = []
        with torch.no_grad():
            for row in rows:
                d = load_station(row)
                if config.get("no_geo"):
                    d["geo"] = np.zeros_like(d["geo"])
                qref = np.load(hydro_root / "参考流量" / (row["station_id"] + ".npy"))
                t = pd.to_datetime(d["time_ns"], utc=True)
                origin = int(
                    np.searchsorted(t.asi8, pd.Timestamp(periods["stage_validation_origin"], tz="UTC").value)
                )
                try:
                    val_support = config.get("validation_support_hours", 720)
                    inp, label = source_episode(
                        d, origin - val_support, val_support, config.get("query_hours", 168)
                    )
                    valid = np.isfinite(label)
                    if valid.mean() < 0.9:
                        continue
                    pred, _, scale = stage_forward(
                        model, inp, qref, device, config, fold_hydros.get(row["crossfit_fold"], hydro)
                    )
                    validation.append(
                        float(ensemble_crps(pred[:, valid] / scale, tensor(label[valid], device) / scale))
                    )
                except ValueError:
                    continue
        score = float(np.mean(validation)) if validation else float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "source_temporal_validation": score,
            }
        )
        write_json(root / "history.json", history)
        print(f"{model_kind} epoch={epoch + 1} source_validation={score:.6f}", flush=True)
        if score < best_score:
            best_score, best = score, copy.deepcopy(model.state_dict())
    if best is None:
        raise ValueError("没有合格训练与验证episode")
    torch.save(
        {
            "state": best,
            "config": config,
            "geo_dim": dimension,
            "trained_systems": sorted({r["terminal_system_id"] for r in rows}),
            "index_hash": digest(index_path),
        },
        root / "model.pt",
    )
    import psutil

    write_json(
        root / "metadata.json",
        {
            **run_metadata(config, [index_path]),
            "scope": scope,
            "actual_seconds": time.monotonic() - begin,
            "actual_effective_hours": actual_hours,
            "rss_bytes_at_end": psutil.Process().memory_info().rss,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            "torch": torch.__version__,
            "device": str(device),
            "parameters": sum(p.numel() for p in model.parameters()),
            "truncated_history_gradient_hours": 720,
        },
    )


def stage_forward(model, inp, qref, device, config, hydro):
    if isinstance(model, EchoStage):
        return local_prediction(
            model,
            inp,
            qref,
            device,
            config.get("train_members", 4),
            hydro if model.probabilistic else None,
            gradients=torch.is_grad_enabled(),
        )
    center, scale = support_scaling(inp.support_h, inp.support_mask)
    start, end = inp.support_start, inp.query_end
    size = end - start
    h = np.zeros(size, dtype="float32")
    mask = np.zeros(size, dtype="float32")
    h[: len(inp.support_h)] = np.where(inp.support_mask, (inp.support_h - center) / scale, 0)
    mask[: len(inp.support_h)] = inp.support_mask
    pred, _ = model(
        tensor(inp.weather[None, start:end] / np.array([10.0, 20.0, 1.0]), device),
        tensor(inp.geography[None], device),
        tensor(h[None], device),
        tensor(mask[None], device),
        tensor(hours_since(mask)[None], device),
        tensor(np.sin(np.arange(size)[None] / 8766 * 2 * np.pi), device),
    )
    return (center + scale * pred[:, inp.query_start - start :]), pred.new_tensor(0.0), scale
