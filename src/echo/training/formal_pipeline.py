"""Small-sample end-to-end runner for the complete ECHO architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from echo.data.episodes import tokens
from echo.data.permissions import assert_source_permissions, support_scaling
from echo.evaluation.report import evaluate, render_report
from echo.models.formal import FormalECHO
from echo.training.pipeline import load_station, tensor
from echo.utils import digest, read_json, run_metadata, write_json


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _geography(spatial_row, device):
    data = _load_npz(spatial_row["path"])
    climate = np.asarray(data["climate_features"], dtype="float32")
    climate = np.pad(climate[:64], (0, max(0, 64 - len(climate))))
    parents = data["downstream_parent"].astype(int).tolist()
    edge_index = torch.tensor(
        [[i, parent] for i, parent in enumerate(parents) if parent >= 0],
        dtype=torch.long,
        device=device,
    ).T.contiguous()
    geography = {
        "node_features": tensor(data["node_features"][None], device),
        "node_areas": tensor(data["node_areas"][None], device),
        "edge_index": edge_index,
        "terrain_tokens": tensor(data["terrain_tokens"][None], device),
        "terrain_padding_mask": torch.as_tensor(data["terrain_padding_mask"][None], device=device),
        "climate_features": tensor(climate[None], device),
    }
    return geography, data, parents


def _station_context(row, spatial_row, device):
    data = load_station(row, exclude_q=row["split"] != "train")
    geography, spatial, parents = _geography(spatial_row, device)
    return data, geography, spatial, parents


def _weather_cube(weather, nodes, device):
    base = tensor(weather, device)
    return base[None, None].expand(1, nodes, -1, -1).contiguous()


def _support_heights(data, start, end):
    if "h" in data:
        return data["h"][start:end]
    support_start = int(data["support_start"])
    offset = start - support_start
    values = data["support_h"][offset:offset + (end - start)]
    if len(values) != end - start:
        raise ValueError("请求的历史H超出目标许可支持期")
    return values


def _run_hydrology(model, weather, geography, parents, code, warmup_hours=0):
    nodes = geography["node_features"].shape[1]
    node_features = geography["node_features"]
    node_areas = geography["node_areas"]
    if warmup_hours:
        warm = _weather_cube(weather[:warmup_hours], nodes, weather.device)
        neutral = weather.new_zeros((1, 32))
        with torch.no_grad():
            spin = model.simulate_hydrology(warm, node_features, node_areas, parents, neutral)
        state = spin["state"].expand(len(code), -1, -1).contiguous()
        route_state = spin["routing_state"].expand(len(code), -1).contiguous()
    else:
        state = route_state = None
    cube = _weather_cube(weather[warmup_hours:], nodes, weather.device)
    result = model.simulate_hydrology(
        cube, node_features, node_areas, parents, code, state=state, routing_state=route_state
    )
    return result


def _predict_window(model, data, geography, spatial, parents, reference_q, start, support_end,
                    query_end, members, device, warmup_hours):
    h_support = _support_heights(data, start, support_end)
    mask = np.isfinite(h_support)
    if mask.mean() < 0.8:
        raise ValueError("支持期H有效率低于80%")
    center, scale = support_scaling(h_support, mask)
    hist_tokens, _, _ = tokens(
        data["weather"][start:support_end], reference_q[start:support_end],
        float(data["area"]), h_support, mask,
    )
    history = tensor(hist_tokens[None], device)
    codes, kl, environment = model.infer(history, geography, members=members, stochastic=True)
    if codes.shape[1] != 1:
        raise ValueError("本地逐站流水线期望每个episode只有一个水系样本")
    codes = codes[:, 0]
    forcing = tensor(data["weather"][start:query_end], device)
    warm_start = max(0, start - warmup_hours)
    if warm_start < start:
        weather = tensor(data["weather"][warm_start:query_end], device)
        offset = start - warm_start
        hydro = _run_hydrology(model, weather, geography, parents, codes, offset)
    else:
        hydro = _run_hydrology(model, forcing, geography, parents, codes, 0)
    q = hydro["q_outlet"]
    predictions = []
    area = tensor([float(data["area"])], device)
    center_t, scale_t = tensor([center], device), tensor([scale], device)
    uref = torch.log1p(q[:, :support_end - start].detach().clamp_min(0) / (area[:, None] / 86.4))
    uref = uref.median(-1).values
    for member in range(members):
        pred, _ = model.generate(q[member:member + 1], area, environment,
                                 codes[member:member + 1], center_t, scale_t,
                                 uref[member:member + 1])
        predictions.append(pred[0])
    return torch.stack(predictions), q, kl, center, scale, mask


def _choose_episode(data, rng, support_hours, query_hours, train_start, train_end):
    times = pd.to_datetime(data["time_ns"], utc=True).asi8
    first = int(np.searchsorted(times, pd.Timestamp(train_start, tz="UTC").value))
    stop = int(np.searchsorted(times, pd.Timestamp(train_end, tz="UTC").value))
    candidates = np.arange(first + support_hours, stop - query_hours, 24)
    if not len(candidates):
        raise ValueError(f"{data['station_id']}没有符合配置的来源训练窗口")
    for origin in rng.permutation(candidates)[: min(60, len(candidates))]:
        start = int(origin - support_hours)
        h = data["h"][start:origin]
        q = data["q"][start:origin + query_hours]
        if np.isfinite(h).mean() >= 0.8 and np.isfinite(q).mean() >= 0.5:
            return start, int(origin), int(origin + query_hours)
    raise ValueError(f"{data['station_id']}找不到H/Q覆盖合格的来源窗口")


def _make_model(config):
    return FormalECHO(
        climate_dim=64,
        width=config["width"],
        state_dim=config["state_dim"],
        head_dim=config["head_dim"],
        rank=config["mimo_rank"],
        mamba_blocks=config["mamba_blocks"],
        pooling_queries=config["pooling_queries"],
        pooling_heads=config["pooling_heads"],
        posterior_layers=config["posterior_layers"],
        spline_bins=config["spline_bins"],
        posterior_hidden=config["posterior_hidden"],
        curve_segments=config["curve_segments"],
        dynamic_blocks=config["dynamic_blocks"],
        mamba_backend="torch_reference",
    )


def run(index_path, spatial_index_path, config_path, output):
    index_path, spatial_index_path = Path(index_path), Path(spatial_index_path)
    config_path, output = Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空，为防止覆盖已有结果而停止: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index, spatial_index = read_json(index_path), read_json(spatial_index_path)
    if digest(index_path) != spatial_index["h_index_sha256"]:
        raise ValueError("正式空间输入与水位索引不匹配")
    rows = {row["station_id"]: row for row in index["stations"]}
    spatial_rows = {row["station_id"]: row for row in spatial_index["stations"]}
    train_ids = config["training_station_ids"]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("来源站列表有重复站号")
    train_rows = [rows[sid] for sid in train_ids]
    for row in train_rows:
        spatial_row = spatial_rows.get(row["station_id"])
        if row["split"] != "train" or spatial_row is None:
            raise ValueError(f"来源站角色或正式空间输入不匹配: {row['station_id']}")
        if row["terminal_system_id"] != spatial_row["terminal_system_id"] or row["crossfit_fold"] != spatial_row["crossfit_fold"]:
            raise ValueError(f"来源站水系/交叉拟合折号不一致: {row['station_id']}")
    forbidden = set(index.get("forbidden_systems", [])) | {
        row["terminal_system_id"] for row in index["stations"] if row["split"] != "train"
    }
    assert_source_permissions({row["terminal_system_id"] for row in train_rows}, forbidden)

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    device = torch.device(config.get("device", "cpu"))
    model = _make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    hydro_root = Path(config["hydrology_run"])
    ref_root = hydro_root / "参考流量"
    rng = np.random.default_rng(int(config["seed"]))
    station_payload = []
    for row in train_rows:
        data, geography, spatial, parents = _station_context(row, spatial_rows[row["station_id"]], device)
        qref_path = ref_root / f"{row['station_id']}.npy"
        qref = np.load(qref_path, mmap_mode="r")
        if len(qref) != len(data["time_ns"]):
            raise ValueError(f"来源参考Q与气象时间长度不匹配: {row['station_id']}")
        station_payload.append((row, data, geography, spatial, parents, qref))

    train_start = config["train_start"]
    train_end = config["train_end"]
    support_hours = int(config["support_days"] * 24)
    query_hours = int(config["query_hours"])
    warmup_hours = int(config["hydro_warmup_days"] * 24)
    history, steps = [], []
    model.train()
    for epoch in range(int(config["epochs"])):
        for row, data, geography, spatial, parents, qref in station_payload:
            start, support_end, query_end = _choose_episode(
                data, rng, support_hours, query_hours, train_start, train_end
            )
            # The posterior sees support H only. Query H/Q remain separate training labels.
            pred, qpred, kl, center, scale, mask = _predict_window(
                model, data, geography, spatial, parents, qref, start, support_end,
                query_end, int(config["train_members"]), device, warmup_hours,
            )
            support_y = tensor(data["h"][start:support_end], device)
            query_y = tensor(data["h"][support_end:query_end], device)
            support_valid = torch.as_tensor(mask, device=device)
            query_valid = torch.isfinite(query_y)
            h_loss = pred[:, :support_hours][:, support_valid].sub(support_y[support_valid]).div(scale).square().mean()
            query_pred = pred[:, support_hours:]
            if query_valid.any():
                h_loss = h_loss + query_pred[:, query_valid].sub(query_y[query_valid]).div(scale).square().mean()
            observed_q = tensor(data["q"][start:query_end], device)
            q_valid = torch.isfinite(observed_q) & (observed_q >= 0)
            if q_valid.any():
                q_loss = (torch.log1p(qpred[:, q_valid]) - torch.log1p(observed_q[q_valid])).square().mean()
            else:
                q_loss = h_loss.new_zeros(())
            loss = h_loss + float(config["q_loss_weight"]) * q_loss + float(config["kl_weight"]) * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"训练损失非有限值: {row['station_id']}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            record = {"epoch": epoch + 1, "station_id": row["station_id"],
                      "terminal_system_id": row["terminal_system_id"],
                      "crossfit_fold": row["crossfit_fold"], "start_index": start,
                      "support_hours": support_hours, "query_hours": query_hours,
                      "h_loss": float(h_loss.detach()), "q_loss": float(q_loss.detach()),
                      "kl": float(kl.detach()), "loss": float(loss.detach())}
            history.append(record)
            steps.append(record)
            print(f"formal epoch={epoch + 1}/{config['epochs']} station={row['station_id']} loss={record['loss']:.5f}", flush=True)
        write_json(output / "训练过程.json", history)

    checkpoint = {
        "state": model.state_dict(), "config": config,
        "index_hash": digest(index_path), "spatial_index_hash": digest(spatial_index_path),
        "training_station_ids": train_ids,
        "trained_systems": sorted({row["terminal_system_id"] for row in train_rows}),
        "crossfit_folds": {row["station_id"]: row["crossfit_fold"] for row in train_rows},
        "mamba_backend": model.mamba_backend,
        "architecture_scope": "complete_protocol_module_graph_reduced_local_configuration",
        "future_H_used_for_prediction": False,
        "source_query_HQ_used_as_training_labels": True,
    }
    torch.save(checkpoint, output / "model.pt")
    write_json(output / "run.json", {
        "scope": "formal_architecture_real_small_sample_smoke",
        "architecture": {"basin_encoder": "three_layer_GATv2",
                         "terrain_encoder": "16_token_SetAttention",
                         "history_encoder": "MIMO_selective_SSM_torch_reference",
                         "joint_posterior": "32D_conditional_rational_quadratic_spline_flow",
                         "hydrology": "semi_distributed_HBV_with_local_posterior_parameter_adaptation",
                         "response": "32_segment_monotone_curve_plus_8_block_dynamic_state"},
        "training_steps": steps,
        "warmup_days": config["hydro_warmup_days"],
        "spatial_weather": "basin_mean_forcing_repeated_across_incremental_nodes",
        "terrain_resolution": spatial_index["terrain_resolution"],
        "topology_area_quality": {r["station_id"]: r["topology_unmapped_fraction"]
                                   for r in spatial_index["stations"] if r["station_id"] in train_ids},
        "metadata": run_metadata(config, [index_path, spatial_index_path, config_path]),
    })

    model.eval()
    prediction_root = output / "预测"
    records = []
    members = int(config["prediction_members"])
    with torch.no_grad():
        for row in index["stations"]:
            if row["split"] not in set(config["prediction_splits"]):
                continue
            if row["station_id"] not in spatial_rows:
                continue
            assert_source_permissions(checkpoint["trained_systems"], {row["terminal_system_id"]})
            data, geography, spatial, parents = _station_context(row, spatial_rows[row["station_id"]], device)
            qref_path = ref_root / f"{row['station_id']}.npy"
            qref = np.load(qref_path, mmap_mode="r")
            end = int(data["support_end"])
            start = end - support_hours
            query_end = end + query_hours
            prediction, _, _, center, scale, _ = _predict_window(
                model, data, geography, spatial, parents, qref, start, end, query_end,
                members, device, warmup_hours,
            )
            future = prediction[:, support_hours:].cpu().numpy()
            station_dir = prediction_root / row["split"]
            station_dir.mkdir(parents=True, exist_ok=True)
            path = station_dir / f"{row['station_id']}.npz"
            np.savez_compressed(path, members=future,
                                time_ns=data["time_ns"][end:query_end],
                                support_h=_support_heights(data, start, end),
                                station_id=np.array(row["station_id"]),
                                system=np.array(row["terminal_system_id"]))
            records.append({"station_id": row["station_id"], "terminal_system_id": row["terminal_system_id"],
                            "path": str(path.resolve()), "split": row["split"],
                            "support_days": config["support_days"], "query_hours": query_hours,
                            "center_m": center, "scale_m": scale})
            print(f"predicted station={row['station_id']} split={row['split']}", flush=True)
    write_json(prediction_root / "predictions.json", {
        "scope": "formal_architecture_real_small_sample_smoke",
        "mode": "conditional_simulation_no_future_H_updates",
        "model": "formal_echo_local_reduced",
        "members": members, "budget_days": config["support_days"],
        "query_hours": query_hours, "uncertainty_scope": "conditional_joint_local_code_only",
        "stations": records, "failures": [],
    })
    labels_root = index_path.parent / "评分标签"
    for split in config["prediction_splits"]:
        split_index = prediction_root / "predictions.json"
        # The evaluator consumes its own manifest; score only one split at a time.
        doc = read_json(split_index)
        chosen = [r for r in doc["stations"] if r["split"] == split]
        if not chosen:
            continue
        split_root = prediction_root / split
        write_json(split_root / "predictions.json", {**doc, "stations": chosen})
        score_root = output / "评估" / split
        summary = evaluate(split_root, labels_root, score_root)
        render_report(score_root, split_root, labels_root)
        print(f"scored split={split} sites={summary['sites']}", flush=True)
    return read_json(output / "run.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--spatial-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.index, args.spatial_index, args.config, args.output)


if __name__ == "__main__":
    main()
