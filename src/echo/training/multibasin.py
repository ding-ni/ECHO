"""多流域批量HBV预训练；复用同一个HBV方程，按冻结水系留折。"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from echo.hydro.hbv import RegionalHBV
from echo.utils import digest, read_json, run_metadata, utc_now, write_json


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def station_logq_loss(predicted, observed, minimum_coverage=.8):
    valid = torch.isfinite(observed) & (observed >= 0)
    counts = valid.sum(1)
    eligible = counts >= observed.shape[1] * minimum_coverage
    safe = torch.where(valid, observed, torch.zeros_like(observed))
    squared = (torch.log1p(predicted) - torch.log1p(safe)).square()
    means = (squared * valid).sum(1) / counts.clamp_min(1)
    return (means[eligible].mean() if eligible.any() else None), int(counts[eligible].sum())


def load_sources(index_path):
    index = read_json(index_path)
    if index.get("scope") != "real_source_Q_pretraining":
        raise ValueError("只接受明确标注的来源Q预训练数据")
    forbidden = set(index["forbidden_systems"])
    arrays, rows, times = [], [], None
    for row in index["stations"]:
        if row["split"] != "train" or row["terminal_system_id"] in forbidden:
            raise ValueError("非训练水系不得进入Q预训练")
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("来源数据哈希改变")
        with np.load(row["path"], allow_pickle=False) as z:
            if {"h", "support_h", "future_h"} & set(z.files):
                raise ValueError("Q预训练文件不应读取或携带H")
            if str(z["system"]) != row["terminal_system_id"] or str(z["station_id"]) != row["station_id"]:
                raise ValueError("数据站点/水系与索引不一致")
            if times is None:
                times = z["time_ns"].copy()
            elif not np.array_equal(times, z["time_ns"]):
                raise ValueError("批量训练时间轴不一致")
            arrays.append({k: z[k].copy() for k in ["weather", "q", "geo", "area"]})
        rows.append(row)
    if len({r["terminal_system_id"] for r in rows}) < 3:
        raise ValueError("来源水系不足三折")
    if set(r["crossfit_fold"] for r in rows) != {0, 1, 2}:
        raise ValueError("冻结三折不完整")
    return index, rows, arrays, times


def advance(model, weather, geo, area, stop):
    state = None
    with torch.no_grad():
        for start in range(0, stop, 720):
            _, state, _ = model(weather[:, start:min(start + 720, stop)], geo, area, state=state)
    return state


def validate(model, arrays, first, end, device, batch_size, details_path=None, station_ids=None):
    per_site, details = [], []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            group = arrays[start:start + batch_size]
            weather, q, geo, area = pack(group, device)
            state = advance(model, weather, geo, area, first)
            sums = torch.zeros(len(group), dtype=torch.float64, device=device)
            counts = torch.zeros_like(sums)
            for position in range(first, end, 720):
                stop = min(position + 720, end)
                predicted, state, _ = model(weather[:, position:stop], geo, area, state=state)
                truth = q[:, position:stop]
                valid = torch.isfinite(truth) & (truth >= 0)
                safe = torch.where(valid, truth, torch.zeros_like(truth))
                error = (torch.log1p(predicted) - torch.log1p(safe)).double().square()
                sums += (error * valid).sum(1)
                counts += valid.sum(1)
            good = counts >= (end - first) * .25
            per_site.extend((sums[good] / counts[good]).cpu().tolist())
            for i in range(len(group)):
                details.append({"station_id": station_ids[start + i] if station_ids else str(start + i),
                                "valid_Q_hours": int(counts[i]), "query_hours": end - first,
                                "logQ_MSE": float(sums[i] / counts[i]) if bool(good[i]) else None})
    if not per_site:
        raise ValueError("来源全年验证没有有效站点")
    if details_path:
        write_json(details_path, {"stations": details, "station_equal_mean_logQ_MSE": float(np.mean(per_site))})
    return float(np.mean(per_site))


def pack(arrays, device):
    return tuple(torch.as_tensor(np.stack([d[key] for d in arrays]), dtype=torch.float32, device=device)
                 for key in ["weather", "q", "geo", "area"])


def fit_one(rows, arrays, times, forbidden, config, output, index_hash):
    device = torch.device(config.get("device", "cpu"))
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    model = RegionalHBV(len(arrays[0]["geo"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", .002))
    latest = output.with_suffix(".latest.pt")
    trained = sorted({r["terminal_system_id"] for r in rows})
    if set(trained) & set(forbidden):
        raise ValueError("禁止水系进入来源预训练")
    history, best, best_state, first_epoch = [], float("inf"), None, 0
    elapsed = 0.0
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["index_hash"] != index_hash or saved["config"] != config or saved["trained_systems"] != trained:
            raise ValueError("恢复检查点的数据/配置/水系不同")
        model.load_state_dict(saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        history, first_epoch = saved["history"], saved["next_epoch"]
        best, best_state, elapsed = saved["best_score"], saved["best_state"], saved["elapsed_seconds"]
    time_index = pd.to_datetime(times, utc=True)
    def position(date):
        return int(np.searchsorted(time_index.asi8, pd.Timestamp(date, tz="UTC").value))
    train_start, train_end, validation_end = (position(d) for d in ["2013-10-01", "2018-10-01", "2019-10-01"])
    query, blocks = config.get("query_hours", 168), config.get("blocks_per_epoch", 8)
    batch_size = config.get("basin_batch_size", 32)
    begin = time.monotonic()
    for epoch in range(first_epoch, config["epochs"]):
        rng = np.random.default_rng(config["seed"] + epoch)
        # 均匀轮转五个来源训练水文年，各年内随机取连续窗口，不读取目标表现。
        year = 2013 + epoch % 5
        low, high = max(train_start, position(f"{year}-10-01")), min(train_end, position(f"{year + 1}-10-01"))
        origin = int(rng.integers(low, max(low + 1, high - blocks * query)))
        order = rng.permutation(len(rows))
        losses, points, updates = [], 0, 0
        model.train()
        for start in range(0, len(order), batch_size):
            selected = order[start:start + batch_size]
            group = [arrays[i] for i in selected]
            weather, q, geo, area = pack(group, device)
            state = advance(model, weather, geo, area, origin)
            for block in range(blocks):
                left, right = origin + block * query, origin + (block + 1) * query
                predicted, state, balance = model(weather[:, left:right], geo, area, state=state)
                if not torch.isfinite(predicted).all() or not torch.isfinite(balance).all():
                    raise FloatingPointError("水文训练出现非有限值")
                loss, valid_points = station_logq_loss(predicted, q[:, left:right])
                if loss is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    points += valid_points
                    updates += 1
                state = state.detach()
        if not losses:
            raise ValueError("本轮没有合格来源Q训练块")
        score = None
        if epoch == 0 or (epoch + 1) % config.get("validate_every", 5) == 0 or epoch + 1 == config["epochs"]:
            model.eval()
            score = validate(model, arrays, train_end, validation_end, device, batch_size,
                             output.with_suffix(".validation.json"), [r["station_id"] for r in rows])
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
                save_checkpoint(output, {"state": best_state, "geo_dim": len(arrays[0]["geo"]), "config": config,
                            "index_hash": index_hash, "trained_systems": trained,
                            "forbidden_systems": sorted(forbidden), "best_epoch": epoch + 1})
        history.append({"epoch": epoch + 1, "train_logQ_mse": float(np.mean(losses)),
                        "source_full_year_validation": score, "valid_Q_training_points": points,
                        "optimizer_steps": updates, "origin_utc": str(time_index[origin])})
        elapsed_now = elapsed + time.monotonic() - begin
        save_checkpoint(latest, {"state": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": config, "index_hash": index_hash, "trained_systems": trained,
                    "best_state": best_state, "best_score": best, "history": history,
                    "next_epoch": epoch + 1, "elapsed_seconds": elapsed_now})
        write_json(output.with_suffix(".history.json"), history)
        write_json(output.with_suffix(".status.json"), {"updated_utc": utc_now(), "epoch": epoch + 1,
                   "epochs": config["epochs"], "sites": len(rows), "systems": len(trained),
                   "best_source_validation": best, "elapsed_seconds": elapsed_now,
                   "total_valid_Q_points": sum(h["valid_Q_training_points"] for h in history),
                   "rss_bytes": psutil.Process().memory_info().rss,
                   "complete": epoch + 1 == config["epochs"]})
        print(f"{output.stem} epoch={epoch + 1}/{config['epochs']} sites={len(rows)} "
              f"train={np.mean(losses):.6f} source_year_validation={score} updates={updates}", flush=True)
    return {"sites": len(rows), "systems": len(trained), "best_validation": best}


def train(index_path, config_path, output):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, rows, arrays, times = load_sources(index_path)
    write_json(output / "metadata.json", {**run_metadata(config, [index_path, config_path]),
               "scope": index["scope"], "sites": len(rows), "systems": len({r['terminal_system_id'] for r in rows}),
               "source_H_read": False, "target_Q_read": False,
               "states": "float32; long spinup detached, TBPTT168h as in existing frontend"})
    results = {}
    for fold in config.get("folds", [None, 0, 1, 2]):
        selected = [i for i, r in enumerate(rows) if fold is None or r["crossfit_fold"] != fold]
        forbidden = set(index["forbidden_systems"]) | {
            r["terminal_system_id"] for r in rows if fold is not None and r["crossfit_fold"] == fold}
        name = "hydro_full" if fold is None else f"hydro_fold{fold}"
        print(f"开始{name}: {len(selected)}站，按冻结水系分折", flush=True)
        results[name] = fit_one([rows[i] for i in selected], [arrays[i] for i in selected], times,
                                forbidden, config, output / (name + ".pt"), digest(index_path))
        write_json(output / "progress.json", {"updated_utc": utc_now(), "completed": results,
                   "all_complete": len(results) == len(config.get("folds", [None, 0, 1, 2]))})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多流域HBV批量训练和三折，自动从最近完整轮次恢复")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(train(arguments.index, arguments.config, arguments.output))
