import numpy as np
import torch


def ensemble_crps(members, observed):
    """成员第一维；O(K log K) CRPS，无K²矩阵。"""
    k = members.shape[0]
    ordered = members.sort(dim=0).values
    weights = torch.arange(1, k + 1, device=members.device, dtype=members.dtype) * 2 - k - 1
    spread = (ordered * weights.reshape((-1,) + (1,) * (members.ndim - 1))).sum(0) / k**2
    return ((members - observed).abs().mean(0) - spread).mean()


def event_windows(h, threshold, merge_gap=24):
    hits = np.flatnonzero(np.isfinite(h) & (h >= threshold))
    if not len(hits):
        return []
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > merge_gap) + 1)
    windows = [(max(0, int(g[0]) - 24), min(len(h), int(g[-1]) + 49)) for g in groups]
    merged = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def score_site(observed, members, support_h, block_hours=168):
    observed, members = np.asarray(observed), np.asarray(members)
    if members.ndim == 1:
        members = members[None]
    median = np.median(members, axis=0)
    if members.shape[1] != len(observed) or not np.isfinite(members).all():
        raise ValueError("预测长度不匹配或包含非有限值，禁止通过排除预测失败时段改善评分")
    valid = np.isfinite(observed)
    for start in range(0, len(valid), block_hours):
        end = min(start + block_hours, len(valid))
        if np.isfinite(observed[start:end]).mean() < 0.9:
            valid[start:end] = False
    if not valid.any():
        raise ValueError("没有达到查询覆盖门槛的评分窗口")
    error = median[valid] - observed[valid]
    result = {
        "MAE_m": float(np.abs(error).mean()),
        "RMSE_m": float(np.sqrt(np.mean(error**2))),
        "valid_hours": int(valid.sum()),
        "query_hours": len(valid),
    }
    finite_support = np.asarray(support_h)[np.isfinite(support_h)]
    threshold = float(np.quantile(finite_support, 0.95))
    event_scores, peak_errors, peak_times = [], [], []
    for start, end in event_windows(observed, threshold):
        if valid[start:end].mean() < 0.9:
            continue
        obs = observed[start:end].copy()
        pred = median[start:end].copy()
        mask = valid[start:end]
        obs[~mask], pred[~mask] = np.nan, np.nan
        event_scores.append(float(np.sqrt(np.nanmean((pred - obs) ** 2))))
        peak_errors.append(float(abs(np.nanmax(pred) - np.nanmax(obs))))
        peak_times.append(float(abs(np.nanargmax(pred) - np.nanargmax(obs))))
    result.update(
        event_RMSE_m=float(np.mean(event_scores)) if event_scores else None,
        peak_error_m=float(np.mean(peak_errors)) if peak_errors else None,
        peak_time_error_h=float(np.mean(peak_times)) if peak_times else None,
        events=len(event_scores),
        research_threshold_m=threshold,
    )
    result["CRPS_m"] = float(ensemble_crps(torch.tensor(members[:, valid]), torch.tensor(observed[valid])))
    if len(members) > 1:
        for level in [0.5, 0.9]:
            low, high = np.quantile(members[:, valid], [(1 - level) / 2, (1 + level) / 2], axis=0)
            result[f"coverage_{int(level * 100)}"] = float(
                ((observed[valid] >= low) & (observed[valid] <= high)).mean()
            )
            result[f"width_{int(level * 100)}_m"] = float((high - low).mean())
    return result


def system_bootstrap(frame, metric="event_RMSE_m", seed=20261019, samples=1000):
    clean = frame.dropna(subset=[metric])
    systems = clean.terminal_system_id.unique()
    if not len(systems):
        return {"mean": None, "ci95": None, "systems": 0}
    rng = np.random.default_rng(seed)
    blocks = {s: clean.loc[clean.terminal_system_id == s, metric].to_numpy(float) for s in systems}
    draws = [
        np.concatenate([blocks[s] for s in rng.choice(systems, len(systems), replace=True)]).mean()
        for _ in range(samples)
    ]
    return {
        "mean": float(clean[metric].mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "sites": len(clean),
        "systems": len(systems),
        "resampling_unit": "terminal_river_system",
    }
