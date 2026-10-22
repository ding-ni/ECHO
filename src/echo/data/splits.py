"""终端水系级固定划分与三折交叉拟合。"""

import hashlib

import pandas as pd


def group_key(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def build_splits(manifest, seed=20261019, max_sites=100):
    df = manifest.copy()
    required = [
        "station_id",
        "terminal_system_id",
        "topology_evidence",
        "area_km2",
        "support_coverage",
        "query_coverage",
        "time_support_verified",
        "datum_verified",
    ]
    if any(k not in df for k in required):
        raise ValueError(f"站点清单必须包含: {required}")
    if df.station_id.duplicated().any():
        raise ValueError("站点清单重复")
    df["exclusion_reason"] = ""
    checks = {
        "unknown_terminal": df.terminal_system_id.isna() | (df.terminal_system_id.astype(str) == ""),
        "unknown_topology": df.topology_evidence.isna() | (df.topology_evidence.astype(str) == ""),
        "area_outside_range": ~df.area_km2.between(20, 5000),
        "support_coverage": ~df.support_coverage.ge(0.8),
        "query_coverage": ~df.query_coverage.ge(0.9),
        "unverified_time": ~df.time_support_verified.eq(True),
        "unverified_datum": ~df.datum_verified.eq(True),
    }
    if "weather_available" in df:
        checks["weather_unavailable"] = ~df.weather_available.eq(True)
    if "topology_verified" in df:
        checks["unverified_topology_crosscheck"] = ~df.topology_verified.eq(True)
    for reason, mask in checks.items():
        df.loc[mask, "exclusion_reason"] += reason + ";"
    df["split"] = "excluded"
    good = df[df.exclusion_reason == ""]
    systems = sorted(good.terminal_system_id.unique(), key=lambda x: group_key(x, seed))
    if len(systems) < 8:
        raise ValueError("至少需8个独立终端水系，才能预留盲测并划分开发池；不以随机站点替代")
    blind_n = max(1, round(len(systems) * 0.2))
    blind = set(systems[:blind_n])
    df.loc[df.terminal_system_id.isin(blind) & (df.exclusion_reason == ""), "split"] = "blind_holdout"
    pool = good[~good.terminal_system_id.isin(blind)].copy()
    # 环境分层轮转抽样：在各面积/坡度层中按固定哈希取样，无性能驱动选择。
    pool["stratum"] = pd.cut(pool.area_km2, [0, 100, 500, 2000, float("inf")], labels=False).astype(str)
    if "slope_degrees" in pool:
        pool["stratum"] += "_" + (pool.slope_degrees.fillna(-1) >= 5).astype(str)
    pool["order"] = pool.station_id.map(lambda x: group_key(x, seed))
    groups = [g.sort_values("order").index.tolist() for _, g in pool.groupby("stratum")]
    selected = []
    while any(groups) and len(selected) < max_sites:
        for g in groups:
            if g and len(selected) < max_sites:
                selected.append(g.pop(0))
    dev_systems = sorted(pool.loc[selected].terminal_system_id.unique(), key=lambda x: group_key(x, seed + 1))
    # 整水系分配。小开发集按组数近似60/20/20，绝不切开水系。
    n = len(dev_systems)
    if n < 5:
        raise ValueError("抽样后开发水系不足5个，应增加候选覆盖")
    train_n, val_n = max(3, round(n * 0.6)), max(1, round(n * 0.2))
    train_n = min(train_n, n - 2)
    val_n = min(val_n, n - train_n - 1)
    labels = {
        s: ("train" if i < train_n else "validation" if i < train_n + val_n else "development_test")
        for i, s in enumerate(dev_systems)
    }
    # 水系归属覆盖全部站点，抽样/质量排除是另一维；同水系的未选站不能成为另一集合来源。
    system_labels = {**dict.fromkeys(blind, "blind_holdout"), **labels}
    df["system_split"] = df.terminal_system_id.map(system_labels).fillna("excluded")
    df.loc[selected, "split"] = pool.loc[selected].terminal_system_id.map(labels)
    df.loc[(df.split == "excluded") & (df.exclusion_reason == ""), "exclusion_reason"] = (
        "not_selected_development_budget;"
    )
    df["crossfit_fold"] = -1
    trains = [s for s in dev_systems if labels[s] == "train"]
    fold = {s: i % 3 for i, s in enumerate(trains)}
    df.loc[df.split == "train", "crossfit_fold"] = df.loc[df.split == "train", "terminal_system_id"].map(fold)
    df["target_Q_forbidden"] = df.system_split != "train"
    return df
