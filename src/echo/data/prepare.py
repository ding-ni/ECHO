"""按站标准化，来源/目标输入/评分标签物理分开。"""

from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.netcdf import open_netcdf
from echo.geo.features import FeatureScaler, validate_features
from echo.utils import digest, read_json, write_json


def convert(values, unit, kind):
    unit = unit.strip().lower().replace(" ", "")
    x = np.asarray(values, dtype="float64")
    if kind == "H":
        factors = {"m": 1.0, "meter": 1.0, "meters": 1.0, "ft": 0.3048, "feet": 0.3048}
    elif kind == "Q":
        factors = {
            "m3/s": 1.0,
            "m^3/s": 1.0,
            "m3s-1": 1.0,
            "cms": 1.0,
            "cfs": 0.028316846592,
            "ft3/s": 0.028316846592,
            "ft^3/s": 0.028316846592,
        }
    elif kind in ("P", "PET"):
        factors = {
            "mm/h": 1.0,
            "mm/hr": 1.0,
            "mm/hour": 1.0,
            "kg/m2/s": 3600.0,
            "kgm-2s-1": 3600.0,
            "mm/s": 3600.0,
            "mm/day": 1 / 24.0,
            "m/h": 1000.0,
        }
    elif kind == "T":
        if unit in {"k", "kelvin"}:
            return x - 273.15
        factors = {"c": 1.0, "degc": 1.0, "celsius": 1.0, "°c": 1.0, "degree_celsius": 1.0}
    else:
        raise ValueError(f"未知变量 {kind}")
    if unit not in factors:
        raise ValueError(f"未支持或未核验单位 {kind}: {unit}")
    return x * factors[unit]


def read_series(path, spec, time_name, kind):
    with open_netcdf(path) as ds:
        variable = spec["variable"]
        if variable not in ds:
            raise KeyError(f"{path}: 缺少{variable}")
        value = ds[variable].squeeze()
        if value.ndim != 1 or value.dims != ds[time_name].dims:
            raise ValueError("只支持同一时间维度上的逐小时序列")
        t = pd.DatetimeIndex(ds[time_name].values)
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        if t.has_duplicates or not t.is_monotonic_increasing:
            raise ValueError("重复或逆序时间必须先审计，禁止静默整理")
        raw_units = str(value.attrs.get("units", ""))
        if raw_units and raw_units != spec["unit"] and not spec.get("metadata_override_evidence"):
            raise ValueError(f"映射与文件单位不一致: {raw_units} / {spec['unit']}")
        return pd.Series(convert(value.values, spec["unit"], kind), index=t)


def date_mask(times, start, end):
    return (times >= pd.Timestamp(start, tz="UTC")) & (times < pd.Timestamp(end, tz="UTC"))


def prepare(manifest_path, mapping_path, output, features=None):
    mapping = read_json(mapping_path)
    if not mapping.get("verified") or not mapping.get("evidence"):
        raise ValueError("字段映射必须具有真实核验依据")
    if mapping.get("timezone") != "UTC" or mapping.get("time_support") not in {
        "step_start",
        "step_end",
        "instantaneous",
    }:
        raise ValueError("时间支持和UTC尚未核验")
    features = features or ["area_km2", "latitude", "longitude"]
    validate_features(features)
    df = pd.read_csv(manifest_path, dtype={"station_id": str, "terminal_system_id": str})
    if df.groupby("terminal_system_id").split.nunique().max() > 1:
        raise ValueError("同终端水系跨集合")
    active = df[df.split.isin(["train", "validation", "development_test"])].copy()
    if active.empty or not (active.split == "train").any():
        raise ValueError("没有许可来源站")
    scaler = FeatureScaler().fit(active.loc[active.split == "train", features].to_numpy(float))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, failures = [], []
    for _, row in active.iterrows():
        sid = row.station_id
        try:
            if not row.time_support_verified or not row.datum_verified:
                raise ValueError("时间支持或基准未核验")
            times = pd.date_range("2011-10-01", "2021-10-01", freq="h", inclusive="left", tz="UTC")
            h = (
                read_series(row.qh_path, mapping["H"], mapping["time_variable"], "H")
                .reindex(times)
                .to_numpy()
            )
            weather = np.column_stack(
                [
                    read_series(
                        row.weather_path,
                        mapping[k],
                        mapping.get("weather_time_variable", mapping["time_variable"]),
                        k,
                    )
                    .reindex(times)
                    .to_numpy()
                    for k in ["P", "T", "PET"]
                ]
            )
            if not np.isfinite(weather).all():
                raise ValueError("预热或任务气象缺测；不得静默填零")
            if np.any(weather[:, (0, 2)] < 0):
                raise ValueError("降水或PET负值，需核验符号与质量")
            support = date_mask(times, "2019-10-01", "2020-10-01")
            future = date_mask(times, "2020-10-01", "2021-10-01")
            if np.isfinite(h[support]).mean() < 0.8:
                raise ValueError("支持H覆盖不足80%")
            geo = scaler.transform(row[features].to_numpy(float)[None])[0]
            payload = {
                "station_id": np.array(sid),
                "system": np.array(row.terminal_system_id),
                "time_ns": times.asi8,
                "weather": weather.astype("float32"),
                "geo": geo,
                "area": np.float32(row.area_km2),
                "support_start": np.int64(np.flatnonzero(support)[0]),
                "support_end": np.int64(np.flatnonzero(future)[0]),
            }
            if row.split == "train":
                allowed = date_mask(times, "2013-10-01", "2019-10-01")
                source_h = np.where(allowed, h, np.nan)
                q = (
                    read_series(row.qh_path, mapping["Q"], mapping["time_variable"], "Q")
                    .reindex(times)
                    .to_numpy()
                )
                payload.update(h=source_h.astype("float32"), q=np.where(allowed, q, np.nan).astype("float32"))
                location = output / "来源训练" / f"{sid}.npz"
            else:
                # 只存过去支持H，数组长度与支持窗口相同，未来标签根本不在预测文件里。
                payload["support_h"] = h[support].astype("float32")
                location = output / "目标上下文" / f"{sid}.npz"
                labels = output / "评分标签" / f"{sid}.parquet"
                labels.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"time_ns": times.asi8[future], "H": h[future]}).to_parquet(labels, index=False)
            location.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(location, **payload)
            index.append(
                {
                    "station_id": sid,
                    "terminal_system_id": row.terminal_system_id,
                    "split": row.split,
                    "crossfit_fold": int(row.crossfit_fold),
                    "path": str(location.resolve()),
                    "sha256": digest(location),
                }
            )
        except Exception as exc:
            failures.append({"station_id": sid, "error": f"{type(exc).__name__}: {exc}"})
    write_json(
        output / "index.json",
        {
            "scope": "real_development",
            "stations": index,
            "failures": failures,
            "features": features,
            "manifest_sha256": digest(manifest_path),
            "mapping_sha256": digest(mapping_path),
            "geo_center": scaler.center.tolist(),
            "geo_scale": scaler.scale.tolist(),
        },
    )
    return {"prepared": len(index), "failures": failures}
