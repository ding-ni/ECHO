"""按站标准化，来源/目标输入/评分标签物理分开。"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.datum import validate_registry_station
from echo.data.netcdf import open_netcdf
from echo.data.usgs_evidence import validate_support_evidence
from echo.data.usgs_q_evidence import validate_source_evidence
from echo.geo.features import FeatureScaler, validate_features
from echo.utils import digest, read_json, write_json


@lru_cache(maxsize=32)
def verified_extraction_root(directory):
    marker = Path(directory) / "extraction.json"
    record = read_json(marker)
    archive = Path(record["archive"])
    audit = read_json(archive.with_name(archive.name + ".audit.json"))
    if record.get("preview"):
        raise ValueError("提取目录与完整校验归档不一致")
    if audit.get("status") not in {"downloaded_verified", "verified_existing"}:
        raise ValueError("原始归档未完成校验")
    stat = archive.stat()
    # 提取完成之后未变化的归档复用当时的校验收据，避免每个准备进程重读21GB。
    # 大小/修改时间或收据变化时仍重算内容哈希。
    unchanged = (audit.get("sha256") == record["archive_sha256"]
                 and audit.get("bytes") == stat.st_size
                 and stat.st_mtime_ns <= marker.stat().st_mtime_ns)
    if not unchanged and digest(archive) != record["archive_sha256"]:
        raise ValueError("提取目录与完整校验归档不一致")
    return True


def require_verified_origin(path):
    path = Path(path).resolve()
    for parent in path.parents:
        if (parent / "extraction.json").is_file():
            return verified_extraction_root(str(parent))
    raise ValueError("只接受从完整校验归档提取的数据，局部模式探查不能用于训练")


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


def read_series(path, spec, time_name, kind, time_reference="step_start"):
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
        hour_ns = 3600 * 10**9
        if not t.equals(t.floor("h")) or np.any(np.diff(t.asi8) % hour_ns):
            raise ValueError("输入不是已核验的UTC整小时序列，不能通过取样冒充小时聚合")
        raw_units = str(value.attrs.get("units", ""))
        if raw_units and raw_units != spec["unit"] and not spec.get("metadata_override_evidence"):
            raise ValueError(f"映射与文件单位不一致: {raw_units} / {spec['unit']}")
        unit = spec["unit"]
        if kind in {"P", "PET"} and unit in {"kg/m2", "kg m-2"}:
            if spec.get("accumulation_hours") != 1 or not spec.get("unit_evidence"):
                raise ValueError("累计量必须有一小时累积时长及单位依据，不能当成每秒速率")
            unit = "mm/h"
        x = convert(value.values, unit, kind)
        policy = spec.get("negative_policy", "reject")
        if policy != "reject":
            if kind != "PET" or policy != "positive_part" or not spec.get("negative_policy_evidence"):
                raise ValueError("负值变换仅允许有依据的PET正部蒸发需求，不能用于降水或水位")
            x = np.maximum(x, 0)
        support = spec.get("time_support")
        if support is not None:
            if support not in {"step_start", "step_end", "instantaneous"} or not spec.get("time_evidence"):
                raise ValueError("变量时间支持未核验")
            if time_reference not in {"step_start", "step_end"}:
                raise ValueError("统一时间参考必须为步首或步末")
            if support == "step_end" and time_reference == "step_start":
                t = t - pd.Timedelta(hours=1)
            elif support == "step_start" and time_reference == "step_end":
                t = t + pd.Timedelta(hours=1)
        return pd.Series(x, index=t)


def date_mask(times, start, end):
    return (times >= pd.Timestamp(start, tz="UTC")) & (times < pd.Timestamp(end, tz="UTC"))


def prepare(manifest_path, mapping_path, output, features=None, datum_registry=None,
            system_roles=None, source_evidence=None, height_evidence=None, height_snapshots=None):
    mapping = read_json(mapping_path)
    if not mapping.get("verified") or not mapping.get("evidence"):
        raise ValueError("字段映射必须具有真实核验依据")
    if datum_registry is None:
        raise ValueError("真实数据准备必须指定历史水尺基准登记，不接受仅布尔放行")
    if system_roles is None or source_evidence is None:
        raise ValueError("真实数据准备必须指定冻结水系角色和来源Q/H核验记录")
    if height_evidence is None or height_snapshots is None:
        raise ValueError("真实数据准备必须指定支持期H核验记录与快照目录")
    if mapping.get("timezone") != "UTC" or mapping.get("time_support") not in {
        "step_start",
        "step_end",
        "instantaneous",
    }:
        raise ValueError("时间支持和UTC尚未核验")
    features = features or ["area_km2", "slope_degrees", "relief_m", "latitude", "longitude"]
    time_reference = mapping.get("time_reference", "step_start")
    validate_features(features)
    df = pd.read_csv(manifest_path, dtype={"station_id": str, "terminal_system_id": str})
    assigned = df[df.split != "excluded"]
    if assigned.groupby("terminal_system_id").split.nunique().max() > 1:
        raise ValueError("同终端水系跨集合")
    if "system_split" in df:
        if (
            df.groupby("terminal_system_id").system_split.nunique().max() > 1
            or not assigned.split.eq(assigned.system_split).all()
        ):
            raise ValueError("站点集合与完整水系归属不一致")
    active = df[df.split.isin(["train", "validation", "development_test"])].copy()
    if active.empty or not (active.split == "train").any():
        raise ValueError("没有许可来源站")
    roles = {r["terminal_system_id"]: r for r in read_json(system_roles)["systems"]}
    for _, row in active.iterrows():
        role = roles.get(row.terminal_system_id, {})
        if role.get("system_split") != row.split or role.get("crossfit_fold") != row.crossfit_fold:
            raise ValueError("活动清单与冻结水系角色或交叉拟合折号不一致")
    allowed_stations = mapping.get("allowed_station_ids")
    if allowed_stations is None or not set(active.station_id).issubset(set(allowed_stations)):
        raise ValueError("字段映射尚未核验这些站点")
    scaler = FeatureScaler().fit(active.loc[active.split == "train", features].to_numpy(float))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index, failures = [], []
    for _, row in active.iterrows():
        sid = row.station_id
        try:
            if row.time_support_verified not in (True,):
                raise ValueError("时间支持未核验")
            if "topology_verified" in row.index and row.topology_verified not in (True,):
                raise ValueError("终端水系边界复核未通过")
            if row.datum_verified not in (True,):
                raise ValueError("时间支持或基准未核验")
            if (
                str(row.get("datum_id", "unknown")) in {"unknown", "", "nan"}
                or not str(row.get("datum_evidence", "")).strip()
            ):
                raise ValueError("缺少基准标识及独立依据")
            datum_start, datum_end = (
                ("2013-10-01", "2019-10-01") if row.split == "train"
                else ("2019-10-01", "2021-10-01")
            )
            actual_datum = validate_registry_station(datum_registry, sid, datum_start, datum_end)
            if actual_datum != row.datum_id:
                raise ValueError("清单基准标识与实际任务时期不一致")
            if row.split == "train":
                if row.get("source_time_support_verified") not in (True,):
                    raise ValueError("来源训练期Q/H时间支持未逐站核验")
                validate_source_evidence(source_evidence, row, system_roles)
            else:
                validate_support_evidence(height_evidence, height_snapshots, row)
            require_verified_origin(row.qh_path)
            require_verified_origin(row.weather_path)
            times = pd.date_range("2011-10-01", "2021-10-01", freq="h", inclusive="left", tz="UTC")
            h = (
                read_series(row.qh_path, mapping["H"], mapping["time_variable"], "H", time_reference)
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
                        time_reference,
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
                    read_series(row.qh_path, mapping["Q"], mapping["time_variable"], "Q", time_reference)
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
            "forbidden_systems": sorted(s for s, r in roles.items() if r["system_split"] != "train"),
            "failures": failures,
            "features": features,
            "manifest_sha256": digest(manifest_path),
            "mapping_sha256": digest(mapping_path),
            "datum_registry_sha256": digest(datum_registry),
            "system_roles_sha256": digest(system_roles),
            "source_evidence_sha256": digest(source_evidence),
            "height_evidence_sha256": digest(height_evidence),
            "geo_center": scaler.center.tolist(),
            "geo_scale": scaler.scale.tolist(),
        },
    )
    return {"prepared": len(index), "failures": failures}
