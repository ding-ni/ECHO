"""气象质量探查；只读取气象白名单，不读取文件内的 Streamflow。"""

import numpy as np
import pandas as pd

from echo.data.netcdf import open_netcdf
from echo.data.prepare import read_series, require_verified_origin
from echo.data.splits import group_key
from echo.utils import digest, utc_now, write_json


def audit_meteorology(manifest, output, limit=20, seed=20261019):
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    pool = frame[frame.weather_available.eq(True)].copy()
    pool["order"] = pool.station_id.map(lambda sid: group_key(sid, seed))
    # 跨终端水系轮转，避免仅按站号取到同一地区的前20站。
    groups = [g.sort_values("order") for _, g in pool.groupby("terminal_system_id", dropna=False)]
    groups.sort(key=lambda g: group_key(str(g.terminal_system_id.iloc[0]), seed))
    selected = []
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank < len(group) and len(selected) < limit:
                selected.append(group.iloc[rank])
        if len(selected) >= limit:
            break
    expected = pd.date_range("2011-10-01", "2021-10-01", inclusive="left", freq="h")
    records, failures = [], []
    for row in selected:
        try:
            with open_netcdf(row.weather_path) as ds:
                times = pd.DatetimeIndex(ds.DateTime.values)
                if times.has_duplicates or not times.is_monotonic_increasing:
                    raise ValueError("气象时间重复或逆序")
                if not expected.isin(times).all():
                    raise ValueError("预热和任务期缺少整小时时间坐标")
                # 通过变量白名单选择，气象归档中的观测Q不能进入审计统计或模型视图。
                values = ds[["Rainf", "Tair", "PotEvap"]].sel(DateTime=expected)
                record = {
                    "station_id": row.station_id,
                    "terminal_system_id": row.terminal_system_id,
                    "file_sha256": digest(row.weather_path),
                    "hours": len(expected),
                    "variables": {},
                }
                for name in ("Rainf", "Tair", "PotEvap"):
                    x = np.asarray(values[name].values, dtype=float)
                    finite = x[np.isfinite(x)]
                    record["variables"][name] = {
                        "file_unit": values[name].attrs.get("units", "missing"),
                        "missing": int((~np.isfinite(x)).sum()),
                        "negative_hours": int((x < 0).sum()),
                        "minimum": float(finite.min()) if len(finite) else None,
                        "maximum": float(finite.max()) if len(finite) else None,
                        "negative_magnitude_sum": float(-finite[finite < 0].sum()),
                    }
                records.append(record)
        except Exception as exc:
            failures.append({"station_id": row.station_id, "error": str(exc)})
    result = {
        "scope": "meteorological_quality_probe_not_training_approval",
        "created_utc": utc_now(),
        "manifest_sha256": digest(manifest),
        "seed": seed,
        "requested_files": limit,
        "files": records,
        "failures": failures,
        "period": [str(expected[0]), str(expected[-1])],
        "observed_Q_read": False,
        "units_inferred_from_values": False,
    }
    write_json(output, result)
    return result


def prepare_weather(manifest, mapping, output):
    """只生成已核验气象视图，水位/基准未知时仍可独立完成；不能作为训练索引。"""
    from pathlib import Path

    from echo.utils import read_json

    specs = read_json(mapping)
    if not specs.get("weather_semantics_verified") or specs.get("timezone") != "UTC":
        raise ValueError("气象单位、UTC和时间支持必须独立核验")
    time_reference = specs.get("time_reference", "step_start")
    frame = pd.read_csv(manifest, dtype={"station_id": str})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2011-10-01", "2021-10-01", freq="h", inclusive="left", tz="UTC")
    records, failures = [], []
    for _, row in frame.iterrows():
        sid = row.station_id
        try:
            require_verified_origin(row.weather_path)
            weather = pd.DataFrame(index=times)
            for kind in ["P", "T", "PET"]:
                spec = specs[kind]
                if not spec.get("units_verified"):
                    raise ValueError(f"{kind}单位未核验")
                weather[kind] = read_series(
                    row.weather_path, spec, specs["weather_time_variable"], kind, time_reference
                ).reindex(times)
            # 显式保留有符号原始潜在通量；PET列为所有模型共用的非负需求。
            raw_spec = {k: v for k, v in specs["PET"].items() if not k.startswith("negative_policy")}
            weather["PET_signed_raw"] = read_series(
                row.weather_path, raw_spec, specs["weather_time_variable"], "PET", time_reference
            ).reindex(times)
            if not np.isfinite(weather.to_numpy()).all() or (weather[["P", "PET"]] < 0).any().any():
                raise ValueError("标准化气象缺测或存在未处理的负降水/蒸发需求")
            target = output / f"{sid}.parquet"
            weather.index.name = "time_utc"
            weather.to_parquet(target)
            past = weather[weather.index < pd.Timestamp("2018-10-01", tz="UTC")]
            p_mean, pet_mean = float(past.P.mean() * 24), float(past.PET.mean() * 24)
            records.append(
                {
                    "station_id": sid,
                    "path": str(target.resolve()),
                    "sha256": digest(target),
                    "source_sha256": digest(row.weather_path),
                    "hours": len(weather),
                    "PET_negative_hours": int((weather.PET_signed_raw < 0).sum()),
                    "PET_negative_magnitude_mm": float(-weather.PET_signed_raw.clip(upper=0).sum()),
                    "historical_climate": {
                        "start": "2011-10-01",
                        "end_exclusive": "2018-10-01",
                        "P_mean_mm_day": p_mean,
                        "PET_mean_mm_day": pet_mean,
                        "T_mean_degC": float(past["T"].mean()),
                        "aridity": pet_mean / p_mean if p_mean > 0 else None,
                    },
                }
            )
        except Exception as exc:
            failures.append({"station_id": sid, "error": str(exc)})
    result = {
        "scope": "verified_weather_only_not_model_training_index",
        "created_utc": utc_now(),
        "manifest_sha256": digest(manifest),
        "mapping_sha256": digest(mapping),
        "stations": records,
        "failures": failures,
        "units": {"P": "mm/h", "T": "degC", "PET": "mm/h", "PET_signed_raw": "mm/h"},
        "time_reference": time_reference,
        "time_alignment": "按已核验time_support转换到time_reference；瞬时变量保留整点，不以整点值冒充小时均值",
        "observed_Q_read": False,
        "H_read": False,
        "real_training_complete": False,
    }
    write_json(output / "weather_index.json", result)
    return {"prepared": len(records), "failures": failures}
