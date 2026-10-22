"""固定训练水系的多流域Q预训练数据，不要求或读取地方H基准。"""

import argparse
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.download import request
from echo.data.prepare import date_mask, read_series, require_verified_origin
from echo.geo.features import FeatureScaler, validate_features
from echo.utils import digest, read_json, utc_now, write_json

FEATURES = ["area_km2", "slope_degrees", "relief_m", "latitude", "longitude"]


def source_pool(manifest, roles_path):
    frame = pd.read_csv(manifest, dtype={"station_id": str, "terminal_system_id": str})
    roles = read_json(roles_path)
    if digest(manifest) != roles["manifest_sha256"]:
        raise ValueError("使用原始冻结候选清单，不能在扩展时重排水系")
    train = {r["terminal_system_id"]: r for r in roles["systems"] if r["system_split"] == "train"}
    forbidden = {r["terminal_system_id"] for r in roles["systems"] if r["system_split"] != "train"}
    pool = frame[frame.terminal_system_id.isin(train) & frame.weather_available.eq(True)
                 & frame.topology_verified.eq(True) & frame.area_km2.between(20, 5000)].copy()
    pool["crossfit_fold"] = pool.terminal_system_id.map(lambda x: train[x]["crossfit_fold"])
    pool = pool.sort_values("station_id")
    if pool.station_id.duplicated().any() or set(pool.terminal_system_id) & forbidden:
        raise ValueError("来源站重复或包含非训练水系")
    return pool, forbidden


def prepare_sources(manifest, roles_path, mapping_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    if index_path.exists():
        previous = read_json(index_path)
        expected = {str(p): digest(p) for p in [manifest, roles_path, mapping_path]}
        if previous.get("inputs") != expected:
            raise ValueError("数据版本已经存在且输入变化，请另建版本")
        if all(Path(r["path"]).is_file() and digest(r["path"]) == r["sha256"]
               for r in previous["stations"]):
            return previous
        raise ValueError("既有数据视图不完整，请另建版本")
    mapping = read_json(mapping_path)
    if not mapping.get("verified") or not mapping.get("weather_semantics_verified"):
        raise ValueError("气象字段须已核验")
    pool, forbidden = source_pool(manifest, roles_path)
    validate_features(FEATURES)
    times = pd.date_range("2011-10-01", "2019-10-01", inclusive="left", freq="h", tz="UTC")
    training = date_mask(times, "2013-10-01", "2018-10-01")
    validation = date_mask(times, "2018-10-01", "2019-10-01")
    allowed = training | validation
    records, failures, pending = [], [], []
    for number, (_, row) in enumerate(pool.iterrows(), 1):
        sid = row.station_id
        try:
            require_verified_origin(row.qh_path)
            require_verified_origin(row.weather_path)
            # 明确只读来源Q和P/T/PET，完全不读取H或任何目标水系Q。
            q = read_series(row.qh_path, mapping["Q"], mapping["time_variable"], "Q",
                            mapping["time_reference"]).reindex(times).to_numpy()
            weather = np.column_stack([
                read_series(row.weather_path, mapping[k], mapping["weather_time_variable"], k,
                            mapping["time_reference"]).reindex(times).to_numpy()
                for k in ["P", "T", "PET"]])
            valid_q = np.isfinite(q) & (q >= 0)
            coverage = {"train": float(valid_q[training].mean()), "validation": float(valid_q[validation].mean())}
            # H支持期的80%门槛不套用于全年来源Q。缺测Q通过损失掩码处理，
            # 寒区冬季缺测站仍可贡献有效季节的监督；保留逐站覆盖记录。
            if coverage["train"] < .5 or coverage["validation"] < .25:
                raise ValueError(f"来源Q训练覆盖不足50%或验证不足25%: {coverage}")
            if not np.isfinite(weather).all() or (weather[:, [0, 2]] < 0).any():
                raise ValueError("气象缺测或P/PET需求有负值")
            raw_path = output / "来源训练" / (sid + ".npz")
            raw_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(raw_path, station_id=np.array(sid), system=np.array(row.terminal_system_id),
                                time_ns=times.asi8, weather=weather.astype("float32"),
                                q=np.where(allowed & valid_q, q, np.nan).astype("float32"),
                                geo_raw=row[FEATURES].to_numpy(float), area=np.float32(row.area_km2))
            records.append({"station_id": sid, "terminal_system_id": row.terminal_system_id,
                            "split": "train", "crossfit_fold": int(row.crossfit_fold),
                            "path": str(raw_path.resolve()), "Q_coverage": coverage,
                            "qh_sha256": digest(row.qh_path), "weather_sha256": digest(row.weather_path)})
            pending.append(row[FEATURES].to_numpy(float))
        except Exception as exc:
            failures.append({"station_id": sid, "error": f"{type(exc).__name__}: {exc}"})
        if number % 10 == 0 or number == len(pool):
            print(f"Q预训练准备 {number}/{len(pool)} 合格={len(records)} 排除={len(failures)}", flush=True)
        write_json(output / "progress.json", {"checked": number, "candidates": len(pool),
                   "qualified": len(records), "failures": failures, "updated_utc": utc_now()})
    if len(records) < 3:
        raise ValueError("有效来源站不足")
    scaler = FeatureScaler().fit(np.stack(pending))
    for row in records:
        path = Path(row["path"])
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        raw = data.pop("geo_raw")
        data["geo"] = scaler.transform(raw[None])[0]
        np.savez_compressed(path, **data)
        row["sha256"] = digest(path)
    result = {"scope": "real_source_Q_pretraining", "created_utc": utc_now(), "stations": records,
              "forbidden_systems": sorted(forbidden), "excluded_candidates": failures,
              "features": FEATURES, "geo_center": scaler.center.tolist(), "geo_scale": scaler.scale.tolist(),
              "inputs": {str(p): digest(p) for p in [manifest, roles_path, mapping_path]},
              "H_read": False, "target_Q_read": False,
              "time_semantics": "archive_instantaneous_Q_contract_with_existing_USGS_source_checks",
              "Q_coverage_rule": {"training_minimum": .5, "validation_minimum": .25,
                                  "training_block_minimum": .8, "missing_Q": "masked_not_imputed"},
              "purpose": "来源Q监督的区域水文预训练；不作为新增H基准放行或目标评估"}
    write_json(index_path, result)
    print(f"完成 {len(records)}站 / {len({r['terminal_system_id'] for r in records})}水系", flush=True)
    return result


def fetch_manuscripts(manifest, roles_path, output):
    pool, _ = source_pool(manifest, roles_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        sid = row.station_id
        path = output / (sid + ".xml")
        url = f"https://sims-manuscripts-bucket.s3.us-west-2.amazonaws.com/sims_USGS_{sid}.xml"
        try:
            if path.exists():
                receipt = read_json(path.with_suffix(".provenance.json"))
                if receipt["url"] != url or receipt["sha256"] != digest(path):
                    raise ValueError("快照收据不匹配")
            else:
                for attempt in range(3):
                    try:
                        with request(url) as response:
                            body = response.read()
                        tree = ET.fromstring(body)
                        if tree.findtext(".//site_no") != sid:
                            raise ValueError("站号不匹配")
                        path.write_bytes(body)
                        write_json(path.with_suffix(".provenance.json"),
                                   {"url": url, "retrieved_utc": utc_now(), "sha256": digest(path)})
                        break
                    except OSError:
                        if attempt == 2:
                            raise
                        time.sleep(attempt + 1)
            tree = ET.parse(path)
            return {"station_id": sid, "terminal_system_id": row.terminal_system_id,
                    "snapshot": str(path.resolve()), "snapshot_sha256": digest(path), "source_url": url,
                    "gage_quotes": [e.findtext("info") for e in tree.findall(".//Element")
                                    if e.findtext("name") == "GAGE"],
                    "review_status": "fetched_pending_epoch_review"}
        except Exception as exc:
            return {"station_id": sid, "error": f"{type(exc).__name__}: {exc}"}

    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        jobs = [executor.submit(fetch, row) for _, row in pool.iterrows()]
        for future in as_completed(jobs):
            records.append(future.result())
            if len(records) % 20 == 0:
                print(f"历史稿获取 {len(records)}/{len(pool)}", flush=True)
    write_json(output / "index.json", {"created_utc": utc_now(), "stations": sorted(records, key=lambda x:x["station_id"])})
    return {"fetched": sum("error" not in r for r in records), "total": len(records)}


def main():
    parser = argparse.ArgumentParser(description="扩展固定来源训练水系的数据与站点资料")
    parser.add_argument("action", choices=["prepare", "manuscripts"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.mapping is None:
            parser.error("prepare需要--mapping")
        result = prepare_sources(args.manifest, args.roles, args.mapping, args.output)
        print({"stations": len(result["stations"]), "excluded": len(result["excluded_candidates"])})
    else:
        print(fetch_manuscripts(args.manifest, args.roles, args.output))


if __name__ == "__main__":
    main()
