"""从现有评分生成公开聚合快照，校验比较具有相同站点、时间和支持预算。"""

import argparse
from pathlib import Path

import numpy as np

from echo.utils import ROOT, digest, read_json, utc_now, write_json


def snapshot(output):
    output = Path(output)
    if output.exists():
        raise ValueError("公开结果快照已存在，请使用新版本")
    run = ROOT / "本地运行/扩展水位训练_v2"
    names = {"persistence": "持续性", "direct_lstm": "直接水位LSTM",
             "ECHO_静态": "ECHO静态", "ECHO_动态": "ECHO动态"}
    records, identities, inputs = [], None, {}
    for name, label in names.items():
        folder = run / "预测" / name / "development_test"
        predicted = read_json(folder / "predictions.json")
        summary_path = run / "评估" / name / "development_test/summary.json"
        summary = read_json(summary_path)
        if predicted["failures"] or summary["failures"] or predicted["budget_days"] != "complete_water_year":
            raise ValueError("比较含失败或支持预算不一致")
        keys = {}
        for row in predicted["stations"]:
            with np.load(row["path"], allow_pickle=False) as data:
                keys[row["station_id"]] = (data["time_ns"].copy(), data["support_h"].copy())
        if identities is None:
            identities = keys
        elif set(keys) != set(identities) or any(
            not np.array_equal(keys[sid][0], identities[sid][0])
            or not np.array_equal(keys[sid][1], identities[sid][1], equal_nan=True) for sid in keys
        ):
            raise ValueError("模型比较的站点、查询时间或历史支持水位不相同")
        inputs[str(summary_path.relative_to(ROOT))] = digest(summary_path)
        inputs[str((folder / "predictions.json").relative_to(ROOT))] = digest(folder / "predictions.json")
        records.append({"model": label, "sites": summary["sites"],
                        **summary["all_period"], "event_RMSE_m": summary["primary"]["mean"]})
    result = {"created_utc": utc_now(), "scope": "three_development_systems_single_seed_deterministic",
              "same_station_support_and_future_time_verified": True,
              "station_ids": sorted(identities), "models": records, "inputs": inputs,
              "support_period_UTC": ["2019-10-01", "2020-10-01"],
              "query_period_UTC": ["2020-10-01", "2021-10-01"],
              "period_end_exclusive": True,
              "joint_posterior_or_formal_architecture_tested": False}
    write_json(output, result)
    return result


def joint_snapshot(output):
    output = Path(output)
    if output.exists():
        raise ValueError("联合逻辑结果快照已存在，请使用新版本")
    folder = ROOT / "本地运行/扩展水位训练_v2/ECHO联合推断_本地小样例_v1"
    result = read_json(folder / "联合逻辑验证结果.json")
    diagnostics = read_json(folder / "地方后验与HBV适配诊断.json")
    history = read_json(folder / "history.json")
    metadata = read_json(folder / "metadata.json")
    if not result.get("complete") or result["completed_target_sites"] != len(diagnostics["stations"]):
        raise ValueError("联合小样例尚未完整完成")
    scores = {}
    inputs = {}
    for split in ["validation", "development_test"]:
        import pandas as pd

        path = folder / "短查询评分" / split / "逐站指标.csv"
        frame = pd.read_csv(path, dtype={"station_id": str})
        summary_path = folder / "短查询评分" / split / "summary.json"
        summary = read_json(summary_path)
        if summary["failures"]:
            raise ValueError("联合短查询存在评分失败")
        scores[split] = {"sites": len(frame), "all_period": summary["all_period"],
                         "probability": {key: float(frame[key].mean()) for key in
                                         ["coverage_50", "coverage_90", "width_50_m", "width_90_m"]}}
        inputs[str(path.relative_to(ROOT))] = digest(path)
        inputs[str(summary_path.relative_to(ROOT))] = digest(summary_path)
    for name in ["model.pt", "metadata.json", "history.json", "地方后验与HBV适配诊断.json", "联合逻辑验证结果.json"]:
        path = folder / name
        inputs[str(path.relative_to(ROOT))] = digest(path)
    snapshot = {
        **result, "created_utc": utc_now(), "source_training_epochs": len(history),
        "source_training_episodes": sum(r["train_episodes"] for r in history),
        "source_validation_episodes_per_epoch": [r["validation_episodes"] for r in history],
        "source_validation_scores": [r["source_temporal_validation"] for r in history],
        "training_seconds": metadata["actual_seconds"],
        "training_base_commit": metadata["commit"], "training_working_tree_dirty": metadata["working_tree_dirty"],
        "posterior": "32_dimensions_diagonal_plus_rank4_gaussian",
        "minimum_covariance_eigenvalue": min(r["min_covariance_eigenvalue"] for r in diagnostics["stations"]),
        "hydro_curve_cross_covariance_norms": [r["hydro_curve_cross_covariance_norm"] for r in diagnostics["stations"]],
        "scores": scores, "inputs": inputs,
        "uncertainty_scope": "conditional_local_parameters_only_not_calibrated_full_predictive_uncertainty",
        "caution": "仅未来首24小时；不同于完整支持年/未来年的确定性比较，不能据此宣称年度或洪水性能提升",
    }
    write_json(output, snapshot)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    print(joint_snapshot(args.output) if args.joint else snapshot(args.output))
