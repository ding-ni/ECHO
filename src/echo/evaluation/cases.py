"""评分侧的固定洪水案例选择；选例函数不接收模型预测或误差。"""

import argparse
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

from echo.evaluation.metrics import event_windows
from echo.utils import digest, read_json, utc_now, write_json


def select_windows(observed, support_h, count=2):
    observed, support_h = np.asarray(observed), np.asarray(support_h)
    if count < 1 or not np.isfinite(support_h).any():
        raise ValueError("案例数量必须为正，支持期必须有有效H")
    valid = np.isfinite(observed)
    for start in range(0, len(valid), 168):
        if valid[start:start + 168].mean() < .9:
            valid[start:start + 168] = False
    threshold = float(np.quantile(support_h[np.isfinite(support_h)], .95))
    windows = [(a, b) for a, b in event_windows(observed, threshold) if valid[a:b].mean() >= .9]
    return threshold, windows[:count]


def freeze_cases(index_path, labels_root, output):
    output = Path(output)
    if output.exists():
        raise ValueError("案例清单已经存在，禁止看过误差后覆盖选例")
    index = read_json(index_path)
    cases, inputs, shortages = [], {str(index_path): digest(index_path)}, []
    for row in sorted(index["stations"], key=lambda x: x["station_id"]):
        if row["split"] not in {"validation", "development_test"}:
            continue
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("模型视图哈希变更")
        label_path = Path(labels_root) / (row["station_id"] + ".parquet")
        labels = pd.read_parquet(label_path)
        if labels.time_ns.duplicated().any():
            raise ValueError("评分标签重复时间")
        with np.load(row["path"], allow_pickle=False) as data:
            if {"q", "h", "future_h"} & set(data.files):
                raise ValueError("目标视图含禁止字段")
            times = data["time_ns"][int(data["support_end"]):]
            truth = labels.set_index("time_ns").reindex(times).H.to_numpy()
            threshold, windows = select_windows(truth, data["support_h"])
        inputs[str(label_path)] = digest(label_path)
        if len(windows) < 2:
            shortages.append({"station_id": row["station_id"], "available": len(windows)})
        for number, (start, end) in enumerate(windows, 1):
            cases.append({"station_id": row["station_id"], "split": row["split"], "number": number,
                          "start": start, "end": end, "start_ns": int(times[start]),
                          "end_ns_inclusive": int(times[end - 1]), "threshold_m": threshold})
    record = {"created_utc": utc_now(), "index_sha256": digest(index_path), "inputs": inputs,
              "rule": "station_id_order_first_two_chronological_eligible_events_per_target",
              "selection_uses_predictions": False, "threshold_support_quantile": .95,
              "merge_gap_hours": 24, "pre_hours": 24, "post_hours": 48,
              "coverage": "same_168h_90pct_scoring_mask_and_event_90pct",
              "scope": "eight_site_development_case_selection_not_prospective_preregistration",
              "shortages": shortages, "cases": cases}
    write_json(output, record)
    return record


def render_cases(manifest, runs, labels_root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    record = read_json(manifest)
    for path, expected in record["inputs"].items():
        if digest(path) != expected:
            raise ValueError("选例数据版本变更")
    indexes = {}
    for name, folder in runs.items():
        indexes[name] = {}
        for split in ["validation", "development_test"]:
            idx = read_json(Path(folder) / split / "predictions.json")
            if idx["failures"] or idx["budget_days"] != "complete_water_year":
                raise ValueError("案例对照含失败或不同预算")
            if record["index_sha256"] not in idx["metadata"]["inputs"].values():
                raise ValueError("预测与选例数据版本不同")
            for row in idx["stations"]:
                indexes[name][row["station_id"]] = row["path"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not record["cases"]:
        raise ValueError("没有合格事件")
    fig, axes = plt.subplots(len(record["cases"]), 1, figsize=(11, 2.7 * len(record["cases"])), squeeze=False)
    colors = ["#808080", "#4477AA", "#228833", "#AA3377", "#EE7733"]
    rows, input_hashes = [], {str(manifest): digest(manifest)}
    for ax, case in zip(axes[:, 0], record["cases"], strict=True):
        sid, start, end = case["station_id"], case["start"], case["end"]
        expected_times, expected_support = None, None
        label_path = Path(labels_root) / (sid + ".parquet")
        expected_label = [sha for path, sha in record["inputs"].items() if Path(path).name == sid + ".parquet"]
        if len(expected_label) != 1 or digest(label_path) != expected_label[0]:
            raise ValueError("绘图标签与选例版本不同")
        labels = pd.read_parquet(label_path).set_index("time_ns")
        if labels.index.duplicated().any():
            raise ValueError("评分标签重复时间")
        for position, (name, stations) in enumerate(indexes.items()):
            color = colors[position % len(colors)]
            path = stations[sid]
            with np.load(path, allow_pickle=False) as data:
                if expected_times is None:
                    expected_times, expected_support = data["time_ns"].copy(), data["support_h"].copy()
                    if int(expected_times[start]) != case["start_ns"] or int(expected_times[end - 1]) != case["end_ns_inclusive"]:
                        raise ValueError("预测时刻与固定案例不一致")
                    truth = labels.reindex(expected_times).H.to_numpy()[start:end]
                    time = pd.to_datetime(expected_times[start:end], utc=True)
                    ax.plot(time, truth, color="black", linewidth=1.3, label="Observed")
                else:
                    np.testing.assert_array_equal(data["time_ns"], expected_times)
                    np.testing.assert_array_equal(data["support_h"], expected_support)
                if not np.isfinite(data["members"]).all():
                    raise ValueError("预测含非有限值")
                values = np.median(data["members"][:, start:end], axis=0)
                ax.plot(time, values, color=color, linewidth=1.0,
                        linestyle="--" if position >= 4 else "-", label=name)
            input_hashes[str(path)] = digest(path)
        ax.set(title=f"{sid} | {case['split']} | event {case['number']} | {time[0].year}", ylabel="Stage (m)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M UTC", tz=timezone.utc))
        rows.append({**case, "start_utc": str(time[0]), "end_utc": str(time[-1])})
    axes[0, 0].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = output / "固定洪水案例对照.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    write_json(output / "案例绘图记录.json", {"created_utc": utc_now(), "cases": rows,
               "inputs": input_hashes, "models": list(runs), "shortages": record["shortages"]})
    return path


def main():
    parser = argparse.ArgumentParser(description="冻结固定规则案例，或绘制已有预测；仅评分侧读取未来H。")
    parser.add_argument("action", choices=["freeze", "render"])
    parser.add_argument("--index", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", nargs="+", help="图例名=包含validation/development_test的预测目录")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.index is None:
            parser.error("freeze需要--index")
        result = freeze_cases(args.index, args.labels, args.manifest)
        print({"cases": len(result["cases"]), "shortages": result["shortages"]})
    else:
        if not args.predictions or args.output is None:
            parser.error("render需要--predictions和--output")
        print(render_cases(args.manifest, dict(x.split("=", 1) for x in args.predictions), args.labels, args.output))


if __name__ == "__main__":
    main()
