"""评分标签只在此模块读取；开发证据与合成证据明确标识。"""

from pathlib import Path

import numpy as np
import pandas as pd

from echo.evaluation.metrics import score_site, system_bootstrap
from echo.utils import read_json, write_json


def evaluate(predictions, labels_root, output):
    pred_index = read_json(Path(predictions) / "predictions.json")
    scores, failures = [], list(pred_index.get("failures", []))
    for row in pred_index["stations"]:
        try:
            with np.load(row["path"], allow_pickle=False) as z:
                labels = pd.read_parquet(Path(labels_root) / (row["station_id"] + ".parquet"))
                if labels.time_ns.duplicated().any():
                    raise ValueError("评分标签重复时间")
                truth = labels.set_index("time_ns").reindex(z["time_ns"]).H.to_numpy()
                metrics = score_site(truth, z["members"], z["support_h"])
                scores.append({**row, **metrics})
        except Exception as exc:
            failures.append({"station_id": row["station_id"], "error": str(exc)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scores)
    frame.to_csv(output / "逐站指标.csv", index=False, encoding="utf-8-sig")
    summary = {
        "scope": pred_index["scope"],
        "mode": pred_index["mode"],
        "model": pred_index["model"],
        "budget_days": pred_index["budget_days"],
        "sites": len(frame),
        "failures": failures,
    }
    if len(frame):
        summary["primary"] = system_bootstrap(frame)
        summary["all_period"] = {k: float(frame[k].mean()) for k in ["MAE_m", "RMSE_m", "CRPS_m"]}
        summary["tolerance_pass_fraction"] = {
            str(v): float((frame.RMSE_m <= v).mean()) for v in [0.1, 0.2, 0.5]
        }
    write_json(output / "summary.json", summary)
    return summary


def render_report(evaluation_root, predictions=None, labels_root=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(evaluation_root)
    summary = read_json(root / "summary.json")
    label = (
        "合成流程检查，不是实测性能"
        if summary["scope"] == "synthetic_demo"
        else "真实数据开发评估，非正式盲测"
    )
    lines = [
        "# ECHO 第一阶段报告",
        "",
        f"证据范围：**{label}**",
        "",
        f"任务：{summary['mode']}；模型：{summary['model']}；支持预算：{summary['budget_days']}。",
        "",
        f"成功评分站点：{summary['sites']}；失败或缺测站点：{len(summary['failures'])}。",
        "",
    ]
    if summary["sites"]:
        frame = pd.read_csv(root / "逐站指标.csv", dtype={"station_id": str, "terminal_system_id": str})
        lines += ["```json", __import__("json").dumps(summary, ensure_ascii=False, indent=2), "```", ""]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(frame.RMSE_m, bins=min(15, max(3, len(frame))), color="#347F89")
        ax.set(xlabel="Stage RMSE (m)", ylabel="Sites", title=label if label.isascii() else summary["scope"])
        fig.tight_layout()
        fig.savefig(root / "error_distribution.png", dpi=160)
        plt.close(fig)
        lines += ["![站点误差分布](error_distribution.png)", ""]
        if predictions and labels_root:
            idx = read_json(Path(predictions) / "predictions.json")
            # 固定站点ID顺序，预先选例，不按预测表现选好案例。
            for row in sorted(idx["stations"], key=lambda x: x["station_id"])[:10]:
                with np.load(row["path"], allow_pickle=False) as z:
                    labels = pd.read_parquet(Path(labels_root) / (row["station_id"] + ".parquet"))
                    truth = labels.set_index("time_ns").reindex(z["time_ns"]).H.to_numpy()
                    end = min(720, len(truth))
                    fig, ax = plt.subplots(figsize=(9, 3))
                    ax.plot(truth[:end], color="black", lw=1.0, label="Observed")
                    ax.plot(np.median(z["members"][:, :end], axis=0), color="#347F89", label="Median")
                    if z["members"].shape[0] > 1:
                        lo, hi = np.quantile(z["members"][:, :end], [0.05, 0.95], axis=0)
                        ax.fill_between(np.arange(end), lo, hi, alpha=0.2)
                    ax.set(xlabel="Hours from origin", ylabel="Stage (m)", title=row["station_id"])
                    ax.legend()
                    fig.tight_layout()
                    image = "process_" + row["station_id"] + ".png"
                    fig.savefig(root / image, dpi=150)
                    plt.close(fig)
                    lines += [f"![站点{row['station_id']}]({image})", ""]
    lines += [
        "## 当前限制",
        "",
        "- 以上只支持标注的证据范围。",
        "- 参考Q与曲线是有效模型解释，不等于唯一真实率定关系。",
        "- 本阶段未运行正式Mamba-3、空间降雨、独立水动力机制和真实天气预报试验。",
        "",
    ]
    (root / "阶段报告.md").write_text("\n".join(lines), encoding="utf-8")
    return root / "阶段报告.md"
