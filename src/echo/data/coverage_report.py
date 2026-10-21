"""候选站点覆盖图表；仅描述已审计数据，不推断训练性能。"""

from pathlib import Path

import pandas as pd


def render_coverage(inventory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(inventory, dtype={"station_id": str})
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(frame.support_coverage, bins=10, color="#347F89", label="Support")
    axes[0].hist(frame.query_coverage, bins=10, color="#D69549", alpha=0.55, label="Future labels")
    axes[0].set(xlabel="Finite stage coverage", ylabel="Sites", xlim=(0, 1))
    axes[0].legend()
    points = axes[1].scatter(
        frame.longitude, frame.latitude, c=frame.support_coverage, cmap="viridis", vmin=0, vmax=1, s=20
    )
    axes[1].set(xlabel="Longitude", ylabel="Latitude", title="Audited candidate locations")
    fig.colorbar(points, ax=axes[1], label="Support coverage")
    fig.tight_layout()
    fig.savefig(root / "data_coverage.png", dpi=160)
    plt.close(fig)
    text = (
        "# 数据覆盖报告\n\n这是候选站数据审计，不是模型验证。\n\n"
        f"候选站：{len(frame)}；支持H覆盖达到80%：{int((frame.support_coverage >= 0.8).sum())}。\n\n"
        "![覆盖与位置](data_coverage.png)\n\n"
        "站点顺序来自当前审计输入，不能据此宣称已完成气候地貌分层抽样。"
        "时间支持、基准与终端水系仍需独立证据。\n"
    )
    (root / "数据覆盖报告.md").write_text(text, encoding="utf-8")
    return root / "数据覆盖报告.md"
