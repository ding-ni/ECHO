"""同站配对效应与水系置信区间。"""

from pathlib import Path

import pandas as pd

from echo.evaluation.metrics import system_bootstrap
from echo.utils import read_json, write_json


def compare_runs(evaluations, output):
    rows = []
    for name, path in evaluations.items():
        summary = read_json(Path(path) / "summary.json")
        frame = pd.read_csv(Path(path) / "逐站指标.csv", dtype={"station_id": str, "terminal_system_id": str})
        if frame.empty:
            continue
        frame["run"] = name
        frame["budget_days"] = summary["budget_days"]
        frame["scope"] = summary["scope"]
        rows.append(frame)
    if not rows:
        raise ValueError("没有可比较结果")
    combined = pd.concat(rows, ignore_index=True)
    if combined.scope.nunique() != 1:
        raise ValueError("不得把合成和真实结果放入同一比较")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "模型与预算指标.csv", index=False, encoding="utf-8-sig")
    names, effects = list(evaluations), []
    for other in names[1:]:
        left = combined[combined.run == names[0]]
        right = combined[combined.run == other]
        pair = left.merge(right, on=["station_id", "terminal_system_id"], suffixes=("_base", "_other"))
        if len(pair) and not (pair.valid_hours_base == pair.valid_hours_other).all():
            raise ValueError("评分有效小时不同，不能直接配对比较")
        if len(pair):
            pair["event_RMSE_m"] = pair.event_RMSE_m_other - pair.event_RMSE_m_base
            effect = system_bootstrap(pair)
            effects.append(
                {
                    "reference": names[0],
                    "other": other,
                    "difference_other_minus_reference": effect,
                    "paired_sites": len(pair),
                    "reference_sites": len(left),
                    "other_sites": len(right),
                }
            )
    result = {"scope": combined.scope.iloc[0], "effects": effects}
    write_json(output / "配对效应.json", result)
    return result
