"""Pre-origin climate summaries; never use the query's weather in environment g."""

import numpy as np
import pandas as pd


def historical_climate(weather, time_ns, origin_ns):
    weather, time_ns = np.asarray(weather), np.asarray(time_ns)
    values = weather[time_ns < origin_ns].astype("float64")
    times = pd.to_datetime(time_ns[time_ns < origin_ns], utc=True)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 365 * 24 or not np.isfinite(values).all():
        raise ValueError("环境气候摘要需要至少一年起报前有效P/T/PET")
    monthly = []
    for month in range(1, 13):
        selected = values[times.month == month]
        if not len(selected):
            raise ValueError("起报前气候缺少月份，不以0伪造统计")
        monthly.append(selected.mean(0))
    p, t, pet = values.T
    initial45 = np.r_[np.asarray(monthly).ravel(), values.mean(0), values.std(0),
                       np.quantile(values, .9, axis=0)]
    quantiles12 = np.quantile(values, [.05, .25, .75, .95], axis=0).ravel()
    extra7 = np.asarray([(p > .1).mean(), (p == 0).mean(), (t < 0).mean(),
                         (t > 25).mean(), p.sum() / max(pet.sum(), 1e-6),
                         np.ptp(np.asarray(monthly)[:, 1]),
                         p[t < 0].sum() / max(p.sum(), 1e-6)])
    result = np.r_[initial45, quantiles12, extra7].astype("float32")
    if result.shape != (64,) or not np.isfinite(result).all():
        raise ValueError("气候摘要不满足64个有限统计量")
    return result
