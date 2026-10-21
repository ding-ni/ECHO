"""真实NetCDF模式探查，不根据变量值猜单位。"""

from pathlib import Path

import numpy as np
import pandas as pd

from echo.data.netcdf import open_netcdf


def inspect_file(path):
    row = {"file": str(path), "station_id": Path(path).stem.removesuffix("_hourly"), "status": "ok"}
    try:
        with open_netcdf(path) as ds:
            row["sizes"] = dict(ds.sizes)
            row["global_attributes"] = dict(ds.attrs)
            row["variables"] = {
                k: {"dims": list(v.dims), "dtype": str(v.dtype), "attrs": dict(v.attrs)}
                for k, v in ds.variables.items()
            }
            times = [k for k, v in ds.variables.items() if np.issubdtype(v.dtype, np.datetime64)]
            if times:
                t = pd.DatetimeIndex(ds[times[0]].values)
                row.update(
                    start=str(t.min()),
                    end=str(t.max()),
                    duplicates=int(t.duplicated().sum()),
                    hourly_regular=bool((np.diff(t.asi8) == 3600 * 10**9).all()),
                )
    except Exception as exc:
        row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return row


def audit_tree(root, limit=20):
    paths = sorted(Path(root).rglob("*.nc"))
    if not paths:
        raise ValueError("没有真实NetCDF文件")
    return {
        "scope": "schema_probe_not_full_quality_audit",
        "files_found": len(paths),
        "files": [inspect_file(p) for p in (paths[:limit] if limit else paths)],
    }
