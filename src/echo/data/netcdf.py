"""通过Python文件对象打开NetCDF，避免旧Windows原生库的中文路径问题。"""

from contextlib import contextmanager
from pathlib import Path

import xarray as xr


@contextmanager
def open_netcdf(path):
    with Path(path).open("rb") as stream:
        magic = stream.read(8)
        stream.seek(0)
        engine = "h5netcdf" if magic.startswith(b"\x89HDF") else "scipy"
        with xr.open_dataset(stream, engine=engine) as dataset:
            yield dataset
