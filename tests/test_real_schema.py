import numpy as np
import pandas as pd
import pytest
import xarray as xr

from echo.data.audit import inspect_file
from echo.data.prepare import prepare, read_series, require_verified_origin
from echo.utils import write_json


def test_chinese_netcdf_path_and_explicit_units(tmp_path):
    path = tmp_path / "中文站点_01000100.nc"
    ds = xr.Dataset(
        {"water_level": ("time", np.array([1.0, 2.0, 3.0]))},
        coords={"time": pd.date_range("2020-01-01", periods=3, freq="h")},
    )
    ds.water_level.attrs["units"] = "m"
    ds.to_netcdf(path, engine="h5netcdf")
    report = inspect_file(path)
    assert report["status"] == "ok"
    assert report["hourly_regular"]
    series = read_series(path, {"variable": "water_level", "unit": "m"}, "time", "H")
    assert str(series.index.tz) == "UTC"
    with pytest.raises(ValueError):
        read_series(path, {"variable": "water_level", "unit": "ft"}, "time", "H")


def test_unverified_mapping_cannot_prepare_training_data(tmp_path):
    path = tmp_path / "mapping.json"
    write_json(path, {"verified": False})
    with pytest.raises(ValueError, match="核验"):
        prepare(tmp_path / "nonexistent.csv", path, tmp_path / "out")


def test_partial_schema_probe_cannot_become_training_input(tmp_path):
    path = tmp_path / "probe.nc"
    path.write_bytes(b"schema probe")
    with pytest.raises(ValueError, match="完整校验"):
        require_verified_origin(path)
