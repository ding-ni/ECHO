import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from echo.geo.nldas_inputs import build_weights, grid_identity, read_hour


def test_original_grid_intersections_and_signed_pet_time_support(tmp_path):
    lat = np.array([40.0625, 40.1875])
    lon = np.array([-100.9375, -100.8125])
    nodes = gpd.GeoDataFrame(geometry=[box(-101, 40, -100.875, 40.25),
                                      box(-100.875, 40, -100.75, 40.25)], crs=4326)
    weights, cells, coverage = build_weights(nodes, lat, lon)
    np.testing.assert_allclose(weights.sum(1), 1)
    np.testing.assert_allclose(coverage, 1)
    stamp = np.array(["2011-10-01T00:00:00"], dtype="datetime64[ns]")
    shape = (1, 2, 2)
    ds = xr.Dataset({
        "Rainf": (("time", "lat", "lon"), np.full(shape, 3.),
                  {"units": "kg m-2", "cell_methods": "time: sum"}),
        "PotEvap": (("time", "lat", "lon"), np.full(shape, -.1),
                    {"units": "kg m-2", "cell_methods": "time: sum"}),
        "Tair": (("time", "lat", "lon"), np.full(shape, 273.15),
                 {"units": "K", "cell_methods": "time: point"}),
        "time_bnds": (("time", "bnds"), np.array([[stamp[0] - np.timedelta64(1, "h"), stamp[0]]])),
    }, coords={"lat": lat, "lon": lon, "time": stamp})
    ds.time.attrs["bounds"] = "time_bnds"
    for var in ["time", "time_bnds"]:
        ds[var].encoding["units"] = "hours since 1979-01-01"
    path = tmp_path / "hour.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    end, forcing, signed = read_hour(path, cells, grid_identity(lat, lon))
    assert end == stamp[0].astype("int64")
    np.testing.assert_allclose(forcing, np.tile([3, 0, 0], (4, 1)), atol=1e-8)
    np.testing.assert_allclose(signed, -.1)
    ds.Rainf.attrs["cell_methods"] = "time: mean"
    ds.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="单位或时间支持"):
        read_hour(path, cells, grid_identity(lat, lon))
