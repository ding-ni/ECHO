import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from echo.geo.climate import historical_climate
from echo.geo.grid_weather import aggregate_weather, intersect_grid
from echo.geo.valley_terrain import patch_statistics


def test_area_weighted_original_weather_keeps_rainfall_location():
    cells = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=6933)
    nodes = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 100), box(50, 0, 200, 100)], crs=6933)
    weights, coverage = intersect_grid(nodes, cells)
    weather = np.zeros((2, 2, 3))
    weather[:, 0, 0], weather[:, 1, 0] = 3, 9
    result = aggregate_weather(weights, weather)
    np.testing.assert_allclose(coverage, 1)
    np.testing.assert_allclose(result[:, 0, 0], [3, 7])
    assert result.shape == (2, 2, 3)
    whole_mean = np.average(result[:, 0, 0], weights=[5000, 15000])
    assert whole_mean == 6
    with pytest.raises(ValueError, match="覆盖不足"):
        intersect_grid(nodes, cells.iloc[:1])


def test_real_dem_patch_uses_pixel_heights_and_coverage():
    yy, xx = np.indices((51, 51))
    elevation = 100 + xx.astype(float) * 3
    circle = (xx - 25)**2 + (yy - 25)**2 <= 20**2
    token, mask = patch_statistics(elevation, circle, 30, 100, 0, 0, 0, 1)
    assert not mask
    assert token[0] < token[1] < token[2]
    assert abs(token[6] - np.degrees(np.arctan(.1))) < 1e-4
    elevation[:, :40] = np.nan
    _, mask = patch_statistics(elevation, circle, 30, 100, 0, 0, 0, 1)
    assert mask


def test_climate_is_64_real_summaries_and_ignores_query_weather():
    origin = np.datetime64("2020-01-01T00", "h")
    time = np.arange(np.datetime64("2018-01-01T00", "h"), np.datetime64("2020-01-03T00", "h"))
    values = np.random.default_rng(19).random((len(time), 3))
    reference = historical_climate(values, time.astype("datetime64[ns]").astype("int64"),
                                    origin.astype("datetime64[ns]").astype("int64"))
    values[time >= origin] = 1e5
    changed = historical_climate(values, time.astype("datetime64[ns]").astype("int64"),
                                  origin.astype("datetime64[ns]").astype("int64"))
    assert reference.shape == (64,)
    np.testing.assert_array_equal(reference, changed)
