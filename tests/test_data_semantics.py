import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from echo.data.datum import assert_comparable_datum
from echo.data.meteorology import prepare_weather
from echo.data.prepare import read_series
from echo.data.splits import build_splits
from echo.data.usgs_evidence import hourly_height
from echo.geo.overlap import crosscheck_boundaries
from echo.utils import write_json


def test_hourly_accumulation_alignment_and_pet_policy(tmp_path):
    path = tmp_path / "forcing.nc"
    xr.Dataset(
        {"Rainf": ("time", [1.0, 5.0]), "PotEvap": ("time", [-0.1, 0.3])},
        coords={"time": pd.date_range("2020-01-01 01:00", periods=2, freq="h")},
    ).to_netcdf(path, engine="h5netcdf")
    spec = {
        "variable": "Rainf",
        "unit": "kg/m2",
        "accumulation_hours": 1,
        "unit_evidence": "hourly accumulation documentation",
        "time_support": "step_end",
        "time_evidence": "test fixture interval definition",
    }
    result = read_series(path, spec, "time", "P")
    assert result.index[0] == pd.Timestamp("2020-01-01", tz="UTC")
    np.testing.assert_allclose(result, [1.0, 5.0])  # 不能错误乘3600。
    end_result = read_series(path, spec, "time", "P", time_reference="step_end")
    assert end_result.index[0] == pd.Timestamp("2020-01-01 01:00", tz="UTC")
    with pytest.raises(ValueError, match="累积"):
        read_series(path, {**spec, "accumulation_hours": 3}, "time", "P")
    policy = {"negative_policy": "positive_part", "negative_policy_evidence": "HBV evaporative demand"}
    result = read_series(path, {**spec, **policy, "variable": "PotEvap"}, "time", "PET")
    np.testing.assert_allclose(result, [0.0, 0.3])
    with pytest.raises(ValueError, match="负值变换"):
        read_series(path, {**spec, **policy}, "time", "P")
    with pytest.raises(ValueError, match="时间支持"):
        read_series(path, {**spec, "time_support": "unknown"}, "time", "P")


def test_interior_datum_change_is_not_hidden_by_matching_endpoints():
    long = {"start": "2010-01-01", "end": "2030-01-01", "datum_id": "A", "evidence": "fixture"}
    nested = {"start": "2020-04-01", "end": "2020-05-01", "datum_id": "B", "evidence": "fixture"}
    with pytest.raises(ValueError, match="跨基准"):
        assert_comparable_datum("2019-10-01", "2021-10-01", [long, nested])


def test_usgs_hourly_height_uses_offset_and_excludes_right_endpoint():
    payload = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableCode": [{"value": "00065"}],
                        "unit": {"unitCode": "ft"},
                        "noDataValue": -999999.0,
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2019-10-01T00:00:00-04:00", "value": "1"},
                                {"dateTime": "2019-10-01T00:15:00-04:00", "value": "3"},
                                {"dateTime": "2019-10-01T01:00:00-04:00", "value": "100"},
                            ]
                        }
                    ],
                }
            ]
        }
    }
    values = hourly_height(payload, "2019-10-01 04:00", "2019-10-01 05:00")
    assert len(values) == 1
    assert values.index[0] == pd.Timestamp("2019-10-01 04:00", tz="UTC")
    assert values.iloc[0] == pytest.approx(1 * 0.3048)
    payload["value"]["timeSeries"][0]["variable"]["variableCode"][0]["value"] = "00060"
    with pytest.raises(ValueError, match="00065"):
        hourly_height(payload, "2019-10-01 04:00", "2019-10-01 05:00")


def test_cross_terminal_overlap_quarantines_whole_system():
    shapes = gpd.GeoDataFrame(
        {
            "station_id": ["01", "02", "03", "04"],
            "geometry": [
                box(-81, 39, -80, 40),
                box(-80.8, 39.2, -80.2, 39.8),
                box(-90, 40, -89.8, 40.2),
                box(-100, 40, -99.8, 40.2),
            ],
        },
        crs="EPSG:4326",
    )
    stations = pd.DataFrame(
        {
            "station_id": ["01", "02", "03", "04", "05"],
            "terminal_system_id": ["A", "B", "A", "C", "D"],
            "topology_evidence": ["fixture"] * 5,
            "longitude": [-80.5, -80.5, -89.9, -99.9, -110],
            "latitude": [39.5, 39.5, 40.1, 40.1, 40],
        }
    )
    out, report = crosscheck_boundaries(stations, shapes)
    out = out.set_index("station_id")
    assert not out.loc[["01", "02", "03", "05"], "topology_verified"].any()
    assert out.loc["04", "topology_verified"]
    assert report["quarantined_terminal_systems"] == ["A", "B"]
    assert len(report["cross_terminal_conflicts"]) == 1


def test_site_budget_preserves_unselected_connected_system_permissions():
    frame = pd.DataFrame(
        [
            {
                "station_id": f"{i:08}",
                "terminal_system_id": f"R{i // 3}",
                "topology_evidence": "fixture",
                "topology_verified": True,
                "area_km2": 200.0,
                "support_coverage": 1.0,
                "query_coverage": 1.0,
                "time_support_verified": True,
                "datum_verified": True,
            }
            for i in range(60)
        ]
    )
    split = build_splits(frame, max_sites=15)
    assert split.groupby("terminal_system_id").system_split.nunique().max() == 1
    active = split[split.split.isin(["train", "validation", "development_test"])]
    assert len(active) == 15
    assert active.split.eq(active.system_split).all()
    targets = set(active.loc[active.split != "train", "terminal_system_id"])
    assert split.loc[split.terminal_system_id.isin(targets), "target_Q_forbidden"].all()


def test_weather_only_view_preserves_signed_pet_and_aligns_pulse(tmp_path, monkeypatch):
    import echo.data.meteorology as module

    monkeypatch.setattr(module, "require_verified_origin", lambda path: True)
    times = pd.date_range("2011-10-01", "2021-10-01", freq="h")
    rain, pet = np.zeros(len(times)), np.full(len(times), 0.2)
    rain[1], pet[1] = 3.0, -0.1
    source = tmp_path / "forcing.nc"
    xr.Dataset(
        {
            "Rainf": ("time", rain),
            "Tair": ("time", np.full(len(times), 20.0)),
            "PotEvap": ("time", pet),
            "Streamflow": ("time", np.full(len(times), 123.0)),
        },
        coords={"time": times},
    ).to_netcdf(source, engine="h5netcdf")
    manifest = tmp_path / "stations.csv"
    pd.DataFrame([{"station_id": "00000100", "weather_path": str(source)}]).to_csv(manifest, index=False)
    base = {
        "units_verified": True,
        "time_support": "step_end",
        "time_evidence": "fixture",
        "unit": "kg/m2",
        "accumulation_hours": 1,
        "unit_evidence": "fixture",
    }
    mapping = tmp_path / "mapping.json"
    write_json(
        mapping,
        {
            "verified": False,
            "weather_semantics_verified": True,
            "time_reference": "step_end",
            "timezone": "UTC",
            "weather_time_variable": "time",
            "P": {**base, "variable": "Rainf"},
            "T": {**base, "variable": "Tair", "unit": "degC", "time_support": "instantaneous"},
            "PET": {
                **base,
                "variable": "PotEvap",
                "negative_policy": "positive_part",
                "negative_policy_evidence": "fixture demand definition",
            },
        },
    )
    result = prepare_weather(manifest, mapping, tmp_path / "out")
    assert result == {"prepared": 1, "failures": []}
    output = pd.read_parquet(tmp_path / "out/00000100.parquet")
    assert set(output.columns) == {"P", "T", "PET", "PET_signed_raw"}
    assert output.P.iloc[0] == 0.0
    assert output.P.iloc[1] == 3.0
    assert output.PET.iloc[1] == 0.0
    assert output.PET_signed_raw.iloc[1] == -0.1
    assert output["T"].iloc[0] == 20.0
