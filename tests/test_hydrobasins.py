import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from echo.geo.hydrobasins import map_terminals


def test_spatial_terminal_mapping_traces_graph_and_keeps_unknowns(tmp_path):
    shape = tmp_path / "basins.gpkg"
    basins = gpd.GeoDataFrame(
        {
            "HYBAS_ID": [1, 2, 3],
            "NEXT_DOWN": [2, 0, 0],
            "MAIN_BAS": [2, 2, 3],
            "ENDO": [0, 0, 0],
            "COAST": [0, 1, 1],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1)],
        crs="EPSG:4326",
    )
    basins.to_file(shape, driver="GPKG")
    stations = tmp_path / "stations.csv"
    pd.DataFrame(
        {
            "station_id": ["0001", "0002", "0003", "0004"],
            "longitude": [0.5, 1.5, 3.5, 10.0],
            "latitude": [0.5] * 4,
        }
    ).to_csv(stations, index=False)
    output = tmp_path / "topology.csv"
    result = map_terminals(shape, stations, output)
    assert result["matched_stations"] == 3
    assert result["terminal_systems"] == 2
    assert len(result["failures"]) == 1
    actual = pd.read_csv(output, dtype=str).set_index("station_id")
    assert actual.loc["0001", "terminal_system_id"] == actual.loc["0002", "terminal_system_id"]
    assert actual.loc["0003", "terminal_system_id"] != actual.loc["0001", "terminal_system_id"]
