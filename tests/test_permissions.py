import numpy as np
import pandas as pd
import pytest

from echo.data.permissions import ForecastInput, assert_source_permissions, support_scaling
from echo.data.prepare import convert
from echo.data.splits import build_splits
from echo.geo.features import validate_features
from echo.geo.topology import terminal_of


def test_no_future_labels_or_q_in_forecast_object():
    fields = ForecastInput.__dataclass_fields__
    assert not ({"q", "future_h", "query_h"} & set(fields))
    h = np.array([1.0, np.nan, 3.0])
    assert support_scaling(h, np.isfinite(h)) == (2.0, 1.0)
    with pytest.raises(ValueError):
        assert_source_permissions(["A", "B"], ["B"])


def test_only_approved_independent_features():
    validate_features(["area_km2", "latitude"])
    for name in ["baseflow_index", "observed_runoff_ratio", "Q_quantile", "station_id"]:
        with pytest.raises(ValueError):
            validate_features([name])


def test_units_and_leap_water_year():
    assert convert([1], "ft", "H")[0] == 0.3048
    assert convert([1], "cfs", "Q")[0] == 0.028316846592
    assert convert([273.15], "K", "T")[0] == 0
    assert convert([1], "mm/day", "P")[0] == 1 / 24
    assert len(pd.date_range("2019-10-01", "2020-10-01", freq="h", inclusive="left")) == 8784
    with pytest.raises(ValueError):
        convert([1], "unknown", "H")


def test_topology_requires_complete_downstream_evidence():
    assert terminal_of("a", {"a": "b", "b": "sea"}, {"sea"}) == "sea"
    with pytest.raises(ValueError):
        terminal_of("a", {}, {"sea"})
    with pytest.raises(ValueError):
        terminal_of("a", {"a": "b", "b": "a"}, {"sea"})


def test_group_splits_and_crossfit_are_disjoint():
    frame = pd.DataFrame(
        [
            {
                "station_id": f"{i:08}",
                "terminal_system_id": f"R{i // 2}",
                "topology_evidence": "synthetic_verified_graph",
                "area_km2": 200.0,
                "support_coverage": 0.95,
                "query_coverage": 0.99,
                "time_support_verified": True,
                "datum_verified": True,
            }
            for i in range(40)
        ]
    )
    split = build_splits(frame, max_sites=100)
    assert split.groupby("terminal_system_id").split.nunique().max() == 1
    train = split[split.split == "train"]
    assert train.groupby("terminal_system_id").crossfit_fold.nunique().max() == 1
    assert set(train.crossfit_fold) == {0, 1, 2}
    assert split[split.split == "blind_holdout"].target_Q_forbidden.all()
