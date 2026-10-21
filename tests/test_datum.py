import pytest

from echo.data.datum import assert_comparable_datum


def test_datum_changes_cannot_be_silently_joined():
    epochs = [
        {"start": "2010-01-01", "end": "2020-01-01", "datum_id": "A", "evidence": "synthetic"},
        {"start": "2020-01-01", "end": "2025-01-01", "datum_id": "B", "evidence": "synthetic"},
    ]
    assert assert_comparable_datum("2011-01-01", "2015-01-01", epochs) == "A"
    with pytest.raises(ValueError):
        assert_comparable_datum("2019-01-01", "2021-01-01", epochs)
