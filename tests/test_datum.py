import json

import pytest

from echo.data.datum import assert_comparable_datum, validate_registry_station
from echo.utils import digest


def test_datum_changes_cannot_be_silently_joined():
    epochs = [
        {"start": "2010-01-01", "end": "2020-01-01", "datum_id": "A", "evidence": "synthetic"},
        {"start": "2020-01-01", "end": "2025-01-01", "datum_id": "B", "evidence": "synthetic"},
    ]
    assert assert_comparable_datum("2011-01-01", "2015-01-01", epochs) == "A"
    with pytest.raises(ValueError):
        assert_comparable_datum("2019-01-01", "2021-01-01", epochs)


def test_registry_validates_snapshot_and_epoch(tmp_path):
    snapshot = tmp_path / "station.xml"
    snapshot.write_text(
        "<USGS><site_no>00000001</site_no><Element><name>GAGE</name>"
        "<info>Datum is stable.</info></Element></USGS>",
        encoding="utf-8",
    )
    (tmp_path / "station.provenance.json").write_text(
        json.dumps(
            {
                "url": "https://sims-manuscripts-bucket.s3.us-west-2.amazonaws.com/"
                "sims_USGS_00000001.xml",
                "sha256": digest(snapshot),
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "station_id": "00000001",
                        "review_status": "historical_epoch_reviewed",
                        "snapshot": "station.xml",
                        "snapshot_sha256": digest(snapshot),
                        "gage_quote": "Datum is stable.",
                        "epochs": [
                            {
                                "start": "2013-10-01",
                                "end": "2021-10-01",
                                "datum_id": "A",
                                "evidence": "official",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert validate_registry_station(registry, "00000001", "2019-10-01", "2020-10-01") == "A"
    with pytest.raises(ValueError, match="跨基准|缺失"):
        validate_registry_station(registry, "00000001", "2021-01-01", "2022-01-01")
