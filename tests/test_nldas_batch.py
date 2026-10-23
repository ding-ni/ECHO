import numpy as np
import pytest

from echo.data import nldas_batch
from echo.utils import digest, write_json


def test_real_calendar_and_fixed_product_urls():
    assert len(nldas_batch.hour_index("2019-10-01", "2020-10-01")) == 8784
    assert nldas_batch.acquisition_plan("2011-10-01", "2021-10-01")["hours"] == 87672
    spec = nldas_batch.granule(nldas_batch.hour_index("2020-02-29T23:00Z", "2020-03-01")[0])
    assert "/2020/060/" in spec["url"] and "A20200229.2300.020.nc" in spec["url"]
    with pytest.raises(ValueError):
        nldas_batch.hour_index("2020-01-01T00:01Z", "2020-02-01")


def test_verified_hour_restart_and_wrong_digest_blocks_reuse(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(token, folder, *, granule_url, receipt_name):
        calls.append(granule_url)
        folder.mkdir(parents=True, exist_ok=True)
        filename = granule_url.rsplit("/", 1)[1]
        target = folder / filename
        target.write_bytes(b"synthetic-hourly-data")
        day, hour = filename.split(".A", 1)[1].split(".")[:2]
        stamp = np.datetime64(f"{day[:4]}-{day[4:6]}-{day[6:]}T{hour[:2]}:00:00", "ns")
        receipt = {"status": "downloaded_netcdf_verified", "url": granule_url,
                   "bytes": target.stat().st_size, "sha256": digest(target), "time_values": [str(stamp)]}
        write_json(folder / receipt_name, receipt)
        return receipt

    monkeypatch.setattr(nldas_batch, "verify_access", fake_fetch)
    result = nldas_batch.download_hours("2011-10-01", "2011-10-01T02:00Z", tmp_path, "fake", workers=1)
    assert result["completed"] == 2 and result["failed"] == 0
    nldas_batch.download_hours("2011-10-01", "2011-10-01T02:00Z", tmp_path, "fake", workers=1)
    assert len(calls) == 2
    file = next(tmp_path.rglob("*.nc"))
    file.write_bytes(b"changed")
    result = nldas_batch.download_hours("2011-10-01", "2011-10-01T02:00Z", tmp_path, "fake", workers=1)
    assert result["failed"] == 1 and len(calls) == 2
