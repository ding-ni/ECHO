import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from echo.data import dem_assets


@pytest.mark.parametrize("wrong_range", [False, True])
def test_timeout_prefix_only_appended_when_range_is_verified(tmp_path, monkeypatch, wrong_range):
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=16, height=16, count=1, dtype="float32",
                         crs="EPSG:4326", transform=from_origin(-70, 45, .001, .001)) as ds:
            ds.write(np.arange(256, dtype="float32").reshape(16, 16), 1)
        payload = memory.read()
    etag = hashlib.md5(payload).hexdigest()

    class Session:
        def head(self, *args, **kwargs):
            return SimpleNamespace(headers={"Content-Length": str(len(payload)), "ETag": f'"{etag}"'},
                                   raise_for_status=lambda: None)

    monkeypatch.setattr(dem_assets.requests, "Session", Session)
    calls = []

    def curl(args, **kwargs):
        start, end = map(int, args[args.index("--range") + 1].split("-"))
        size = (end - start + 1) // 2 if not calls else end - start + 1
        calls.append(start)
        header = args[args.index("--dump-header") + 1]
        block = args[args.index("--output") + 1]
        from pathlib import Path
        Path(header).write_text(f"HTTP/1.1 206 Partial Content\nContent-Range: bytes "
                                f"{start + int(wrong_range)}-{end}/{len(payload)}\n", encoding="latin-1")
        Path(block).write_bytes(payload[start:start + size])
        return SimpleNamespace(returncode=28 if len(calls) == 1 else 0, stderr="timeout")

    monkeypatch.setattr(dem_assets.subprocess, "run", curl)
    spec = {"url": "https://prd-tnm.s3.amazonaws.com/USGS_1_n45w070_20200101.tif",
            "catalogue_bytes": len(payload)}
    if wrong_range:
        with pytest.raises(OSError):
            dem_assets.download_asset(spec, tmp_path)
        assert not list(tmp_path.glob("*.receipt.json"))
        assert not list(tmp_path.glob("*.part"))
    else:
        receipt = dem_assets.download_asset(spec, tmp_path)
        assert len(calls) == 2 and calls[1] == len(payload) // 2
        assert receipt["bytes"] == len(payload)
        with rasterio.open(receipt["path"]) as ds:
            assert ds.read(1)[-1, -1] == 255
