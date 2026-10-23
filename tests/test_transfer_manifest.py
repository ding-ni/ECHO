import pytest

from echo.data.readiness import verify_transfer
from echo.utils import digest, write_json


def test_transfer_hash_and_path_boundaries(tmp_path):
    payload = tmp_path / "source.npz"
    payload.write_bytes(b"example data")
    manifest = tmp_path / "transfer.json"
    entry = {"path": "source.npz", "sha256": digest(payload), "bytes": payload.stat().st_size}
    write_json(manifest, {"files": [entry]})
    assert verify_transfer(manifest, tmp_path)["verified_files"] == 1
    payload.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="哈希"):
        verify_transfer(manifest, tmp_path)
    for name in ["../source.npz", "F:/source.npz", "/source.npz", "..\\source.npz"]:
        write_json(manifest, {"files": [{**entry, "path": name}]})
        with pytest.raises(ValueError, match="路径"):
            verify_transfer(manifest, tmp_path)
