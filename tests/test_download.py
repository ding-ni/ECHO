import hashlib
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from echo.data.archive import extract_verified, safe_member
from echo.data.download import fetch, fetch_parallel
from echo.utils import write_json


@pytest.fixture
def server():
    content = b"echo-test-data" * 10000

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            begin, stop = (requested or "bytes=0-").split("=")[1].split("-")
            offset = int(begin)
            end = int(stop) + 1 if stop else len(content)
            ranged = self.path != "/ignore" and requested is not None
            self.send_response(206 if ranged else 200)
            if ranged:
                self.send_header("Content-Range", f"bytes {offset}-{end - 1}/{len(content)}")
            else:
                offset = 0
                end = len(content)
            self.send_header("Content-Length", str(end - offset))
            self.end_headers()
            self.wfile.write(content[offset:end])

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{http.server_port}", content
    http.shutdown()
    thread.join()


@pytest.mark.parametrize("endpoint", ["/range", "/ignore"])
def test_resume_and_server_ignores_range(server, tmp_path, endpoint):
    url, content = server
    (tmp_path / "a.bin.part").write_bytes(content[:1234])
    spec = {
        "filename": "a.bin",
        "bytes": len(content),
        "md5": hashlib.md5(content).hexdigest(),
        "url": url + endpoint,
        "record": "local-test",
    }
    result = fetch(spec, tmp_path, snapshot=False, progress=False)
    assert result["status"] == "downloaded_verified"
    assert (tmp_path / "a.bin").read_bytes() == content
    assert fetch(spec, tmp_path, snapshot=False)["status"] == "verified_existing"


def test_checksum_failure_retains_partial(server, tmp_path):
    url, content = server
    with pytest.raises(ValueError):
        fetch(
            {"filename": "a.bin", "bytes": len(content), "md5": "bad", "url": url},
            tmp_path,
            snapshot=False,
            progress=False,
        )
    assert not (tmp_path / "a.bin").exists()
    assert (tmp_path / "a.bin.part").is_file()


def test_parallel_range_assembly_retains_verified_prefix(server, tmp_path):
    url, content = server
    (tmp_path / "a.bin.part").write_bytes(content[:1234])
    spec = {
        "filename": "a.bin",
        "bytes": len(content),
        "md5": hashlib.md5(content).hexdigest(),
        "url": url + "/range",
        "record": "test",
    }
    result = fetch_parallel(spec, tmp_path, connections=3, chunk_bytes=8192, snapshot=False)
    assert result["status"] == "downloaded_verified"
    assert (tmp_path / "a.bin").read_bytes() == content


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "/absolute", "a/../../b", "a\\..\\..\\b"])
def test_archive_rejects_traversal(name):
    with pytest.raises(ValueError):
        safe_member(name)


@pytest.mark.parametrize("name", ["..", "../a", "C:relative", "a\\b"])
def test_download_asset_name_is_one_safe_file(tmp_path, name):
    with pytest.raises(ValueError):
        fetch({"filename": name, "md5": "unused", "bytes": 0}, tmp_path, snapshot=False)


def test_verified_extraction(tmp_path):
    zpath = tmp_path / "test.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("safe/file.txt", "hello")
    write_json(
        tmp_path / "test.zip.audit.json",
        {"bytes": zpath.stat().st_size, "md5": hashlib.md5(zpath.read_bytes()).hexdigest()},
    )
    assert extract_verified(zpath, tmp_path / "out", preview=True)["members"] == 1
    assert not (tmp_path / "out").exists()
    extract_verified(zpath, tmp_path / "out")
    assert (tmp_path / "out/safe/file.txt").read_text() == "hello"
