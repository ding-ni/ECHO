import json

import pytest
import xarray as xr

from echo.data.nldas_access import verify_access


class Response:
    def __init__(self, content=b"", status=200, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.content = content

    def iter_content(self, size):
        yield self.content

    def close(self):
        pass


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


@pytest.mark.parametrize("engine", ["scipy", "h5netcdf"])
def test_complete_netcdf_inspected_without_secret(tmp_path, engine):
    payload = xr.Dataset({"Rainf": ("time", [1.0], {"units": "kg m-2"})}).to_netcdf(engine=engine)
    session = Session(Response(payload))
    receipt = verify_access("local-test-secret", tmp_path, session=session)
    assert receipt["status"] == "downloaded_netcdf_verified"
    assert receipt["variables"]["Rainf"]["units"] == "kg m-2"
    assert not receipt["training_ready"]
    assert not receipt["schema_mapping_verified"]
    assert "local-test-secret" not in json.dumps(receipt)
    assert session.calls[0][1]["headers"] == {"Authorization": "Bearer local-test-secret"}
    assert not session.calls[0][1]["allow_redirects"]


@pytest.mark.parametrize("url", [
    "https://evil.example/collect", "http://urs.earthdata.nasa.gov/",
    "https://urs.earthdata.nasa.gov/?token=do-not-log",
    "https://urs.earthdata.nasa.gov:444/",
])
def test_redirect_never_forwards_credentials_to_unapproved_url(tmp_path, url):
    session = Session(Response(status=302, headers={"Location": url}))
    with pytest.raises(RuntimeError):
        verify_access("local-test-secret", tmp_path, session=session)
    assert len(session.calls) == 1
    receipt = (tmp_path / "NLDAS_认证验证.json").read_text(encoding="utf-8")
    assert "do-not-log" not in receipt and "local-test-secret" not in receipt


@pytest.mark.parametrize("response", [
    Response(b"<html>local-test-secret</html>"),
    Response(b"local-test-secret", status=401),
    Response(b"CDF\x01truncated"),
    Response(b"bad", headers={"Content-Length": "100"}),
])
def test_bad_data_or_auth_cannot_be_success(tmp_path, response):
    with pytest.raises(RuntimeError) as error:
        verify_access("local-test-secret", tmp_path, session=Session(response))
    assert "local-test-secret" not in str(error.value)
    receipt = json.loads((tmp_path / "NLDAS_认证验证.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "local-test-secret" not in json.dumps(receipt)
    assert not list(tmp_path.glob("*.nc"))
    for part in tmp_path.glob("*.part"):
        assert b"local-test-secret" not in part.read_bytes()
