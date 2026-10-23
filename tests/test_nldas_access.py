import json

import pytest
import xarray as xr

from echo.data.nldas_access import SAMPLE_URL, SIGNED_DATA_HOST, verify_access


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


def test_official_signed_cdn_receives_no_bearer_and_signature_is_not_logged(tmp_path):
    data_path = SAMPLE_URL.split("/data/", 1)[1]
    redirect = (
        f"https://{SIGNED_DATA_HOST}/storage/gesdisc-cumulus-prod-protected.s3.us-west-2.amazonaws.com/"
        f"{data_path}?Expires=123&Signature=private-signature&Key-Pair-Id=key"
    )
    payload = xr.Dataset({"Rainf": ("time", [1.0])}).to_netcdf(engine="scipy")
    session = Session(Response(status=303, headers={"Location": redirect}), Response(payload))
    receipt = verify_access("local-test-secret", tmp_path, session=session)
    assert receipt["status"] == "downloaded_netcdf_verified"
    assert receipt["signed_data_redirect_used"]
    assert session.calls[1][1]["headers"] == {}
    assert "private-signature" not in json.dumps(receipt)


@pytest.mark.parametrize("suffix", ["other-file.nc?Signature=x&Expires=1&Key-Pair-Id=k", "file.nc"])
def test_cdn_redirect_requires_exact_sample_path_and_signature(tmp_path, suffix):
    session = Session(Response(status=303, headers={"Location": f"https://{SIGNED_DATA_HOST}/{suffix}"}))
    with pytest.raises(RuntimeError):
        verify_access("local-test-secret", tmp_path, session=session)
    assert len(session.calls) == 1


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
