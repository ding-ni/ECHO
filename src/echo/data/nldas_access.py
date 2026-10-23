"""Validate one fixed NASA NLDAS granule without logging credentials."""

import argparse
import os
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
import xarray as xr

from echo.utils import digest, utc_now, write_json

SAMPLE_URL = (
    "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/"
    "2011/274/NLDAS_FORA0125_H.A20111001.0000.020.nc"
)
ALLOWED_HOSTS = {"data.gesdisc.earthdata.nasa.gov", "urs.earthdata.nasa.gov"}
# Observed in an authenticated redirect from the official GES DISC endpoint.
# Never send the Earthdata bearer to this CDN or persist its signed URL.
SIGNED_DATA_HOST = "d2b3c3wh8s6en5.cloudfront.net"
MAX_BYTES = 64 * 1024 * 1024


def _allowed(url, *, signed_data=False):
    parts = urlsplit(url)
    hosts = {SIGNED_DATA_HOST} if signed_data else ALLOWED_HOSTS
    if (
        parts.scheme != "https" or parts.hostname not in hosts
        or parts.port not in {None, 443} or parts.username or parts.password
        or (parts.query and not signed_data) or parts.fragment
    ):
        raise ValueError("拒绝将认证发送到非许可地址或带参数的重定向；需另行核验认证流程")
    if signed_data:
        expected_suffix = (
            "/gesdisc-cumulus-prod-protected.s3.us-west-2.amazonaws.com"
            + urlsplit(SAMPLE_URL).path.removeprefix("/data")
        )
        if not parts.path.endswith(expected_suffix) or not {
            "Expires", "Signature", "Key-Pair-Id"
        }.issubset(parse_qs(parts.query)):
            raise ValueError("下载签名或固定样例路径不匹配")


def verify_access(token, output, *, session=None):
    """Use a caller-supplied token in headers only; return a public-safe receipt.

    Success checks the complete NetCDF and inventories its schema. It does not
    certify variable mapping, time support, or training-data readiness.
    """
    token = token.strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("未提供有效的本机令牌；请通过本机隐藏输入录入")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "NLDAS_认证验证.json"
    target = output / Path(urlsplit(SAMPLE_URL).path).name
    part = target.with_suffix(target.suffix + ".part")
    receipt = {
        "schema_version": 1, "checked_utc": utc_now(), "url": SAMPLE_URL,
        "status": "not_verified", "schema_mapping_verified": False,
        "training_ready": False, "credential_saved_in_receipt": False,
    }
    owned = session is None
    session = requests.Session() if owned else session
    response = None
    try:
        url = SAMPLE_URL
        signed_data = False
        for _ in range(6):
            _allowed(url, signed_data=signed_data)
            response = session.get(
                url, headers={} if signed_data else {"Authorization": f"Bearer {token}"},
                allow_redirects=False, stream=True, timeout=(20, 60),
            )
            receipt["http_status"] = response.status_code
            if response.status_code in {301, 302, 303, 307, 308}:
                next_url = urljoin(url, response.headers.get("Location", ""))
                response.close()
                if signed_data:
                    raise ValueError("签名下载端再次重定向，需重新核验")
                if urlsplit(next_url).hostname == SIGNED_DATA_HOST:
                    if urlsplit(url).hostname != "data.gesdisc.earthdata.nasa.gov":
                        raise ValueError("签名地址并非由GES DISC数据端点返回")
                    signed_data = True
                _allowed(next_url, signed_data=signed_data)
                receipt["signed_data_redirect_used"] = signed_data
                url = next_url
                continue
            break
        else:
            raise ValueError("认证重定向过多，未取得数据")
        if response.status_code != 200:
            raise ValueError(f"NASA返回HTTP {response.status_code}；需核对令牌、有效期及应用授权")
        if "html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("返回HTML页面，未取得NetCDF数据")
        size = 0
        prefix = b""
        identified = False
        with part.open("wb") as stream:
            for chunk in response.iter_content(256 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("样例超过64MiB限制，停止读取")
                if not identified:
                    prefix += chunk
                    if len(prefix) < 8:
                        continue
                    if not (
                        prefix[:4] in {b"CDF\x01", b"CDF\x02", b"CDF\x05"}
                        or prefix[:8] == b"\x89HDF\r\n\x1a\n"
                    ):
                        raise ValueError("响应不是NetCDF，未保存响应正文")
                    identified = True
                    chunk, prefix = prefix, b""
                stream.write(chunk)
        if not identified:
            raise ValueError("未取得完整NetCDF文件头")
        declared = response.headers.get("Content-Length")
        if declared and not response.headers.get("Content-Encoding") and size != int(declared):
            raise ValueError("下载长度与服务端声明不符")
        with part.open("rb") as stream:
            magic = stream.read(8)
        if not (magic[:4] in {b"CDF\x01", b"CDF\x02", b"CDF\x05"} or magic == b"\x89HDF\r\n\x1a\n"):
            raise ValueError("文件不是NetCDF，不能标记下载成功")
        # netCDF4's Windows C file API cannot reliably open Chinese paths.
        engine = "h5netcdf" if magic == b"\x89HDF\r\n\x1a\n" else "scipy"
        with xr.open_dataset(part, engine=engine) as dataset:
            dataset.load()  # Verify all data blocks, not just the header.
            receipt["dimensions"] = dict(dataset.sizes)
            receipt["variables"] = {
                name: {"dims": list(var.dims), "units": str(var.attrs.get("units", "")),
                       "long_name": str(var.attrs.get("long_name", ""))}
                for name, var in dataset.variables.items()
            }
            receipt["time_values"] = (
                [str(t) for t in dataset.time.values.reshape(-1)] if "time" in dataset else []
            )
        receipt.update(status="downloaded_netcdf_verified", bytes=size, sha256=digest(part))
        part.replace(target)
        write_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        # Avoid persisting exception text, response bodies, or redirect URLs:
        # upstreams may echo an Authorization header into an error response.
        receipt.update(status="failed", error_type=type(exc).__name__)
        write_json(receipt_path, receipt)
        raise RuntimeError("NLDAS认证或样例核验失败；状态码见本地审计文件，未记录响应正文或凭据") from None
    finally:
        if response is not None:
            response.close()
        if owned:
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("本地数据/审计/NLDAS_认证小样例"))
    args = parser.parse_args()
    token = os.environ.pop("ECHO_EARTHDATA_TOKEN", "")
    try:
        result = verify_access(token, args.output)
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1
    print(f"NLDAS样例已核验：{result['bytes']} bytes；字段与时间语义仍需单独核验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
