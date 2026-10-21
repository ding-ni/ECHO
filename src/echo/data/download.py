"""固定资产下载：续传、大小及MD5验证、元数据快照。可不安装PyTorch独立运行。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from echo.utils import ROOT, digest, read_json, utc_now, write_json


ASSETS = {
    "info": ("15066778", "info.csv", 1162939, "cb5872491b17c077a27fa24dc9aaac98"),
    "attributes": ("15066778", "attributes.7z", 19067073, "3c43c3df3809e69f5f1798f58d5dc62a"),
    "shapefiles": ("15066778", "shapefiles.7z", 530604106, "b16c335e73a5532f1c0f8117a3c53448"),
    "qh": ("16729675", "Hourly2.zip", 4176462827, "8bab1c99329db3fa6b1b6cb464a2a573"),
    "weather": ("15066778", "timeseries.7z", 21247136353, "68faf47c89cb96c513e89343f87f3f0e"),
    "weather_extra": ("15070091", "timeseries_nonobs.7z", 35655472043, "ef76e71d97953ed62c2cb24c0ccb23b1"),
}
DEFAULT_ASSETS = ["info", "attributes", "shapefiles", "qh", "weather"]


def asset_specs():
    return {key: {"record": v[0], "filename": v[1], "bytes": v[2], "md5": v[3],
                  "url": f"https://zenodo.org/api/records/{v[0]}/files/{v[1]}/content"}
            for key, v in ASSETS.items()}


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "ECHO-research/0.1 (auditable scientific download)", **(headers or {})}), timeout=90)


def fetch(spec, root, attempts=6, progress=True, snapshot=True):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = spec["filename"]
    if Path(name).name != name:
        raise ValueError("资产名不能包含路径")
    dest = root / name
    audit = root / (name + ".audit.json")
    part = root / (name + ".part")
    status_file = root / (name + ".status.json")
    if dest.exists():
        if dest.stat().st_size != spec["bytes"] or digest(dest, "md5") != spec["md5"]:
            raise ValueError(f"已有文件校验失败，保留现场: {dest}")
        result = {**spec, "status": "verified_existing", "sha256": digest(dest), "verified_utc": utc_now()}
        write_json(audit, result)
        return result
    free = shutil.disk_usage(root).free
    needed = spec["bytes"] - (part.stat().st_size if part.exists() else 0)
    if free < needed + 1024**3:
        raise OSError("可用磁盘不足，需额外保留1GiB")
    if snapshot:
        snap = root / ("record_" + spec["record"] + ".json")
        if not snap.exists():
            with request("https://zenodo.org/api/records/" + spec["record"]) as response:
                meta = json.load(response)
            matches = [f for f in meta["files"] if f["key"] == name]
            if not matches or matches[0]["size"] != spec["bytes"] or matches[0]["checksum"] != "md5:" + spec["md5"]:
                raise ValueError("远端资产与固定清单不一致，禁止静默更新")
            write_json(snap, meta)
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        if offset == spec["bytes"]:
            break
        if offset > spec["bytes"]:
            raise ValueError("临时文件大于固定资产，保留检查")
        write_json(status_file, {"status": "downloading", "bytes": offset, "expected": spec["bytes"],
                                 "attempt": attempt + 1, "updated_utc": utc_now()})
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with request(spec["url"], headers) as response:
                code = response.status
                if offset and code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("续传起点不匹配")
                    mode = "ab"
                elif code == 200:
                    # 服务端忽略Range：从头重写未验收临时文件，不能把完整文件附加到旧片段。
                    mode, offset = "wb", 0
                else:
                    raise ValueError(f"不支持的HTTP状态 {code}")
                size, last_log = offset, time.monotonic()
                with part.open(mode) as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        size += len(block)
                        if size > spec["bytes"]:
                            raise ValueError("下载体量超出清单")
                        if time.monotonic() - last_log >= 15:
                            write_json(status_file, {"status": "downloading", "bytes": size,
                                                     "expected": spec["bytes"], "updated_utc": utc_now()})
                            if progress:
                                print(f"{name}: {size / spec['bytes']:.1%} ({size:,} bytes)", flush=True)
                            last_log = time.monotonic()
            if part.stat().st_size != spec["bytes"]:
                raise OSError("连接提前结束，稍后从临时文件继续")
            break
        except (OSError, urllib.error.URLError) as exc:
            write_json(status_file, {"status": "retry_pending", "error": str(exc), "updated_utc": utc_now()})
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 30))
    md5, sha = digest(part, "md5"), digest(part)
    if md5 != spec["md5"]:
        write_json(status_file, {"status": "checksum_failed", "actual_md5": md5})
        raise ValueError(f"MD5不匹配，临时文件保留: {part}")
    part.replace(dest)
    result = {**spec, "status": "downloaded_verified", "sha256": sha, "retrieved_utc": utc_now()}
    write_json(audit, result)
    write_json(status_file, result)
    if progress:
        print(f"完成并校验: {dest}", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", nargs="+", choices=list(ASSETS), default=DEFAULT_ASSETS)
    p.add_argument("--output", type=Path, default=ROOT / "本地数据/原始/下载")
    p.add_argument("--list", action="store_true")
    a = p.parse_args(argv)
    specs = asset_specs()
    if a.list:
        print(json.dumps({k: specs[k] for k in a.assets}, indent=2, ensure_ascii=False))
        print(f"总计 {sum(specs[k]['bytes'] for k in a.assets) / 1e9:.2f} GB（十进制）")
        return
    for key in a.assets:
        fetch(specs[key], a.output)


if __name__ == "__main__":
    main()
