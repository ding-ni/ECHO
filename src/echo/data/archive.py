"""档案路径、展开大小与校验；不覆盖已有提取结果。"""

import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from echo.utils import digest, read_json, write_json


def safe_member(name):
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or any(":" in part for part in p.parts):
        raise ValueError(f"危险档案路径: {name}")
    return p


def extract_verified(archive, target, preview=False):
    archive, target = Path(archive), Path(target)
    audit_path = archive.with_name(archive.name + ".audit.json")
    if not audit_path.exists():
        raise ValueError("缺少下载校验记录")
    audit = read_json(audit_path)
    if archive.stat().st_size != audit["bytes"] or digest(archive, "md5") != audit["md5"]:
        raise ValueError("档案校验失败")
    marker = target / "extraction.json"
    if marker.exists():
        previous = read_json(marker)
        if previous["archive_sha256"] == digest(archive):
            return previous
        raise ValueError("目标目录已有不同版本")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            entries = z.infolist()
            for f in entries:
                safe_member(f.filename)
                if stat.S_ISLNK(f.external_attr >> 16):
                    raise ValueError("不接受符号链接")
            total = sum(f.file_size for f in entries)
    else:
        import py7zr

        with py7zr.SevenZipFile(archive) as z:
            entries = z.list()
            for f in entries:
                safe_member(f.filename)
                if getattr(f, "is_symlink", False):
                    raise ValueError("不接受符号链接")
            total = sum(f.uncompressed or 0 for f in entries)
    result = {
        "archive": str(archive),
        "members": len(entries),
        "uncompressed_bytes": total,
        "archive_sha256": digest(archive),
        "preview": preview,
    }
    if preview:
        return result
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < total + 1024**3:
        raise OSError("解压空间不足")
    if target.exists() and any(target.iterdir()):
        raise ValueError("目标提取目录非空且未完成；请指定新目录，保留旧现场")
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)  # 上面已逐项核验路径与链接。
    else:
        import py7zr

        with py7zr.SevenZipFile(archive) as z:
            z.extractall(target)
    result["preview"] = False
    write_json(marker, result)
    return result
