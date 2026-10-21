"""下载中ZIP的前部只读探查。每个成员验证CRC，但不代表完整资产MD5已通过；只供模式核验。"""

import struct
import zlib
from pathlib import Path

from echo.data.archive import safe_member
from echo.utils import write_json


def probe_partial_zip(path, output, limit=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with Path(path).open("rb") as stream:
        while len(records) < limit:
            raw = stream.read(30)
            if len(raw) != 30:
                break
            sig, version, flags, method, tm, date, crc, compressed, size, namelen, extralen = struct.unpack(
                "<IHHHHHIIIHH", raw
            )
            if sig != 0x04034B50:
                break
            if flags & 1 or compressed == 0xFFFFFFFF:
                raise ValueError("不支持加密或ZIP64的局部探查；请等待完整归档")
            name = stream.read(namelen).decode("utf-8" if flags & 2048 else "cp437")
            stream.read(extralen)
            member = safe_member(name)
            if not name.lower().endswith(".nc") and not flags & 8:
                stream.seek(compressed, 1)
                continue
            if compressed > 100 * 1024**2 or size > 100 * 1024**2:
                raise ValueError("探查成员超过100MiB限制")
            if flags & 8:
                if method != 8:
                    raise ValueError("未知长度仅支持deflate")
                decoder, chunks, expanded = zlib.decompressobj(-15), [], 0
                while not decoder.eof:
                    chunk = stream.read(65536)
                    if not chunk:
                        break
                    result = decoder.decompress(chunk)
                    expanded += len(result)
                    if expanded > 100 * 1024**2:
                        raise ValueError("成员超过展开限制")
                    chunks.append(result)
                if not decoder.eof:
                    break
                stream.seek(-len(decoder.unused_data), 1)
                first = struct.unpack("<I", stream.read(4))[0]
                if first == 0x08074B50:
                    crc, compressed, size = struct.unpack("<III", stream.read(12))
                else:
                    crc = first
                    compressed, size = struct.unpack("<II", stream.read(8))
                content = b"".join(chunks)
            else:
                content = stream.read(compressed)
                if len(content) < compressed:
                    break
                content = content if method == 0 else zlib.decompress(content, -15) if method == 8 else None
            if content is None or len(content) != size or zlib.crc32(content) != crc:
                raise ValueError("成员CRC或大小失败")
            target = output.joinpath(*member.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            records.append({"member": name, "bytes": size, "crc32": crc})
    write_json(
        output / "probe.json",
        {
            "scope": "partial_archive_schema_probe_only_not_training",
            "full_archive_md5_verified": False,
            "members": records,
        },
    )
    return records
