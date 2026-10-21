"""路径、审计与可复现元数据。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if ROOT.name == "src":
    ROOT = ROOT.parent


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_metadata(config, inputs=()):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "uncommitted", True
    return {
        "created_utc": utc_now(),
        "commit": commit,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config,
        "inputs": {str(p): digest(p) for p in inputs if Path(p).is_file()},
    }
