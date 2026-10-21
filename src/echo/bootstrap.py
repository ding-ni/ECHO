"""等待官方CUDA wheel校验后安装、诊断、完整测试并生成合成样例。"""

import subprocess
import sys
import time

from echo.utils import ROOT, utc_now, write_json


def main():
    wheel = ROOT / "本地缓存/wheel/torch-2.8.0+cu126-cp311-cp311-win_amd64.whl"
    started = time.monotonic()
    while not wheel.with_name(wheel.name + ".audit.json").exists():
        if time.monotonic() - started > 3600:
            raise TimeoutError("CUDA wheel仍未校验完成，请检查下载日志")
        time.sleep(10)
    logs = ROOT / "本地运行/环境与测试"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("安装CUDA", ["-m", "pip", "install", str(wheel), "--no-deps", "--progress-bar", "off"]),
        ("依赖一致性", ["-m", "pip", "check"]),
        ("环境诊断", ["-m", "echo", "doctor"]),
        ("完整测试", ["-m", "pytest", "-q", "--basetemp", str(logs / "测试临时"), "-p", "no:cacheprovider"]),
        ("合成闭环", ["-m", "echo", "demo"]),
    ]
    statuses = []
    for name, args in tasks:
        with (logs / (name + ".log")).open("w", encoding="utf-8") as out:
            result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        statuses.append({"task": name, "exit_code": result.returncode})
        write_json(logs / "执行状态.json", {"tasks": statuses, "updated_utc": utc_now()})
        if result.returncode:
            raise SystemExit(result.returncode)
    with (ROOT / "07_环境配置/windows实测依赖锁.txt").open("w", encoding="utf-8") as out:
        subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], stdout=out, check=True)


if __name__ == "__main__":
    main()
