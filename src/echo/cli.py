"""统一命令入口；各命令延迟加载，下载不需要加载GPU库。"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="ECHO 数据、训练、预测与评估")
    parser.add_argument(
        "command",
        choices=[
            "doctor",
            "download",
            "extract",
            "audit",
            "inventory",
            "topology",
            "boundary-audit",
            "weather-audit",
            "station-evidence",
            "weather-prepare",
            "split",
            "prepare",
            "train",
            "predict",
            "evaluate",
            "report",
            "compare",
            "demo",
        ],
    )
    args, rest = parser.parse_known_args()
    from echo.commands import dispatch

    dispatch(args.command, rest)
