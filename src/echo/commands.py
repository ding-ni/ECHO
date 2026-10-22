"""CLI命令参数与任务分派。"""

import argparse
import json
import platform
import shutil
from pathlib import Path

from echo.utils import ROOT, write_json


def doctor(output=None):
    import psutil
    import torch

    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "free_disk_gb": round(shutil.disk_usage(ROOT).free / 1024**3, 2),
        "project": str(ROOT),
        "gpu_forward_backward": False,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 8, device=device, requires_grad=True)
    y = torch.nn.Linear(8, 4).to(device)(x)
    y.square().mean().backward()
    passed = bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all())
    result["forward_backward_passed"] = passed
    result["forward_backward_device"] = device
    result["gpu_forward_backward"] = passed and device == "cuda"
    write_json(output or ROOT / "本地运行/环境诊断.json", result)
    return result


def dispatch(command, argv):
    if command == "download":
        from echo.data.download import main

        return main(argv)
    p = argparse.ArgumentParser(prog="echo " + command)
    if command == "doctor":
        p.add_argument("--output", type=Path)
        a = p.parse_args(argv)
        result = doctor(a.output)
    elif command == "extract":
        from echo.data.archive import extract_verified
        from echo.data.download import asset_specs

        p.add_argument("--asset", choices=list(asset_specs()), required=True)
        p.add_argument("--archive", type=Path)
        p.add_argument("--output", type=Path)
        p.add_argument("--preview", action="store_true")
        a = p.parse_args(argv)
        spec = asset_specs()[a.asset]
        archive = a.archive or ROOT / "本地数据/原始/下载" / spec["filename"]
        result = extract_verified(archive, a.output or ROOT / "本地数据/原始" / a.asset, a.preview)
    elif command == "audit":
        from echo.data.audit import audit_tree

        p.add_argument("--root", required=True, type=Path)
        p.add_argument("--limit", type=int, default=20)
        p.add_argument("--output", type=Path, default=ROOT / "本地运行/字段探查.json")
        a = p.parse_args(argv)
        result = audit_tree(a.root, a.limit)
        write_json(a.output, result)
        result = {"output": str(a.output), "files_found": result["files_found"]}
    elif command == "inventory":
        from echo.data.inventory import build_inventory

        p.add_argument("--qh-root", type=Path, required=True)
        p.add_argument("--weather-root", type=Path, required=True)
        p.add_argument("--attributes-root", type=Path, required=True)
        p.add_argument("--topology", type=Path)
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args(argv)
        result = build_inventory(a.qh_root, a.weather_root, a.attributes_root, a.output, a.topology)
    elif command == "topology":
        from echo.geo.hydrobasins import download_topology, map_terminals

        p.add_argument("--stations", required=True, type=Path)
        p.add_argument("--shapefile", type=Path)
        p.add_argument("--output", required=True, type=Path)
        a = p.parse_args(argv)
        shape = a.shapefile or download_topology(ROOT / "本地数据/原始/HydroBASINS")
        result = map_terminals(shape, a.stations, a.output)
    elif command == "boundary-audit":
        from echo.geo.overlap import audit_boundaries

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--shapefile", required=True, type=Path)
        p.add_argument("--source-crs")
        p.add_argument("--crs-evidence")
        p.add_argument("--output", required=True, type=Path)
        a = p.parse_args(argv)
        result = audit_boundaries(a.manifest, a.shapefile, a.output, a.source_crs, a.crs_evidence)
        result = {k: v for k, v in result.items() if k != "cross_terminal_conflicts"}
    elif command == "weather-audit":
        from echo.data.meteorology import audit_meteorology

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--limit", type=int, default=20)
        a = p.parse_args(argv)
        if a.limit < 1:
            p.error("limit必须为正")
        result = audit_meteorology(a.manifest, a.output, a.limit)
        result = {"output": str(a.output), "files": len(result["files"]), "failures": result["failures"]}
    elif command == "weather-prepare":
        from echo.data.meteorology import prepare_weather

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--mapping", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        a = p.parse_args(argv)
        result = prepare_weather(a.manifest, a.mapping, a.output)
    elif command == "station-evidence":
        from echo.data.usgs_evidence import audit_station_evidence

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--snapshots", required=True, type=Path)
        p.add_argument("--limit", type=int, default=20)
        a = p.parse_args(argv)
        if a.limit < 1:
            p.error("limit必须为正")
        result = audit_station_evidence(a.manifest, a.output, a.snapshots, a.limit)
    elif command == "split":
        import pandas as pd

        from echo.data.splits import build_splits

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--output", required=True, type=Path)
        p.add_argument("--max-sites", type=int, default=100)
        a = p.parse_args(argv)
        frame = build_splits(
            pd.read_csv(a.manifest, dtype={"station_id": str, "terminal_system_id": str}),
            max_sites=a.max_sites,
        )
        a.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(a.output, index=False, encoding="utf-8-sig")
        result = {
            "split_counts": frame.split.value_counts().to_dict(),
            "independent_systems": int(frame.terminal_system_id.nunique()),
        }
        write_json(a.output.with_suffix(".summary.json"), result)
    elif command == "prepare":
        from echo.data.prepare import prepare

        p.add_argument("--manifest", required=True, type=Path)
        p.add_argument("--mapping", required=True, type=Path)
        p.add_argument("--output", type=Path, default=ROOT / "本地数据/模型视图")
        a = p.parse_args(argv)
        result = prepare(a.manifest, a.mapping, a.output)
    elif command == "train":
        import yaml

        from echo.training.pipeline import fit_stage

        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args(argv)
        config = yaml.safe_load(a.config.read_text(encoding="utf-8-sig"))
        fit_stage(a.index, config, a.output)
        result = {"run": str(a.output)}
    elif command == "predict":
        from echo.training.predict import predict

        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--hydro-run", type=Path, required=True)
        p.add_argument("--model-run", type=Path)
        p.add_argument("--baseline", choices=["persistence", "local_spline"])
        p.add_argument("--split", choices=["validation", "development_test"], default="development_test")
        p.add_argument("--budget-days", type=int, choices=[7, 30, 90])
        p.add_argument("--members", type=int, default=64)
        p.add_argument("--device", default="cpu")
        p.add_argument("--output", required=True, type=Path)
        a = p.parse_args(argv)
        result = predict(
            a.index,
            a.hydro_run,
            a.output,
            a.model_run,
            a.baseline,
            a.split,
            a.budget_days,
            a.members,
            a.device,
        )
    elif command == "evaluate":
        from echo.evaluation.report import evaluate

        p.add_argument("--predictions", type=Path, required=True)
        p.add_argument("--labels", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args(argv)
        result = evaluate(a.predictions, a.labels, a.output)
    elif command == "report":
        from echo.evaluation.report import render_report

        p.add_argument("--evaluation", type=Path, required=True)
        p.add_argument("--predictions", type=Path)
        p.add_argument("--labels", type=Path)
        a = p.parse_args(argv)
        result = str(render_report(a.evaluation, a.predictions, a.labels))
    elif command == "compare":
        from echo.evaluation.comparison import compare_runs

        p.add_argument("--evaluations", nargs="+", required=True, help="名称=评估目录")
        p.add_argument("--output", type=Path, required=True)
        a = p.parse_args(argv)
        result = compare_runs(dict(item.split("=", 1) for item in a.evaluations), a.output)
    elif command == "demo":
        from echo.demo import run_demo

        p.add_argument("--output", type=Path, default=ROOT / "本地运行/合成样例")
        a = p.parse_args(argv)
        result = run_demo(a.output)
    else:
        raise ValueError(command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
