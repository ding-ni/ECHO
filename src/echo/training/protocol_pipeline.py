"""One unchanged A--E implementation for complete local/server protocol data."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from echo.models.protocol import ProtocolECHO, ProtocolHydrology
from echo.training.protocol_engine import (
    ProtocolEpisode,
    SourceLabels,
    cache_references,
    fit_hydrology_bank,
    fit_response_stages,
    predict_forecast,
)
from echo.utils import digest, read_json, run_metadata, write_json


def _verified_path(base, row, key):
    path = Path(row[key])
    path = path if path.is_absolute() else base / path
    if not path.is_file() or digest(path) != row[key + "_sha256"]:
        raise ValueError(f"输入文件缺失或哈希不符: {key}")
    return path


def load_episodes(index_path):
    index_path = Path(index_path)
    index = read_json(index_path)
    if not index.get("complete_spatial_inputs_verified"):
        raise ValueError("完整空间资料尚未验收：需要真实节点属性、河谷DEM和完整逐小时网格气象")
    episodes, train, validation = [], [], []
    for row in index["episodes"]:
        if row["role"] not in {"train", "validation", "development_test"}:
            raise ValueError("完整主模型数据清单含盲测或未知角色")
        inputs_path = _verified_path(index_path.parent, row, "input_path")
        provenance_path = _verified_path(index_path.parent, row, "provenance_path")
        with np.load(inputs_path, allow_pickle=False) as archive:
            inputs = {name: archive[name] for name in archive.files}
        episode = ProtocolEpisode(row["station_id"], row["terminal_system_id"], row["role"],
                                  row["crossfit_fold"], inputs, read_json(provenance_path),
                                  row["support_start"], row["support_end"], row["query_end"])
        episode.validate(require_reference=False)
        episodes.append(episode)
        if episode.role in {"train", "validation"}:
            labels_path = _verified_path(index_path.parent, row, "labels_path")
            with np.load(labels_path, allow_pickle=False) as archive:
                if episode.role == "validation" and "q_supervision" in archive.files:
                    raise ValueError("来源验证H任务不允许载入验证水系Q")
                labels = SourceLabels(archive["query_h"], archive["q_supervision"] if episode.role == "train" else None)
            (train if episode.role == "train" else validation).append((episode, labels))
        elif "labels_path" in row:
            raise ValueError("开发预测输入清单不能指向未来评分标签")
    # No duplicate station episode in this first reference-cache interface.
    if len({e.station_id for e in episodes}) != len(episodes):
        raise ValueError("当前参考缓存按站点存储；一个运行内每站一个固定episode，预算批次分别运行")
    if not train or not validation:
        raise ValueError("需要许可来源训练和完整留出来源验证水系")
    return index, episodes, train, validation


def run(index_path, config_path, output, *, stage_a_only=False, device="cuda", members=None, profile="local"):
    index_path, config_path, output = Path(index_path), Path(config_path), Path(output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if config["model"]["class"] != "echo.models.protocol.ProtocolECHO":
        raise ValueError("此入口只运行统一的完整ProtocolECHO结构")
    index, episodes, train, validation = load_episodes(index_path)
    if output.exists():
        raise ValueError("运行目录已存在，请新建运行版本")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA；阶段A可以显式使用--device cpu")
    if not stage_a_only:
        # Fail before A work if the full requested environment is unavailable.
        import importlib.util
        import platform
        if platform.system() != "Linux" or importlib.util.find_spec("mamba_ssm") is None:
            raise RuntimeError("完整B—E需要已安装固定官方Mamba-3的Linux CUDA环境；可先单独执行阶段A")
    output.mkdir(parents=True)
    seed = int(config.get("seed", 20261019))
    if profile not in {"local", "server"}:
        raise ValueError("运行规模需为local或server")
    runtime = config[profile + "_runtime"]
    steps = runtime["optimizer_steps_per_stage"]
    stage_a_steps = int(steps if isinstance(steps, int) else steps["A"])
    torch.set_num_threads(int(runtime.get("cpu_threads", 2)))
    node_dim = episodes[0].inputs["node_features"].shape[-1]
    terrain_dim = episodes[0].inputs["terrain_tokens"].shape[-1]

    def factory():
        return ProtocolHydrology(node_dim, terrain_dim).to(device)

    forbidden = set(index.get("forbidden_systems", [])) | {e.system_id for e in episodes if e.role != "train"}
    write_json(output / "run.json", run_metadata({"mode": "full_scientific_structure_local_steps",
                                                 "device": device, "seed": seed, "runtime_profile": profile,
                                                 "stage_A_only": stage_a_only}, [index_path, config_path]))
    bank = fit_hydrology_bank(factory, train, output / "stage_A", stage_a_steps,
                               config["training"]["learning_rate"], seed, forbidden)
    references, curve_setup = cache_references(factory, bank, episodes, output / "reference_Q")
    by_station = {r["station_id"]: r for r in references}
    for episode in episodes:
        row = by_station[episode.station_id]
        episode.inputs["reference_q"] = np.load(row["path"], allow_pickle=False)
        episode.provenance.update(reference_checkpoint_sha256=row["reference_checkpoint_sha256"],
                                  reference_trained_systems=row["reference_trained_systems"],
                                  source_fixed_reference_u=curve_setup["source_fixed_reference_u"])
    if stage_a_only:
        return {"completed_stages": ["A"], "full_model_trained": False, "reference_sites": len(references)}
    model = ProtocolECHO(node_dim, terrain_dim, curve_setup["knot_width"]).to(device)
    result = fit_response_stages(
        model, train, validation, output / "response", steps_per_stage=steps,
        learning_rate=config["training"]["learning_rate"],
        joint_learning_rate=config["training"]["joint_learning_rate"],
        members=config["training"]["members_start"], seed=seed,
        stage_a_receipt=next(r for r in bank if r["excluded_fold"] is None))
    target_files, budgets = [], set()
    for episode in episodes:
        if episode.role != "development_test":
            continue
        predictions = predict_forecast(model, episode, {e.system_id for e, _ in train},
                                       members or config["inference"]["members"], result["residual_model"], seed)
        predictions["members"] = predictions["members_with_source_calibrated_residual"]
        predictions["support_h"] = episode.inputs["support_h"]
        target = output / "predictions" / f"{episode.station_id}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **predictions)
        target_files.append({"station_id": episode.station_id, "terminal_system_id": episode.system_id,
                             "path": str(target.resolve()), "sha256": digest(target)})
        budgets.add(len(episode.inputs["support_h"]))
    prediction_metadata = {
        "scope": "real_protocol_local_development" if profile == "local" else "real_protocol_development",
        "mode": "conditional_simulation", "model": "protocol_echo",
        "budget_days": sorted(budgets)[0] / 24 if len(budgets) == 1 else "mixed_recorded_per_station",
        "stations": target_files, "failures": [], "member_field": "members",
        "architecture_status": "ProtocolECHO_official_Mamba3_true_spatial_inputs_A_to_E",
        "uncertainty_scope": "joint_local_codes_plus_source_calibrated_AR1_residual_not_weather_ensemble",
        "metadata": run_metadata({"profile": profile, "seed": seed}, [index_path, config_path]),
        "stage_E_checkpoint_sha256": digest(output / "response/model_E.pt"),
    }
    write_json(output / "predictions/predictions.json", prediction_metadata)
    write_json(output / "predictions_member_only/predictions.json", {
        **prediction_metadata, "member_field": "members_without_residual",
        "uncertainty_scope": "joint_local_code_member_disagreement_without_residual_or_weather_ensemble"})
    summary = {"completed_stages": list("ABCDE"), "targets": target_files,
               "source_index_sha256": digest(index_path), "predictions_contain_future_H": False,
               "prediction_interfaces_ready_for_separate_scoring": True,
               "scoring_and_statistical_report_completed": False}
    write_json(output / "completion.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--stage-A-only", action="store_true")
    parser.add_argument("--profile", default="local", choices=["local", "server"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        _, episodes, train, validation = load_episodes(args.index)
        print({"episodes": len(episodes), "source": len(train), "validation": len(validation)})
    else:
        if args.output is None:
            parser.error("--output is required for training")
        print(run(args.index, args.config, args.output, stage_a_only=args.stage_A_only,
                  device=args.device, profile=args.profile))
