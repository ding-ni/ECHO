"""Executable data contract shared by local and server full-architecture runs."""

import numpy as np
import pandas as pd

PROHIBITED = {"observed_target_Q_quantiles", "observed_runoff_ratio", "observed_baseflow_index",
              "q", "h", "future_h", "target_q", "rating_curve"}


def validate_protocol_inputs(data, provenance, support_start, support_end, query_end, target=True,
                             require_reference=True):
    required = {"weather_nodes", "time_ns", "node_features", "node_areas", "downstream_parent",
                "edge_index", "edge_attr", "edge_attr_mask", "terrain_tokens", "terrain_padding_mask",
                "climate_features", "support_h"}
    if require_reference:
        required.add("reference_q")
    missing = required - data.keys()
    if missing:
        raise ValueError(f"完整主模型缺少输入字段: {sorted(missing)}")
    if target and PROHIBITED & {k.lower() for k in data}:
        raise ValueError("目标模型输入出现禁用的Q、率定曲线或未来H字段")
    features = set(provenance.get("node_feature_names", [])) | set(provenance.get("terrain_feature_names", []))
    if PROHIBITED & features:
        raise ValueError("地理字段含观测Q派生特征")
    expected = {"weather_kind": "incremental_area_weighted_original_grid",
                "terrain_kind": "DEM_local_valley_multiscale",
                "graph_kind": "connected_child_parent_edge_contraction"}
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"{key}未满足定稿数据语义: {value}")
    if not provenance.get("source_snapshots") or (
        require_reference and not provenance.get("reference_checkpoint_sha256")
    ):
        raise ValueError("缺少空间输入来源与参考流量检查点收据")
    times = np.asarray(data["time_ns"], dtype="int64")
    if len(times) < 2 or not np.all(np.diff(times) == 3600 * 10**9):
        raise ValueError("气象时间必须为逐小时UTC连续时间轴")
    if not 0 <= support_start < support_end < query_end <= len(times):
        raise ValueError("支持和查询边界无效")
    if provenance.get("climate_cutoff_ns", int(times[-1]) + 1) > int(times[support_end]):
        raise ValueError("环境气候统计使用了起报之后的天气")
    if pd.Timestamp(times[0], tz="UTC") > pd.Timestamp(times[support_start], tz="UTC") - pd.DateOffset(years=2):
        raise ValueError("支持前真实气象不足两个日历年")
    weather = np.asarray(data["weather_nodes"])
    areas = np.asarray(data["node_areas"])
    nodes = len(areas)
    if weather.shape != (nodes, len(times), 3) or nodes < 1 or nodes > 32:
        raise ValueError("子流域气象必须为[N,T,3]且节点数在1—32")
    if not np.isfinite(weather).all() or (weather[:, :, [0, 2]] < 0).any():
        raise ValueError("强迫有缺测或P/PET需求为负")
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("增量面积无效")
    node_features = np.asarray(data["node_features"])
    climate = np.asarray(data["climate_features"])
    if node_features.ndim != 2 or node_features.shape[0] != nodes or not np.isfinite(node_features).all():
        raise ValueError("节点属性形状或有限值无效")
    if climate.shape != (64,) or not np.isfinite(climate).all():
        raise ValueError("气候摘要需要64个起报前有限统计量")
    parents = np.asarray(data["downstream_parent"])
    if parents.shape != (nodes,) or sum(parents == -1) != 1:
        raise ValueError("真实河网必须为单出口")
    for node, parent in enumerate(parents):
        if parent != -1 and not node < parent < nodes:
            raise ValueError("河网非上游到下游排序")
    edges = np.asarray(data["edge_index"])
    expected_edges = {(i, int(p)) for i, p in enumerate(parents) if p >= 0}
    if edges.shape != (2, nodes - 1) or set(map(tuple, edges.T)) != expected_edges:
        raise ValueError("图编码边与守恒路由边不一致")
    if data["edge_attr"].shape != (nodes - 1, 3) or data["edge_attr_mask"].shape != (nodes - 1, 3):
        raise ValueError("边属性需含长度、落差、坡度及各自覆盖掩码")
    if not np.isfinite(data["edge_attr"]).all():
        raise ValueError("边属性缺值必须显式占位并附掩码")
    terrain = np.asarray(data["terrain_tokens"])
    terrain_mask = np.asarray(data["terrain_padding_mask"], dtype=bool)
    if terrain.ndim != 2 or terrain.shape[0] != 16 or terrain_mask.shape != (16,) or terrain_mask.all():
        raise ValueError("需要16个真实河谷片段及有效掩码")
    if not np.isfinite(terrain).all():
        raise ValueError("地形缺值必须显式占位并附掩码")
    if len(data["support_h"]) != support_end - support_start:
        raise ValueError("H仅应包含许可支持窗口，不得扩展到查询")
    if np.isfinite(data["support_h"]).mean() < .8:
        raise ValueError("支持H覆盖不足80%")
    if require_reference and (
        data["reference_q"].shape != (len(times),) or not np.isfinite(data["reference_q"]).all()
    ):
        raise ValueError("参考Q与气象长度或有限值不一致")
    return {"nodes": nodes, "support_hours": support_end - support_start,
            "query_hours": query_end - support_end,
            "warmup_hours": support_start, "target_Q_read": False}
