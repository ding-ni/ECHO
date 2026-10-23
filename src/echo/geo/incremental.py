"""Topology-preserving contractions of an outlet-rooted incremental catchment graph."""

from __future__ import annotations

from collections import deque

import numpy as np


def induced_parents(selected, downstream, outlet):
    """Trace through clipped-out cells without inventing a connection between branches."""
    selected = set(map(int, selected))
    if outlet not in selected:
        raise ValueError("站点出口单元不在流域增量单元内")
    parents = {}
    for node in sorted(selected):
        if node == outlet:
            parents[node] = -1
            continue
        current, seen = node, {node}
        while True:
            current = int(downstream.get(current, 0))
            if not current or current in seen:
                raise ValueError("单元下游路径有环或未到达目标站出口")
            if current in selected:
                parents[node] = current
                break
            seen.add(current)
    _topological_order(parents)
    return parents


def _topological_order(parents):
    counts = dict.fromkeys(parents, 0)
    for node, parent in parents.items():
        if parent != -1:
            if parent not in parents or parent == node:
                raise ValueError("增量图含未知下游或自环")
            counts[parent] += 1
    queue = deque(sorted(node for node, count in counts.items() if count == 0))
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        parent = parents[node]
        if parent != -1:
            counts[parent] -= 1
            if counts[parent] == 0:
                queue.append(parent)
    if len(order) != len(parents) or sum(p == -1 for p in parents.values()) != 1:
        raise ValueError("增量图必须是单出口无环河网")
    return order


def contract_incremental_graph(parents, areas, maximum_nodes=8):
    """Merge only connected child-parent pairs; retain a reproducible quotient map.

    Unary reaches are contracted first. When branching exceeds the budget, small
    tributary increments are merged into their real downstream parent. No new
    cross-branch routing edge is introduced. All source cells remain accounted for.
    """
    if maximum_nodes < 1 or set(parents) != set(areas):
        raise ValueError("节点预算或增量面积索引无效")
    _topological_order(parents)
    if any(not np.isfinite(a) or a <= 0 for a in areas.values()):
        raise ValueError("增量面积必须有限且为正")
    graph = dict(parents)
    groups = {node: {node} for node in graph}
    totals = {node: float(areas[node]) for node in graph}
    while len(graph) > maximum_nodes:
        children = {node: [] for node in graph}
        for node, parent in graph.items():
            if parent != -1:
                children[parent].append(node)
        candidates = [node for node in graph if graph[node] != -1]
        node = min(candidates, key=lambda n: (
            len(children[graph[n]]) != 1, totals[n], n))
        parent = graph[node]
        groups[parent].update(groups.pop(node))
        totals[parent] += totals.pop(node)
        for child in children[node]:
            graph[child] = parent
        del graph[node]
    order = _topological_order(graph)
    positions = {node: i for i, node in enumerate(order)}
    groups = [sorted(groups[node]) for node in order]
    parent_indices = np.asarray([positions[graph[node]] if graph[node] != -1 else -1
                                 for node in order], dtype="int64")
    area_array = np.asarray([totals[node] for node in order], dtype="float64")
    mapping = {cell: index for index, cells in enumerate(groups) for cell in cells}
    for node, parent in parents.items():
        if parent != -1 and mapping[node] != mapping[parent]:
            if parent_indices[mapping[node]] != mapping[parent]:
                raise AssertionError("合并后改变了原河网路径")
    return {"groups": groups, "representative_ids": order,
            "parents": parent_indices, "areas_km2": area_array, "cell_to_node": mapping}
