"""Area intersections of original weather cells and real incremental polygons."""

import numpy as np
from scipy.sparse import csr_matrix


def intersect_grid(nodes, cells, minimum_coverage=.98):
    """Inputs are GeoDataFrames with declared CRS; never fabricate subgrid rainfall."""
    if nodes.crs is None or cells.crs is None or not 0 < minimum_coverage <= 1:
        raise ValueError("气象网格及增量流域须声明CRS与有效覆盖阈值")
    nodes, cells = nodes.to_crs(6933), cells.to_crs(6933)
    if not nodes.geometry.is_valid.all() or not cells.geometry.is_valid.all():
        raise ValueError("源网格或流域几何无效")
    rows, columns, areas, coverages = [], [], [], []
    for i, polygon in enumerate(nodes.geometry):
        total = float(polygon.area)
        if total <= 0:
            raise ValueError("增量流域面积为零")
        intersections = []
        for j in cells.sindex.query(polygon, predicate="intersects"):
            area = float(polygon.intersection(cells.geometry.iloc[j]).area)
            if area > 0:
                intersections.append((int(j), area))
        covered = sum(a for _, a in intersections)
        coverage = covered / total
        if coverage > 1.00001:
            raise ValueError("原始气象网格相互重叠，面积重复累计")
        if coverage < minimum_coverage:
            raise ValueError(f"节点{i}原始天气网格覆盖不足: {coverage:.6f}")
        coverages.append(coverage)
        for j, area in intersections:
            rows.append(i)
            columns.append(j)
            areas.append(area / covered)
    weights = csr_matrix((areas, (rows, columns)), shape=(len(nodes), len(cells)))
    return weights, np.asarray(coverages)


def aggregate_weather(weights, values):
    """values[T, original_cell, channel] -> [incremental_node,T,channel]."""
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1] != weights.shape[1]:
        raise ValueError("气象数据网格数与几何权重不匹配")
    used = np.unique(weights.indices)
    if not np.isfinite(values[:, used]).all():
        raise ValueError("有效原始气象格点存在缺测，不能把缺测静默填零")
    result = np.stack([weights @ values[t] for t in range(len(values))], axis=1)
    return result.astype("float32")
