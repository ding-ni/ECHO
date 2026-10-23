import numpy as np
import torch

from echo.training.formal_pipeline import _geography, _support_heights


def test_target_support_height_is_mapped_without_future_labels():
    support = np.arange(100, dtype="float32")
    data = {"support_start": np.int64(1000), "support_h": support}
    np.testing.assert_array_equal(_support_heights(data, 1040, 1060), support[40:60])


def test_formal_geography_accepts_real_spatial_view_and_pads_climate():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    index = root / "本地数据/模型视图/正式空间输入_小样本_v6/index.json"
    if not index.exists():
        import pytest
        pytest.skip("本地真实数据视图不随代码仓库分发")
    import json

    doc = json.loads(index.read_text(encoding="utf-8-sig"))
    row = next(x for x in doc["stations"] if x["station_id"] == "01081500")
    geography, spatial, parents = _geography(row, torch.device("cpu"))
    assert geography["node_features"].shape == (1, 8, 8)
    assert geography["terrain_tokens"].shape == (1, 16, 10)
    assert geography["climate_features"].shape == (1, 64)
    assert torch.isfinite(geography["climate_features"]).all()
    assert np.isclose(spatial["node_areas"].sum(), row["reported_station_area_km2"], rtol=1e-5)
    assert parents[-1] == -1
