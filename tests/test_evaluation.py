import numpy as np
import pandas as pd
import torch

from echo.evaluation.metrics import ensemble_crps, score_site, system_bootstrap


def test_crps_agrees_with_pairwise_definition():
    m = torch.tensor([[1.0, 3.0], [2.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([2.0, 3.0])
    expected = (m - y).abs().mean() - 0.5 * (m[:, None] - m[None]).abs().mean()
    torch.testing.assert_close(ensemble_crps(m, y), expected)


def test_missing_labels_and_no_events_do_not_make_fake_event_score():
    h = np.ones(168)
    result = score_site(h, h[None], np.full(100, 3.0))
    assert result["RMSE_m"] == 0
    assert result["events"] == 0
    assert result["event_RMSE_m"] is None


def test_bootstrap_uses_system_blocks():
    frame = pd.DataFrame({"terminal_system_id": ["A", "A", "B"], "event_RMSE_m": [1.0, 2.0, 3.0]})
    result = system_bootstrap(frame)
    assert result["systems"] == 2 and result["sites"] == 3
    assert result["mean"] == 2.0
