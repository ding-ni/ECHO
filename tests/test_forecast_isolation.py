import numpy as np
import torch

from echo.data.permissions import ForecastInput
from echo.models.echo import EchoStage
from echo.training.pipeline import local_prediction


def test_scoring_file_changes_do_not_change_prediction(tmp_path):
    torch.manual_seed(19)
    model = EchoStage(2, dynamic=True).eval()
    weather = np.ones((80, 3), dtype="float32")
    q = np.linspace(0.1, 5, 80).astype("float32")
    h = np.linspace(1.0, 2.0, 24).astype("float32")
    inp = ForecastInput(
        "S", "R", weather, np.zeros(2, dtype="float32"), 100.0, h, np.ones(24, dtype=bool), 16, 40, 40, 80
    )
    scoring = tmp_path / "futureH.txt"
    scoring.write_text("1,2,3")
    with torch.no_grad():
        first = local_prediction(model, inp, q, "cpu")[0]
        scoring.write_text("999,999,999")
        second = local_prediction(model, inp, q, "cpu")[0]
    torch.testing.assert_close(first, second, atol=0, rtol=0)
