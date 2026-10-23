import json

from echo.training import protocol_doctor


def test_failed_gpu_exercise_leaves_failure_receipt(tmp_path, monkeypatch):
    def inspect(exercise=False):
        if exercise:
            raise RuntimeError("official CUDA unavailable")
        return {"official_backbone_forward_backward_passed": False}

    monkeypatch.setattr(protocol_doctor, "inspect", inspect)
    output = tmp_path / "receipt.json"
    assert protocol_doctor.main(["--exercise", "--output", str(output)]) == 1
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["official_backbone_forward_backward_passed"] is False
    assert receipt["exercise_requested"] is True
    assert receipt["error_type"] == "RuntimeError"


def test_readonly_does_not_exercise_gpu(tmp_path, monkeypatch):
    calls = []

    def inspect(exercise=False):
        calls.append(exercise)
        return {"official_backbone_forward_backward_passed": False}

    monkeypatch.setattr(protocol_doctor, "inspect", inspect)
    assert protocol_doctor.main(["--output", str(tmp_path / "receipt.json")]) == 0
    assert calls == [False]
