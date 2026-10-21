from echo.demo import run_demo


def test_synthetic_train_predict_score_report(tmp_path):
    result = run_demo(tmp_path)
    assert result["successfully_scored"] == 1
    assert result["scope"] == "synthetic_demo_not_real_performance"
    assert (tmp_path / "评估/阶段报告.md").is_file()
    assert len(result["losses"]) == 4
