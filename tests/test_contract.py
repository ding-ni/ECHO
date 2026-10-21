from pathlib import Path


def test_package_has_single_core():
    import echo

    assert echo.__version__.startswith("0.1.")
    assert (Path(__file__).parents[1] / "AGENTS.md").is_file()
