"""Tests for experiment runner error handling and tune utilities."""

import pytest
from pathlib import Path
from yolocc.experiment.runner import (
    ExperimentRunner,
    TUNE_PRESETS,
    resolve_search_space,
)


class TestExperimentRunner:
    def test_run_experiment_raises_on_missing_dataset(self, tmp_path):
        """run_experiment raises FileNotFoundError, not SystemExit."""
        runner = ExperimentRunner(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            runner.run_experiment(
                overrides={"lr0": 0.01},
                dataset=str(tmp_path / "nonexistent"),
            )

    def test_run_tune_raises_on_missing_dataset(self, tmp_path):
        """run_tune raises FileNotFoundError when dataset doesn't exist."""
        runner = ExperimentRunner(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            runner.run_tune(
                search_space={},
                dataset=str(tmp_path / "nonexistent"),
            )


class TestResolveSearchSpace:
    def test_resolve_search_space_preset(self):
        """Preset names resolve to the correct search space dict."""
        assert resolve_search_space("lr") == TUNE_PRESETS["lr"]
        assert resolve_search_space("all") == {}
        assert resolve_search_space(None) == {}

    def test_resolve_search_space_custom(self):
        """Custom 'key=min:max' format is parsed correctly."""
        result = resolve_search_space("lr0=0.001:0.01 momentum=0.8:0.98")
        assert result == {"lr0": (0.001, 0.01), "momentum": (0.8, 0.98)}

    def test_resolve_search_space_scientific_notation(self):
        """Scientific notation in bounds is parsed correctly."""
        result = resolve_search_space("lr0=1e-5:1e-1")
        assert result == {"lr0": (1e-5, 1e-1)}

    def test_resolve_search_space_invalid(self):
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError):
            resolve_search_space("bad_format")


class TestFindTuneDir:
    def test_find_tune_dir_in_exp_output(self, tmp_path):
        """_find_tune_dir locates directory containing best_hyperparameters.yaml."""
        tune_dir = tmp_path / "tune"
        tune_dir.mkdir()
        (tune_dir / "best_hyperparameters.yaml").write_text("lr0: 0.01\n")

        found = ExperimentRunner._find_tune_dir(tmp_path)
        assert found == tune_dir

    def test_find_tune_dir_not_found(self, tmp_path):
        """_find_tune_dir raises FileNotFoundError when no tune output exists."""
        with pytest.raises(FileNotFoundError):
            ExperimentRunner._find_tune_dir(tmp_path)
