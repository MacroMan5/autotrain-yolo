"""Tests for experiment tracker."""

import pytest
import yaml
from pathlib import Path
from yolocc.experiment.tracker import (
    ExperimentResult,
    ExperimentTracker,
    safe_experiment_dir_name,
)


@pytest.fixture
def tracker(tmp_path):
    return ExperimentTracker(experiments_dir=tmp_path)


@pytest.fixture
def sample_result():
    return ExperimentResult(
        experiment_id="exp_001",
        name="lr0=0.005",
        overrides={"lr0": 0.005},
        metrics={
            "mAP50": 0.76,
            "mAP50-95": 0.48,
            "precision": 0.82,
            "recall": 0.71,
        },
        per_class_ap={"cat": 0.87, "dog": 0.78, "bird": 0.55},
        epochs_run=23,
        epochs_max=100,
        is_baseline=False,
    )


@pytest.fixture
def baseline_result():
    return ExperimentResult(
        experiment_id="exp_000",
        name="baseline",
        overrides={},
        metrics={
            "mAP50": 0.72,
            "mAP50-95": 0.45,
            "precision": 0.80,
            "recall": 0.68,
        },
        per_class_ap={"cat": 0.85, "dog": 0.72, "bird": 0.42},
        epochs_run=50,
        epochs_max=100,
        is_baseline=True,
    )


class TestExperimentResult:
    def test_primary_metric(self, sample_result):
        assert sample_result.primary_metric == 0.48

    def test_improved_over(self, sample_result, baseline_result):
        assert sample_result.improved_over(baseline_result) is True

    def test_delta(self, sample_result, baseline_result):
        delta = sample_result.delta(baseline_result)
        assert abs(delta["mAP50-95"] - 0.03) < 0.001

    def test_tune_fields_default_false(self):
        result = ExperimentResult(
            experiment_id="exp_010",
            name="minimal",
            overrides={"lr0": 0.01},
            metrics={"mAP50": 0.70, "mAP50-95": 0.40, "precision": 0.75, "recall": 0.65},
        )
        assert result.is_tune is False
        assert result.tune_iterations == 0
        assert result.tune_search_space == {}
        assert result.tune_best_params == {}
        assert result.tune_dir is None


class TestExperimentTracker:
    def test_log_creates_directory(self, tracker, sample_result):
        tracker.log(sample_result)
        exp_dir = tracker.experiments_dir / "exp_001_lr0_0.005"
        assert exp_dir.exists()
        assert (exp_dir / "config.yaml").exists()
        assert (exp_dir / "metrics.yaml").exists()

    def test_log_and_get_best(self, tracker, baseline_result, sample_result):
        tracker.log(baseline_result)
        tracker.log(sample_result)
        best = tracker.get_best()
        assert best.experiment_id == "exp_001"

    def test_get_history(self, tracker, baseline_result, sample_result):
        tracker.log(baseline_result)
        tracker.log(sample_result)
        history = tracker.get_history()
        assert len(history) == 2

    def test_generate_report(self, tracker, sample_result, baseline_result):
        tracker.log(baseline_result)
        tracker.log(sample_result)
        report = tracker.generate_report(sample_result, baseline_result)
        assert "lr0=0.005" in report
        assert "IMPROVED" in report

    def test_generate_summary(self, tracker, baseline_result, sample_result):
        tracker.log(baseline_result)
        tracker.log(sample_result)
        summary = tracker.generate_summary()
        assert "Experiment Dashboard" in summary
        assert "exp_001" in summary

    def test_next_experiment_id(self, tracker):
        assert tracker.next_experiment_id() == "exp_000"

    def test_next_experiment_id_increments(self, tracker, baseline_result):
        tracker.log(baseline_result)
        assert tracker.next_experiment_id() == "exp_001"

    def test_next_experiment_id_respects_orphan_dirs(self, tracker):
        """ID generation accounts for dirs with missing/corrupt metrics."""
        orphan = tracker.experiments_dir / "exp_000_orphan"
        orphan.mkdir()
        assert tracker.next_experiment_id() == "exp_001"

    def test_next_experiment_id_respects_orphan_and_loaded(self, tracker, baseline_result):
        """ID generation uses max of both in-memory and filesystem."""
        tracker.log(baseline_result)  # exp_000
        orphan = tracker.experiments_dir / "exp_005_orphan"
        orphan.mkdir()
        assert tracker.next_experiment_id() == "exp_006"

    def test_get_baseline_returns_most_recent(self, tracker):
        """get_baseline returns the most recently logged baseline, not the first."""
        old_baseline = ExperimentResult(
            experiment_id="exp_000",
            name="baseline_old",
            overrides={},
            metrics={"mAP50": 0.50, "mAP50-95": 0.30, "precision": 0.60, "recall": 0.50},
            is_baseline=True,
        )
        new_baseline = ExperimentResult(
            experiment_id="exp_002",
            name="baseline_new",
            overrides={},
            metrics={"mAP50": 0.72, "mAP50-95": 0.45, "precision": 0.80, "recall": 0.68},
            is_baseline=True,
        )
        tracker.log(old_baseline)
        tracker.log(new_baseline)
        baseline = tracker.get_baseline()
        assert baseline is not None
        assert baseline.experiment_id == "exp_002"

    def test_tune_result_persists_through_log_load(self, tmp_path):
        """Tune fields survive a log-to-disk then load-from-disk round trip."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        tune_result = ExperimentResult(
            experiment_id="exp_001",
            name="tune_lr",
            overrides={"lr0": 0.005},
            metrics={"mAP50": 0.80, "mAP50-95": 0.52, "precision": 0.85, "recall": 0.75},
            is_tune=True,
            tune_iterations=20,
            tune_search_space={"lr0": [0.001, 0.01]},
            tune_best_params={"lr0": 0.005},
            tune_dir="/some/path",
        )
        tracker.log(tune_result)

        # Create a fresh tracker from the same directory
        tracker2 = ExperimentTracker(experiments_dir=tmp_path)
        loaded = tracker2.get_history()
        assert len(loaded) == 1
        r = loaded[0]
        assert r.is_tune is True
        assert r.tune_iterations == 20
        assert r.tune_search_space == {"lr0": [0.001, 0.01]}
        assert r.tune_best_params == {"lr0": 0.005}
        assert r.tune_dir == "/some/path"

    def test_tune_in_summary_output(self, tmp_path):
        """Tune results show 'tune(N)' in the Change column of the summary."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        baseline = ExperimentResult(
            experiment_id="exp_000",
            name="baseline",
            overrides={},
            metrics={"mAP50": 0.72, "mAP50-95": 0.45, "precision": 0.80, "recall": 0.68},
            is_baseline=True,
        )
        tune_result = ExperimentResult(
            experiment_id="exp_001",
            name="tune_lr",
            overrides={"lr0": 0.005},
            metrics={"mAP50": 0.80, "mAP50-95": 0.52, "precision": 0.85, "recall": 0.75},
            is_tune=True,
            tune_iterations=20,
            tune_search_space={"lr0": [0.001, 0.01]},
            tune_best_params={"lr0": 0.005},
        )
        tracker.log(baseline)
        tracker.log(tune_result)
        summary = tracker.generate_summary()
        assert "tune(20)" in summary

    def test_tune_in_report_output(self, tmp_path):
        """Report for a tune result includes tune summary section."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        baseline = ExperimentResult(
            experiment_id="exp_000",
            name="baseline",
            overrides={},
            metrics={"mAP50": 0.72, "mAP50-95": 0.45, "precision": 0.80, "recall": 0.68},
            is_baseline=True,
        )
        tune_result = ExperimentResult(
            experiment_id="exp_001",
            name="tune_lr",
            overrides={"lr0": 0.005},
            metrics={"mAP50": 0.80, "mAP50-95": 0.52, "precision": 0.85, "recall": 0.75},
            is_tune=True,
            tune_iterations=20,
            tune_search_space={"lr0": [0.001, 0.01]},
            tune_best_params={"lr0": 0.005},
        )
        tracker.log(baseline)
        tracker.log(tune_result)
        report = tracker.generate_report(tune_result, baseline)
        assert "Tune Summary" in report
        assert "20 iterations" in report
        assert "Search space" in report
        assert "Best params" in report


class TestSafeExperimentDirName:
    def test_sanitizes_special_chars(self):
        result = safe_experiment_dir_name("exp_000", "lr0=0.005")
        assert result == "exp_000_lr0_0.005"

    def test_truncates_long_names(self):
        long_name = "a" * 100
        result = safe_experiment_dir_name("exp_001", long_name)
        # The sanitized name portion is truncated to 50 chars
        # Result format: "exp_001_<name[:50]>"
        name_part = result[len("exp_001_"):]
        assert len(name_part) <= 50

    def test_empty_name_returns_just_id(self):
        result = safe_experiment_dir_name("exp_000", "")
        assert result == "exp_000"

    def test_preserves_word_chars(self):
        result = safe_experiment_dir_name("exp_001", "my_experiment-v2.0")
        assert result == "exp_001_my_experiment-v2.0"
