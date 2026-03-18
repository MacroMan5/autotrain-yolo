"""
Smoke tests — verify core components can be instantiated and wired up
without running actual YOLO training.
"""

from pathlib import Path

import pytest

from yolocc.experiment.runner import ExperimentRunner
from yolocc.experiment.tracker import ExperimentResult, ExperimentTracker
from yolocc.project import ProjectConfig


@pytest.mark.integration
@pytest.mark.slow
class TestExperimentRunnerSmoke:
    """Smoke tests for ExperimentRunner initialization."""

    def test_runner_initialization(self, tmp_path, yolo_dataset):
        """ExperimentRunner can be created with a valid ProjectConfig."""
        config = ProjectConfig(
            name="test",
            classes={0: "cat", 1: "dog"},
            defaults={"base_model": "yolo11n.pt", "dataset": str(yolo_dataset)},
        )
        runner = ExperimentRunner(project_config=config, experiments_dir=tmp_path)
        assert runner.config.name == "test"
        assert runner.tracker is not None

    def test_runner_config_classes(self, tmp_path, yolo_dataset):
        """Runner should inherit class mapping from project config."""
        config = ProjectConfig(
            name="smoke-test",
            classes={0: "cat", 1: "dog"},
            defaults={"base_model": "yolo11n.pt", "dataset": str(yolo_dataset)},
        )
        runner = ExperimentRunner(project_config=config, experiments_dir=tmp_path)
        assert runner.config.num_classes == 2
        assert runner.config.class_names == ["cat", "dog"]

    def test_runner_tracker_experiments_dir(self, tmp_path, yolo_dataset):
        """Tracker should use the experiments_dir provided to the runner."""
        config = ProjectConfig(
            name="test",
            classes={0: "cat", 1: "dog"},
            defaults={},
        )
        runner = ExperimentRunner(project_config=config, experiments_dir=tmp_path)
        assert runner.tracker.experiments_dir == tmp_path


@pytest.mark.integration
@pytest.mark.slow
class TestTrackerSmoke:
    """Smoke tests for ExperimentTracker on empty state."""

    def test_summary_on_empty_state(self, tmp_path):
        """generate_summary should work with no experiments logged."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        summary = tracker.generate_summary()
        assert "No experiments logged yet." in summary

    def test_get_best_on_empty_state(self, tmp_path):
        """get_best should return None when no experiments exist."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        assert tracker.get_best() is None

    def test_get_baseline_on_empty_state(self, tmp_path):
        """get_baseline should return None when no experiments exist."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        assert tracker.get_baseline() is None

    def test_next_experiment_id_starts_at_zero(self, tmp_path):
        """First experiment ID should be exp_000."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        assert tracker.next_experiment_id() == "exp_000"

    def test_log_and_retrieve(self, tmp_path):
        """Logging a result should make it retrievable via get_history."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        result = ExperimentResult(
            experiment_id="exp_000",
            name="baseline",
            overrides={},
            metrics={"mAP50": 0.5, "mAP50-95": 0.35, "precision": 0.6, "recall": 0.5},
            is_baseline=True,
        )
        tracker.log(result)
        history = tracker.get_history()
        assert len(history) == 1
        assert history[0].name == "baseline"
        assert history[0].primary_metric == 0.35

    def test_summary_after_logging(self, tmp_path):
        """Summary should include experiment data after logging."""
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        result = ExperimentResult(
            experiment_id="exp_000",
            name="baseline",
            overrides={},
            metrics={"mAP50": 0.5, "mAP50-95": 0.35, "precision": 0.6, "recall": 0.5},
            is_baseline=True,
        )
        tracker.log(result)
        summary = tracker.generate_summary()
        assert "baseline" in summary
        assert "0.3500" in summary
