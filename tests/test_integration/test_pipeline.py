"""
End-to-end pipeline test — config → experiment runner → tracker → summary.
Mocks YOLO.train() so no GPU or real training is needed.
"""
from unittest.mock import patch, MagicMock

import pytest

from yolocc.experiment.runner import ExperimentRunner
from yolocc.project import ProjectConfig


def _make_mock_train_results():
    """Create a mock matching the fields runner.py actually accesses."""
    results = MagicMock()
    results.results_dict = {
        "metrics/mAP50(B)": 0.75,
        "metrics/mAP50-95(B)": 0.52,
        "metrics/precision(B)": 0.80,
        "metrics/recall(B)": 0.70,
        "epoch": 10,
    }
    results.maps = [0.55, 0.49]  # per-class mAP, 2 classes
    return results


@pytest.mark.integration
class TestPipelineEndToEnd:
    """Prove config → ExperimentRunner → tracker → summary connects."""

    def test_baseline_experiment_logs_and_generates_summary(self, tmp_path, yolo_dataset):
        """AC-005: full pipeline from config to summary.md."""
        config = ProjectConfig(
            name="e2e-test",
            classes={0: "cat", 1: "dog"},
            defaults={"base_model": "yolo11n.pt", "dataset": str(yolo_dataset), "imgsz": 640},
        )

        experiments_dir = tmp_path / "experiments"
        runner = ExperimentRunner(project_config=config, experiments_dir=experiments_dir)

        mock_model = MagicMock()
        mock_model.train.return_value = _make_mock_train_results()

        with patch("ultralytics.YOLO", return_value=mock_model):
            with patch(
                "yolocc.experiment.runner.prepare_ultralytics_data_yaml",
                return_value=(str(yolo_dataset / "data.yaml"), None),
            ):
                result = runner.run_experiment(
                    overrides={}, name="baseline", budget_epochs=10, patience=5
                )

        # Tracker logged 1 experiment
        history = runner.tracker.get_history()
        assert len(history) == 1

        # Result is a baseline with real metrics
        assert result.is_baseline is True
        assert result.primary_metric == 0.52
        assert result.metrics["mAP50"] == 0.75

        # Per-class AP extracted
        assert "cat" in result.per_class_ap
        assert "dog" in result.per_class_ap

        # summary.md was generated
        summary_path = experiments_dir / "summary.md"
        assert summary_path.exists()
        summary_text = summary_path.read_text()
        assert "baseline" in summary_text
        assert "0.5200" in summary_text

        # Experiment directory with metrics.yaml exists
        exp_dirs = list(experiments_dir.glob("exp_000_*"))
        assert len(exp_dirs) == 1
        assert (exp_dirs[0] / "metrics.yaml").exists()
