"""
Integration test for multi-experiment flow.

Mock-based — no GPU or real training needed.
Tests: multi-experiment pipeline, architecture swap, budget counting, tune fields.
"""

from unittest.mock import patch, MagicMock

import pytest

from yolocc.experiment.runner import ExperimentRunner
from yolocc.experiment.tracker import ExperimentResult, ExperimentTracker
from yolocc.project import ProjectConfig


def _make_mock_train_results(mAP50=0.75, mAP50_95=0.52, precision=0.80, recall=0.70, epoch=10):
    """Create a mock matching the fields runner.py actually accesses."""
    results = MagicMock()
    results.results_dict = {
        "metrics/mAP50(B)": mAP50,
        "metrics/mAP50-95(B)": mAP50_95,
        "metrics/precision(B)": precision,
        "metrics/recall(B)": recall,
        "epoch": epoch,
    }
    results.maps = [mAP50_95 + 0.03, mAP50_95 - 0.03]  # 2 classes
    return results


def _make_config(dataset_path):
    return ProjectConfig(
        name="flow-test",
        classes={0: "cat", 1: "dog"},
        defaults={"base_model": "yolo11n.pt", "dataset": str(dataset_path), "imgsz": 640},
    )


@pytest.mark.integration
class TestMultiExperimentFlow:
    """Three-experiment flow: baseline -> HP change -> augmentation."""

    def test_three_experiment_pipeline(self, tmp_path, yolo_dataset):
        config = _make_config(yolo_dataset)
        experiments_dir = tmp_path / "experiments"
        runner = ExperimentRunner(project_config=config, experiments_dir=experiments_dir)

        mock_model = MagicMock()

        # Three experiments with increasing mAP
        results_sequence = [
            _make_mock_train_results(mAP50_95=0.45),  # baseline
            _make_mock_train_results(mAP50_95=0.52),  # HP change
            _make_mock_train_results(mAP50_95=0.55),  # augmentation
        ]
        mock_model.train.side_effect = results_sequence

        with patch("ultralytics.YOLO", return_value=mock_model):
            with patch(
                "yolocc.experiment.runner.prepare_ultralytics_data_yaml",
                return_value=(str(yolo_dataset / "data.yaml"), None),
            ):
                r1 = runner.run_experiment(overrides={}, name="baseline", budget_epochs=10)
                r2 = runner.run_experiment(overrides={"lr0": 0.005}, name="lr0=0.005", budget_epochs=10)
                r3 = runner.run_experiment(overrides={"mosaic": 0.8}, name="mosaic=0.8", budget_epochs=10)

        # All 3 logged
        history = runner.tracker.get_history()
        assert len(history) == 3

        # Best is the last one (highest mAP)
        best = runner.tracker.get_best()
        assert best.experiment_id == "exp_002"
        assert best.primary_metric == 0.55

        # Baseline correctly identified
        assert r1.is_baseline is True
        assert r2.is_baseline is False

        # Summary exists
        summary_path = experiments_dir / "summary.md"
        assert summary_path.exists()

        # experiment_count works
        assert runner.tracker.experiment_count == 3



@pytest.mark.integration
class TestArchitectureSwapFlow:
    """Architecture swap passes YAML config to YOLO constructor."""

    def test_architecture_override_uses_yaml_load(self, tmp_path, yolo_dataset):
        config = _make_config(yolo_dataset)
        experiments_dir = tmp_path / "experiments"
        runner = ExperimentRunner(project_config=config, experiments_dir=experiments_dir)

        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = _make_mock_train_results()
        mock_model_instance.load.return_value = mock_model_instance

        mock_yolo_class = MagicMock(return_value=mock_model_instance)

        with patch("ultralytics.YOLO", mock_yolo_class):
            with patch(
                "yolocc.experiment.runner.prepare_ultralytics_data_yaml",
                return_value=(str(yolo_dataset / "data.yaml"), None),
            ):
                # Create a fake architecture YAML
                arch_yaml = tmp_path / "yolo11-p2.yaml"
                arch_yaml.write_text("# fake config")

                result = runner.run_experiment(
                    overrides={"model": str(arch_yaml)},
                    name="arch_swap",
                    budget_epochs=10,
                )

        # YOLO was called with the YAML path (architecture config)
        first_call_arg = str(mock_yolo_class.call_args_list[0][0][0])
        assert first_call_arg.endswith(".yaml")

        # .load() was called on the returned model (loading weights)
        mock_model_instance.load.assert_called_once()

        # Result records the architecture config
        assert result.architecture_config == str(arch_yaml)

    def test_resume_from_recorded_in_result(self, tmp_path, yolo_dataset):
        config = _make_config(yolo_dataset)
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
                    overrides={"lr0": 0.005},
                    name="from_checkpoint",
                    budget_epochs=10,
                    resume_from="experiments/exp_003/train/weights/best.pt",
                )

        assert result.resume_from == "experiments/exp_003/train/weights/best.pt"



class TestBudgetCounting:
    """experiment_count property matches logged count."""

    def test_empty_tracker_count_zero(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        assert tracker.experiment_count == 0

    def test_count_matches_logged(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        for i in range(5):
            tracker.log(ExperimentResult(
                experiment_id=f"exp_{i:03d}",
                name=f"exp_{i}",
                overrides={},
                metrics={"mAP50-95": 0.50 + i * 0.01},
            ))
        assert tracker.experiment_count == 5



class TestNewFields:
    """Verify architecture_config and resume_from persist through log/load cycle."""

    def test_architecture_config_persists(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        tracker.log(ExperimentResult(
            experiment_id="exp_000",
            name="arch_test",
            overrides={"model": "configs/architectures/yolo11-p2.yaml"},
            metrics={"mAP50-95": 0.50},
            architecture_config="configs/architectures/yolo11-p2.yaml",
        ))

        # Reload from disk
        tracker2 = ExperimentTracker(experiments_dir=tmp_path)
        loaded = tracker2.get_history()[0]
        assert loaded.architecture_config == "configs/architectures/yolo11-p2.yaml"

    def test_resume_from_persists(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        tracker.log(ExperimentResult(
            experiment_id="exp_001",
            name="resume_test",
            overrides={"lr0": 0.005},
            metrics={"mAP50-95": 0.55},
            resume_from="experiments/exp_000/train/weights/best.pt",
        ))

        # Reload from disk
        tracker2 = ExperimentTracker(experiments_dir=tmp_path)
        loaded = tracker2.get_history()[0]
        assert loaded.resume_from == "experiments/exp_000/train/weights/best.pt"

    def test_report_shows_architecture(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        result = ExperimentResult(
            experiment_id="exp_000",
            name="arch_test",
            overrides={},
            metrics={"mAP50-95": 0.50},
            architecture_config="configs/architectures/yolo11-p2.yaml",
            is_baseline=True,
        )
        tracker.log(result)
        report = tracker.generate_report(result)
        assert "yolo11-p2.yaml" in report
