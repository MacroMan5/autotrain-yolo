"""
Tests for yolocc.training.analyzer — TrainingAnalyzer init and validation.
"""

import pytest

from yolocc.training.analyzer import TrainingAnalyzer


class TestTrainingAnalyzerInit:
    """Tests for TrainingAnalyzer initialization and threshold validation."""

    def test_default_thresholds(self, tmp_path):
        """TrainingAnalyzer should initialize with documented defaults."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
        )
        assert analyzer.low_conf == 0.4
        assert analyzer.high_conf == 0.65
        assert analyzer.model is None  # lazily loaded
        assert analyzer.results["total_images"] == 0
        assert analyzer.results["total_detections"] == 0

    def test_custom_thresholds(self, tmp_path):
        """Custom thresholds should be stored correctly."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
            low_conf_threshold=0.3,
            high_conf_threshold=0.7,
        )
        assert analyzer.low_conf == 0.3
        assert analyzer.high_conf == 0.7

    def test_low_conf_equals_high_conf_raises(self, tmp_path):
        """low_conf_threshold == high_conf_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            TrainingAnalyzer(
                model_path=str(tmp_path / "model.pt"),
                dataset_path=str(tmp_path / "dataset"),
                low_conf_threshold=0.5,
                high_conf_threshold=0.5,
            )

    def test_low_conf_greater_than_high_conf_raises(self, tmp_path):
        """low_conf_threshold > high_conf_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="must be less than"):
            TrainingAnalyzer(
                model_path=str(tmp_path / "model.pt"),
                dataset_path=str(tmp_path / "dataset"),
                low_conf_threshold=0.8,
                high_conf_threshold=0.3,
            )

    def test_negative_threshold_raises(self, tmp_path):
        """Negative thresholds should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            TrainingAnalyzer(
                model_path=str(tmp_path / "model.pt"),
                dataset_path=str(tmp_path / "dataset"),
                low_conf_threshold=-0.1,
                high_conf_threshold=0.65,
            )

    def test_threshold_above_one_raises(self, tmp_path):
        """Thresholds above 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            TrainingAnalyzer(
                model_path=str(tmp_path / "model.pt"),
                dataset_path=str(tmp_path / "dataset"),
                low_conf_threshold=0.4,
                high_conf_threshold=1.5,
            )

    def test_output_dir_none_disables_saving(self, tmp_path):
        """Setting output_dir=None should disable report saving."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
            output_dir=None,
        )
        assert analyzer.output_dir is None

    def test_medium_high_boundary_default_thresholds(self, tmp_path):
        """medium_high_boundary should be computed correctly with defaults (0.4, 0.65) -> 0.825."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
        )
        assert analyzer.medium_high_boundary == pytest.approx(0.825)

    def test_medium_high_boundary_custom_thresholds(self, tmp_path):
        """medium_high_boundary with custom thresholds (0.3, 0.85) -> 0.925."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
            low_conf_threshold=0.3,
            high_conf_threshold=0.85,
        )
        assert analyzer.medium_high_boundary == pytest.approx(0.925)

    def test_paths_are_stored_as_path_objects(self, tmp_path):
        """model_path and dataset_path should be stored as Path objects."""
        analyzer = TrainingAnalyzer(
            model_path=str(tmp_path / "model.pt"),
            dataset_path=str(tmp_path / "dataset"),
        )
        from pathlib import Path
        assert isinstance(analyzer.model_path, Path)
        assert isinstance(analyzer.dataset_path, Path)
