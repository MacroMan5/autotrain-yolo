"""Tests for training utilities."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolocc.training.utils import (
    check_gpu,
    copy_model_safe,
    get_device,
    save_training_summary,
    validate_checkpoint,
    EXIT_CHECKPOINT_NOT_FOUND,
)


class TestGetDevice:
    def test_returns_string(self):
        device = get_device()
        assert isinstance(device, str)

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("0", "cpu")


class TestCheckGpu:
    def test_check_gpu_returns_bool(self):
        result = check_gpu(warn_only=True)
        assert isinstance(result, bool)


class TestValidateCheckpoint:
    def test_validate_checkpoint_missing_file_exits(self, tmp_path):
        missing = tmp_path / "nonexistent.pt"
        with pytest.raises(SystemExit) as exc_info:
            validate_checkpoint(str(missing))
        assert exc_info.value.code == EXIT_CHECKPOINT_NOT_FOUND

    def test_validate_checkpoint_existing_file_ok(self, tmp_path):
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake model data")
        # Should not raise
        validate_checkpoint(str(checkpoint))


class TestCopyModelSafe:
    def test_copy_model_safe_success(self, tmp_path):
        src = tmp_path / "source.pt"
        src.write_bytes(b"model weights")
        dst = tmp_path / "dest" / "model.pt"

        result = copy_model_safe(src, dst)

        assert result is True
        assert dst.exists()
        assert dst.read_bytes() == b"model weights"

    def test_copy_model_safe_missing_source(self, tmp_path):
        src = tmp_path / "missing.pt"
        dst = tmp_path / "dest.pt"

        result = copy_model_safe(src, dst)

        assert result is False
        assert not dst.exists()


class TestSaveTrainingSummary:
    def _make_results(self, metrics=None):
        """Create a mock results object with a results_dict attribute."""
        if metrics is None:
            metrics = {
                "metrics/mAP50(B)": 0.85,
                "metrics/mAP50-95(B)": 0.60,
                "metrics/precision(B)": 0.90,
                "metrics/recall(B)": 0.80,
                "epoch": 50,
            }
        return SimpleNamespace(results_dict=metrics)

    def test_save_training_summary_creates_json(self, tmp_path):
        # Create a fake model file so the summary can read its size
        model_file = tmp_path / "best.pt"
        model_file.write_bytes(b"\x00" * 1024)

        output_dir = tmp_path / "reports"
        results = self._make_results()

        summary = save_training_summary(
            results,
            model_path=str(model_file),
            output_dir=str(output_dir),
        )

        # Verify the return value
        assert summary["metrics"]["mAP50"] == 0.85
        assert summary["metrics"]["mAP50-95"] == 0.60
        assert summary["metrics"]["precision"] == 0.90
        assert summary["metrics"]["recall"] == 0.80

        # Verify the JSON file was created
        json_path = output_dir / "best_metrics.json"
        assert json_path.exists()

        with open(json_path) as f:
            saved = json.load(f)
        assert saved["metrics"]["mAP50"] == 0.85
        assert saved["training"]["epochs"] == 50
        assert "timestamp" in saved

    def test_save_training_summary_with_variant(self, tmp_path):
        model_file = tmp_path / "variant.pt"
        model_file.write_bytes(b"\x00" * 512)

        output_dir = tmp_path / "reports"
        results = self._make_results()

        summary = save_training_summary(
            results,
            model_path=str(model_file),
            output_dir=str(output_dir),
            variant="indoor",
        )

        assert summary["variant"] == "indoor"

    def test_save_training_summary_no_results_dict(self, tmp_path):
        """A results object without results_dict should still produce output."""
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"\x00" * 256)

        output_dir = tmp_path / "reports"
        # Object without results_dict attribute
        results = SimpleNamespace()

        summary = save_training_summary(
            results,
            model_path=str(model_file),
            output_dir=str(output_dir),
        )

        # Metrics should default to 0
        assert summary["metrics"]["mAP50"] == 0
        assert summary["metrics"]["mAP50-95"] == 0
