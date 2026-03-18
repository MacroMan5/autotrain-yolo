"""
Tests for yolocc.export.onnx — pure-logic helpers only (no real YOLO model).
"""

from pathlib import Path

import pytest

from yolocc.export.onnx import (
    DEFAULT_IMG_SIZE,
    _get_default_export_dir,
    export_onnx,
)


class TestExportOnnx:
    """Tests for the ONNX export function."""

    def test_missing_model_returns_none(self, tmp_path: Path):
        """A nonexistent model path should return None (not raise)."""
        result = export_onnx(
            model_path=str(tmp_path / "nonexistent_model.pt"),
            export_dir=tmp_path / "exports",
        )
        assert result is None

    def test_missing_model_does_not_create_export_dir(self, tmp_path: Path):
        """When the model is missing, the export dir should not be created."""
        export_dir = tmp_path / "exports"
        export_onnx(
            model_path=str(tmp_path / "nonexistent_model.pt"),
            export_dir=export_dir,
        )
        # The function returns early before mkdir, so exports/ should not exist
        assert not export_dir.exists()


class TestExportDefaults:
    """Tests for export module defaults and helpers."""

    def test_default_img_size_is_640(self):
        """The default image size should be 640."""
        assert DEFAULT_IMG_SIZE == 640

    def test_get_default_export_dir_returns_path(self):
        """_get_default_export_dir should return a Path ending with 'exports'."""
        result = _get_default_export_dir()
        assert isinstance(result, Path)
        assert result.name == "exports"
