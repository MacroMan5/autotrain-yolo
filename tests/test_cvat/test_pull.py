"""Tests for CVAT pull — pure logic, no CVAT connection needed."""
import zipfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from yolocc.cvat.pull import pull_task, pull_project


class TestPullOutputPath:
    """Test that output paths are constructed correctly."""

    def test_default_output_for_task(self, tmp_path):
        """Default output dir should be datasets/cvat_task_<id>."""
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        # Make export_dataset create a valid zip
        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("data.yaml", yaml.safe_dump({"nc": 1, "names": {0: "obj"}}))

        mock_task.export_dataset.side_effect = fake_export

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: tmp_path / p,
                ):
                    result = pull_task(42)

        assert "cvat_task_42" in str(result)
        assert (result / "data.yaml").exists()

    def test_custom_output_dir(self, tmp_path):
        """Explicit output_dir should be used."""
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("data.yaml", yaml.safe_dump({"nc": 1, "names": {0: "a"}}))

        mock_task.export_dataset.side_effect = fake_export

        custom_out = str(tmp_path / "my_custom_output")

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p,
                ):
                    result = pull_task(42, output_dir=custom_out)

        assert "my_custom_output" in str(result)


class TestPullProject:
    """Test pull_project path construction."""

    def test_default_output_for_project(self, tmp_path):
        """Default output dir should be datasets/cvat_project_<id>."""
        mock_client = MagicMock()
        mock_project = MagicMock()
        mock_client.projects.retrieve.return_value = mock_project

        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("data.yaml", yaml.safe_dump({"nc": 2, "names": {0: "a", 1: "b"}}))

        mock_project.export_dataset.side_effect = fake_export

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: tmp_path / p,
                ):
                    result = pull_project(7)

        assert "cvat_project_7" in str(result)


class TestPullExtraction:
    """Test that ZIP extraction works correctly."""

    def test_extracts_zip_and_removes_it(self, tmp_path):
        """After pull, the zip should be extracted and deleted."""
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("data.yaml", yaml.safe_dump({"nc": 1, "names": {0: "obj"}}))
                zf.writestr("images/train/img_001.jpg", b"fake image data")
                zf.writestr("labels/train/img_001.txt", "0 0.5 0.5 0.3 0.2\n")

        mock_task.export_dataset.side_effect = fake_export

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: tmp_path / p,
                ):
                    result = pull_task(1)

        # ZIP should be removed
        assert not (result / "dataset.zip").exists()
        # Files should be extracted
        assert (result / "data.yaml").exists()
        assert (result / "images" / "train" / "img_001.jpg").exists()
        assert (result / "labels" / "train" / "img_001.txt").exists()

    def test_warns_when_no_data_yaml(self, tmp_path, capsys):
        """Should print a warning if exported zip has no data.yaml."""
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("images/img.jpg", b"fake")

        mock_task.export_dataset.side_effect = fake_export

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: tmp_path / p,
                ):
                    pull_task(99)

        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "data.yaml" in captured.out


class TestPullExportFormat:
    """Verify the correct export format is requested."""

    def test_requests_ultralytics_format(self, tmp_path):
        """Should request 'Ultralytics YOLO Detection 1.0' format."""
        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        def fake_export(format_name, filename, include_images):
            zp = Path(filename)
            zp.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("data.yaml", "nc: 1\nnames:\n  0: x\n")

        mock_task.export_dataset.side_effect = fake_export

        with patch("yolocc.cvat.pull.require_cvat"):
            with patch("yolocc.cvat.pull.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.pull.resolve_workspace_path",
                    side_effect=lambda p: tmp_path / p,
                ):
                    pull_task(1)

        mock_task.export_dataset.assert_called_once()
        call_kwargs = mock_task.export_dataset.call_args
        assert call_kwargs[1]["format_name"] == "Ultralytics YOLO Detection 1.0" or \
            call_kwargs.kwargs.get("format_name") == "Ultralytics YOLO Detection 1.0"
