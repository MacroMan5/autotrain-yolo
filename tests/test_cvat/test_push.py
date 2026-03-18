"""Tests for CVAT push — pure logic, no CVAT connection needed."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from yolocc.cvat.push import (
    IMAGE_EXTENSIONS,
    push_task,
    push_from_analysis,
)


class TestImageExtensions:
    """Validate the IMAGE_EXTENSIONS constant."""

    def test_common_extensions_present(self):
        """Should include common image formats."""
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS

    def test_extensions_are_lowercase(self):
        """All extensions should be lowercase."""
        for ext in IMAGE_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")


class TestPushTaskImageCollection:
    """Test image file collection logic."""

    def test_collects_images_from_directory(self, tmp_path):
        """Should find all image files in the directory."""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "a.jpg").write_bytes(b"fake")
        (images_dir / "b.png").write_bytes(b"fake")
        (images_dir / "c.txt").write_text("not an image")

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = 42
        mock_client.tasks.create_from_data.return_value = mock_task

        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=mock_client):
                with patch("yolocc.cvat.push.get_cvat_config", return_value={}):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        side_effect=lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p,
                    ):
                        task_id = push_task(str(images_dir))

        assert task_id == 42
        # Verify create_from_data was called with 2 image files (a.jpg, b.png), not the .txt
        call_args = mock_client.tasks.create_from_data.call_args
        resources = call_args.kwargs.get("resources") or call_args[1].get("resources")
        assert len(resources) == 2
        resource_names = [Path(r).name for r in resources]
        assert "a.jpg" in resource_names
        assert "b.png" in resource_names
        assert "c.txt" not in resource_names

    def test_raises_on_missing_directory(self, tmp_path):
        """Should raise FileNotFoundError for non-existent image dir."""
        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=MagicMock()):
                with patch("yolocc.cvat.push.get_cvat_config", return_value={}):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        return_value=tmp_path / "nonexistent",
                    ):
                        with pytest.raises(FileNotFoundError, match="not found"):
                            push_task("nonexistent")

    def test_raises_on_empty_directory(self, tmp_path):
        """Should raise FileNotFoundError when no images found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=MagicMock()):
                with patch("yolocc.cvat.push.get_cvat_config", return_value={}):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        return_value=empty_dir,
                    ):
                        with pytest.raises(FileNotFoundError, match="No images"):
                            push_task(str(empty_dir))


class TestPushTaskName:
    """Test task naming and project assignment."""

    def test_custom_task_name(self, tmp_path):
        """Task name should be passed to CVAT."""
        images_dir = tmp_path / "imgs"
        images_dir.mkdir()
        (images_dir / "test.jpg").write_bytes(b"fake")

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = 1
        mock_client.tasks.create_from_data.return_value = mock_task

        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=mock_client):
                with patch("yolocc.cvat.push.get_cvat_config", return_value={}):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        side_effect=lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p,
                    ):
                        push_task(str(images_dir), task_name="My Custom Task")

        call_args = mock_client.tasks.create_from_data.call_args
        spec = call_args.kwargs.get("spec") or call_args[1].get("spec")
        assert spec["name"] == "My Custom Task"

    def test_project_id_from_config(self, tmp_path):
        """Should use project_id from CVAT config when not explicitly provided."""
        images_dir = tmp_path / "imgs"
        images_dir.mkdir()
        (images_dir / "test.png").write_bytes(b"fake")

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = 5
        mock_client.tasks.create_from_data.return_value = mock_task

        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.push.get_cvat_config",
                    return_value={"project_id": 99, "url": "http://test:8080"},
                ):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        side_effect=lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p,
                    ):
                        push_task(str(images_dir))

        call_args = mock_client.tasks.create_from_data.call_args
        spec = call_args.kwargs.get("spec") or call_args[1].get("spec")
        assert spec["project_id"] == 99

    def test_explicit_project_id_overrides_config(self, tmp_path):
        """Explicitly passed project_id should override config."""
        images_dir = tmp_path / "imgs"
        images_dir.mkdir()
        (images_dir / "test.jpg").write_bytes(b"fake")

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = 10
        mock_client.tasks.create_from_data.return_value = mock_task

        with patch("yolocc.cvat.push.require_cvat"):
            with patch("yolocc.cvat.push.get_client", return_value=mock_client):
                with patch(
                    "yolocc.cvat.push.get_cvat_config",
                    return_value={"project_id": 99},
                ):
                    with patch(
                        "yolocc.cvat.push.resolve_workspace_path",
                        side_effect=lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p,
                    ):
                        push_task(str(images_dir), project_id=55)

        call_args = mock_client.tasks.create_from_data.call_args
        spec = call_args.kwargs.get("spec") or call_args[1].get("spec")
        assert spec["project_id"] == 55


class TestPushFromAnalysis:
    """Test the push_from_analysis batching logic."""

    def test_raises_on_missing_analysis_file(self, tmp_path):
        """Should raise FileNotFoundError when analysis file doesn't exist."""
        with patch("yolocc.cvat.push.require_cvat"):
            with patch(
                "yolocc.cvat.push.resolve_workspace_path",
                return_value=tmp_path / "nonexistent.txt",
            ):
                with pytest.raises(FileNotFoundError, match="Analysis file"):
                    push_from_analysis("nonexistent.txt")

    def test_empty_analysis_returns_empty_list(self, tmp_path):
        """Analysis file with no valid paths should return empty list."""
        analysis_file = tmp_path / "empty_analysis.txt"
        analysis_file.write_text("# Just comments\n# No real paths\n")

        with patch("yolocc.cvat.push.require_cvat"):
            with patch(
                "yolocc.cvat.push.resolve_workspace_path",
                return_value=analysis_file,
            ):
                result = push_from_analysis(str(analysis_file))

        assert result == []

    def test_skips_nonexistent_images(self, tmp_path):
        """Should skip image paths that don't actually exist."""
        analysis_file = tmp_path / "analysis.txt"
        analysis_file.write_text(
            "/nonexistent/path/a.jpg\n"
            "/nonexistent/path/b.jpg\n"
        )

        with patch("yolocc.cvat.push.require_cvat"):
            with patch(
                "yolocc.cvat.push.resolve_workspace_path",
                return_value=analysis_file,
            ):
                result = push_from_analysis(str(analysis_file))

        assert result == []

    def test_batches_images(self, tmp_path):
        """Should split images into batches of max_per_task."""
        # Create 5 real image files
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        image_paths = []
        for i in range(5):
            img = images_dir / f"img_{i:03d}.jpg"
            img.write_bytes(b"fake")
            image_paths.append(str(img))

        analysis_file = tmp_path / "analysis.txt"
        analysis_file.write_text("\n".join(image_paths) + "\n")

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.id = 100
        mock_client.tasks.create_from_data.return_value = mock_task

        with patch("yolocc.cvat.push.require_cvat"):
            with patch(
                "yolocc.cvat.push.resolve_workspace_path",
                side_effect=lambda p: Path(p) if Path(p).is_absolute() else analysis_file,
            ):
                with patch("yolocc.cvat.push.get_client", return_value=mock_client):
                    with patch("yolocc.cvat.push.get_cvat_config", return_value={}):
                        result = push_from_analysis(
                            str(analysis_file),
                            max_per_task=2,
                        )

        # 5 images with max_per_task=2 -> 3 batches (2+2+1)
        assert len(result) == 3

    def test_skips_comment_lines(self, tmp_path):
        """Lines starting with # should be ignored."""
        img = tmp_path / "test.jpg"
        img.write_bytes(b"fake")

        analysis_file = tmp_path / "analysis.txt"
        analysis_file.write_text(f"# This is a comment\n{img}\n# Another comment\n")

        # Mock push_task to avoid needing the full CVAT client pipeline
        with patch("yolocc.cvat.push.require_cvat"):
            with patch(
                "yolocc.cvat.push.resolve_workspace_path",
                return_value=analysis_file,
            ):
                with patch("yolocc.cvat.push.push_task", return_value=42) as mock_push:
                    result = push_from_analysis(str(analysis_file))

        # Only 1 real image (comments skipped), so 1 task
        assert len(result) == 1
        # push_task should have been called once
        mock_push.assert_called_once()


class TestUploadAnnotations:
    """Test pre-annotation upload logic."""

    def test_label_file_matching(self, tmp_path):
        """_upload_annotations should match image stems to label files."""
        from yolocc.cvat.push import _upload_annotations

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.3 0.2\n")
        (labels_dir / "img_002.txt").write_text("1 0.3 0.3 0.2 0.2\n")

        image_files = [
            tmp_path / "img_001.jpg",
            tmp_path / "img_002.jpg",
            tmp_path / "img_003.jpg",  # No matching label
        ]

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        _upload_annotations(mock_client, 42, labels_dir, image_files)

        # Should have called import_annotations because 2 labels matched
        mock_task.import_annotations.assert_called_once()

    def test_no_matching_labels_skips_upload(self, tmp_path, capsys):
        """When no labels match, should skip the upload."""
        from yolocc.cvat.push import _upload_annotations

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # No label files at all

        image_files = [tmp_path / "img_001.jpg"]

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_client.tasks.retrieve.return_value = mock_task

        _upload_annotations(mock_client, 42, labels_dir, image_files)

        # Should NOT have called import_annotations
        mock_task.import_annotations.assert_not_called()
        captured = capsys.readouterr()
        assert "No matching label" in captured.out or "without pre-annotations" in captured.out

    def test_zip_structure_matches_yolo_1_1_format(self, tmp_path):
        """AC-006: verify the zip content _upload_annotations creates."""
        import zipfile
        from yolocc.cvat.push import _upload_annotations

        # Create label files
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_content_1 = "0 0.5 0.5 0.3 0.2\n"
        label_content_2 = "1 0.3 0.3 0.2 0.2\n"
        (labels_dir / "img_001.txt").write_bytes(label_content_1.encode())
        (labels_dir / "img_002.txt").write_bytes(label_content_2.encode())

        image_files = [
            tmp_path / "img_001.jpg",
            tmp_path / "img_002.jpg",
        ]

        # Capture the zip file path when import_annotations is called
        captured_zip = {}

        def capture_import(format_name, filename):
            # Read zip content before it gets deleted in the finally block
            with zipfile.ZipFile(filename, "r") as zf:
                captured_zip["namelist"] = zf.namelist()
                captured_zip["train_txt"] = zf.read("train.txt").decode()
                captured_zip["label_1"] = zf.read("obj_train_data/img_001.txt").decode()
                captured_zip["label_2"] = zf.read("obj_train_data/img_002.txt").decode()
            captured_zip["format"] = format_name

        mock_client = MagicMock()
        mock_task = MagicMock()
        mock_task.import_annotations.side_effect = capture_import
        mock_client.tasks.retrieve.return_value = mock_task

        _upload_annotations(mock_client, 42, labels_dir, image_files)

        # Verify zip structure
        assert "train.txt" in captured_zip["namelist"]
        assert "obj_train_data/img_001.txt" in captured_zip["namelist"]
        assert "obj_train_data/img_002.txt" in captured_zip["namelist"]

        # Verify train.txt lists image paths
        train_lines = captured_zip["train_txt"].strip().split("\n")
        assert len(train_lines) == 2
        assert "obj_train_data/img_001.jpg" in train_lines[0]
        assert "obj_train_data/img_002.jpg" in train_lines[1]

        # Verify label content preserved (YOLO bbox format)
        assert captured_zip["label_1"] == label_content_1
        assert captured_zip["label_2"] == label_content_2

        # Verify format name
        assert captured_zip["format"] == "YOLO 1.1"

        # TODO: CVAT YOLO 1.1 importer may require obj.data and obj.names
        # If CVAT import fails on real instance, this is the likely cause.
        # File as separate fix — don't scope-creep this test.
        assert "obj.data" not in captured_zip["namelist"]
        assert "obj.names" not in captured_zip["namelist"]
