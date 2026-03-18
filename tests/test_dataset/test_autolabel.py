"""
Tests for yolocc.dataset.autolabel — pure-logic helpers only (no YOLO model).
"""

from pathlib import Path

import pytest
from PIL import Image

from yolocc.dataset.autolabel import (
    COCO_CLASSES,
    find_images,
    get_base_name,
    get_coco_overlap,
    is_augmented,
)


class TestGetBaseName:
    """Tests for the Roboflow suffix stripping helper."""

    def test_strips_roboflow_suffix(self):
        """A filename with '.rf.<32-hex>' should have the suffix removed."""
        filename = "0020005_jpg.rf.3f0ea87de18c8560dee2ddfdfb20dcce.jpg"
        assert get_base_name(filename) == "0020005_jpg.jpg"

    def test_normal_filename_unchanged(self):
        """A normal filename without '.rf.' should be returned as-is."""
        assert get_base_name("img001.jpg") == "img001.jpg"

    def test_preserves_extension(self):
        """The original extension should be preserved after stripping."""
        filename = "photo.rf.aaaabbbbccccdddd1111222233334444.png"
        assert get_base_name(filename) == "photo.png"

    def test_no_extension(self):
        """Filenames without extensions should also work."""
        assert get_base_name("image_no_ext") == "image_no_ext"


class TestIsAugmented:
    """Tests for Roboflow augmentation detection."""

    def test_detects_roboflow_augmented(self):
        """Filenames containing '.rf.<32-hex-chars>' should be detected."""
        filename = "0020005_jpg.rf.3f0ea87de18c8560dee2ddfdfb20dcce.jpg"
        assert is_augmented(filename) is True

    def test_normal_file_not_augmented(self):
        """Normal filenames should not be flagged as augmented."""
        assert is_augmented("img001.jpg") is False

    def test_partial_rf_not_matched(self):
        """'.rf.' followed by fewer than 32 hex chars should not match."""
        assert is_augmented("img.rf.abc123.jpg") is False

    def test_rf_in_directory_name_not_matched(self):
        """Only the filename is checked, not directory components."""
        # The function operates on filenames, not full paths
        assert is_augmented("normal_image.jpg") is False


class TestFindImages:
    """Tests for recursive image discovery."""

    def test_finds_images_in_nested_dirs(self, tmp_path: Path):
        """Images in nested subdirectories should all be found."""
        # Create nested directory structure
        (tmp_path / "subdir1").mkdir()
        (tmp_path / "subdir2" / "deep").mkdir(parents=True)

        # Create images in various locations
        Image.new("RGB", (10, 10)).save(tmp_path / "root.jpg")
        Image.new("RGB", (10, 10)).save(tmp_path / "subdir1" / "a.png")
        Image.new("RGB", (10, 10)).save(tmp_path / "subdir2" / "deep" / "b.jpeg")

        # Create a non-image file that should be ignored
        (tmp_path / "readme.txt").write_text("not an image")

        images = find_images(tmp_path)
        image_names = sorted(p.name for p in images)

        assert len(images) == 3
        assert "root.jpg" in image_names
        assert "a.png" in image_names
        assert "b.jpeg" in image_names

    def test_finds_uppercase_extensions(self, tmp_path: Path):
        """Images with uppercase extensions (e.g., .JPG) should also be found."""
        Image.new("RGB", (10, 10)).save(tmp_path / "photo_lower.jpg")
        # Create an uppercase extension file — use a distinct stem to avoid
        # case-insensitive filesystem collisions on Windows/macOS.
        Image.new("RGB", (10, 10)).save(tmp_path / "PHOTO_UPPER.JPG")

        images = find_images(tmp_path)
        assert len(images) == 2

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        """An empty directory should return an empty list."""
        images = find_images(tmp_path)
        assert images == []

    def test_all_supported_extensions(self, tmp_path: Path):
        """All supported image extensions should be found."""
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            Image.new("RGB", (10, 10)).save(tmp_path / f"test{ext}")

        images = find_images(tmp_path)
        assert len(images) == 5


class TestCOCOClasses:
    """Tests for COCO class list and overlap checking."""

    def test_coco_classes_has_80_entries(self):
        """The COCO class list should have exactly 80 entries."""
        assert len(COCO_CLASSES) == 80

    def test_get_coco_overlap_all_match(self):
        """All-COCO classes should return full overlap, empty non-overlap."""
        overlapping, non_overlapping = get_coco_overlap(["person", "car", "dog"])
        assert overlapping == ["person", "car", "dog"]
        assert non_overlapping == []

    def test_get_coco_overlap_none_match(self):
        """Custom classes should return empty overlap, full non-overlap."""
        overlapping, non_overlapping = get_coco_overlap(["smoke", "fire", "drone"])
        assert overlapping == []
        assert non_overlapping == ["smoke", "fire", "drone"]

    def test_get_coco_overlap_partial(self):
        """Mixed classes should split correctly between overlap and non-overlap."""
        overlapping, non_overlapping = get_coco_overlap(["person", "smoke", "car"])
        assert overlapping == ["person", "car"]
        assert non_overlapping == ["smoke"]

    def test_get_coco_overlap_case_insensitive(self):
        """COCO overlap matching should be case-insensitive."""
        overlapping, _ = get_coco_overlap(["Person", "CAR"])
        assert len(overlapping) == 2
