"""
Tests for yolocc.dataset.cleaner
"""

from pathlib import Path

import pytest
import numpy as np
from PIL import Image


# The cleaner module imports cv2 at module level, so guard the entire test
# module behind an import check.
cv2 = pytest.importorskip("cv2")

from yolocc.dataset.cleaner import clean_dataset, dhash, find_label_path


def _make_image(path: Path, color: tuple, size: tuple = (32, 32)) -> None:
    """Create a small PNG image with PIL."""
    img = Image.new("RGB", size, color=color)
    img.save(path)


class TestDhash:
    """Tests for the dhash function."""

    def test_dhash_consistent(self, tmp_path: Path):
        """Same image content should always produce the same hash."""
        img_path = tmp_path / "img.png"
        _make_image(img_path, (100, 150, 200))

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h1 = dhash(img)
        h2 = dhash(img)
        assert h1 == h2

    def test_dhash_different_images(self, tmp_path: Path):
        """Visually different images should produce different hashes."""
        # Create two images with very different patterns to ensure different hashes
        # Use numpy to create images with strong horizontal gradients
        img_a = np.zeros((32, 32), dtype=np.uint8)
        img_a[:, :16] = 0
        img_a[:, 16:] = 255

        img_b = np.zeros((32, 32), dtype=np.uint8)
        img_b[:16, :] = 0
        img_b[16:, :] = 255

        h_a = dhash(img_a)
        h_b = dhash(img_b)
        assert h_a != h_b


class TestFindLabelPath:
    """Tests for finding corresponding label files."""

    def test_find_label_path_standard_structure(self, tmp_path: Path):
        """Should find labels in standard YOLO images/train -> labels/train structure."""
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)

        img_path = tmp_path / "images" / "train" / "photo.png"
        lbl_path = tmp_path / "labels" / "train" / "photo.txt"
        _make_image(img_path, (100, 100, 100))
        lbl_path.write_text("0 0.5 0.5 0.3 0.2\n")

        found = find_label_path(img_path)
        assert found is not None
        assert found == lbl_path

    def test_find_label_path_missing_returns_none(self, tmp_path: Path):
        """Should return None when no label file exists."""
        (tmp_path / "images" / "train").mkdir(parents=True)
        img_path = tmp_path / "images" / "train" / "orphan.png"
        _make_image(img_path, (50, 50, 50))

        found = find_label_path(img_path)
        assert found is None


class TestCleanDataset:
    """Tests for the clean_dataset function."""

    def test_dry_run_no_deletions(self, tmp_path: Path):
        """In dry-run mode, no files should be deleted."""
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)

        # Image with empty label (would normally be removed)
        img_path = tmp_path / "images" / "train" / "empty_lbl.png"
        lbl_path = tmp_path / "labels" / "train" / "empty_lbl.txt"
        _make_image(img_path, (200, 100, 50))
        lbl_path.write_text("")

        # Image with valid label
        img2_path = tmp_path / "images" / "train" / "good.png"
        lbl2_path = tmp_path / "labels" / "train" / "good.txt"
        _make_image(img2_path, (50, 100, 200))
        lbl2_path.write_text("0 0.5 0.5 0.3 0.2\n")

        result = clean_dataset(str(tmp_path), dry_run=True)

        # Files should still exist because it was a dry run
        assert img_path.exists()
        assert lbl_path.exists()
        assert result["removed_empty"] > 0

    def test_removes_empty_labels(self, tmp_path: Path):
        """Images with empty label files should be removed (not dry run)."""
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)

        # Image with empty label
        img_path = tmp_path / "images" / "train" / "empty_lbl.png"
        lbl_path = tmp_path / "labels" / "train" / "empty_lbl.txt"
        _make_image(img_path, (200, 100, 50))
        lbl_path.write_text("")

        # Image with valid label
        img2_path = tmp_path / "images" / "train" / "good.png"
        lbl2_path = tmp_path / "labels" / "train" / "good.txt"
        _make_image(img2_path, (50, 100, 200))
        lbl2_path.write_text("0 0.5 0.5 0.3 0.2\n")

        result = clean_dataset(
            str(tmp_path),
            remove_empty=True,
            remove_duplicates=False,
            dry_run=False,
        )

        assert result["removed_empty"] == 1
        assert not img_path.exists()
        assert img2_path.exists()
