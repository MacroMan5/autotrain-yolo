"""
Tests for yolocc.dataset.splitter
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image

from yolocc.dataset.splitter import (
    categorize_by_annotations,
    parse_label_file,
    split_dataset,
    stratified_split,
)


def _create_flat_source(tmp_path: Path, n_images: int = 10) -> Path:
    """
    Create a flat source directory with images/ and labels/ (no train/val split)
    suitable as input to split_dataset.
    """
    src = tmp_path / "source"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)

    for i in range(n_images):
        # Alternate class so we get both class 0 and class 1
        cls_id = i % 2
        color = ((i * 25) % 256, (i * 50) % 256, (i * 75) % 256)
        img = Image.new("RGB", (32, 32), color=color)
        img.save(src / "images" / f"img_{i:03d}.png")
        (src / "labels" / f"img_{i:03d}.txt").write_text(
            f"{cls_id} 0.5 0.5 0.3 0.2\n"
        )

    return src


class TestSplitOutput:
    """Tests for the overall split_dataset output."""

    def test_split_creates_output_structure(self, tmp_path: Path):
        """Output should contain images/{train,val,test} + labels/{train,val,test}."""
        src = _create_flat_source(tmp_path)
        out = tmp_path / "split_out"
        split_dataset(
            source_dir=str(src),
            output_dir=str(out),
            train_ratio=0.8,
            val_ratio=0.1,
            class_names=["cat", "dog"],
        )

        for split in ("train", "val", "test"):
            assert (out / "images" / split).is_dir()
            assert (out / "labels" / split).is_dir()

    def test_split_creates_data_yaml(self, tmp_path: Path):
        """A valid data.yaml should be created in the output directory."""
        src = _create_flat_source(tmp_path)
        out = tmp_path / "split_out"
        split_dataset(
            source_dir=str(src),
            output_dir=str(out),
            class_names=["cat", "dog"],
        )

        data_yaml = out / "data.yaml"
        assert data_yaml.exists()
        cfg = yaml.safe_load(data_yaml.read_text())
        assert cfg["nc"] == 2
        assert "train" in cfg
        assert "val" in cfg

    def test_split_ratios_approximately_correct(self, tmp_path: Path):
        """Train/val/test counts should roughly match the requested ratios."""
        n = 20
        src = _create_flat_source(tmp_path, n_images=n)
        out = tmp_path / "split_out"
        result = split_dataset(
            source_dir=str(src),
            output_dir=str(out),
            train_ratio=0.7,
            val_ratio=0.15,
            class_names=["cat", "dog"],
        )

        train_count = result["train_count"]
        val_count = result["val_count"]
        total = train_count + val_count + (n - train_count - val_count)

        # Allow generous tolerance for small datasets
        assert train_count >= n * 0.5
        assert val_count >= 1

    def test_split_reproducible_with_seed(self, tmp_path: Path):
        """Two splits with the same seed should produce identical file sets."""
        src = _create_flat_source(tmp_path, n_images=12)

        out1 = tmp_path / "split_a"
        out2 = tmp_path / "split_b"

        for out in (out1, out2):
            split_dataset(
                source_dir=str(src),
                output_dir=str(out),
                train_ratio=0.8,
                val_ratio=0.1,
                seed=99,
                class_names=["cat", "dog"],
            )

        for split in ("train", "val", "test"):
            files_a = sorted(p.name for p in (out1 / "images" / split).iterdir())
            files_b = sorted(p.name for p in (out2 / "images" / split).iterdir())
            assert files_a == files_b

    def test_empty_source_raises(self, tmp_path: Path):
        """Splitting an empty directory should raise an error."""
        empty = tmp_path / "empty_src"
        empty.mkdir()
        out = tmp_path / "split_out"

        with pytest.raises(FileNotFoundError):
            split_dataset(
                source_dir=str(empty),
                output_dir=str(out),
                class_names=["cat", "dog"],
            )


class TestSplitterHelpers:
    """Tests for helper functions."""

    def test_categorize_by_annotations(self, tmp_path: Path):
        """Images should be grouped by the class IDs in their labels."""
        src = _create_flat_source(tmp_path, n_images=6)
        categories = categorize_by_annotations(
            src / "images", src / "labels"
        )
        # Even-indexed images have class 0, odd have class 1
        assert "class_0_only" in categories
        assert "class_1_only" in categories
        assert len(categories["class_0_only"]) == 3
        assert len(categories["class_1_only"]) == 3

    def test_parse_label_file(self, tmp_path: Path):
        """parse_label_file should extract class IDs from a label file."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.3 0.2\n1 0.6 0.4 0.2 0.3\n0 0.1 0.1 0.1 0.1\n")
        classes = parse_label_file(label)
        assert classes == [0, 1, 0]

    def test_parse_label_file_empty(self, tmp_path: Path):
        """parse_label_file should return empty list for empty file."""
        label = tmp_path / "empty.txt"
        label.write_text("")
        classes = parse_label_file(label)
        assert classes == []
