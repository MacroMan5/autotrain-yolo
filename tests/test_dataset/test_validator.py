"""
Tests for yolocc.dataset.validator
"""

from pathlib import Path

import pytest
import yaml

from yolocc.dataset.validator import (
    DatasetState,
    DatasetValidator,
    detect_dataset_state,
    validate_dataset,
)


class TestDatasetValidatorStructure:
    """Tests for dataset structure validation."""

    def test_valid_dataset_passes(self, yolo_dataset: Path):
        """A well-formed dataset should pass validation."""
        # The fixture has an empty label and an orphaned image which produce
        # warnings (not errors), plus out-of-bounds/invalid-class counts of 0.
        # However the empty label and orphaned image trigger warnings, and
        # the default (non-strict) mode only fails on errors.
        result = validate_dataset(str(yolo_dataset))
        assert result is True

    def test_missing_data_yaml_fails(self, yolo_dataset: Path):
        """Validation should fail when data.yaml is missing."""
        (yolo_dataset / "data.yaml").unlink()
        result = validate_dataset(str(yolo_dataset))
        assert result is False

    def test_missing_images_dir_fails(self, yolo_dataset: Path, tmp_path: Path):
        """Validation should fail when images directory is missing."""
        # Create a dataset with data.yaml but no images dir
        broken = tmp_path / "broken_ds"
        broken.mkdir()
        data_cfg = {
            "train": "images/train",
            "val": "images/val",
            "nc": 2,
            "names": {0: "cat", 1: "dog"},
        }
        (broken / "data.yaml").write_text(yaml.safe_dump(data_cfg))
        # labels but no images
        (broken / "labels" / "train").mkdir(parents=True)
        (broken / "labels" / "val").mkdir(parents=True)

        result = validate_dataset(str(broken))
        assert result is False


class TestDatasetValidatorAnnotations:
    """Tests for annotation-level validation."""

    def test_out_of_bounds_annotation_warns(self, yolo_dataset: Path):
        """Annotations with coords > 1.0 should be flagged."""
        # Write an out-of-bounds annotation
        oob_label = yolo_dataset / "labels" / "train" / "img_000.txt"
        oob_label.write_text("0 1.5 0.5 0.3 0.2\n")

        validator = DatasetValidator(str(yolo_dataset))
        validator.validate()
        # Out-of-bounds annotations are counted as errors in this implementation
        assert validator.stats["annotations"]["out_of_bounds"] > 0

    def test_invalid_class_id_warns(self, yolo_dataset: Path):
        """Annotations with class_id >= nc should be flagged."""
        bad_label = yolo_dataset / "labels" / "train" / "img_000.txt"
        bad_label.write_text("99 0.5 0.5 0.3 0.2\n")

        validator = DatasetValidator(str(yolo_dataset))
        validator.validate()
        assert validator.stats["annotations"]["invalid_class"] > 0

    def test_empty_label_file_warns(self, yolo_dataset: Path):
        """Empty label files should produce a warning."""
        validator = DatasetValidator(str(yolo_dataset))
        validator.validate()
        # The fixture includes img_004.txt which is empty
        assert len(validator.stats["labels"]["empty"]) > 0
        assert any("empty label" in w for w in validator.warnings)


class TestDatasetValidatorIntegrity:
    """Tests for integrity and stats."""

    def test_orphaned_image_warns(self, yolo_dataset: Path):
        """Images without a matching label should produce a warning."""
        validator = DatasetValidator(str(yolo_dataset))
        validator.validate()
        # img_005 in train has no label file
        assert len(validator.stats["images"]["missing_label"]) > 0
        assert any("without labels" in w for w in validator.warnings)

    def test_strict_mode_fails_on_warnings(self, yolo_dataset: Path):
        """In strict mode, warnings should cause validation to fail."""
        # The fixture naturally has warnings (empty label, orphaned image)
        result = validate_dataset(str(yolo_dataset), strict=True)
        assert result is False

    def test_class_distribution_stats(self, yolo_dataset: Path):
        """Validator should report class distribution statistics."""
        validator = DatasetValidator(str(yolo_dataset))
        validator.validate()
        dist = validator.stats.get("class_distribution", {})
        # Should have entries for class 0 (cat) and class 1 (dog)
        assert 0 in dist
        assert 1 in dist
        assert dist[0] > 0
        assert dist[1] > 0

    def test_validation_returns_bool(self, yolo_dataset: Path):
        """validate_dataset should always return a bool."""
        result = validate_dataset(str(yolo_dataset))
        assert isinstance(result, bool)


class TestDetectDatasetState:
    """Tests for detect_dataset_state()."""

    def test_complete_dataset_detected(self, yolo_dataset: Path):
        """A full YOLO dataset should be detected as 'complete'."""
        state = detect_dataset_state(yolo_dataset)
        assert state.structure == "complete"
        assert state.has_data_yaml is True
        assert state.has_splits is True
        assert state.has_images is True

    def test_complete_dataset_with_negatives(self, tmp_path: Path):
        """data.yaml + splits = complete, even with 30% images missing labels."""
        from PIL import Image

        # Create YOLO structure with splits
        for split in ("train", "val"):
            (tmp_path / "images" / split).mkdir(parents=True)
            (tmp_path / "labels" / split).mkdir(parents=True)

        # 10 train images, only 7 have labels (30% intentional negatives)
        for i in range(10):
            Image.new("RGB", (32, 32)).save(
                tmp_path / "images" / "train" / f"img_{i:03d}.jpg"
            )
            if i < 7:
                (tmp_path / "labels" / "train" / f"img_{i:03d}.txt").write_text(
                    "0 0.5 0.5 0.3 0.2\n"
                )

        # 4 val images, all labeled
        for i in range(4):
            Image.new("RGB", (32, 32)).save(
                tmp_path / "images" / "val" / f"val_{i:03d}.jpg"
            )
            (tmp_path / "labels" / "val" / f"val_{i:03d}.txt").write_text(
                "0 0.5 0.5 0.3 0.2\n"
            )

        data_cfg = {"train": "images/train", "val": "images/val", "nc": 1, "names": {0: "obj"}}
        (tmp_path / "data.yaml").write_text(yaml.safe_dump(data_cfg))

        state = detect_dataset_state(tmp_path)
        assert state.structure == "complete"

    def test_unlabeled_images_detected(self, tmp_path: Path):
        """A directory with only images should be detected as 'unlabeled'."""
        from PIL import Image

        (tmp_path / "images").mkdir()
        for i in range(5):
            Image.new("RGB", (32, 32)).save(tmp_path / "images" / f"img_{i}.jpg")

        state = detect_dataset_state(tmp_path)
        assert state.structure == "unlabeled"
        assert state.image_count == 5
        assert state.label_count == 0
        assert state.has_labels is False

    def test_labeled_unsplit_detected(self, tmp_path: Path):
        """Images + labels without train/val split should be 'labeled_unsplit'."""
        from PIL import Image

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(5):
            Image.new("RGB", (32, 32)).save(tmp_path / "images" / f"img_{i}.jpg")
            (tmp_path / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.3 0.2\n")

        state = detect_dataset_state(tmp_path)
        assert state.structure == "labeled_unsplit"
        assert state.has_images is True
        assert state.has_labels is True
        assert state.has_splits is False

    def test_partial_labels_detected(self, tmp_path: Path):
        """Few labels relative to images (no data.yaml) should be 'partial_labels'."""
        from PIL import Image

        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        for i in range(10):
            Image.new("RGB", (32, 32)).save(tmp_path / "images" / f"img_{i}.jpg")
        # Only label 2 of 10
        for i in range(2):
            (tmp_path / "labels" / f"img_{i}.txt").write_text("0 0.5 0.5 0.3 0.2\n")

        state = detect_dataset_state(tmp_path)
        assert state.structure == "partial_labels"
        assert state.label_coverage == pytest.approx(0.2)

    def test_empty_directory_detected(self, tmp_path: Path):
        """An empty directory should be detected as 'empty'."""
        state = detect_dataset_state(tmp_path)
        assert state.structure == "empty"
        assert state.image_count == 0

    def test_flat_structure_detected(self, tmp_path: Path):
        """Images and .txt files side by side should be 'labeled_unsplit'."""
        from PIL import Image

        for i in range(5):
            Image.new("RGB", (32, 32)).save(tmp_path / f"img_{i}.jpg")
            (tmp_path / f"img_{i}.txt").write_text("0 0.5 0.5 0.3 0.2\n")

        state = detect_dataset_state(tmp_path)
        assert state.structure == "labeled_unsplit"
        assert state.has_images is True
        assert state.has_labels is True

    def test_next_steps_populated(self, tmp_path: Path):
        """next_steps should be non-empty for every structure type."""
        from PIL import Image

        # Empty
        state = detect_dataset_state(tmp_path)
        assert len(state.next_steps) > 0

        # Unlabeled
        (tmp_path / "images").mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / "images" / "img.jpg")
        state = detect_dataset_state(tmp_path)
        assert len(state.next_steps) > 0
