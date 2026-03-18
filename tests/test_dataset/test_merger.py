"""
Tests for yolocc.dataset.merger
"""

from pathlib import Path

import pytest

from yolocc.dataset.merger import (
    BBox,
    deduplicate_bboxes,
    load_annotations,
    merge_annotations,
    parse_remap,
    save_annotations,
)


class TestBBox:
    """Tests for the BBox dataclass."""

    def test_bbox_from_yolo_line(self):
        """Should correctly parse a standard YOLO annotation line."""
        bbox = BBox.from_yolo_line("0 0.5 0.5 0.3 0.2")
        assert bbox.class_id == 0
        assert bbox.x_center == pytest.approx(0.5)
        assert bbox.y_center == pytest.approx(0.5)
        assert bbox.width == pytest.approx(0.3)
        assert bbox.height == pytest.approx(0.2)

    def test_bbox_to_yolo_line_roundtrip(self):
        """Parsing and serializing should produce equivalent values."""
        original_line = "1 0.45 0.55 0.25 0.35"
        bbox = BBox.from_yolo_line(original_line)
        output_line = bbox.to_yolo_line()
        # Re-parse to compare numerically (formatting may differ)
        bbox2 = BBox.from_yolo_line(output_line)
        assert bbox2.class_id == bbox.class_id
        assert bbox2.x_center == pytest.approx(bbox.x_center)
        assert bbox2.y_center == pytest.approx(bbox.y_center)
        assert bbox2.width == pytest.approx(bbox.width)
        assert bbox2.height == pytest.approx(bbox.height)

    def test_bbox_from_yolo_line_invalid(self):
        """Parsing an incomplete line should raise ValueError."""
        with pytest.raises(ValueError):
            BBox.from_yolo_line("0 0.5 0.5")


class TestBBoxIoU:
    """Tests for IoU calculation."""

    def test_bbox_iou_identical(self):
        """Identical boxes should have IoU of 1.0."""
        a = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        b = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4)
        assert a.iou(b) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        a = BBox(class_id=0, x_center=0.1, y_center=0.1, width=0.1, height=0.1)
        b = BBox(class_id=0, x_center=0.9, y_center=0.9, width=0.1, height=0.1)
        assert a.iou(b) == pytest.approx(0.0)

    def test_bbox_iou_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        # Two boxes of width=0.4, height=0.4 offset by 0.2 in x
        a = BBox(class_id=0, x_center=0.4, y_center=0.5, width=0.4, height=0.4)
        b = BBox(class_id=0, x_center=0.6, y_center=0.5, width=0.4, height=0.4)

        # a spans [0.2, 0.6] x [0.3, 0.7]
        # b spans [0.4, 0.8] x [0.3, 0.7]
        # intersection: [0.4, 0.6] x [0.3, 0.7] = 0.2 * 0.4 = 0.08
        # union: 0.16 + 0.16 - 0.08 = 0.24
        # IoU = 0.08 / 0.24 = 1/3
        iou = a.iou(b)
        assert iou == pytest.approx(1.0 / 3.0, abs=1e-6)


class TestParseRemap:
    """Tests for remap parsing."""

    def test_parse_remap_valid(self):
        """'7:1' should map to {7: 1}."""
        result = parse_remap(["7:1"])
        assert result == {7: 1}

    def test_parse_remap_multiple(self):
        """Multiple remap args should all be parsed."""
        result = parse_remap(["7:1", "3:0", "5:2"])
        assert result == {7: 1, 3: 0, 5: 2}

    def test_parse_remap_invalid_raises(self):
        """Invalid format (no colon) should raise ValueError."""
        with pytest.raises(ValueError):
            parse_remap(["7-1"])


class TestLoadAnnotations:
    """Tests for loading annotation files."""

    def test_load_annotations(self, tmp_path: Path):
        """Should load valid YOLO annotations from a file."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.3 0.2\n1 0.6 0.4 0.2 0.3\n")

        bboxes = load_annotations(label)
        assert len(bboxes) == 2
        assert bboxes[0].class_id == 0
        assert bboxes[1].class_id == 1

    def test_load_annotations_missing_file(self, tmp_path: Path):
        """Loading from a non-existent file should return empty list."""
        bboxes = load_annotations(tmp_path / "nonexistent.txt")
        assert bboxes == []

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Saving and loading should preserve bbox data."""
        bboxes = [
            BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.3, height=0.2),
            BBox(class_id=1, x_center=0.6, y_center=0.4, width=0.2, height=0.3),
        ]
        label_path = tmp_path / "labels" / "roundtrip.txt"
        save_annotations(label_path, bboxes)

        loaded = load_annotations(label_path)
        assert len(loaded) == 2
        for orig, loaded_b in zip(bboxes, loaded):
            assert loaded_b.class_id == orig.class_id
            assert loaded_b.x_center == pytest.approx(orig.x_center)
            assert loaded_b.y_center == pytest.approx(orig.y_center)


class TestDeduplicate:
    """Tests for deduplication logic."""

    def test_deduplicate_removes_high_iou(self):
        """Boxes with IoU above threshold should be deduplicated."""
        # Two nearly identical boxes
        a = BBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4, source_idx=0)
        b = BBox(class_id=0, x_center=0.51, y_center=0.51, width=0.4, height=0.4, source_idx=1)

        result, stats = deduplicate_bboxes([a, b], iou_threshold=0.5)
        assert len(result) == 1
        assert stats["total_conflicts"] == 1

    def test_deduplicate_keeps_non_overlapping(self):
        """Non-overlapping boxes should both be kept."""
        a = BBox(class_id=0, x_center=0.2, y_center=0.2, width=0.1, height=0.1)
        b = BBox(class_id=0, x_center=0.8, y_center=0.8, width=0.1, height=0.1)

        result, stats = deduplicate_bboxes([a, b], iou_threshold=0.5)
        assert len(result) == 2
        assert stats["total_conflicts"] == 0


class TestMergeWithRemap:
    """Tests for merge_annotations with class remapping."""

    def test_merge_with_remap(self, tmp_path: Path):
        """Merging with a remap should change class IDs in the output."""
        # Create two source directories
        src_a = tmp_path / "labels_a"
        src_b = tmp_path / "labels_b"
        output = tmp_path / "merged"
        src_a.mkdir()
        src_b.mkdir()

        # Source A: class 7
        (src_a / "img_001.txt").write_text("7 0.5 0.5 0.3 0.2\n")
        # Source B: class 0
        (src_b / "img_001.txt").write_text("0 0.1 0.1 0.1 0.1\n")

        # Remap class 7 -> class 1
        stats = merge_annotations(
            sources=[src_a, src_b],
            output=output,
            remap={7: 1},
            iou_threshold=0.5,
        )

        assert stats["bboxes_remapped"] == 1
        assert stats["total_bboxes_output"] == 2

        # Verify the output file was written with remapped class
        out_file = output / "img_001.txt"
        assert out_file.exists()
        content = out_file.read_text()
        lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
        class_ids = [int(l.split()[0]) for l in lines]
        assert 1 in class_ids  # remapped from 7
        assert 0 in class_ids  # unchanged
        assert 7 not in class_ids  # should have been remapped

    def test_remap_does_not_mutate_original_bboxes(self, tmp_path: Path):
        """Remapping should not mutate original BBox objects."""
        src = tmp_path / "labels"
        output = tmp_path / "merged"
        src.mkdir()

        (src / "img.txt").write_text("7 0.5 0.5 0.3 0.2\n")

        # Load bboxes before merge so we have references to the originals
        originals = load_annotations(src / "img.txt", source_idx=0)
        assert originals[0].class_id == 7

        # Run merge with remap 7 -> 1
        merge_annotations(
            sources=[src],
            output=output,
            remap={7: 1},
        )

        # Original BBox objects must remain untouched
        assert originals[0].class_id == 7
