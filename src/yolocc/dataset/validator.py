#!/usr/bin/env python3
"""
Dataset Validator
=================

Validates YOLO datasets before training.

Checks performed:
1. Structure - data.yaml, images/labels directories
2. Integrity - readable images, non-empty labels
3. Annotations - valid coordinates, correct class IDs
4. Statistics - class distribution, balance

Classes:
    DatasetValidator: Main validator class

Functions:
    validate_dataset: Convenience function for validation
    validate_cli: CLI entry point
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

import yaml

from yolocc.paths import resolve_workspace_path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# For checking corrupted images
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DatasetValidator:
    """Validates a YOLO dataset."""

    def __init__(self, dataset_path: str, strict: bool = False):
        """
        Initialize the validator.

        Args:
            dataset_path: Path to the dataset directory
            strict: Strict mode - fail on warnings too
        """
        self.dataset_path = Path(dataset_path)
        self.strict = strict

        # Results
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.stats: Dict = {}

        # Dataset config
        self.data_yaml: Optional[Dict] = None
        self.num_classes: int = 0
        self.class_names: List[str] = []
        self.split_dirs: Dict[str, Dict[str, Path]] = {}

    def validate(self) -> bool:
        """
        Execute all validations.

        Returns:
            True if the dataset is valid (no critical errors)
        """
        print("=" * 60)
        print("Dataset Validation")
        print("=" * 60)
        print(f"Path: {self.dataset_path}")
        print(f"Mode: {'Strict' if self.strict else 'Normal'}")
        print()

        # 1. Structure
        print("[1/4] Checking structure...")
        self._check_structure()

        # 2. Integrity
        print("[2/4] Checking integrity...")
        self._check_integrity()

        # 3. Annotations
        print("[3/4] Checking annotations...")
        self._check_annotations()

        # 4. Statistics
        print("[4/4] Computing statistics...")
        self._compute_stats()

        # Report
        self._print_report()

        # Result
        if self.errors:
            return False
        if self.strict and self.warnings:
            return False
        return True

    def _check_structure(self):
        """Check the dataset structure."""
        # data.yaml exists?
        data_yaml_path = self.dataset_path / "data.yaml"
        if not data_yaml_path.exists():
            self.errors.append(f"data.yaml not found at {data_yaml_path}")
            return

        # Parse data.yaml
        try:
            with open(data_yaml_path) as f:
                self.data_yaml = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to parse data.yaml: {e}")
            return

        # Required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in self.data_yaml:
                self.errors.append(f"Missing required field in data.yaml: {field}")

        if self.errors:
            return

        self.num_classes = self.data_yaml['nc']
        self.class_names = list(self.data_yaml['names'].values()) if isinstance(self.data_yaml['names'], dict) else self.data_yaml['names']

        # Directories exist?
        for split in ['train', 'val']:
            # Check both 'val' and 'valid' for the validation split
            potential_splits = [split]
            if split == 'val':
                potential_splits.append('valid')

            found_img = None
            found_lbl = None

            for s in potential_splits:
                # Check for images/s and labels/s (standard structure)
                img_dir = self.dataset_path / 'images' / s
                lbl_dir = self.dataset_path / 'labels' / s

                # Alternative: s/images and s/labels (common alternative structure)
                alt_img_dir = self.dataset_path / s / 'images'
                alt_lbl_dir = self.dataset_path / s / 'labels'

                if img_dir.exists():
                    found_img = img_dir
                elif alt_img_dir.exists():
                    found_img = alt_img_dir

                if lbl_dir.exists():
                    found_lbl = lbl_dir
                elif alt_lbl_dir.exists():
                    found_lbl = alt_lbl_dir

                if found_img and found_lbl:
                    break

            if not found_img:
                self.errors.append(f"Images directory not found for split '{split}' (checked {potential_splits})")
            if not found_lbl:
                self.errors.append(f"Labels directory not found for split '{split}' (checked {potential_splits})")

            if found_img and found_lbl:
                self.split_dirs[split] = {'images': found_img, 'labels': found_lbl}

        print(f"  >> data.yaml valid ({self.num_classes} classes: {', '.join(self.class_names)})")

    def _check_integrity(self):
        """Check file integrity."""
        if self.errors:  # Skip if structure invalid
            return

        self.stats['images'] = {'total': 0, 'valid': 0, 'corrupted': [], 'missing_label': []}
        self.stats['labels'] = {'total': 0, 'valid': 0, 'empty': [], 'missing_image': []}

        for split, dirs in self.split_dirs.items():
            img_dir = dirs['images']
            lbl_dir = dirs['labels']

            # Check images
            for img_path in img_dir.glob('*'):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    continue

                self.stats['images']['total'] += 1

                # Corrupted image?
                if PIL_AVAILABLE:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        self.stats['images']['valid'] += 1
                    except Exception:
                        self.stats['images']['corrupted'].append(str(img_path))
                        continue
                else:
                    self.stats['images']['valid'] += 1

                # Corresponding label exists?
                label_path = lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    self.stats['images']['missing_label'].append(str(img_path))

            # Check labels
            for lbl_path in lbl_dir.glob('*.txt'):
                self.stats['labels']['total'] += 1

                # Empty label?
                content = lbl_path.read_text().strip()
                if not content:
                    self.stats['labels']['empty'].append(str(lbl_path))
                else:
                    self.stats['labels']['valid'] += 1

                # Corresponding image exists?
                img_found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    if (img_dir / f"{lbl_path.stem}{ext}").exists():
                        img_found = True
                        break
                if not img_found:
                    self.stats['labels']['missing_image'].append(str(lbl_path))

        # Report
        corrupted = len(self.stats['images']['corrupted'])
        missing_lbl = len(self.stats['images']['missing_label'])
        empty_lbl = len(self.stats['labels']['empty'])
        missing_img = len(self.stats['labels']['missing_image'])

        if corrupted > 0:
            self.warnings.append(f"{corrupted} corrupted images found")
        if missing_lbl > 0:
            self.warnings.append(f"{missing_lbl} images without labels")
        if empty_lbl > 0:
            self.warnings.append(f"{empty_lbl} empty label files")
        if missing_img > 0:
            self.warnings.append(f"{missing_img} labels without images")

        print(f"  >> Images: {self.stats['images']['valid']}/{self.stats['images']['total']} valid")
        print(f"  >> Labels: {self.stats['labels']['valid']}/{self.stats['labels']['total']} valid")

    def _check_annotations(self):
        """Check annotation validity."""
        if self.errors:
            return

        self.stats['annotations'] = {
            'total': 0,
            'valid': 0,
            'out_of_bounds': 0,
            'invalid_class': 0,
            'malformed': 0
        }
        self.stats['class_distribution'] = Counter()

        for split, dirs in self.split_dirs.items():
            lbl_dir = dirs['labels']

            for lbl_path in lbl_dir.glob('*.txt'):
                content = lbl_path.read_text().strip()
                if not content:
                    continue

                for line_num, line in enumerate(content.split('\n'), 1):
                    self.stats['annotations']['total'] += 1
                    parts = line.strip().split()

                    # Format: class_id x_center y_center width height [confidence] [...]
                    if len(parts) < 5:
                        self.stats['annotations']['malformed'] += 1
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                    except ValueError:
                        self.stats['annotations']['malformed'] += 1
                        continue

                    # Valid class ID?
                    if class_id < 0 or class_id >= self.num_classes:
                        self.stats['annotations']['invalid_class'] += 1
                        continue

                    # Coordinates in [0, 1]?
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        self.stats['annotations']['out_of_bounds'] += 1
                        continue

                    # Valid!
                    self.stats['annotations']['valid'] += 1
                    self.stats['class_distribution'][class_id] += 1

        # Report
        oob = self.stats['annotations']['out_of_bounds']
        inv_cls = self.stats['annotations']['invalid_class']
        malformed = self.stats['annotations']['malformed']

        if oob > 0:
            self.errors.append(f"{oob} annotations with out-of-bounds coordinates")
        if inv_cls > 0:
            self.errors.append(f"{inv_cls} annotations with invalid class IDs")
        if malformed > 0:
            self.warnings.append(f"{malformed} malformed annotation lines")

        valid = self.stats['annotations']['valid']
        total = self.stats['annotations']['total']
        print(f"  >> Annotations: {valid}/{total} valid")

    def _compute_stats(self):
        """Compute dataset statistics."""
        if self.errors:
            return

        # Class distribution
        dist = self.stats['class_distribution']
        if dist:
            max_count = max(dist.values())
            min_count = min(dist.values()) if min(dist.values()) > 0 else 1
            ratio = max_count / min_count

            if ratio > 10:
                self.errors.append(f"Severe class imbalance: ratio {ratio:.1f}:1 (max 10:1 recommended)")
            elif ratio > 5:
                self.warnings.append(f"Class imbalance: ratio {ratio:.1f}:1")

            print("  >> Class distribution:")
            for class_id, count in sorted(dist.items()):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                pct = count / sum(dist.values()) * 100
                print(f"      {class_name}: {count} ({pct:.1f}%)")

        # Avg annotations per image
        if self.stats['images']['valid'] > 0:
            avg = self.stats['annotations']['valid'] / self.stats['images']['valid']
            self.stats['avg_annotations_per_image'] = avg
            print(f"  >> Avg annotations/image: {avg:.2f}")

    def _print_report(self):
        """Print the final report."""
        print()
        print("=" * 60)
        print("Validation Report")
        print("=" * 60)

        if self.errors:
            print("\n[X] ERRORS (must fix before training):")
            for err in self.errors:
                print(f"   - {err}")

        if self.warnings:
            print("\n[!] WARNINGS (recommended to fix):")
            for warn in self.warnings:
                print(f"   - {warn}")

        if not self.errors and not self.warnings:
            print("\n[OK] Dataset is valid! Ready for training.")
        elif not self.errors:
            print("\n[OK] Dataset is valid with warnings. Training can proceed.")
        else:
            print("\n[X] Dataset has errors. Fix before training.")

        print()


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


@dataclass
class DatasetState:
    """Result of probing a directory to determine its dataset readiness."""

    has_data_yaml: bool = False
    has_splits: bool = False
    has_images: bool = False
    has_labels: bool = False
    image_count: int = 0
    label_count: int = 0
    label_coverage: float = 0.0
    detected_classes: list = field(default_factory=list)
    structure: str = "empty"
    data_yaml_path: Optional[Path] = None
    images_dir: Optional[Path] = None
    labels_dir: Optional[Path] = None
    next_steps: list = field(default_factory=list)


def detect_dataset_state(path: Path) -> DatasetState:
    """
    Probe a directory to determine its dataset starting state.

    This is a fast, filesystem-only check. It does NOT load any YOLO model
    or read image pixels — only counts files and peeks at label text.

    Args:
        path: Directory to probe.

    Returns:
        DatasetState with structure classification and next_steps.
    """
    state = DatasetState()
    path = Path(path)

    if not path.exists() or not path.is_dir():
        state.next_steps = ["Provide a valid directory path"]
        return state

    # --- Check for data.yaml ---
    data_yaml_path = path / "data.yaml"
    if data_yaml_path.exists():
        state.has_data_yaml = True
        state.data_yaml_path = data_yaml_path

    # --- Check for train/val split directories ---
    # Standard: images/train, images/val
    # Alternative: train/images, val/images
    for split in ("train", "val"):
        candidates = [
            path / "images" / split,
            path / split / "images",
        ]
        if split == "val":
            candidates.extend([
                path / "images" / "valid",
                path / "valid" / "images",
            ])
        if any(d.exists() and d.is_dir() for d in candidates):
            state.has_splits = True
            break

    # --- Find images ---
    images_dir = None
    image_files: list[Path] = []

    # Try structured locations first
    for candidate in (path / "images", path):
        if candidate.is_dir():
            count = 0
            for f in candidate.rglob("*"):
                if count >= 10000:
                    break
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    image_files.append(f)
                    count += 1
            if image_files:
                images_dir = candidate
                break

    state.image_count = len(image_files)
    state.has_images = state.image_count > 0
    state.images_dir = images_dir

    # --- Find labels ---
    label_files: list[Path] = []
    labels_dir = None

    # Try structured labels/ directory
    labels_candidate = path / "labels"
    if labels_candidate.is_dir():
        labels_dir = labels_candidate
        count = 0
        for f in labels_candidate.rglob("*.txt"):
            if count >= 10000:
                break
            if f.is_file():
                label_files.append(f)
                count += 1

    # Try flat structure: .txt files alongside images
    if not label_files and images_dir:
        search_dir = images_dir if images_dir != path else path
        for img in image_files[:100]:
            txt_path = img.with_suffix(".txt")
            if txt_path.exists():
                label_files.append(txt_path)
            # Also check in same directory with different nesting
            alt_txt = search_dir / f"{img.stem}.txt"
            if alt_txt.exists() and alt_txt not in label_files:
                label_files.append(alt_txt)

    state.label_count = len(label_files)
    state.has_labels = state.label_count > 0
    state.labels_dir = labels_dir

    # --- Label coverage ---
    if state.image_count > 0:
        state.label_coverage = state.label_count / state.image_count

    # --- Sample labels to extract class IDs ---
    if label_files:
        sample_size = min(50, len(label_files))
        sample = random.sample(label_files, sample_size)
        class_ids: set[int] = set()
        for lbl_path in sample:
            try:
                content = lbl_path.read_text().strip()
                for line in content.split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(int(parts[0]))
            except (ValueError, OSError):
                continue
        state.detected_classes = sorted(class_ids)

    # --- Classify structure (priority order — first match wins) ---
    #
    # Critical: data.yaml presence overrides coverage analysis.
    # Datasets with intentional negative images (no objects) are normal
    # in detection and should not be classified as partial/unlabeled.
    if state.has_data_yaml and state.has_splits:
        state.structure = "complete"
        state.next_steps = [
            f"yolo-validate {path}",
            "Continue to profiling and baseline",
        ]
    elif (
        state.has_images
        and state.has_labels
        and not state.has_data_yaml
        and state.label_coverage < 0.5
    ):
        state.structure = "partial_labels"
        state.next_steps = [
            "Auto-label remaining images with trained model",
            "Or label manually",
        ]
    elif state.has_labels and state.has_images and not state.has_splits:
        state.structure = "labeled_unsplit"
        state.next_steps = [
            f"yolo-split --source {path} --output datasets/{path.name}_split",
        ]
    elif state.has_images and not state.has_labels:
        state.structure = "unlabeled"
        state.next_steps = [
            "Check COCO class overlap for auto-labeling",
            "Or label manually in CVAT/Roboflow/Label Studio",
        ]
    else:
        state.structure = "empty"
        state.next_steps = ["Provide a directory with images"]

    return state


def validate_dataset(dataset_path: str, strict: bool = False) -> bool:
    """
    Validate a YOLO dataset.

    Args:
        dataset_path: Path to the dataset
        strict: Strict mode (fail on warnings too)

    Returns:
        True if valid, False otherwise
    """
    validator = DatasetValidator(dataset_path, strict)
    return validator.validate()


def validate_cli() -> None:
    """CLI entry point for dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate YOLO dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-validate datasets/my_dataset
  yolo-validate datasets/my_dataset --strict
        """
    )
    parser.add_argument("dataset", help="Path to dataset directory (relative paths are resolved from workspace root)")
    parser.add_argument("--strict", action="store_true", help="Strict mode (fail on warnings)")
    args = parser.parse_args()

    if not PIL_AVAILABLE:
        print("Warning: Pillow not installed. Image corruption check disabled.")
        print("Install with: pip install Pillow")

    dataset_path = resolve_workspace_path(args.dataset)

    is_valid = validate_dataset(str(dataset_path), args.strict)
    sys.exit(0 if is_valid else 1)


__all__ = [
    "DatasetState",
    "DatasetValidator",
    "detect_dataset_state",
    "validate_dataset",
    "validate_cli",
]


if __name__ == "__main__":
    validate_cli()
