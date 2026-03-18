#!/usr/bin/env python3
"""
Dataset Splitter
================

Splits a dataset into train/val with stratified sampling.

Ensures each category (normal, occluded, edge cases) maintains
the same ratio in both train and val sets.

Functions:
    split_dataset: Main function to split dataset
    stratified_split: Perform stratified split
    categorize_by_annotations: Categorize by annotation content
    categorize_by_prefix: Categorize by filename prefix
    split_cli: CLI entry point
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from yolocc.paths import resolve_workspace_path


def parse_label_file(label_path: Path) -> List[int]:
    """Extract class IDs from a YOLO label file."""
    classes = []
    if label_path.exists():
        for line in label_path.read_text().strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if parts:
                    classes.append(int(parts[0]))
    return classes


def categorize_by_annotations(
    images_dir: Path,
    labels_dir: Path,
) -> Dict[str, List[Path]]:
    """
    Categorize images based on which class IDs appear in their annotations.

    Categories are generated dynamically from actual class IDs found:
    - 'class_0_only', 'class_1_only', etc. for single-class images
    - 'classes_0_1', 'classes_0_2_3', etc. for multi-class images
    - 'empty' for images without annotations
    """
    categories = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        classes = parse_label_file(label_path)

        if not classes:
            categories['empty'].append(img_path)
        else:
            class_set = sorted(set(classes))
            if len(class_set) == 1:
                key = f"class_{class_set[0]}_only"
            else:
                key = "classes_" + "_".join(str(c) for c in class_set)
            categories[key].append(img_path)

    return categories


def categorize_by_prefix(images_dir: Path) -> Dict[str, List[Path]]:
    """
    Categorize images by filename prefix.

    Expected naming conventions:
    - normal_*.jpg -> 'normal'
    - occluded_*.jpg -> 'occluded'
    - edge_*.jpg -> 'edge'
    - smoke_*.jpg -> 'edge'
    - dark_*.jpg -> 'edge'

    If no prefix matches, uses 'default' category.
    """
    categories = defaultdict(list)

    prefix_mapping = {
        'normal': 'normal',
        'occluded': 'occluded',
        'partial': 'occluded',
        'edge': 'edge',
        'smoke': 'edge',
        'dark': 'edge',
        'far': 'edge',
        'blur': 'edge',
    }

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Check prefix
        filename_lower = img_path.stem.lower()
        category = 'default'

        for prefix, cat in prefix_mapping.items():
            if filename_lower.startswith(prefix):
                category = cat
                break

        categories[category].append(img_path)

    return categories


def stratified_split(
    categories: Dict[str, List[Path]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split each category maintaining the ratios.

    Args:
        categories: Dict mapping category names to lists of image paths
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    random.seed(seed)

    train_images = []
    val_images = []
    test_images = []

    print("\nStratified Split:")
    print("-" * 50)

    for category, images in sorted(categories.items()):
        # Shuffle within category
        shuffled = images.copy()
        random.shuffle(shuffled)

        # Split
        total = len(shuffled)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_part = shuffled[:train_end]
        val_part = shuffled[train_end:val_end]
        test_part = shuffled[val_end:]

        train_images.extend(train_part)
        val_images.extend(val_part)
        test_images.extend(test_part)

        print(f"  {category:15s}: {total:4d} total -> "
              f"{len(train_part):4d} train, {len(val_part):4d} val, {len(test_part):4d} test")

    # Final shuffle to mix categories
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)

    return train_images, val_images, test_images


def copy_image_and_label(
    img_path: Path,
    src_labels_base: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path
):
    """Copy an image and its corresponding label to destination."""
    # Copy image
    dst_img = dst_images_dir / img_path.name
    shutil.copy2(img_path, dst_img)

    # Find label - check multiple possible locations
    label_name = f"{img_path.stem}.txt"
    possible_label_paths = [
        src_labels_base / img_path.parent.name / label_name,  # labels/train/img.txt
        src_labels_base / label_name,                          # labels/img.txt
        img_path.parent / label_name,                          # same dir as image
    ]

    label_path = None
    for p in possible_label_paths:
        if p.exists():
            label_path = p
            break

    if label_path:
        dst_label = dst_labels_dir / label_name
        shutil.copy2(label_path, dst_label)
    else:
        # Create empty label file for images without annotations
        dst_label = dst_labels_dir / label_name
        dst_label.write_text("")


def create_data_yaml(output_dir: Path, class_names: Optional[List[str]] = None):
    """Create data.yaml for the split dataset."""
    if class_names is None:
        raise ValueError("class_names is required — provide via --classes or yolo-project.yaml")

    names_str = '\n'.join(f"  {i}: {name}" for i, name in enumerate(class_names))

    yaml_content = f"""# Auto-generated dataset configuration
path: .
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names:
{names_str}
"""
    (output_dir / "data.yaml").write_text(yaml_content)


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    categorize_method: str = 'annotations',
    seed: int = 42,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Split dataset into train/val/test with stratified sampling.

    Args:
        source_dir: Directory containing images/ and labels/
        output_dir: Where to create the split dataset
        train_ratio: Fraction for training (default 0.8 = 80%)
        val_ratio: Fraction for validation (default 0.1 = 10%)
        categorize_method: 'annotations' or 'prefix'
        seed: Random seed for reproducibility
        class_names: List of class names for data.yaml

    Returns:
        Dict with split statistics
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Find source directories - handle multiple structures
    src_images = None
    src_labels = None

    # Structure 1: images/ and labels/ folders
    if (source_path / "images").exists():
        src_images = [source_path / "images"]
        src_labels = source_path / "labels"

    # Structure 2: flat structure
    elif any(source_path.glob("*.jpg")) or any(source_path.glob("*.png")):
        src_images = [source_path]
        src_labels = source_path

    if not src_images:
        raise FileNotFoundError(f"No images found in: {source_path}")

    print(f"\n{'='*60}")
    print("DATASET SPLITTER")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Split:  {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val / {(1-train_ratio-val_ratio)*100:.0f}% test")
    print(f"Method: {categorize_method}")
    print(f"Seed:   {seed}")

    # Categorize images from all source directories
    categories = defaultdict(list)

    for img_dir in src_images:
        labels_dir = src_labels if src_labels else img_dir

        if categorize_method == 'annotations':
            dir_categories = categorize_by_annotations(img_dir, labels_dir)
        else:
            dir_categories = categorize_by_prefix(img_dir)

        for cat, imgs in dir_categories.items():
            categories[cat].extend(imgs)

    if not any(categories.values()):
        raise ValueError("No images found in source directory")

    # Perform stratified split
    train_images, val_images, test_images = stratified_split(categories, train_ratio, val_ratio, seed)

    total = len(train_images) + len(val_images) + len(test_images)
    if total == 0:
        raise ValueError("No images found after stratified split")
    print(f"\n{'='*50}")
    print(f"Total: {total} images")
    print(f"  Train: {len(train_images)} ({len(train_images)/total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} ({len(val_images)/total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} ({len(test_images)/total*100:.1f}%)")

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\nCopying files...")

    splits_data = [
        (train_images, "train"),
        (val_images, "val"),
        (test_images, "test")
    ]

    for images, split_name in splits_data:
        dst_img_dir = output_path / "images" / split_name
        dst_lbl_dir = output_path / "labels" / split_name
        for img_path in images:
            copy_image_and_label(img_path, src_labels, dst_img_dir, dst_lbl_dir)

    # Create data.yaml
    create_data_yaml(output_path, class_names)

    print(f"\n{'='*60}")
    print("SPLIT COMPLETE")
    print(f"{'='*60}")
    print("Output structure:")
    print(f"  {output_path}/")
    print("  +-- data.yaml")
    print("  +-- images/")
    print(f"  |   +-- train/  ({len(train_images)} images)")
    print(f"  |   +-- val/    ({len(val_images)} images)")
    print("  +-- labels/")
    print("      +-- train/")
    print("      +-- val/")
    print(f"{'='*60}\n")

    return {
        'train_count': len(train_images),
        'val_count': len(val_images),
        'categories': {k: len(v) for k, v in categories.items()}
    }


def split_cli() -> None:
    """CLI entry point for dataset splitting."""
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val with stratified sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-split --source datasets/raw --output datasets/split
  yolo-split --source datasets/raw --output datasets/split --ratio 0.8
  yolo-split --source datasets/raw --output datasets/split --method prefix --seed 123
        """
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Source directory containing images and labels (relative paths are resolved from workspace root)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for split dataset (relative paths are resolved from workspace root)"
    )
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--val-ratio", "-v",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=['annotations', 'prefix'],
        default='annotations',
        help="Categorization method: 'annotations' (by class content) or 'prefix' (by filename)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs='+',
        default=None,
        help="Class names for data.yaml (required unless yolo-project.yaml exists)"
    )

    args = parser.parse_args()

    # Try to load class names from project config if not provided
    if args.classes is None:
        from yolocc.project import load_project_config
        config = load_project_config()
        if config and config.class_names:
            args.classes = config.class_names
        else:
            print("ERROR: --classes required (or create yolo-project.yaml)")
            sys.exit(1)

    # Validate ratios
    if not 0.1 <= args.ratio <= 0.95:
        print("ERROR: Ratio must be between 0.1 and 0.95")
        sys.exit(1)
    if not 0.0 <= args.val_ratio <= 0.45:
        print("ERROR: val-ratio must be between 0.0 and 0.45")
        sys.exit(1)
    if args.ratio + args.val_ratio >= 1.0:
        print("ERROR: ratio + val-ratio must be less than 1.0 (to leave room for test)")
        sys.exit(1)

    source_dir = resolve_workspace_path(args.source)
    output_dir = resolve_workspace_path(args.output)

    split_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        train_ratio=args.ratio,
        val_ratio=args.val_ratio,
        categorize_method=args.method,
        seed=args.seed,
        class_names=args.classes
    )


__all__ = [
    "split_dataset",
    "stratified_split",
    "categorize_by_annotations",
    "categorize_by_prefix",
    "split_cli",
]


if __name__ == "__main__":
    split_cli()
