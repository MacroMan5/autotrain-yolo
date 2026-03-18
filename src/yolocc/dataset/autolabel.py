#!/usr/bin/env python3
"""
Auto-Label Module
=================

Auto-annotate images using a trained YOLO model.

Features:
- Merge images from multiple dataset sources
- Auto-generate YOLO annotations using a trained model
- Create proper data.yaml with configurable classes
- Handle various dataset structures (flat, nested)

Functions:
    autolabel_dataset: Main function to auto-annotate images
    autolabel_cli: CLI entry point
"""

import argparse
import shutil
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from PIL import Image, ImageOps
from ultralytics import YOLO

from yolocc.paths import resolve_workspace_path

import re

# COCO pretrained model class names (80 classes from yolo11n.pt / yolov8n.pt)
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_coco_overlap(target_classes: list[str]) -> tuple[list[str], list[str]]:
    """
    Check which target classes overlap with COCO's 80 classes.

    Args:
        target_classes: List of class names the user wants to detect.

    Returns:
        (overlapping, non_overlapping) — two lists of class names.
    """
    coco_lower = {c.lower() for c in COCO_CLASSES}
    overlapping = [c for c in target_classes if c.lower() in coco_lower]
    non_overlapping = [c for c in target_classes if c.lower() not in coco_lower]
    return overlapping, non_overlapping

# Roboflow augmentation pattern: filename.rf.{hash}.ext
ROBOFLOW_PATTERN = re.compile(r'\.rf\.[a-f0-9]{32}')


def get_base_name(filename: str) -> str:
    """
    Extract base name without Roboflow augmentation suffix.

    Example:
        "0020005_jpg.rf.3f0ea87de18c8560dee2ddfdfb20dcce.jpg" -> "0020005_jpg.jpg"
    """
    return ROBOFLOW_PATTERN.sub('', filename)


def is_augmented(filename: str) -> bool:
    """Check if filename contains Roboflow augmentation suffix."""
    return bool(ROBOFLOW_PATTERN.search(filename))


def preprocess_image(
    img_path: Path,
    output_path: Path,
    auto_orient: bool = False,
    resize: Optional[int] = None
) -> Tuple[int, int]:
    """
    Preprocess an image with optional auto-orient and resize.

    Args:
        img_path: Source image path
        output_path: Output image path
        auto_orient: Fix EXIF orientation
        resize: Target size (square, with letterbox padding)

    Returns:
        Tuple of (original_width, original_height)
    """
    img = Image.open(img_path)

    # Auto-orient based on EXIF
    if auto_orient:
        img = ImageOps.exif_transpose(img)

    # Convert RGBA/palette to RGB for JPEG compatibility
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')

    orig_size = img.size  # (width, height) — after transpose/conversion

    # Resize with letterbox (maintain aspect ratio, pad to square)
    if resize:
        # Calculate new size maintaining aspect ratio
        w, h = img.size
        scale = resize / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Create square canvas and paste centered
        canvas = Image.new('RGB', (resize, resize), (114, 114, 114))  # YOLO gray
        paste_x = (resize - new_w) // 2
        paste_y = (resize - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)

    return orig_size


def find_images(source_path: Path) -> List[Path]:
    """
    Recursively find all images in a directory.

    Args:
        source_path: Path to search for images

    Returns:
        List of image paths
    """
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []

    for ext in extensions:
        images.extend(source_path.rglob(f'*{ext}'))
        images.extend(source_path.rglob(f'*{ext.upper()}'))

    return sorted(set(images))


def collect_images_from_sources(
    sources: List[Path],
    verbose: bool = False
) -> Tuple[List[Path], Dict[str, int]]:
    """
    Collect all images from multiple source directories.

    Args:
        sources: List of source directories
        verbose: Print progress

    Returns:
        Tuple of (list of image paths, stats dict)
    """
    all_images = []
    stats = defaultdict(int)

    for source in sources:
        if not source.exists():
            print(f"Warning: Source not found: {source}")
            continue

        images = find_images(source)
        stats[str(source)] = len(images)
        all_images.extend(images)

        if verbose:
            print(f"  {source}: {len(images)} images")

    return all_images, dict(stats)


def autolabel_dataset(
    sources: List[Path],
    output_dir: Path,
    model_path: Path,
    class_names: Optional[List[str]] = None,
    confidence: float = 0.25,
    split_ratio: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    seed: int = 42,
    copy_images: bool = True,
    skip_existing: bool = True,
    skip_augmented: bool = False,
    auto_orient: bool = False,
    resize: Optional[int] = None,
    save_confidence: bool = False,
    review_threshold: Optional[float] = None,
    verbose: bool = False,
    dry_run: bool = False
) -> Dict:
    """
    Auto-annotate images from multiple sources using a YOLO model.

    Args:
        sources: List of source directories containing images
        output_dir: Output directory for the merged dataset
        model_path: Path to the YOLO model for prediction
        class_names: Class names for data.yaml (None = use model's classes)
        confidence: Confidence threshold for detections
        split_ratio: (train, val, test) split ratios
        seed: Random seed for reproducible splits
        copy_images: If True, copy images; if False, create symlinks
        skip_existing: Skip images that already exist in output
        skip_augmented: Skip Roboflow augmented images (keep only originals)
        auto_orient: Fix EXIF orientation issues
        resize: Resize images to square (e.g., 640 for 640x640)
        save_confidence: Save confidence scores in label files (6th column)
        review_threshold: Images with avg confidence below this go to review/ folder
        verbose: Print detailed progress
        dry_run: Sample 20 images, report expected results, write nothing

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_images': 0,
        'images_with_detections': 0,
        'images_without_detections': 0,
        'images_for_review': 0,
        'total_detections': 0,
        'detections_per_class': defaultdict(int),
        'source_stats': {},
        'split_counts': {'train': 0, 'val': 0, 'test': 0}
    }

    # Validate
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    train_r, val_r, test_r = split_ratio
    if abs(train_r + val_r + test_r - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratio)}")

    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))

    # Get class names from model if not provided
    if class_names is None:
        class_names = list(model.names.values())
    print(f"Classes: {class_names}")

    # Collect images from all sources
    print(f"\nCollecting images from {len(sources)} sources...")
    all_images, source_stats = collect_images_from_sources(sources, verbose)
    stats['source_stats'] = source_stats
    stats['total_images'] = len(all_images)

    if not all_images:
        print("No images found!")
        return stats

    print(f"Total images: {len(all_images)}")

    # Filter augmented images if requested.
    #
    # Important: some Roboflow exports contain ONLY augmented filenames (".rf.<hash>").
    # In that case, "skip_augmented" would remove everything, so we keep them and warn.
    augmented_found = 0
    augmented_skipped = 0
    if skip_augmented:
        filtered_images = []
        original_found = 0
        for img_path in all_images:
            if is_augmented(img_path.name):
                augmented_found += 1
            else:
                original_found += 1
                filtered_images.append(img_path)

        if original_found == 0 and augmented_found > 0:
            print(
                f"Warning: all {augmented_found} images look Roboflow-augmented "
                f"(.rf.<hash>); keeping them (nothing to skip)."
            )
        else:
            augmented_skipped = augmented_found
            all_images = filtered_images
            if augmented_skipped > 0:
                print(f"Skipped {augmented_skipped} Roboflow augmented images")
                print(f"Remaining: {len(all_images)} original images")

    # Deduplicate by (parent directory, base name) to avoid dropping
    # different images from different source directories that share a filename.
    # Prefer the non-augmented original when both exist for the same key.
    selected_by_key: Dict[Tuple[str, str], Path] = {}
    is_augmented_by_key: Dict[Tuple[str, str], bool] = {}
    duplicates = 0

    for img_path in all_images:
        base_name = get_base_name(img_path.name)
        dedup_key = (str(img_path.parent), base_name)
        current_is_aug = is_augmented(img_path.name)

        if dedup_key not in selected_by_key:
            selected_by_key[dedup_key] = img_path
            is_augmented_by_key[dedup_key] = current_is_aug
            continue

        duplicates += 1
        existing_is_aug = is_augmented_by_key[dedup_key]
        if existing_is_aug and not current_is_aug:
            # Replace augmented variant with original image.
            selected_by_key[dedup_key] = img_path
            is_augmented_by_key[dedup_key] = current_is_aug

    unique_images = list(selected_by_key.values())

    if duplicates > 0:
        print(f"Removed {duplicates} duplicate images (by base name)")

    stats['unique_images'] = len(unique_images)
    stats['augmented_found'] = augmented_found
    stats['augmented_skipped'] = augmented_skipped

    # --- Dry-run: sample inference, report stats, write nothing ---
    if dry_run:
        sample_size = min(20, len(unique_images))
        sample_images = random.sample(unique_images, sample_size) if sample_size else []
        sample_detections = 0
        sample_classes_seen: set[int] = set()

        for img_path in sample_images:
            results = model.predict(str(img_path), verbose=False, conf=confidence)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    sample_detections += 1
                    sample_classes_seen.add(int(box.cls.item()))

        avg_det = sample_detections / sample_size if sample_size else 0
        estimated_total = int(avg_det * len(unique_images))
        seen_names = [
            class_names[c] for c in sorted(sample_classes_seen)
            if c < len(class_names)
        ]

        stats['dry_run'] = True
        stats['sample_size'] = sample_size
        stats['sample_detections'] = sample_detections
        stats['avg_detections_per_image'] = avg_det
        stats['estimated_total_detections'] = estimated_total
        stats['classes_seen'] = sorted(sample_classes_seen)
        stats['classes_seen_names'] = seen_names

        print(f"\n[DRY RUN] Sampled {sample_size} of {len(unique_images)} images")
        print(f"  Avg detections/image: {avg_det:.1f}")
        print(f"  Classes seen: {seen_names}")
        print(f"  Estimated total labels: {estimated_total}")
        print("\nNo files written.")
        return stats

    # Shuffle and split
    random.seed(seed)
    random.shuffle(unique_images)

    n_total = len(unique_images)
    n_train = int(n_total * train_r)
    n_val = int(n_total * val_r)

    splits = {
        'train': unique_images[:n_train],
        'val': unique_images[n_train:n_train + n_val],
        'test': unique_images[n_train + n_val:]
    }

    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ['train', 'val', 'test']:
        if splits[split_name]:
            (output_dir / 'images' / split_name).mkdir(parents=True, exist_ok=True)
            (output_dir / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

    # Create review folder if threshold is set
    review_dir = None
    review_images = []  # Track images needing review
    if review_threshold is not None:
        review_dir = output_dir / 'review'
        (review_dir / 'images').mkdir(parents=True, exist_ok=True)
        (review_dir / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"\nReview threshold: {review_threshold} (images below this go to review/)")

    # Process each split
    print("\nAuto-labeling...")

    for split_name, split_images in splits.items():
        if not split_images:
            continue

        print(f"\n{split_name}: {len(split_images)} images")
        stats['split_counts'][split_name] = len(split_images)

        img_out_dir = output_dir / 'images' / split_name
        lbl_out_dir = output_dir / 'labels' / split_name

        for i, img_path in enumerate(split_images):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(split_images)}")

            # Output paths
            img_out_path = img_out_dir / img_path.name
            lbl_out_path = lbl_out_dir / f"{img_path.stem}.txt"

            # Skip if exists
            if skip_existing and img_out_path.exists() and lbl_out_path.exists():
                continue

            # Copy/preprocess image
            if not img_out_path.exists():
                if auto_orient or resize:
                    # Preprocess: auto-orient and/or resize
                    preprocess_image(img_path, img_out_path, auto_orient, resize)
                elif copy_images:
                    shutil.copy2(img_path, img_out_path)
                else:
                    img_out_path.symlink_to(img_path.resolve())

            # Run prediction on OUTPUT image (after preprocessing)
            predict_path = img_out_path if img_out_path.exists() else img_path
            results = model.predict(str(predict_path), verbose=False, conf=confidence)

            # Generate YOLO format labels
            labels = []
            confidences = []
            needs_review = False

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                img_w, img_h = results[0].orig_shape[1], results[0].orig_shape[0]

                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    confidences.append(conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Convert to YOLO format (center, width, height - normalized)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h

                    # Include confidence if requested
                    if save_confidence:
                        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}")
                    else:
                        labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    stats['detections_per_class'][cls_id] += 1
                    stats['total_detections'] += 1

                stats['images_with_detections'] += 1

                # Check if image needs review (low confidence or no detections)
                if review_threshold is not None and confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    min_conf = min(confidences)
                    # Flag for review if avg or min confidence is below threshold
                    if avg_conf < review_threshold or min_conf < review_threshold * 0.7:
                        needs_review = True

            else:
                stats['images_without_detections'] += 1
                # No detections = needs review
                if review_threshold is not None:
                    needs_review = True

            # Write label file (even if empty)
            with open(lbl_out_path, 'w') as f:
                f.write('\n'.join(labels))

            # Copy to review folder if needed
            if needs_review and review_dir:
                review_img_path = review_dir / 'images' / img_path.name
                review_lbl_path = review_dir / 'labels' / f"{img_path.stem}.txt"

                # Copy image and label to review folder
                if not review_img_path.exists():
                    shutil.copy2(img_out_path, review_img_path)
                if not review_lbl_path.exists():
                    shutil.copy2(lbl_out_path, review_lbl_path)

                # Track for report
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                review_images.append({
                    'path': str(img_path.name),
                    'avg_conf': avg_conf,
                    'min_conf': min(confidences) if confidences else 0,
                    'num_detections': len(confidences),
                    'reason': 'no_detection' if not confidences else 'low_confidence'
                })
                stats['images_for_review'] += 1

    # Generate data.yaml
    import yaml as _yaml

    data_yaml_path = output_dir / 'data.yaml'
    data_yaml_content = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": list(class_names),
    }
    if splits['test']:
        data_yaml_content["test"] = "images/test"

    with open(data_yaml_path, 'w') as f:
        f.write("# Auto-generated dataset\n")
        f.write(f"# Source: {len(sources)} directories merged\n")
        f.write(f"# Model: {model_path.name}\n")
        f.write(f"# Confidence: {confidence}\n\n")
        _yaml.safe_dump(data_yaml_content, f, sort_keys=False)

    # Generate review report if there are images to review
    if review_images and review_dir:
        report_path = review_dir / 'review_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("IMAGES FOR MANUAL REVIEW\n")
            f.write("=" * 70 + "\n")
            f.write(f"Review threshold: {review_threshold}\n")
            f.write(f"Total images for review: {len(review_images)}\n")
            f.write("-" * 70 + "\n\n")

            # Separate by reason
            no_detection = [r for r in review_images if r['reason'] == 'no_detection']
            low_conf = [r for r in review_images if r['reason'] == 'low_confidence']

            if no_detection:
                f.write(f"## NO DETECTIONS ({len(no_detection)} images)\n")
                f.write("These images had no detections - may need manual annotation:\n\n")
                for img in sorted(no_detection, key=lambda x: x['path']):
                    f.write(f"  {img['path']}\n")
                f.write("\n")

            if low_conf:
                f.write(f"## LOW CONFIDENCE ({len(low_conf)} images)\n")
                f.write("These images had low confidence detections - verify annotations:\n\n")
                # Sort by avg confidence (lowest first)
                for img in sorted(low_conf, key=lambda x: x['avg_conf']):
                    f.write(f"  {img['path']}")
                    f.write(f"  (avg: {img['avg_conf']:.2f}, min: {img['min_conf']:.2f}, ")
                    f.write(f"detections: {img['num_detections']})\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("To review these images:\n")
            f.write(f"  1. Open {review_dir}/images in your annotation tool\n")
            f.write(f"  2. Load labels from {review_dir}/labels\n")
            f.write("  3. Correct/add annotations as needed\n")
            f.write("  4. Copy corrected files back to train/val splits\n")

        stats['review_report'] = str(report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("AUTO-LABEL SUMMARY")
    print("=" * 60)
    print(f"Total images:            {stats['total_images']}")
    print(f"Unique images:           {stats.get('unique_images', stats['total_images'])}")
    print(f"With detections:         {stats['images_with_detections']}")
    print(f"Without detections:      {stats['images_without_detections']}")
    print(f"Total detections:        {stats['total_detections']}")

    print("\nDetections per class:")
    for cls_id, count in sorted(stats['detections_per_class'].items()):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"  {cls_id}: {cls_name}: {count}")

    print("\nSplit distribution:")
    for split_name, count in stats['split_counts'].items():
        if count > 0:
            print(f"  {split_name}: {count}")

    # Review summary
    if stats['images_for_review'] > 0:
        print("\n" + "-" * 40)
        print("REVIEW NEEDED")
        print("-" * 40)
        print(f"Images for manual review: {stats['images_for_review']}")
        if review_dir:
            print(f"Review folder:            {review_dir}")
            print(f"Review report:            {review_dir / 'review_report.txt'}")

    print(f"\nOutput: {output_dir}")
    print(f"data.yaml: {data_yaml_path}")

    return stats


def autolabel_cli() -> None:
    """CLI entry point for auto-labeling."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using a trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-label single dataset
  yolo-autolabel --sources datasets/v1 --output datasets/merged --model models/v1.pt

  # Merge and auto-label multiple datasets
  yolo-autolabel --sources datasets/v1 datasets/v2 datasets/v3 \\
                 --output datasets/merged --model models/v1.pt

  # Custom classes and confidence
  yolo-autolabel --sources datasets/v1 --output datasets/merged \\
                 --model models/v1.pt --classes cat dog --confidence 0.3

  # Custom split ratio (70/20/10)
  yolo-autolabel --sources datasets/v1 --output datasets/merged \\
                 --model models/v1.pt --split 0.7 0.2 0.1

  # Include Roboflow augmented images (skipped by default)
  yolo-autolabel --sources datasets/v1 datasets/v2 --output datasets/all \\
                 --model models/v1.pt --include-augmented

  # Auto-orient + resize to 640x640 (recommended for training)
  yolo-autolabel --sources datasets/v1 --output datasets/640 \\
                 --model models/v1.pt --auto-orient --resize 640

  # Save confidence in labels + auto-flag low-confidence for review
  yolo-autolabel --sources datasets/v1 --output datasets/reviewed \\
                 --model models/v1.pt --save-confidence --review-threshold 0.5

  # Full pipeline: resize + confidence + review threshold
  yolo-autolabel --sources datasets/v1 datasets/v2 --output datasets/merged \\
                 --model models/v1.pt --resize 640 --save-confidence --review-threshold 0.5
        """
    )

    parser.add_argument(
        '--sources', '-s',
        nargs='+',
        required=True,
        help='Source directories containing images to annotate (relative paths are resolved from workspace root)'
    )

    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for merged/annotated dataset (relative paths are resolved from workspace root)'
    )

    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to YOLO model for predictions (relative paths are resolved from workspace root)'
    )

    parser.add_argument(
        '--classes', '-c',
        nargs='+',
        default=None,
        help='Class names (default: use model classes). Example: --classes cat dog'
    )

    parser.add_argument(
        '--confidence', '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold for detections (default: 0.25)'
    )

    parser.add_argument(
        '--split',
        nargs=3,
        type=float,
        default=[0.8, 0.2, 0.0],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Split ratios (default: 0.8 0.2 0.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for split (default: 42)'
    )

    parser.add_argument(
        '--no-copy',
        action='store_true',
        help='Create symlinks instead of copying images'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files instead of skipping'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )

    parser.add_argument(
        '--include-augmented',
        action='store_true',
        help='Include Roboflow augmented images (by default they are skipped)'
    )

    parser.add_argument(
        '--auto-orient',
        action='store_true',
        help='Fix EXIF orientation issues (re-orient images)'
    )

    parser.add_argument(
        '--resize',
        type=int,
        default=None,
        metavar='SIZE',
        help='Resize images to square (e.g., --resize 640 for 640x640 with letterbox)'
    )

    parser.add_argument(
        '--save-confidence',
        action='store_true',
        help='Save confidence scores in label files (6th column)'
    )

    parser.add_argument(
        '--review-threshold',
        type=float,
        default=None,
        metavar='CONF',
        help='Images with avg confidence below this threshold go to review/ folder for manual annotation'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Sample 20 images and report expected results without writing files'
    )

    args = parser.parse_args()

    # Convert paths (resolve relative paths from workspace root)
    sources = [resolve_workspace_path(s) for s in args.sources]
    output_dir = resolve_workspace_path(args.output)
    model_path = resolve_workspace_path(args.model)

    # Run
    try:
        autolabel_dataset(
            sources=sources,
            output_dir=output_dir,
            model_path=model_path,
            class_names=args.classes,
            confidence=args.confidence,
            split_ratio=tuple(args.split),
            seed=args.seed,
            copy_images=not args.no_copy,
            skip_existing=not args.overwrite,
            skip_augmented=not args.include_augmented,  # Skip by default
            auto_orient=args.auto_orient,
            resize=args.resize,
            save_confidence=args.save_confidence,
            review_threshold=args.review_threshold,
            verbose=args.verbose,
            dry_run=args.dry_run
        )

        print("\nDone!")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


__all__ = [
    "autolabel_dataset",
    "COCO_CLASSES",
    "collect_images_from_sources",
    "find_images",
    "get_base_name",
    "get_coco_overlap",
    "is_augmented",
    "preprocess_image",
    "autolabel_cli",
]


if __name__ == '__main__':
    autolabel_cli()
