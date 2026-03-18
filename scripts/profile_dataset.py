#!/usr/bin/env python3
"""Dataset profiler for YOLO architecture selection.

Outputs structured YAML profile with object scale distribution at training
resolution, class-wise stats, and architecture suggestion.

Usage:
    python scripts/profile_dataset.py --labels datasets/my_data/labels/train --imgsz 640
    python scripts/profile_dataset.py --labels datasets/my_data/labels/train --images datasets/my_data/images/train --imgsz 640
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)


def find_images_dir(labels_dir: Path) -> Path | None:
    """Infer images directory from labels directory (labels/ -> images/)."""
    if "labels" in labels_dir.parts:
        images_dir = Path(str(labels_dir).replace("labels", "images", 1))
        if images_dir.is_dir():
            return images_dir
    return None


def get_image_dimensions(images_dir: Path, max_samples: int = 1000) -> dict[str, tuple[int, int]]:
    """Sample image dimensions. Returns {stem: (width, height)}."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
    if len(image_files) > max_samples:
        image_files = random.sample(image_files, max_samples)
    dims = {}
    for f in image_files:
        try:
            with Image.open(f) as img:
                dims[f.stem] = img.size  # (width, height)
        except Exception:
            continue
    return dims


def compute_object_sizes(labels_dir: Path, image_dims: dict, imgsz: int) -> list[dict]:
    """Compute object sizes at training resolution with letterbox scaling."""
    objects = []
    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        if stem not in image_dims:
            continue
        native_w, native_h = image_dims[stem]
        scale = imgsz / max(native_w, native_h)
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                norm_w, norm_h = float(parts[3]), float(parts[4])
                pixel_w = norm_w * native_w * scale
                pixel_h = norm_h * native_h * scale
                area = pixel_w * pixel_h
                min_side = min(pixel_w, pixel_h)
                objects.append({
                    "class_id": cls_id,
                    "pixel_w": pixel_w,
                    "pixel_h": pixel_h,
                    "area": area,
                    "min_side": min_side,
                    "image": stem,
                })
    return objects


def classify_scale(area: float) -> str:
    """COCO scale classification by area."""
    if area < 1024:    # 32^2
        return "small"
    elif area < 9216:  # 96^2
        return "medium"
    else:
        return "large"


def suggest_architecture(small_pct: float, large_pct: float, min_obj_px: float, num_classes: int) -> dict:
    """Suggest architecture based on dataset profile."""
    # Head config
    if small_pct > 0.50 and large_pct < 0.10:
        head = "yolo11-p2p3p4.yaml"
        reason = f"{small_pct:.0%} small, {large_pct:.0%} large — shifted pyramid (P2/P3/P4)"
    elif small_pct > 0.30 or min_obj_px < 10:
        head = "yolo11-p2.yaml"
        reason = f"{small_pct:.0%} small objects, min {min_obj_px:.0f}px — P2 head recommended"
    else:
        head = "yolo11.yaml"
        reason = f"{small_pct:.0%} small objects — standard P3/P4/P5 sufficient"

    # Scale
    if num_classes <= 5:
        scale = "n"
    elif num_classes <= 20:
        scale = "s"
    elif num_classes <= 80:
        scale = "m"
    else:
        scale = "l"

    return {"head_config": head, "scale": scale, "reasoning": reason}


def count_splits(labels_dir: Path) -> dict[str, int]:
    """Count images in train/val splits."""
    counts = {}
    parent = labels_dir.parent
    for split in ["train", "val", "test"]:
        split_dir = parent / split
        if split_dir.is_dir():
            counts[split] = len(list(split_dir.glob("*.txt")))
    if not counts:
        counts["train"] = len(list(labels_dir.glob("*.txt")))
    return counts


def main():
    parser = argparse.ArgumentParser(description="Profile YOLO dataset for architecture selection")
    parser.add_argument("--labels", required=True, help="Path to labels directory (e.g., datasets/data/labels/train)")
    parser.add_argument("--images", help="Path to images directory (auto-inferred from --labels if omitted)")
    parser.add_argument("--imgsz", type=int, default=640, help="Training resolution (default: 640)")
    parser.add_argument("--class-names", help="Comma-separated class names (optional)")
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    if not labels_dir.is_dir():
        print(f"ERROR: Labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    label_files = list(labels_dir.glob("*.txt"))
    if len(label_files) < 10:
        print(f"ERROR: Only {len(label_files)} labeled images found (minimum 10 required)", file=sys.stderr)
        sys.exit(1)
    if len(label_files) < 100:
        print(f"WARNING: Only {len(label_files)} labeled images found (100+ recommended)", file=sys.stderr)

    # Find images
    images_dir = Path(args.images) if args.images else find_images_dir(labels_dir)
    if not images_dir or not images_dir.is_dir():
        print(f"ERROR: Cannot find images directory. Use --images to specify.", file=sys.stderr)
        sys.exit(1)

    # Sample image dimensions
    max_samples = 1000 if len(label_files) >= 1000 else len(label_files) + 100
    image_dims = get_image_dimensions(images_dir, max_samples=max_samples)
    if not image_dims:
        print("ERROR: Could not read any image dimensions", file=sys.stderr)
        sys.exit(1)

    # Compute object sizes at training resolution
    objects = compute_object_sizes(labels_dir, image_dims, args.imgsz)
    if not objects:
        print("ERROR: No valid objects found in labels", file=sys.stderr)
        sys.exit(1)

    # Class names
    class_names = {}
    if args.class_names:
        for i, name in enumerate(args.class_names.split(",")):
            class_names[i] = name.strip()

    # Scale distribution
    scales = [classify_scale(o["area"]) for o in objects]
    total = len(scales)
    small_pct = round(scales.count("small") / total, 2)
    medium_pct = round(scales.count("medium") / total, 2)
    large_pct = round(scales.count("large") / total, 2)

    # Min object size (by min side length)
    min_object_px = round(min(o["min_side"] for o in objects), 1)

    # Objects per image
    objects_per_image = defaultdict(int)
    for o in objects:
        objects_per_image[o["image"]] += 1
    avg_per_image = round(sum(objects_per_image.values()) / len(objects_per_image), 1)
    max_per_image = max(objects_per_image.values())

    # Class profiles sorted by avg size ascending
    class_data = defaultdict(list)
    for o in objects:
        class_data[o["class_id"]].append(o)

    class_profiles = []
    for cls_id in sorted(class_data.keys()):
        cls_objects = class_data[cls_id]
        cls_scales = [classify_scale(o["area"]) for o in cls_objects]
        avg_min_side = sum(o["min_side"] for o in cls_objects) / len(cls_objects)
        class_profiles.append({
            "class_id": cls_id,
            "name": class_names.get(cls_id, f"class_{cls_id}"),
            "count": len(cls_objects),
            "avg_size_px": round(avg_min_side, 1),
            "pct_small": round(cls_scales.count("small") / len(cls_scales), 2),
        })
    class_profiles.sort(key=lambda x: x["avg_size_px"])

    num_classes = len(class_data)

    # Train/val split counts
    split_counts = count_splits(labels_dir)

    # Train/val divergence check
    train_val_divergence = {"scale_divergence": False, "note": ""}
    val_labels = labels_dir.parent / "val"
    if val_labels.is_dir():
        val_image_dir = find_images_dir(val_labels)
        if val_image_dir:
            val_dims = get_image_dimensions(val_image_dir, max_samples=500)
            val_objects = compute_object_sizes(val_labels, val_dims, args.imgsz)
            if val_objects:
                val_scales = [classify_scale(o["area"]) for o in val_objects]
                val_small_pct = val_scales.count("small") / len(val_scales)
                diff = abs(small_pct - val_small_pct)
                if diff > 0.15:
                    train_val_divergence = {
                        "scale_divergence": True,
                        "note": f"train small={small_pct:.0%}, val small={val_small_pct:.0%} (diff={diff:.0%} > 15%)",
                    }

    # Architecture suggestion
    suggestion = suggest_architecture(small_pct, large_pct, min_object_px, num_classes)
    suggestion["imgsz"] = args.imgsz

    # Build profile
    profile = {
        "profile": {
            "imgsz": args.imgsz,
            "total_images": split_counts,
            "num_classes": num_classes,
            "scale_distribution": {
                "small_pct": small_pct,
                "medium_pct": medium_pct,
                "large_pct": large_pct,
            },
            "min_object_px": min_object_px,
            "avg_objects_per_image": avg_per_image,
            "max_objects_per_image": max_per_image,
            "class_profiles": class_profiles,
            "train_val_divergence": train_val_divergence,
            "suggested_architecture": suggestion,
        }
    }

    yaml.dump(profile, sys.stdout, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
