#!/usr/bin/env python3
"""
Dataset Cleaner & Deduplicator
==============================

Universally cleans YOLO datasets by:
1. Removing images without valid labels (orphans or empty).
2. Removing visually duplicate images using dHash (Difference Hashing).

This ensures the model doesn't overfit on repetitive data and
maintains high-quality training signals.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from tqdm import tqdm

from yolocc.paths import resolve_workspace_path

def dhash(image, hash_size: int = 8) -> int:
    """
    Compute the Difference Hash for an image.
    Robust against slight compression and resizing.
    """
    # Resize to (hash_size + 1, hash_size)
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    # Compute differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert to integer hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def find_label_path(img_path: Path) -> Optional[Path]:
    """Find the corresponding label path for an image, supporting various structures."""
    # Common YOLO structures:
    # 1. images/train/img.jpg -> labels/train/img.txt
    # 2. images/img.jpg -> labels/img.txt
    # 3. img.jpg -> img.txt (flat)

    label_name = img_path.with_suffix('.txt').name

    # Check current dir (flat)
    if (img_path.parent / label_name).exists():
        return img_path.parent / label_name

    # Check sibling 'labels' folder (split structure)
    parts = list(img_path.parts)
    if 'images' in parts:
        # Use last occurrence — avoids replacing a parent dir also named 'images'
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        lbl_path = Path(*parts).with_suffix('.txt')
        if lbl_path.exists():
            return lbl_path

    return None

def clean_dataset(
    dataset_dir: str,
    remove_empty: bool = True,
    remove_duplicates: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Universally clean and deduplicate a dataset.
    """
    base_path = Path(dataset_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    print(f"\n{'='*60}")
    print(f"DATASET CLEANER: {base_path.name.upper()}")
    print(f"{'='*60}")
    print(f"Path: {base_path}")
    print(f"Options: Empty Labels={remove_empty}, Duplicates={remove_duplicates}")
    if dry_run:
        print("MODE: DRY RUN (No files will be deleted)")
    print("-" * 60)

    if remove_duplicates and not CV2_AVAILABLE:
        print("WARNING: cv2/numpy not available. Skipping deduplication.")
        print("  Install: pip install opencv-python numpy")
        remove_duplicates = False

    # Collect all images recursively
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(base_path.rglob(f"*{ext}")))

    total_found = len(all_images)
    print(f"Found {total_found} images.")

    removed_empty = 0
    removed_dupes = 0
    corrupted_count = 0
    hashes: Dict[int, Path] = {}
    to_delete_imgs: List[Path] = []
    to_delete_lbls: List[Path] = []

    valid_images = []

    # Step 1 & 2: Single pass for efficiency
    for img_path in tqdm(all_images, desc="Processing images"):
        lbl_path = find_label_path(img_path)

        # 1. Check Label Validity
        if remove_empty:
            is_valid_label = False
            if lbl_path and lbl_path.exists():
                if lbl_path.stat().st_size > 0:
                    with open(lbl_path, 'r') as f:
                        if f.read().strip():
                            is_valid_label = True

            if not is_valid_label:
                to_delete_imgs.append(img_path)
                if lbl_path and lbl_path.exists():
                    to_delete_lbls.append(lbl_path)
                removed_empty += 1
                continue

        # 2. Check Duplicates
        if remove_duplicates:
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    corrupted_count += 1
                    print(f"WARNING: Could not read image (corrupted?): {img_path}")
                    continue

                # Skip dedup for uniform images (all same color -> identical hash)
                if np.std(img) < 2.0:
                    valid_images.append(img_path)
                    continue

                h = dhash(img)
                if h in hashes:
                    to_delete_imgs.append(img_path)
                    if lbl_path and lbl_path.exists():
                        to_delete_lbls.append(lbl_path)
                    removed_dupes += 1
                else:
                    hashes[h] = img_path
                    valid_images.append(img_path)
            except Exception as e:
                print(f"Error hashing {img_path}: {e}")
        else:
            valid_images.append(img_path)

    # Execution phase
    if not dry_run:
        for p in to_delete_imgs:
            if p.exists():
                p.unlink()
        for p in to_delete_lbls:
            if p.exists():
                p.unlink()

    print(f"\n{'='*50}")
    print("CLEANING SUMMARY")
    print(f"{'='*50}")
    print(f"Empty/Missing labels removed: {removed_empty}")
    print(f"Duplicates removed:          {removed_dupes}")
    print(f"Corrupted/unreadable:        {corrupted_count}")
    print(f"Total files removed:         {removed_empty + removed_dupes}")
    print(f"Remaining valid images:      {len(valid_images)}")
    print(f"{'='*50}\n")

    return {
        'removed_empty': removed_empty,
        'removed_dupes': removed_dupes,
        'corrupted': corrupted_count,
        'remaining': len(valid_images)
    }

def clean_cli():
    """CLI Entry point."""
    parser = argparse.ArgumentParser(description="Clean and deduplicate YOLO dataset")
    parser.add_argument("dataset", help="Dataset directory")
    parser.add_argument("--no-empty", action="store_false", dest="empty", help="Do not remove empty labels")
    parser.add_argument("--no-dupes", action="store_false", dest="dupes", help="Do not remove duplicates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without deleting")

    args = parser.parse_args()

    dataset_path = resolve_workspace_path(args.dataset)
    clean_dataset(str(dataset_path), args.empty, args.dupes, args.dry_run)

if __name__ == "__main__":
    clean_cli()
