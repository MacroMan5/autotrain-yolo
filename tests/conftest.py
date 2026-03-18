"""
Shared test fixtures for yolocc.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """
    Create a minimal YOLO-format dataset for testing.

    Structure:
        tmp_path/
            data.yaml
            images/
                train/  (6 images)
                val/    (4 images)
            labels/
                train/  (6 label files, 1 empty, 1 image has no label)
                val/    (4 label files)

    Returns the dataset root Path.
    """
    # Create directories
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)

    # data.yaml
    data_cfg = {
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": {0: "cat", 1: "dog"},
    }
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data_cfg))

    # --- Training split (6 images) ---
    train_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    train_annotations = [
        "0 0.5 0.5 0.3 0.2\n",        # img_000: cat
        "1 0.4 0.6 0.25 0.35\n",       # img_001: dog
        "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",  # img_002: cat+dog
        "0 0.5 0.5 0.4 0.4\n",         # img_003: cat
        "",                              # img_004: empty label file
        "1 0.5 0.5 0.3 0.3\n",         # img_005: dog (but img_005 has no label, see below)
    ]

    for i, (color, annotation) in enumerate(zip(train_colors, train_annotations)):
        img_name = f"img_{i:03d}.png"
        img = Image.new("RGB", (32, 32), color=color)
        img.save(tmp_path / "images" / "train" / img_name)

        # img_005: create image but skip the label file (orphaned image)
        if i == 5:
            continue

        lbl_name = f"img_{i:03d}.txt"
        (tmp_path / "labels" / "train" / lbl_name).write_text(annotation)

    # --- Validation split (4 images) ---
    val_colors = [
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
    ]
    val_annotations = [
        "0 0.5 0.5 0.3 0.2\n",
        "1 0.6 0.4 0.2 0.3\n",
        "0 0.5 0.5 0.25 0.25\n",
        "1 0.5 0.5 0.35 0.35\n",
    ]

    for i, (color, annotation) in enumerate(zip(val_colors, val_annotations)):
        img_name = f"val_{i:03d}.png"
        img = Image.new("RGB", (32, 32), color=color)
        img.save(tmp_path / "images" / "val" / img_name)

        lbl_name = f"val_{i:03d}.txt"
        (tmp_path / "labels" / "val" / lbl_name).write_text(annotation)

    return tmp_path


@pytest.fixture
def yolo_project_config(tmp_path: Path) -> Path:
    """
    Create a minimal yolo-project.yaml for testing.

    Returns the path to the config file.
    """
    config = {
        "project": {
            "name": "test-project",
            "description": "A test YOLO project",
        },
        "classes": {
            0: "cat",
            1: "dog",
        },
        "defaults": {
            "epochs": 10,
            "batch": 16,
            "imgsz": 640,
            "lr0": 0.01,
        },
        "variants": {
            "small": {
                "imgsz": 320,
                "batch": 32,
            },
        },
    }
    config_path = tmp_path / "yolo-project.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path
