#!/usr/bin/env python3
"""
Training Utilities
==================

Common functions used by training and fine-tuning scripts.

Exit Codes:
    EXIT_SUCCESS (0): Operation completed successfully
    EXIT_VALIDATION_FAILED (1): Dataset validation failed
    EXIT_TRAINING_FAILED (2): Training process failed
    EXIT_ANALYSIS_FAILED (3): Post-training analysis failed
    EXIT_MODEL_INVALID (4): Model validation failed
    EXIT_CHECKPOINT_NOT_FOUND (5): Checkpoint file not found

Functions:
    check_gpu: Check GPU availability and VRAM
    validate_checkpoint: Validate checkpoint file exists
    save_training_summary: Save training metrics to JSON
    validate_model_classes: Validate model has expected number of classes
    copy_model_safe: Safely copy model files with error handling
"""

import json
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import yaml


# ============================================
# EXIT CODES
# ============================================
# Standardized exit codes for better error handling in CI/CD

EXIT_SUCCESS = 0
EXIT_VALIDATION_FAILED = 1
EXIT_TRAINING_FAILED = 2
EXIT_ANALYSIS_FAILED = 3
EXIT_MODEL_INVALID = 4
EXIT_CHECKPOINT_NOT_FOUND = 5


def check_gpu(warn_only: bool = True) -> bool:
    """
    Check GPU availability and VRAM.

    Args:
        warn_only: If True, print warning but continue.
                   If False, exit if no GPU available.

    Returns:
        True if GPU available, False otherwise.
    """
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected, training will be slow!")
        if not warn_only:
            print("Use --cpu flag to force CPU training, or install CUDA.")
            sys.exit(EXIT_TRAINING_FAILED)
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")

    if gpu_mem < 4:
        print("WARNING: Less than 4GB VRAM. Consider reducing batch size.")
        print("  Use: --batch 8")

    return True


def get_device() -> str:
    """Return the best available device string for training.

    Returns '0' (first GPU) if CUDA is available, otherwise 'cpu'.
    """
    return "0" if torch.cuda.is_available() else "cpu"


def validate_checkpoint(checkpoint_path: str) -> None:
    """
    Validate that a checkpoint file exists before loading.

    Args:
        checkpoint_path: Path to the checkpoint file.

    Raises:
        SystemExit: If checkpoint does not exist.
    """
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Make sure the path is correct and the file exists.")
        sys.exit(EXIT_CHECKPOINT_NOT_FOUND)


def save_training_summary(
    results: Any,
    model_path: str,
    output_dir: str = "reports",
    variant: Optional[str] = None
) -> Dict[str, Any]:
    """
    Save and display training results summary.

    Args:
        results: Results returned by model.train()
        model_path: Path to the saved model
        output_dir: Directory for saving metrics
        variant: Variant name (optional, for fine-tuning)

    Returns:
        Dictionary with metrics summary
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Validate results object
    if not hasattr(results, 'results_dict'):
        print("WARNING: Training results incomplete, metrics may be missing")
        metrics = {}
    else:
        metrics = results.results_dict

    # Get model size safely
    model_path = Path(model_path)
    try:
        model_size = model_path.stat().st_size / 1e6
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {model_path}")
        model_size = 0

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "model_size_mb": round(model_size, 2),
        "metrics": {
            "mAP50": round(metrics.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(metrics.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(metrics.get("metrics/precision(B)", 0), 4),
            "recall": round(metrics.get("metrics/recall(B)", 0), 4),
        },
        "training": {
            "epochs": int(metrics.get("epoch", 0)),
        }
    }

    # Add variant info if provided
    if variant:
        summary["variant"] = variant

    # Save JSON
    model_name = model_path.stem
    json_path = Path(output_dir) / f"{model_name}_metrics.json"

    try:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
    except IOError as e:
        print(f"WARNING: Could not save metrics to {json_path}: {e}")

    # Display summary
    title = f"TRAINING SUMMARY ({variant.upper()})" if variant else "TRAINING SUMMARY"
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"mAP@0.5:      {summary['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5-0.95: {summary['metrics']['mAP50-95']:.4f}")
    print(f"Precision:    {summary['metrics']['precision']:.4f}")
    print(f"Recall:       {summary['metrics']['recall']:.4f}")
    print(f"\nModel size:   {model_size:.1f} MB")
    print(f"Metrics:      {json_path}")

    return summary


def validate_model_classes(model: Any, expected_classes: int) -> None:
    """
    Validate that a model has the expected number of classes.

    Args:
        model: YOLO model instance
        expected_classes: Expected number of classes

    Raises:
        SystemExit: If class count doesn't match.
    """
    if not hasattr(model, 'names'):
        print("ERROR: Model does not have class names attribute")
        sys.exit(EXIT_MODEL_INVALID)

    num_classes = len(model.names)
    if num_classes != expected_classes:
        print(f"ERROR: Model has {num_classes} classes, expected {expected_classes}")
        sys.exit(EXIT_MODEL_INVALID)

    # Get class names safely (handles both dict and list)
    if isinstance(model.names, dict):
        names_str = ', '.join(model.names.values())
    else:
        names_str = ', '.join(model.names)

    print(f"Model validated: {num_classes} classes ({names_str})")


def copy_model_safe(src: Path, dst: Path) -> bool:
    """
    Safely copy a model file with error handling.

    Args:
        src: Source path
        dst: Destination path

    Returns:
        True if successful, False otherwise
    """
    import shutil

    if not src.exists():
        print(f"ERROR: Source model not found: {src}")
        return False

    try:
        # Create parent directories if needed
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Remove destination if exists (for Windows compatibility)
        if dst.exists():
            dst.unlink()
        shutil.copy(src, dst)
        print(f"Model saved to: {dst}")
        return True
    except IOError as e:
        print(f"ERROR: Could not copy model: {e}")
        return False


def prepare_ultralytics_data_yaml(data_yaml: Path, dataset_path: Path) -> tuple:
    """
    Normalize dataset YAML so Ultralytics resolves dataset paths correctly.

    Ultralytics can resolve relative ``path`` values against the current working
    directory. For datasets using ``path: .``, that may incorrectly point to the
    repo root instead of the dataset folder. This helper writes a temporary YAML
    with an absolute ``path`` when needed.

    Returns:
        Tuple of (data_yaml_path_to_use, temp_file_path_or_None)
    """
    try:
        with open(data_yaml, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
    except Exception:
        return str(data_yaml), None

    if not isinstance(data_cfg, dict):
        return str(data_yaml), None

    raw_path = data_cfg.get("path")
    normalized_path: Optional[Path] = None

    if raw_path is None or str(raw_path).strip() == "":
        normalized_path = dataset_path.resolve()
    else:
        path_candidate = Path(str(raw_path))
        if not path_candidate.is_absolute():
            normalized_path = (dataset_path / path_candidate).resolve()

    if normalized_path is None:
        return str(data_yaml), None

    data_cfg["path"] = str(normalized_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="yolo_data_",
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        yaml.safe_dump(data_cfg, tmp_file, sort_keys=False)
        return tmp_file.name, Path(tmp_file.name)


__all__ = [
    "EXIT_SUCCESS",
    "EXIT_VALIDATION_FAILED",
    "EXIT_TRAINING_FAILED",
    "EXIT_ANALYSIS_FAILED",
    "EXIT_MODEL_INVALID",
    "EXIT_CHECKPOINT_NOT_FOUND",
    "check_gpu",
    "get_device",
    "validate_checkpoint",
    "save_training_summary",
    "validate_model_classes",
    "copy_model_safe",
    "prepare_ultralytics_data_yaml",
]
