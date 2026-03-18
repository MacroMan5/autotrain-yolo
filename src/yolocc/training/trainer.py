#!/usr/bin/env python3
"""
Training Module - Base Training and Fine-tuning
================================================

Provides functions for training YOLO models and
fine-tuning them for specific variants via transfer learning.

Functions:
    train: Train a YOLO model
    finetune: Fine-tune a model for a specific variant
    validate_model: Validate an existing trained model
    train_cli: CLI entry point for base training
    finetune_cli: CLI entry point for fine-tuning

Constants:
    DEFAULT_BASE_MODEL: Default base model path
"""

import argparse
import re
import sys
import yaml
from typing import Optional, Any

from ultralytics import YOLO

from yolocc.training.utils import (
    check_gpu,
    get_device,
    validate_checkpoint,
    save_training_summary,
    copy_model_safe,
    prepare_ultralytics_data_yaml,
    EXIT_VALIDATION_FAILED,
    EXIT_TRAINING_FAILED,
    EXIT_MODEL_INVALID,
)
from yolocc.paths import (
    get_models_root,
    get_reports_root,
    resolve_workspace_path,
)
from yolocc.project import load_project_config, get_default, warn_no_config


# ============================================
# CONFIGURATION
# ============================================

DEFAULT_YOLO_MODEL = "yolo11n.pt"

# Default paths now resolved via workspace helper so that all generated
# artifacts can live outside the tool repository when desired.
DEFAULT_DATASET_DIR = "datasets"
DEFAULT_BASE_MODEL = "yolo11n.pt"


def _sanitize_path_arg(path_value: Optional[str]) -> Optional[str]:
    """
    Sanitize path arguments coming from CLI input.

    If a path contains control whitespace (newlines/tabs), collapse all
    whitespace to recover common multiline paste artifacts such as:
    "C:\\n  \\Users\\...".
    """
    if path_value is None:
        return None

    cleaned = str(path_value).strip().strip("\"").strip("'")

    if any(char in cleaned for char in ("\n", "\r", "\t")):
        cleaned = re.sub(r"[\n\r\t]+", "", cleaned)

    return cleaned or None


def train(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_dir: Optional[str] = None,
    base_model: str = DEFAULT_YOLO_MODEL,
    model_name: str = "train",
    config: Optional[str] = None,
    skip_validation: bool = False,
    skip_analysis: bool = False,
    resume: Optional[str] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    patience: Optional[int] = None,
    project_defaults: Optional[dict] = None,
) -> Any:
    """
    Train a YOLO model.

    Args:
        dataset_dir: Path to dataset directory
        output_dir: Directory for model outputs
        base_model: YOLO base model to start from
        model_name: Name for the project/output files
        config: Path to YAML config file with hyperparameter overrides
        skip_validation: Skip dataset validation
        skip_analysis: Skip post-training analysis
        resume: Path to checkpoint for resuming training
        epochs: Number of training epochs (overrides config)
        batch: Batch size (overrides config)
        imgsz: Image size for training (overrides config)
        patience: Training patience (early stopping) (overrides config)

    Returns:
        Trained YOLO model instance
    """
    # Import here to avoid circular imports
    from yolocc.dataset.validator import validate_dataset
    from yolocc.training.analyzer import analyze_training

    dataset_dir = _sanitize_path_arg(dataset_dir) or DEFAULT_DATASET_DIR
    if output_dir is None:
        output_dir = str(get_models_root())
    output_dir = _sanitize_path_arg(output_dir) or str(get_models_root())
    base_model = _sanitize_path_arg(base_model) or DEFAULT_YOLO_MODEL
    config = _sanitize_path_arg(config)
    resume = _sanitize_path_arg(resume)

    print("=" * 60)
    print(f"YOLO Training: {model_name.upper()}")
    print("=" * 60)

    # Check GPU
    check_gpu()

    dataset_path = resolve_workspace_path(dataset_dir)
    data_yaml = dataset_path / "data.yaml"
    output_path = resolve_workspace_path(output_dir)

    # Load config if provided
    overrides = {}
    if config:
        config_path = resolve_workspace_path(config)
        if config_path.exists():
            print(f"Loading config from: {config_path}")
            with open(config_path) as f:
                overrides = yaml.safe_load(f) or {}
        else:
            print(f"WARNING: Config file not found at {config_path}")

    # Override with CLI args if provided
    if epochs is not None:
        overrides['epochs'] = epochs
    if batch is not None:
        overrides['batch'] = batch
    if imgsz is not None:
        overrides['imgsz'] = imgsz
    if patience is not None:
        overrides['patience'] = patience

    # Set defaults if not in config
    training_params = {
        'epochs': 100,
        'imgsz': 640,
        'batch': -1,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'cos_lr': True,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'augment': True,
        'mosaic': 1.0,
        'close_mosaic': 10,
        'mixup': 0.15,
        'copy_paste': 0.1,
        'erasing': 0.4,
        'workers': 8,
        'amp': True,
        'save': True,
        'verbose': True,
        'plots': True,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

    # Layer 2: project config defaults (overrides hardcoded, overridden by --config file)
    if project_defaults:
        training_params.update({k: v for k, v in project_defaults.items()
                                if k in training_params})

    training_params.update(overrides)

    # ========================================
    # STEP 1: Dataset Validation
    # ========================================
    if not skip_validation:
        print("\n[STEP 1/3] Validating dataset...")
        print("-" * 40)

        if not data_yaml.exists():
            print(f"ERROR: Dataset not found at {data_yaml}")
            sys.exit(EXIT_VALIDATION_FAILED)

        is_valid = validate_dataset(str(dataset_path))

        if not is_valid:
            print("\nDataset validation FAILED!")
            sys.exit(EXIT_VALIDATION_FAILED)

        print("\n✓ Dataset validation passed!")
    else:
        print("\n[STEP 1/3] Skipping dataset validation...")

    # ========================================
    # STEP 2: Training
    # ========================================
    print("\n[STEP 2/3] Training model...")
    print("-" * 40)

    # Resume from checkpoint or start fresh
    if resume:
        resume_path = resolve_workspace_path(resume)
        validate_checkpoint(str(resume_path))
        print(f"Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        print(f"Loading base model: {base_model}")
        model = YOLO(base_model)

    print("\nTraining configuration:")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Base model: {base_model}")
    print(f"  - Resume: {resume or 'No'}")
    print(f"  - Output: {output_path}/{model_name}/")
    ultra_data_yaml, temp_data_yaml = prepare_ultralytics_data_yaml(data_yaml, dataset_path)

    # Remove arguments that are explicitly passed to model.train
    # to avoid "got multiple values for keyword argument" errors
    for arg in ['data', 'project', 'name', 'exist_ok', 'resume', 'device']:
        if arg in training_params:
            del training_params[arg]

    try:
        results = model.train(
            data=ultra_data_yaml,
            project=str(output_path),
            name=model_name,
            exist_ok=True,
            resume=bool(resume),
            device=get_device(),
            **training_params
        )
    finally:
        if temp_data_yaml is not None:
            temp_data_yaml.unlink(missing_ok=True)

    # Copy best model
    best_model_file = output_path / model_name / "weights" / "best.pt"
    final_model_file = output_path / f"{model_name}.pt"

    if not copy_model_safe(best_model_file, final_model_file):
        print("Training may have failed or model copy error")
        sys.exit(EXIT_TRAINING_FAILED)

    # Save training metrics
    reports_root = get_reports_root()
    save_training_summary(results, str(final_model_file), output_dir=str(reports_root))

    # ========================================
    # STEP 3: Post-Training Analysis
    # ========================================
    if not skip_analysis:
        print("\n[STEP 3/3] Analyzing training results...")
        print("-" * 40)

        analyze_training(
            model_path=str(final_model_file),
            dataset_path=str(dataset_path),
            output_dir=str(reports_root)
        )

    return model



def finetune(
    variant: str,
    epochs: int = 30,
    base_model: str = DEFAULT_BASE_MODEL,
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_validation: bool = False,
    skip_analysis: bool = False,
    resume: Optional[str] = None,
    batch: int = 16,
    imgsz: int = 640,
    freeze: int = 10,
    version: str = "v1",
) -> Any:
    """
    Fine-tune a model for a specific variant.

    Uses transfer learning with frozen backbone layers
    to adapt the model to a specific variant.

    Args:
        variant: Variant name for fine-tuning
        epochs: Number of fine-tuning epochs
        base_model: Path to base model
        dataset_dir: Path to variant dataset (default: datasets/{variant})
        output_dir: Directory for model outputs
        skip_validation: Skip dataset validation
        skip_analysis: Skip post-training analysis
        resume: Path to checkpoint for resuming
        batch: Batch size
        imgsz: Image size for training
        freeze: Number of layers to freeze (backbone)

    Returns:
        Fine-tuned YOLO model instance
    """
    # Import here to avoid circular imports
    from yolocc.dataset.validator import validate_dataset
    from yolocc.training.analyzer import analyze_training

    base_model = _sanitize_path_arg(base_model) or DEFAULT_BASE_MODEL
    if output_dir is None:
        output_dir = str(get_models_root())
    output_dir = _sanitize_path_arg(output_dir) or str(get_models_root())
    dataset_dir = _sanitize_path_arg(dataset_dir)
    resume = _sanitize_path_arg(resume)

    print("=" * 60)
    print(f"Fine-tuning variant: {variant.upper()}")
    print("=" * 60)

    # Check GPU
    check_gpu()

    # Paths
    if dataset_dir is None:
        dataset_dir = f"datasets/{variant}"
    dataset_path = resolve_workspace_path(dataset_dir)
    data_yaml = dataset_path / "data.yaml"
    output_path = resolve_workspace_path(output_dir)

    # ========================================
    # Preliminary checks
    # ========================================

    base_model_path = resolve_workspace_path(base_model)

    # Base model exists?
    if not base_model_path.exists():
        print(f"ERROR: Base model not found at {base_model_path}")
        print("Run first: yolo-train")
        sys.exit(EXIT_MODEL_INVALID)

    # Validate base model loads correctly
    try:
        test_model = YOLO(str(base_model_path))
        source_classes = len(test_model.names)
        print(f"  Base model: {source_classes} classes")
        del test_model
    except Exception as e:
        print(f"ERROR: Cannot load base model: {e}")
        sys.exit(EXIT_MODEL_INVALID)

    # Dataset exists?
    if not data_yaml.exists():
        print(f"ERROR: Dataset not found at {data_yaml}")
        print(f"Ensure your dataset is at: {dataset_path}")
        sys.exit(EXIT_VALIDATION_FAILED)

    # ========================================
    # STEP 1: Dataset Validation
    # ========================================
    if not skip_validation:
        print("\n[STEP 1/3] Validating dataset...")
        print("-" * 40)

        is_valid = validate_dataset(str(dataset_path))

        if not is_valid:
            print("\nDataset validation FAILED!")
            print("Fix the errors above before fine-tuning.")
            print("Or use --skip-validation to bypass (not recommended)")
            sys.exit(EXIT_VALIDATION_FAILED)

        print("\n✓ Dataset validation passed!")
    else:
        print("\n[STEP 1/3] Skipping dataset validation...")

    # ========================================
    # STEP 2: Fine-tuning
    # ========================================
    print("\n[STEP 2/3] Fine-tuning model...")
    print("-" * 40)

    # Resume from checkpoint or start fresh
    if resume:
        resume_path = resolve_workspace_path(resume)
        validate_checkpoint(str(resume_path))
        print(f"Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
    else:
        print(f"Loading base model: {base_model_path}")
        model = YOLO(str(base_model_path))

    print("\nFine-tuning configuration:")
    print(f"  - Variant: {variant}")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Resume: {resume or 'No'}")
    print(f"  - Frozen layers: {freeze} (backbone)")
    print("  - Learning rate: 0.001 (10x lower than base)")
    ultra_data_yaml, temp_data_yaml = prepare_ultralytics_data_yaml(data_yaml, dataset_path)

    try:
        results = model.train(
            data=ultra_data_yaml,
            project=str(output_path),
            name=f"{variant}_{version}",
            exist_ok=True,
            resume=bool(resume),

            # Fine-tuning params
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=get_device(),
            patience=10,

            # Transfer learning - IMPORTANT
            freeze=freeze,

            # Learning rate (lower for fine-tuning)
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=1.0,
            warmup_momentum=0.8,

            # Reduced augmentation for fine-tuning
            augment=True,
            mosaic=0.5,
            close_mosaic=5,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.3,

            # Performance
            workers=8,
            amp=True,

            # Logging
            save=True,
            verbose=True,
            plots=True,
        )
    finally:
        if temp_data_yaml is not None:
            temp_data_yaml.unlink(missing_ok=True)

    # Copy best model
    best_model_path = output_path / f"{variant}_{version}" / "weights" / "best.pt"
    final_model = output_path / f"{variant}_{version}.pt"

    if not copy_model_safe(best_model_path, final_model):
        print("Training may have failed or model copy error")
        sys.exit(EXIT_TRAINING_FAILED)

    # Save training metrics
    variant_reports_dir = get_reports_root() / variant
    save_training_summary(results, str(final_model), output_dir=str(variant_reports_dir))

    # ========================================
    # STEP 3: Post-Training Analysis
    # ========================================
    if not skip_analysis:
        print("\n[STEP 3/3] Analyzing training results...")
        print("-" * 40)

        analyze_training(
            model_path=str(final_model),
            dataset_path=str(dataset_path),
            output_dir=str(variant_reports_dir)
        )
    else:
        print("\n[STEP 3/3] Skipping post-training analysis...")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(f"Fine-tuning Complete! ({variant.upper()})")
    print("=" * 60)
    print(f"\nModel: {final_model}")
    print(f"Logs:  {output_path}/{variant}_{version}/")

    if not skip_analysis:
        print(f"Report: {variant_reports_dir / 'uncertain_images.txt'}")

    print("\nNext steps:")
    print(f"  1. Export to ONNX: yolo-export --model {final_model}")
    print("  2. Run experiments: yolo-experiment --budget 5")

    return model


def validate_model(
    model_path: Optional[str] = None,
    dataset_dir: str = DEFAULT_DATASET_DIR
) -> Any:
    """
    Validate an existing trained model.

    Args:
        model_path: Path to model (default: yolo11n.pt)
        dataset_dir: Path to dataset for validation

    Returns:
        Validation metrics
    """
    if model_path is None:
        model_path = DEFAULT_BASE_MODEL
    model_path = _sanitize_path_arg(model_path) or DEFAULT_BASE_MODEL
    dataset_dir = _sanitize_path_arg(dataset_dir) or DEFAULT_DATASET_DIR

    print(f"\nValidating model: {model_path}")
    resolved_model = resolve_workspace_path(model_path)
    model = YOLO(str(resolved_model))

    dataset_path = resolve_workspace_path(dataset_dir)
    data_yaml = dataset_path / "data.yaml"
    ultra_data_yaml, temp_data_yaml = prepare_ultralytics_data_yaml(data_yaml, dataset_path)
    try:
        metrics = model.val(data=ultra_data_yaml)
    finally:
        if temp_data_yaml is not None:
            temp_data_yaml.unlink(missing_ok=True)

    print("\nValidation Results:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")

    return metrics


def train_cli() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-train --dataset datasets/my_data --name my_model_v1
  yolo-train --config configs/custom.yaml --name custom_v2
  yolo-train --resume models/my_model_v1/weights/last.pt
        """
    )
    parser.add_argument("--validate", action="store_true",
                        help="Only validate existing model")
    parser.add_argument("--model", type=str,
                        help="Model path for training (base model) or validation")
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Dataset directory (default: from project config or {DEFAULT_DATASET_DIR})")
    parser.add_argument("--name", type=str, default="train",
                        help="Name for the project/output model (default: train)")
    parser.add_argument("--config", type=str,
                        help="Path to YAML config for training hyperparams")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip dataset validation before training")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip post-training analysis")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs (overrides config)")
    parser.add_argument("--batch", type=int,
                        help="Batch size (overrides config)")
    parser.add_argument("--imgsz", type=int,
                        help="Image size (overrides config)")
    parser.add_argument("--patience", type=int,
                        help="Patience for early stopping (overrides config)")
    args = parser.parse_args()

    args.model = _sanitize_path_arg(args.model)
    args.dataset = _sanitize_path_arg(args.dataset)
    args.config = _sanitize_path_arg(args.config)
    args.resume = _sanitize_path_arg(args.resume)

    # Load project config for defaults
    project_config = load_project_config()
    if project_config is None:
        warn_no_config()

    # Resolve parameters: CLI > project config > hardcoded
    dataset = get_default("dataset", cli_value=args.dataset, config=project_config, fallback=DEFAULT_DATASET_DIR)
    base_model = get_default("base_model", cli_value=args.model, config=project_config, fallback=DEFAULT_YOLO_MODEL)

    if args.validate:
        validate_model(args.model, args.dataset)
    else:
        train(
            dataset_dir=dataset,
            output_dir=None,
            base_model=base_model,
            model_name=args.name,
            config=args.config,
            skip_validation=args.skip_validation,
            skip_analysis=args.skip_analysis,
            resume=args.resume,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            project_defaults=project_config.defaults if project_config else None,
        )


def finetune_cli() -> None:
    """CLI entry point for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for a specific variant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-finetune --variant my_variant
  yolo-finetune --variant custom --epochs 50
  yolo-finetune --variant demo --skip-validation
        """
    )
    parser.add_argument("--variant", type=str, required=True,
                        help="Variant name to fine-tune for")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: from variant config or 30)")
    parser.add_argument("--base", type=str, default=None,
                        help=f"Base model path (default: from variant config or {DEFAULT_BASE_MODEL})")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset directory (default: from variant config or datasets/{variant})")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip dataset validation before training")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip post-training analysis")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch size (default: from variant config or 16)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Image size (default: from variant config or 640)")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Layers to freeze (default: from variant config or 10)")
    parser.add_argument("--version", type=str, default="v1",
                        help="Version suffix (default: v1)")

    args = parser.parse_args()

    args.base = _sanitize_path_arg(args.base)
    args.dataset = _sanitize_path_arg(args.dataset)
    args.resume = _sanitize_path_arg(args.resume)

    # Load variant config from yolo-project.yaml
    project_config = load_project_config()
    if project_config is None:
        warn_no_config()
    variant_config = {}
    if project_config:
        variant_config = project_config.get_variant(args.variant) or {}

    try:
        finetune(
            variant=args.variant,
            epochs=args.epochs or variant_config.get("epochs", 30),
            base_model=args.base or variant_config.get("base_model", DEFAULT_BASE_MODEL),
            dataset_dir=args.dataset or variant_config.get("dataset"),
            skip_validation=args.skip_validation,
            skip_analysis=args.skip_analysis,
            resume=args.resume,
            batch=args.batch or variant_config.get("batch", 16),
            imgsz=args.imgsz or variant_config.get("imgsz", 640),
            freeze=args.freeze or variant_config.get("freeze", 10),
            version=args.version,
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(EXIT_VALIDATION_FAILED)


__all__ = [
    "DEFAULT_BASE_MODEL",
    "train",
    "finetune",
    "validate_model",
    "train_cli",
    "finetune_cli",
]


if __name__ == "__main__":
    # Default to train_cli when run directly
    train_cli()
