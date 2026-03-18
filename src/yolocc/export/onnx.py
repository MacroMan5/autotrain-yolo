#!/usr/bin/env python3
"""
ONNX Export Module
==================

Export trained YOLO models to ONNX format for inference.

Functions:
    export_onnx: Export a single model to ONNX
    export_all: Export all models in a directory
    export_cli: CLI entry point
"""

import argparse
import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from yolocc.paths import get_models_root, resolve_workspace_path


# ============================================
# CONFIGURATION
# ============================================

# Resolved lazily to avoid import-time path evaluation before workspace is configured
DEFAULT_EXPORT_DIR = None
DEFAULT_IMG_SIZE = 640


def _get_default_export_dir() -> Path:
    """Resolve the default export directory at call time, not import time."""
    return get_models_root() / "exports"


def export_onnx(
    model_path: str,
    export_dir: Optional[Path] = None,
    img_size: int = DEFAULT_IMG_SIZE,
    dynamic: bool = True,
    simplify: bool = True,
    opset: int = 12
) -> Optional[Path]:
    """
    Export a YOLO model to ONNX format.

    Args:
        model_path: Path to the .pt model file
        export_dir: Directory for ONNX exports (default: models/exports)
        img_size: Image size for export
        dynamic: Use dynamic input shape
        simplify: Simplify ONNX graph
        opset: ONNX opset version

    Returns:
        Path to exported ONNX file, or None if failed
    """
    if export_dir is None:
        export_dir = _get_default_export_dir()

    print("=" * 60)
    print(f"Exporting: {model_path}")
    print("=" * 60)

    model_file = Path(model_path)
    if not model_file.exists():
        print(f"ERROR: Model not found: {model_path}")
        return None

    # Load model
    model = YOLO(str(model_file))

    # Export to ONNX
    print("\nExporting to ONNX...")
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=dynamic,
        simplify=simplify,
        opset=opset,
    )

    if export_path is None:
        print("ERROR: Export failed — model.export() returned None")
        return None

    print(f"Exported: {export_path}")

    # Move to exports folder
    export_dir.mkdir(parents=True, exist_ok=True)
    final_path = export_dir / Path(export_path).name
    if final_path.exists():
        final_path.unlink()
    shutil.move(export_path, final_path)
    print(f"Moved to: {final_path}")

    print(f"\n{'=' * 60}")
    print("Export Complete!")
    print(f"{'=' * 60}")
    print(f"ONNX file: {final_path}")

    return final_path


def export_all(
    models_dir: Optional[str] = None,
    export_dir: Optional[Path] = None,
    img_size: int = DEFAULT_IMG_SIZE
) -> list:
    """
    Export all trained models in a directory.

    Args:
        models_dir: Directory containing .pt models
        export_dir: Directory for ONNX exports
        img_size: Image size for export

    Returns:
        List of exported ONNX file paths
    """
    if models_dir is None:
        models_dir = str(get_models_root())
    models_path = Path(models_dir)
    exported = []

    for model_file in models_path.glob("*.pt"):
        if model_file.name.startswith("yolo"):
            continue  # Skip base YOLO models

        result = export_onnx(str(model_file), export_dir, img_size)
        if result:
            exported.append(result)

    return exported


def export_cli() -> None:
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export YOLO model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yolo-export --model models/my_model.pt
  yolo-export --all
  yolo-export --model models/my_model.pt --imgsz 640
  yolo-export --model models/my_model.pt --deploy-dir /path/to/deploy
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path to export (relative paths are resolved from workspace root)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all models in the workspace models directory",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Export directory (default: workspace models/exports)",
    )
    parser.add_argument(
        "--deploy-dir",
        type=str,
        default=None,
        help="Copy exported ONNX to this directory",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="Image size (default: 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    parser.add_argument("--no-simplify", action="store_true", help="Don't simplify ONNX graph")
    parser.add_argument("--static", action="store_true", help="Use static input shape")

    args = parser.parse_args()

    export_dir = resolve_workspace_path(args.export_dir) if args.export_dir else None

    if args.all:
        export_all(export_dir=export_dir, img_size=args.imgsz)
    elif args.model:
        model_path = resolve_workspace_path(args.model)
        result = export_onnx(
            str(model_path),
            export_dir=export_dir,
            img_size=args.imgsz,
            dynamic=not args.static,
            simplify=not args.no_simplify,
            opset=args.opset,
        )

        if result and args.deploy_dir:
            deploy_path = Path(args.deploy_dir)
            deploy_path.mkdir(parents=True, exist_ok=True)
            dest = deploy_path / result.name
            shutil.copy(result, dest)
            print(f"Deployed to: {dest}")
    else:
        print("Usage:")
        print("  yolo-export --model models/my_model.pt")
        print("  yolo-export --all")
        print("  yolo-export --model models/my_model.pt --deploy-dir /path/to/deploy")


__all__ = [
    "export_onnx",
    "export_all",
    "export_cli",
    "DEFAULT_EXPORT_DIR",
    "DEFAULT_IMG_SIZE",
]


if __name__ == "__main__":
    export_cli()
