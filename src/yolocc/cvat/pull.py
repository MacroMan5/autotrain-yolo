"""Pull CVAT annotations into a local YOLO dataset."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Optional

from yolocc.cvat.client import require_cvat, get_client
from yolocc.paths import resolve_workspace_path


def pull_task(task_id: int, output_dir: Optional[str] = None) -> Path:
    """Export a CVAT task as Ultralytics YOLO dataset."""
    require_cvat()

    client = get_client()

    if output_dir is None:
        output_dir = f"datasets/cvat_task_{task_id}"
    output_path = resolve_workspace_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Pulling CVAT task {task_id} -> {output_path}")

    # Use the high-level SDK to download dataset
    task = client.tasks.retrieve(task_id)
    task.export_dataset(
        format_name="Ultralytics YOLO Detection 1.0",
        filename=str(output_path / "dataset.zip"),
        include_images=True,
    )

    # Extract ZIP
    zip_path = output_path / "dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)
        zip_path.unlink()
        print(f"Extracted to {output_path}")

    # Validate
    data_yaml = output_path / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset ready at {output_path}")
        print(f"Validate with: yolo-validate {output_path}")
    else:
        print("WARNING: No data.yaml found in exported dataset")

    return output_path


def pull_project(project_id: int, output_dir: Optional[str] = None) -> Path:
    """Export a CVAT project as Ultralytics YOLO dataset."""
    require_cvat()

    client = get_client()

    if output_dir is None:
        output_dir = f"datasets/cvat_project_{project_id}"
    output_path = resolve_workspace_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Pulling CVAT project {project_id} -> {output_path}")

    project = client.projects.retrieve(project_id)
    project.export_dataset(
        format_name="Ultralytics YOLO Detection 1.0",
        filename=str(output_path / "dataset.zip"),
        include_images=True,
    )

    zip_path = output_path / "dataset.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)
        zip_path.unlink()

    return output_path


def pull_cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Pull CVAT annotations as YOLO dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="CVAT task ID to pull")
    group.add_argument("--project", type=int, help="CVAT project ID to pull")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    if args.task:
        pull_task(args.task, args.output)
    else:
        pull_project(args.project, args.output)
