"""Push images and pre-annotations to CVAT for review."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from yolocc.cvat.client import require_cvat, get_client, get_cvat_config
from yolocc.paths import resolve_workspace_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def push_task(
    images_dir: str,
    task_name: str = "Review",
    labels_dir: Optional[str] = None,
    project_id: Optional[int] = None,
) -> int:
    """
    Create a CVAT task with images and optional pre-annotations.

    Returns the created task ID.
    """
    require_cvat()

    client = get_client()
    cvat_cfg = get_cvat_config()

    images_path = resolve_workspace_path(images_dir)
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")

    # Collect images
    image_files = sorted([
        f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}")

    print(f"Pushing {len(image_files)} images to CVAT task '{task_name}'")

    # Determine project
    _project_id = project_id or cvat_cfg.get("project_id")

    # Create task
    task_spec = {"name": task_name}
    if _project_id:
        task_spec["project_id"] = _project_id

    task = client.tasks.create_from_data(
        spec=task_spec,
        resource_type="local",
        resources=[str(f) for f in image_files],
    )

    task_id = task.id
    print(f"Created CVAT task {task_id}: {task_name}")

    # Upload pre-annotations if provided
    if labels_dir:
        labels_path = resolve_workspace_path(labels_dir)
        if labels_path.exists():
            _upload_annotations(client, task_id, labels_path, image_files)

    cvat_url = cvat_cfg.get("url", "http://localhost:8080")
    print(f"View in CVAT: {cvat_url}/tasks/{task_id}")

    return task_id


def _upload_annotations(client, task_id: int, labels_path: Path, image_files: list[Path]):
    """Upload YOLO-format labels as pre-annotations."""
    import tempfile
    import zipfile

    # Build a CVAT-compatible annotation archive
    # CVAT expects Darknet YOLO 1.1 format for import
    matched = 0
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with zipfile.ZipFile(tmp_path, "w") as zf:
            names = []
            for img in image_files:
                label_file = labels_path / f"{img.stem}.txt"
                if label_file.exists():
                    zf.write(label_file, f"obj_train_data/{img.stem}.txt")
                    matched += 1
                names.append(f"obj_train_data/{img.name}")

            zf.writestr("train.txt", "\n".join(names))

        if matched > 0:
            task = client.tasks.retrieve(task_id)
            task.import_annotations(
                format_name="YOLO 1.1",
                filename=str(tmp_path),
            )
            print(f"Uploaded {matched} pre-annotations")
        else:
            print("No matching label files found — task created without pre-annotations")
    finally:
        tmp_path.unlink(missing_ok=True)


def push_from_analysis(
    analysis_file: str = "reports/uncertain_images.txt",
    task_prefix: str = "Review",
    max_per_task: int = 50,
) -> list[int]:
    """
    Push uncertain images from yolo-analyze output to CVAT.

    Returns list of created task IDs.
    """
    require_cvat()

    analysis_path = resolve_workspace_path(analysis_file)
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis file not found: {analysis_path}\n"
            "Run yolo-analyze first to generate uncertain_images.txt"
        )

    # Read image paths from analysis
    image_paths = []
    for line in analysis_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            img_path = Path(line)
            if img_path.exists():
                image_paths.append(img_path)

    if not image_paths:
        print("No valid image paths found in analysis file")
        return []

    print(f"Found {len(image_paths)} uncertain images")

    # Split into batches
    task_ids = []
    for i in range(0, len(image_paths), max_per_task):
        batch = image_paths[i:i + max_per_task]
        batch_num = i // max_per_task + 1

        # Copy batch images to temp dir for upload
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            for img in batch:
                shutil.copy2(img, tmp_path / img.name)

            task_id = push_task(
                images_dir=str(tmp_path),
                task_name=f"{task_prefix} batch {batch_num}",
            )
            task_ids.append(task_id)

    print(f"Created {len(task_ids)} CVAT tasks")
    return task_ids


def push_cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Push images to CVAT for review")
    parser.add_argument("--images", type=str, help="Directory of images to push")
    parser.add_argument("--labels", type=str, help="Directory of YOLO labels (pre-annotations)")
    parser.add_argument("--task-name", type=str, default="Review", help="CVAT task name")
    parser.add_argument("--project", type=int, help="CVAT project ID")
    parser.add_argument("--from-analysis", type=str,
                        help="Push uncertain images from analysis file")
    args = parser.parse_args()

    if args.from_analysis:
        push_from_analysis(args.from_analysis)
    elif args.images:
        push_task(args.images, args.task_name, args.labels, args.project)
    else:
        parser.error("Provide --images or --from-analysis")
