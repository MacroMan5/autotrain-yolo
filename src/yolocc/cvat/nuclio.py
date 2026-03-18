"""Generate Nuclio function for CVAT auto-annotation with trained YOLO models."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import yaml

from yolocc.project import load_project_config
from yolocc.paths import resolve_workspace_path


MAIN_PY_TEMPLATE = '''import json
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

model = None

def init_context(context):
    global model
    model_path = "/opt/nuclio/best.onnx"
    model = YOLO(model_path, task="detect")
    context.logger.info(f"Model loaded: {model_path}")

def handler(context, event):
    data = event.body
    buf = BytesIO(base64.b64decode(data))
    image = Image.open(buf)

    results = model(image, conf=0.25)
    detections = []

    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            detections.append({
                "confidence": float(box.conf),
                "label": model.names[int(box.cls)],
                "points": coords,
                "type": "rectangle",
            })

    return context.Response(
        body=json.dumps(detections),
        headers={},
        content_type="application/json",
        status_code=200,
    )
'''


def generate_nuclio_function(
    model_path: str,
    function_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Generate Nuclio function files for CVAT auto-annotation.

    Args:
        model_path: Path to trained YOLO model (.pt or .onnx)
        function_name: Name for the Nuclio function
        output_dir: Output directory for function files
    """
    config = load_project_config()
    model_resolved = resolve_workspace_path(model_path)

    if not model_resolved.exists():
        raise FileNotFoundError(f"Model not found: {model_resolved}")

    if function_name is None:
        function_name = config.name if config else "yolo_detector"

    if output_dir is None:
        output_dir = f"serverless/{function_name}"
    func_dir = resolve_workspace_path(output_dir)
    func_dir.mkdir(parents=True, exist_ok=True)

    # Get class names from config
    classes = {}
    if config and config.classes:
        classes = config.classes
    else:
        # Try to load from model
        try:
            from ultralytics import YOLO
            m = YOLO(str(model_resolved))
            classes = {i: name for i, name in m.names.items()}
            del m
        except Exception:
            pass

    # Generate spec annotations
    spec_annotations = []
    for class_id in sorted(classes.keys()):
        spec_annotations.append({
            "id": class_id,
            "name": classes[class_id],
            "type": "rectangle",
        })

    # Generate function.yaml
    function_yaml = {
        "metadata": {
            "name": f"autotrain-{function_name}",
            "namespace": "cvat",
            "annotations": {
                "name": function_name,
                "type": "detector",
                "spec": json.dumps(spec_annotations, indent=2) if spec_annotations else "[]",
            },
        },
        "spec": {
            "description": f"YOLO detector: {function_name}",
            "runtime": "python:3.10",
            "handler": "main:handler",
            "eventTimeout": "30s",
            "build": {
                "image": f"cvat.autotrain.{function_name}",
                "baseImage": "ubuntu:22.04",
                "directives": {
                    "preCopy": [
                        {"kind": "USER", "value": "root"},
                        {"kind": "RUN", "value": (
                            "apt update && apt install --no-install-recommends -y python3-pip"
                        )},
                        {"kind": "RUN", "value": "pip install ultralytics onnxruntime pillow"},
                        {"kind": "WORKDIR", "value": "/opt/nuclio"},
                        {"kind": "RUN", "value": "ln -s /usr/bin/python3 /usr/bin/python"},
                    ],
                },
            },
            "triggers": {
                "myHttpTrigger": {
                    "numWorkers": 2,
                    "kind": "http",
                    "workerAvailabilityTimeoutMilliseconds": 10000,
                    "attributes": {
                        "maxRequestBodySize": 33554432,
                    },
                },
            },
            "platform": {
                "attributes": {
                    "restartPolicy": {
                        "name": "always",
                        "maximumRetryCount": 3,
                    },
                    "mountMode": "volume",
                },
            },
        },
    }

    # Write function.yaml
    with open(func_dir / "function.yaml", "w") as f:
        yaml.safe_dump(function_yaml, f, sort_keys=False, default_flow_style=False)

    # Write main.py
    (func_dir / "main.py").write_text(MAIN_PY_TEMPLATE)

    # Copy model (prefer ONNX)
    if model_resolved.suffix == ".onnx":
        shutil.copy2(model_resolved, func_dir / "best.onnx")
    else:
        # Export to ONNX
        print(f"Exporting {model_resolved} to ONNX...")
        try:
            from ultralytics import YOLO
            m = YOLO(str(model_resolved))
            onnx_path = m.export(format="onnx")
            shutil.copy2(onnx_path, func_dir / "best.onnx")
        except Exception as e:
            print(f"WARNING: ONNX export failed ({e}). Copy model manually.")
            shutil.copy2(model_resolved, func_dir / "best.pt")

    print(f"\nNuclio function generated at: {func_dir}")
    print(f"Classes: {list(classes.values())}")
    print("\nDeploy with:")
    print(f"  nuctl deploy --path {func_dir} --platform local")

    return func_dir


def deploy_cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate Nuclio function for CVAT")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--name", type=str, help="Function name")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    generate_nuclio_function(args.model, args.name, args.output)
