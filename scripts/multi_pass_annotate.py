#!/usr/bin/env python3
"""Multi-pass annotation: YOLO detection + preparation for Claude review.

Usage:
    python scripts/multi_pass_annotate.py --image img.jpg --model best.pt --conf 0.15 --output annotated.jpg
"""

import argparse
import os
import tempfile
from pathlib import Path


def multi_pass_annotate(image_path, model_path, conf_threshold=0.15, output_path=None):
    """
    Run YOLO at low confidence and prepare annotated image for Claude review.

    Returns dict with detection info and annotated image path.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(str(image_path), conf=conf_threshold, verbose=False)

    if not results:
        return {"detections": [], "annotated_image_path": None}

    result = results[0]

    # Parse detections
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox_xyxy": box.xyxy[0].tolist(),
                "bbox_xywhn": box.xywhn[0].tolist(),
            })

    # Save annotated image
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".jpg", prefix="annotated_")
        os.close(fd)

    annotated = result.plot()  # Ultralytics built-in plotting
    try:
        import cv2
        cv2.imwrite(str(output_path), annotated)
    except ImportError:
        from PIL import Image
        Image.fromarray(annotated[..., ::-1]).save(str(output_path))

    return {
        "detections": detections,
        "annotated_image_path": str(output_path),
        "original_image_path": str(image_path),
        "total_detections": len(detections),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-pass YOLO annotation")
    parser.add_argument("--image", "-i", required=True, help="Input image")
    parser.add_argument("--model", "-m", required=True, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--output", "-o", help="Output annotated image path")
    args = parser.parse_args()

    result = multi_pass_annotate(args.image, args.model, args.conf, args.output)
    print(f"Detections: {result['total_detections']}")
    for d in result["detections"]:
        print(f"  {d['class_name']}: {d['confidence']:.2f}")
    if result["annotated_image_path"]:
        print(f"Annotated image: {result['annotated_image_path']}")


if __name__ == "__main__":
    main()
