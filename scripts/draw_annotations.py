#!/usr/bin/env python3
"""Draw YOLO annotations on an image for visual review.

Usage:
    python scripts/draw_annotations.py --image img.jpg --labels img.txt --classes cat dog bird --output annotated.jpg
"""

import argparse
from pathlib import Path


def draw_annotations(image_path, label_path, class_names, output_path, show_confidence=False):
    """Draw YOLO bounding boxes on an image."""
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required. Install with: pip install opencv-python")
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return False

    h, w = img.shape[:2]
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 255, 0), (255, 128, 0), (0, 128, 255),
    ]

    label_path = Path(label_path)
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            conf = float(parts[5]) if len(parts) > 5 and show_confidence else None

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = colors[cls_id % len(colors)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            if conf is not None:
                label = f"{label} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), img)
    return True


def main():
    parser = argparse.ArgumentParser(description="Draw YOLO annotations on an image")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--labels", "-l", required=True, help="YOLO label file path")
    parser.add_argument("--classes", "-c", nargs="+", required=True, help="Class names")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--confidence", action="store_true", help="Show confidence scores")
    args = parser.parse_args()

    success = draw_annotations(
        args.image, args.labels, args.classes, args.output, args.confidence
    )
    if success:
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
