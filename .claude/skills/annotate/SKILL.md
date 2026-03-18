---
name: annotate
description: "Claude visually inspects images and corrects/creates YOLO bounding box annotations using multimodal vision."
---

# Annotate

Claude vision annotation correction using multi-pass refinement.

**STATUS: EXPERIMENTAL**

## Approach
1. YOLO inference at low confidence (0.15) → precise boxes
2. Claude vision review → keeps good, removes bad, identifies missed
3. For missed objects → YOLO on cropped region
4. Final Claude validation

## Workflow
1. Read `yolo-project.yaml` for class names
2. For each image:
   a. Draw existing annotations: `python scripts/draw_annotations.py`
   b. Claude reads annotated image
   c. Evaluate: keep / correct / remove / add
   d. Write corrected YOLO label

## Coordinate Guide
- (0.0, 0.0) = top-left, (1.0, 1.0) = bottom-right
- (0.5, 0.5) = center
- Width/height relative to image

## Limitations
- ~10-15% coordinate error
- Best for objects > 5% of image area
- Prefer correcting existing boxes over creating from scratch
