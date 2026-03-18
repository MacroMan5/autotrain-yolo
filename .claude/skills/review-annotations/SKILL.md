---
name: review-annotations
description: "AI-assisted annotation review — uses YOLO inference + Claude vision to auto-approve, correct, or flag images for human review in CVAT."
---

# Review Annotations

Multi-pass annotation review: YOLO precision + Claude intelligence.

**STATUS: EXPERIMENTAL**

## Workflow

### 1. Setup
Read `yolo-project.yaml` for classes, model path. Get review folder from user.

### 2. Prepare Output
Create: `review_output/approved/`, `needs_review/`, `rejected/`

### 3. Process Images (batches of 5-10)
For each image:
1. Run YOLO at low confidence: `python scripts/multi_pass_annotate.py --image <path> --model <model> --conf 0.15`
2. Draw existing annotations: `python scripts/draw_annotations.py`
3. Claude reads annotated image (multimodal)
4. Judge: AUTO-APPROVE / AUTO-CORRECT / NEEDS HUMAN / REJECT

### 4. Generate Report
Write `review_output/review_report.md` with counts, corrections, common issues.

## Guardrails
- Batch size 5-10 to avoid context overflow
- Never auto-approve if uncertain
- When uncertain → needs_review
- Generate CVAT export command for needs_review
