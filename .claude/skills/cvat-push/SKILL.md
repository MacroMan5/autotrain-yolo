---
name: cvat-push
description: "Push uncertain or misclassified images to CVAT for human annotation review after training analysis."
---

# Push to CVAT for Review

After training analysis identifies uncertain images, push them to CVAT
for human annotation or correction.

## Pre-Flight Checklist
- [ ] `yolo-analyze` has been run (check for `uncertain_images.txt` or analysis output)
- [ ] CVAT configured in `yolo-project.yaml`
- [ ] `CVAT_ACCESS_TOKEN` env var is set

## Workflow

### 1. Gather Uncertain Images
Read the analysis output:
```bash
cat reports/uncertain_images.txt
```
Or check `experiments/analysis.md` for the list of uncertain predictions.

### 2. Categorize by Priority
Organize images into review batches:
- **False negatives** (highest priority): Images where the model missed detections but labels exist
- **Low confidence** (medium priority): Detections with confidence between low_conf and high_conf thresholds
- **No detection, no label** (low priority): Images where neither model nor labels found anything

### 3. Push to CVAT
```bash
# From analysis file (auto-batches into tasks of 50)
yolo-cvat push --from-analysis reports/uncertain_images.txt

# Or manually specify images
yolo-cvat push --images <path_to_images> --labels <path_to_labels> --task-name "Review: False Negatives"
```

### 4. Generate Report
Write `experiments/cvat_push_report.md`:
```markdown
## CVAT Upload Report
- Date: YYYY-MM-DD
- Tasks created: N
- Total images: N
- False negatives: N (Task ID: X)
- Uncertain: N (Task ID: Y)
- CVAT URL: http://localhost:8080/tasks/<id>
```

## Decision Tree

```
More than 100 images?
├── Yes → Split into batches of 50, create multiple tasks
└── No → Single task

Has existing YOLO labels for these images?
├── Yes → Upload as pre-annotations (saves annotator time)
└── No → Push images only for fresh annotation

False negatives found?
├── Yes → Create separate high-priority task labeled "URGENT: False Negatives"
└── No → Single "Review: Uncertain" task
```

## Guardrails
- NEVER push more than 200 images without asking the user first
- ALWAYS include pre-annotations when available (reduces annotation time)
- Keep a local copy of all images — CVAT is not a backup
- Write the push report before telling the user it's done
