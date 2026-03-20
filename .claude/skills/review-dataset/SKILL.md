---
name: review-dataset
description: "Audit YOLO dataset quality — class distribution, annotation quality, image stats, and improvement suggestions."
---

# Review Dataset

Dataset quality audit + profiling for architecture selection.

## Workflow

### 1. Load Config

Read `yolo-project.yaml` for dataset path, class names, and **imgsz**.
If no config, ask user for dataset path.

### 2. Run Structural Validation

```bash
yolo-validate <dataset_path> --strict
```

### 3. Deep Analysis

Scan label files for:
- **Class Distribution**: Count per class, imbalance ratio. CRITICAL if any class <500, WARNING if <1000.
- **Annotation Quality**: Tiny boxes (<0.02 normalized), edge-clipped, heavily overlapping (IoU >0.9)
- **Empty images**: Count and percentage
- **Split Balance**: Compare class ratios between train and val

### 4. Dataset Profile for Architecture Selection

Run the profiling script with imgsz from yolo-project.yaml (default 640):

```bash
python scripts/profile_dataset.py \
  --labels <dataset>/labels/train \
  --images <dataset>/images/train \
  --imgsz <imgsz> \
  --class-names "<comma-separated from data.yaml>"
```

The script outputs structured YAML with:
- Object scale distribution (% small/medium/large at training resolution)
- Min object size (px) at training resolution
- Class-wise profiles sorted by avg object size ascending (smallest first)
- Train/val scale divergence check (flags >15% difference in small_pct)
- Suggested starting point for architecture (head config + scale + reasoning)

Save the output to `experiments/dataset_profile.yaml`.

### 5. Write to training-plan.md

Fill the **Dataset Summary** section in training-plan.md with profile data:
- Total images: train/val counts
- Classes: N — [list]
- Class balance: most/least represented with counts
- Scale distribution: % small/medium/large at imgsz
- Min object size at training resolution
- Avg objects per image
- Train/val divergence flag

### 6. Write Report

Create `experiments/dataset_audit.md` with:
- Validation findings
- Class distribution table
- Quality issues
- Dataset profile (full YAML output)
- Architecture suggestion (labeled "Suggested starting point", NOT "Recommended")
- Top 3 recommendations

### 7. Print Summary

Key findings, profile highlights, and architecture suggestion.

## Guidelines

- Be specific: "Collect 400+ bird images" not "collect more data"
- Profile uses training-resolution-adjusted sizes, not native pixel sizes
- Architecture suggestion provides data for agent reasoning — it's not a directive
