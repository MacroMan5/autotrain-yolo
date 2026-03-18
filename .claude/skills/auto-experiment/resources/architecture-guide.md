# Architecture Selection Guide

Reference for choosing YOLO11 head configuration and model scale.
Read `experiments/dataset_profile.yaml` for the numbers to apply these thresholds.

## Head Configuration

All thresholds use object sizes **at training imgsz**, not native resolution.

| Condition | Config | File |
|-----------|--------|------|
| >50% COCO-small AND <10% large | `yolo11-p2p3p4.yaml` (shifted, no P5) | `configs/architectures/yolo11-p2p3p4.yaml` |
| >30% COCO-small OR min object <10px | `yolo11-p2.yaml` (4-head P2/P3/P4/P5) | `configs/architectures/yolo11-p2.yaml` |
| Otherwise | `yolo11.yaml` (standard P3/P4/P5) | `configs/architectures/yolo11.yaml` |

**COCO-small** = object area < 32² = 1024 px² at training resolution.
**min object** = minimum side length `min(pixel_w, pixel_h)` at training resolution.

> **Note:** `yolo11-p2p3p4.yaml` is experimental (OPEN-001). If it fails verification, use `yolo11-p2.yaml` instead.

## Model Scale

| Classes | Scale | Rationale |
|---------|-------|-----------|
| ≤5 | n (nano) | Few classes, speed priority |
| ≤20 | s (small) | Balance of speed and accuracy |
| ≤80 | m (medium) | Many classes need capacity |
| >80 | l (large) | COCO-scale datasets |

Override with deployment constraints from program.md (e.g., <5ms inference → force nano).

## Architecture Change Rules

1. **Gate:** 3+ experiments on current architecture with <0.5% mAP improvement before switching
2. **Exception:** If dataset profile shows >50% small objects AND current architecture has no P2 head, immediate change is allowed
3. **Justification:** Must cite ≥2 specific numbers from dataset profile in the journal entry's `architecture_change.justification` field
4. **Usage:** Pass config via `--override "model=configs/architectures/yolo11-p2.yaml"` with pretrained weights

## Do NOT

- Edit `nc` in architecture YAMLs (Ultralytics auto-overrides from data.yaml)
- Generate architecture YAML from scratch (pick from pre-built configs only)
- Use YOLO26 (out of scope)
