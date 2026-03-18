# Complete Project Workflow

Follow the phases in order for a new project, or jump to any phase mid-stream.

---

## Quick Start (30-Second Version)

```bash
pip install -e .

# In Claude Code, run /setup to create yolo-project.yaml + program.md
# Or copy the example configs manually (see Phase 2)

yolo-validate datasets/my_data
yolo-experiment baseline --budget 5

# Edit program.md with your goals, constraints, and strategy
# Then in Claude Code:
#   /experiment   — experiment + HP tuning
#   /analyze      — detailed model analysis + recommendations

yolo-export --model models/best.pt --imgsz 640
```

---

## Ecosystem Overview

```
+---------------------+       +---------------------+       +---------------------+
|                     |       |                     |       |                     |
|    Annotation       |       |      yolocc         |       |    Deployment       |
|    (CVAT/Roboflow)  +------>+                     +------>+    (ONNX Runtime)   |
|                     | pull  |  Train, experiment, |export |                     |
|  Label images       |<------+  analyze, iterate   |       |  Inference app      |
|  Review uncertain   | push  |                     |       |                     |
+---------------------+       +---------------------+       +---------------------+
        ^                              |
        |         /review-annotations  |
        +------------------------------+
              feedback loop
```

- **Annotation platform** (CVAT, Roboflow, or manual) -- where images get labeled. `yolo-cvat pull` downloads datasets, `yolo-cvat push` sends uncertain images back for review.
- **yolocc** -- dataset validation, training, hyperparameter experimentation, analysis, and active learning.
- **Deployment** -- export to ONNX (`yolo-export`) and integrate into your application.

---

## Phase 1: Gather & Prepare Data

> **TL;DR** -- YOLO format: `images/{train,val}/` + `labels/{train,val}/` + `data.yaml`.
> Labels are `class_id x_center y_center width height` (normalized 0-1).
> Run `yolo-validate` to check everything. Run `yolo-clean` to deduplicate.

### YOLO Dataset Format

```
my_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img_0001.jpg
│   │   └── ...
│   └── val/
│       ├── img_0500.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img_0001.txt
    │   └── ...
    └── val/
        ├── img_0500.txt
        └── ...
```

**data.yaml:**

```yaml
path: .                    # Root path (relative to this file)
train: images/train
val: images/val

nc: 3                      # Number of classes
names:
  0: cat
  1: dog
  2: bird
```

**Label file format** -- one line per object, all coordinates normalized to 0-1:

```
class_id  x_center  y_center  width  height
```

**Example** -- 1920x1080 image, bounding box at pixel (400,200)-(600,400):

```
# x_center = (400 + 600) / 2 / 1920 = 0.2604
# y_center = (200 + 400) / 2 / 1080 = 0.2778
# width    = (600 - 400) / 1920      = 0.1042
# height   = (400 - 200) / 1080      = 0.1852

0 0.2604 0.2778 0.1042 0.1852
```

Multiple objects = multiple lines. Empty/missing label file = no objects.

### Creating a Dataset

- **From raw images:** organize into a directory, run `yolo-split --source raw/ --output datasets/my_data --ratio 0.8`, then label with your annotation tool
- **From CVAT:** export as "YOLO 1.1" format, or use `yolo-cvat pull --output datasets/my_data`
- **From Roboflow:** download in "YOLOv11" or "YOLOv8" format (both work)
- **Auto-labeling:** `yolo-autolabel --model models/best.pt --source unlabeled_images/` (always review results before training)

### Dataset Quality Checklist

- [ ] `data.yaml` has correct `nc` and `names` mapping
- [ ] Images are readable (JPEG/PNG/BMP/WebP)
- [ ] Label coordinates are in 0-1 range, class IDs are in 0 to `nc-1`
- [ ] Class distribution is reasonable (not more than 10:1 imbalance)
- [ ] No duplicate or near-duplicate images
- [ ] Train/val split does not leak similar images (e.g., sequential video frames)

```bash
# Validate structure, integrity, annotations, and statistics
yolo-validate datasets/my_data

# Strict mode — treat warnings as errors (good for CI/CD)
yolo-validate datasets/my_data --strict

# Remove orphan labels, empty labels, and near-duplicate images
yolo-clean --dataset datasets/my_data
```

---

## Phase 2: Initialize Project

> **TL;DR** -- Run `/setup` in Claude Code, or manually create `yolo-project.yaml`
> (classes, dataset path, defaults) and `program.md` (goals, constraints, strategy).
> These two files drive the entire experimentation loop.

### Using /setup (Recommended)

Run `/setup` in Claude Code. It walks through project name, classes, dataset path, base model, training defaults, and optional CVAT integration, then generates `yolo-project.yaml` and `program.md`.

### Manual Setup

```bash
cp yolo-project.example.yaml yolo-project.yaml
```

```yaml
project:
  name: "my_detection"
  description: "Detect X in Y"

# Must match your dataset's data.yaml
classes:
  0: cat
  1: dog
  2: bird

defaults:
  base_model: yolo11n.pt           # Auto-downloads from Ultralytics
  imgsz: 640
  epochs: 100
  dataset: datasets/my_dataset     # Relative to workspace root

# Named variants for fine-tuning subsets (optional)
variants:
  indoor:
    dataset: datasets/indoor
    epochs: 30
  outdoor:
    dataset: datasets/outdoor
    epochs: 50

# CVAT Integration (optional — pip install "yolocc[cvat]")
# cvat:
#   url: http://localhost:8080
#   project_id: 1
#   org: ""
```

- `classes` must exactly match `names` in your dataset's `data.yaml`
- `defaults.dataset` is resolved relative to workspace root (`YOLO_WORKSPACE_PATH` or cwd)
- `variants` override `dataset`, `epochs`, and `base_model` per subset

### Writing program.md

`program.md` defines what to optimize, constraints, and allowed actions. Create it from the template (`/setup` generates one) or write it manually. See `examples/fps_detection/program.md` for a real-world example.

Key sections: **Training Mode** (scratch / fine-tune / transfer), **Goals** (target metrics), **Constraints** (what not to change), **Strategy** (what to try and in what order), **Budget** (max epochs, max experiments, patience).

---

## Phase 3: Validate & Establish Baseline

> **TL;DR** -- `yolo-validate datasets/my_data` checks structure, integrity, annotations,
> and statistics. Then `yolo-experiment baseline --budget 5` gets baseline metrics.
> Results land in `experiments/exp_000_baseline/`.

### Dataset Validation

```bash
yolo-validate datasets/my_data
```

Runs four sequential checks: **[1/4] Structure** (data.yaml, directories), **[2/4] Integrity** (image readability, image-label pairing), **[3/4] Annotations** (format, coordinate ranges, class IDs), **[4/4] Statistics** (class counts, imbalance ratio).

Output is one of:
- `[OK] Dataset is valid!` -- safe to proceed
- `[OK] Dataset is valid with warnings.` -- non-critical issues, training will work
- `[X] Dataset has errors.` -- fix listed errors before training

### Running Baseline

```bash
yolo-experiment baseline --budget 5 --patience 3
```

Trains with default settings (no overrides) to establish a reference point. Results go to `experiments/exp_000_baseline/`.

```
experiments/
├── exp_000_baseline/
│   ├── config.yaml       # Training configuration used
│   ├── metrics.yaml      # Final metrics (machine-readable)
│   ├── report.md         # Human-readable report
│   └── train/weights/
│       ├── best.pt       # Best model checkpoint
│       └── last.pt       # Final epoch checkpoint
└── summary.md            # Dashboard of all experiments
```

### Metrics

| Metric | Measures | Range |
|--------|----------|-------|
| **mAP50** | Detection accuracy at 50% IoU threshold | 0.0 - 1.0 |
| **mAP50-95** | Average mAP at IoU 50%-95% in 5% steps (primary target) | 0.0 - 1.0 |
| **Precision** | Fraction of detections that are correct | 0.0 - 1.0 |
| **Recall** | Fraction of real objects that were detected | 0.0 - 1.0 |

### Interpreting Your Baseline

| mAP50 | Interpretation | Next step |
|-------|---------------|-----------|
| > 0.90 | Excellent | Fine-tune, focus on edge cases |
| 0.70 - 0.90 | Good foundation | Hyperparameter experimentation |
| 0.50 - 0.70 | Moderate, possibly data-limited | Check annotation quality, add data |
| < 0.50 | Likely data issues | Run `/review-dataset` and `/review-annotations` first |

If baseline mAP50 < 0.50, fix data before experimenting with hyperparameters.

---

## Phase 4: Experimentation

> **TL;DR** -- Edit `program.md` with goals/constraints. Use `/experiment` in Claude
> Code (assesses bottleneck, tunes via model.tune(), swaps architecture if needed).
> From CLI: `yolo-experiment tune --space lr` for HP optimization,
> or `yolo-experiment run --strategy learning_rate` for grid sweeps.

### How It Works

```
  program.md                    experiments/summary.md
  (goals & constraints)         (what's been tried)
        |                              |
        v                              v
  +--------------------------------------------+
  |         Decide next experiment              |
  |  (bottleneck assessment, model.tune(),      |
  |   or grid sweep)                            |
  +--------------------------------------------+
        |
        v
  Train with modified params (short budget)
        |
        v
  Compare to baseline (mAP50-95, per-class AP)
        |
        v
  Log result to experiments/exp_NNN_name/
        |
        v
  Decide: try another experiment or stop?
```

### Using /experiment (Claude Code)

Type `/experiment`. Claude reads `program.md` and `experiments/summary.md`, picks what to try next, runs experiments, and writes a session report. Quality depends on how specific your `program.md` is.

### Using CLI

```bash
# Run a built-in strategy sweep
yolo-experiment run --strategy learning_rate --budget 10

# Single experiment with specific overrides
yolo-experiment run --override "lr0=0.005 mosaic=0.5" --budget 15

# With a specific dataset or model
yolo-experiment run --override "lr0=0.01" --budget 10 --dataset datasets/custom --model models/base.pt

# View results / list strategies
yolo-experiment summary
yolo-experiment strategies
```

`--budget N` sets max epochs per experiment (default: 50). `--patience N` sets early stopping (default: 10). Strategy mode runs multiple experiments -- total cost is `num_experiments x budget`.

### HP Tuning (model.tune)

Use `model.tune()` for optimized hyperparameter search via Ultralytics' genetic algorithm. Available as presets or custom ranges.

```bash
# Use a preset search space
yolo-experiment tune --space lr --iterations 20 --epochs 10

# Custom parameter ranges
yolo-experiment tune --space "lr0=0.001:0.01 momentum=0.8:0.98" --iterations 15

# Full search (~25 parameters)
yolo-experiment tune --space all --iterations 30 --epochs 10
```

| Preset | Parameters Tuned |
|--------|-----------------|
| `lr` | lr0, lrf, warmup_epochs, warmup_momentum |
| `augmentation` | mosaic, mixup, erasing, hsv_h/s/v, degrees, scale |
| `loss` | box, cls, dfl |
| `optimizer` | lr0, momentum, weight_decay |
| `all` | ~25 parameters (Ultralytics defaults) |

`--iterations N` sets how many tune trials to run (default: 20). `--epochs N` sets epochs per trial (default: 10).

### Grid Sweeps (CLI)

For systematic parameter sweeps with fixed values. For optimized search, prefer model.tune() presets above.

| Strategy | Parameter | Values Swept | When to Use |
|----------|-----------|-------------|-------------|
| `learning_rate` | `lr0` | 0.0005, 0.001, 0.005, 0.01, 0.02 | Try first. Biggest single impact. |
| `optimizer` | `optimizer` | SGD, Adam, AdamW | If LR sweep was inconclusive. |
| `augmentation` | `mosaic` x `mixup` | 3x3 grid (9 combos) | Small datasets, varied scale/context. |
| `resolution` | `imgsz` | 320, 416, 512, 640 | When small objects matter. |
| `batch_size` | `batch` | -1 (auto), 8, 16, 32 | GPU memory sweet spot. |
| `erasing` | `erasing` | 0.0, 0.2, 0.4, 0.6 | Partially occluded objects. |
| `freeze` | `freeze` | 0, 5, 10, 15 | Transfer learning depth. |
| `warmup` | `warmup_epochs` | 0.0, 1.0, 3.0, 5.0 | Unstable early training. |

### Reading Experiment Reports

Each experiment creates `experiments/exp_NNN_name/` with `config.yaml`, `metrics.yaml`, and `report.md`. The report shows a delta table vs baseline and a verdict: **IMPROVED** or **NO IMPROVEMENT**.

The summary dashboard (`experiments/summary.md`) tracks all experiments with status: `baseline`, `improved`, `best`, or `rejected`.

---

## Phase 5: Analyze Results

> **TL;DR** -- `yolo-analyze --model models/best.pt --dataset datasets/my_data`
> identifies false negatives, uncertain predictions, and per-class weak spots.
> Outputs a prioritized image list and recommendations.

### Running Analysis

```bash
yolo-analyze --model models/best.pt --dataset datasets/my_data

# Custom confidence thresholds
yolo-analyze --model models/best.pt --dataset datasets/my_data --low-conf 0.3 --high-conf 0.7

# Combine with CVAT push
yolo-analyze --model models/best.pt --dataset datasets/my_data --upload-cvat
```

Or in Claude Code: `/analyze`

Analysis runs inference on every image with a low threshold (0.1) and produces a report with: general statistics, confidence distribution across four bands, and a prioritized image list (`reports/uncertain_images.txt`) ranked by severity -- HIGH (false negatives: labels exist but no detections), MEDIUM (uncertain confidence range), LOW (no detections, no labels).

Focus review effort on HIGH priority images first. If many uncertain images exist, correcting them and retraining gives the biggest per-image improvement.

### When to Stop Experimenting

```
Has mAP50-95 improved by > 0.5% in the last 3 experiments?
├── YES --> Keep experimenting.
└── NO --> Are there specific weak classes (AP < 0.6)?
    ├── YES --> Bottleneck is data. Run /analyze, fix annotations, add data.
    └── NO --> Hyperparameter gains exhausted. Move to Phase 6 or deploy.
```

---

## Phase 6: CVAT Active Learning Loop

> **TL;DR:** `yolo-analyze` finds uncertain images --> `yolo-cvat push` sends them to CVAT --> human corrects annotations --> `yolo-cvat pull` brings them back --> `yolo-merge` combines with existing data --> retrain. Repeat until diminishing returns.

### Prerequisites

1. Self-hosted CVAT with Nuclio running (see [companion CVAT repo](../CVAT/docs/README.md))
2. `pip install "yolocc[cvat]"`
3. Create a Personal Access Token in CVAT (Settings > Access Tokens)
4. Set the environment variable:
   ```bash
   export CVAT_ACCESS_TOKEN="your-token-here"
   ```
5. Add `cvat:` section to `yolo-project.yaml`:
   ```yaml
   cvat:
     url: http://localhost:8080
     project_id: 1
     org: ""
   ```

### Step 6.1: Find Uncertain Images

```bash
yolo-analyze --model models/best.pt --dataset datasets/my_data
```

Produces `reports/uncertain_images.txt` grouped by priority.

### Step 6.2: Push to CVAT

```bash
# From analysis output (batches into tasks of 50 images)
yolo-cvat push --from-analysis reports/uncertain_images.txt

# Manual push with pre-annotations
yolo-cvat push --images uncertain/ --labels labels/ --task-name "Review batch 1"
```

### Step 6.3: Human Review in CVAT

Open the created tasks in CVAT. Pre-annotations are loaded -- fix wrong classes, adjust boxes, delete false positives, add missed objects.

### Step 6.4: Pull Corrected Annotations

```bash
# Pull a single task
yolo-cvat pull --task 14 --output datasets/corrected_batch1

# Pull an entire project
yolo-cvat pull --project 1 --output datasets/corrected_all

# Always validate after pulling
yolo-validate datasets/corrected_batch1
```

### Step 6.5: Merge with Existing Dataset

```bash
# Default: first source wins on conflicts
yolo-merge --sources datasets/my_data/labels/train datasets/corrected_batch1/labels/train \
           --output datasets/merged/labels/train

# CVAT corrections override originals
yolo-merge --sources datasets/my_data/labels/train datasets/corrected_batch1/labels/train \
           --output datasets/merged/labels/train \
           --source-priority last --clear-output

yolo-validate datasets/merged
```

Run `yolo-merge --help` for all options (class remapping, IoU threshold, dry-run, etc.).

### Step 6.6: Retrain

```bash
yolo-experiment run --override "data=datasets/merged/data.yaml" --budget 10
```

Or use `/experiment` in Claude Code.

### Loop Decision

```
Metrics improved after new data?
├── Yes (+>1% mAP) --> Deploy, continue loop if desired
├── Marginal (<0.5% mAP) --> More diverse images? Annotation issues? Architecture change?
└── Regressed --> STOP. Investigate data quality.

More than 3 iterations with <0.5% gain? Ship what you have.
```

Each iteration: focus on weakest classes, add 50-200 images, document in `experiments/active_learning_log.md`.

### Using /active-learning (Claude Code)

The `/active-learning` skill orchestrates the full loop: checks state, pushes uncertain images, waits for human review, pulls corrections, merges, validates, retrains, compares metrics, and writes an iteration report.

---

## Phase 7: Deploy

> **TL;DR:** Export to ONNX for local/edge deployment, or deploy to CVAT as a Nuclio function for auto-annotation. Or loop back to Phase 6.

### Option A: ONNX Export

```bash
yolo-export --model models/best.pt --imgsz 640

# Custom output, static shape, deploy copy
yolo-export --model models/best.pt --export-dir /path/to/exports --static --deploy-dir /opt/models/production

# Export all models in the workspace
yolo-export --all
```

Run `yolo-export --help` for all options (opset version, simplification, etc.).

### Option B: Deploy to CVAT via Nuclio

```bash
# Generate serverless function (function.yaml + main.py + best.onnx)
yolo-cvat deploy --model models/best.pt --name my_detector

# Deploy to Nuclio
nuctl deploy --path ./serverless/my_detector --platform local
```

### Option C: Keep Improving

Loop back to Phase 6. Each iteration: focus on weakest classes, add 50-200 images, document in `experiments/active_learning_log.md`.

---

## Troubleshooting

### GPU Issues

| Symptom | Fix |
|---------|-----|
| "No CUDA GPUs available" | Run `nvidia-smi` to check driver. Run `python -c "import torch; print(torch.cuda.is_available())"` to check PyTorch. |
| "CUDA out of memory" | Reduce `--batch` (try 8 or 4) or `--imgsz` (try 320). |
| Training slow on GPU | Avoid `--batch -1` on old GPUs. Add `--amp` for mixed precision. |

### Training Problems

| Symptom | Fix |
|---------|-----|
| Loss not decreasing | Try `lr0=0.001`. Check labels are correct. Verify dataset size. |
| Overfitting (train loss low, val loss high) | Increase augmentation, reduce epochs, add more data. |
| mAP stuck at 0 | Verify class IDs in labels match `data.yaml`. Run `yolo-validate`. |
| mAP dropped after changes | Revert to baseline config. Diff what changed. Check for corrupted merge. |

### Dataset Errors

| Symptom | Fix |
|---------|-----|
| "Dataset not found" | Check path in `yolo-project.yaml`. Verify `data.yaml` exists. |
| "Invalid annotation" | Run `yolo-clean` to fix or remove invalid annotations. |
| Class imbalance warning | Add more examples of the underrepresented class, or increase augmentation. |
| Corrupted images | Run `yolo-validate` to identify, then remove. |

### CVAT Connection Issues

| Symptom | Fix |
|---------|-----|
| "CVAT_ACCESS_TOKEN env var is required" | Create a PAT in CVAT (Settings > Access Tokens). `export CVAT_ACCESS_TOKEN="..."` |
| "Connection refused" | Check `curl http://localhost:8080/api/server/about`. Start CVAT with `launch-cvat.bat`. |
| 401 / "Permission denied" | Create a new token. Verify user has access to target project. |
| Import/export stuck | Check `docker ps` for `cvat_worker_*` containers. `docker compose restart`. |
| Nuclio deploy fails | Verify Docker running. Check http://localhost:8070 for error logs. |

---

## Performance Tuning Guide

### Batch Size vs Memory

| GPU VRAM | Recommended batch (imgsz=640) |
|----------|-------------------------------|
| 4 GB     | 4-8                           |
| 8 GB     | 8-16                          |
| 12 GB    | 16-32                         |
| 24 GB    | 32-64                         |

### Image Size Tradeoffs

| imgsz | Speed   | Small Object Detection | Memory |
|-------|---------|------------------------|--------|
| 320   | Fastest | Poor                   | Low    |
| 640   | Balanced | Good                  | Medium |
| 1280  | Slowest | Best                   | High   |

### Augmentation Guide

| Parameter | Default | Effect | When to Use |
|-----------|---------|--------|-------------|
| `mosaic=1.0` | 1.0 | Combines 4 images into one | Small datasets, increases diversity |
| `mixup=0.15` | 0.0 | Blends two images together | Generalization (can confuse similar classes) |
| `erasing=0.4` | 0.4 | Randomly erases patches | Simulates occlusion |
| `copy_paste=0.1` | 0.0 | Copies objects from other images | Rare classes |

Fine-tuning: `mosaic=0.5, mixup=0.0, erasing=0.2`. Small datasets (<500 images): `mosaic=1.0, mixup=0.15, erasing=0.4, copy_paste=0.1`.

### Model Architecture

| Model | Speed | Accuracy | VRAM | Best For |
|-------|-------|----------|------|----------|
| yolo11n | Fastest (<5ms) | Lower | ~2 GB | Edge/mobile, real-time |
| yolo11s | Fast | Balanced | ~4 GB | General use |
| yolo11m | Moderate | Higher | ~8 GB | Accuracy over speed |
| yolo11l | Slower | High | ~12 GB | High-accuracy apps |
| yolo11x | Slowest | Highest | ~16 GB | Maximum accuracy, server |

Start with `yolo11n`/`yolo11s` for experimentation, scale up once dataset and config are solid.

---

## Quick Reference

### CLI Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `yolo-train` | Train from scratch | `yolo-train --dataset ds/ --name v1` |
| `yolo-finetune` | Transfer learning | `yolo-finetune --variant indoor --base best.pt` |
| `yolo-validate` | Validate dataset | `yolo-validate datasets/mine` |
| `yolo-experiment` | Run experiments | `yolo-experiment run --strategy learning_rate` |
| `yolo-analyze` | Find uncertain images | `yolo-analyze --model best.pt --dataset ds/` |
| `yolo-export` | Export to ONNX | `yolo-export --model best.pt --imgsz 640` |
| `yolo-split` | Split into train/val | `yolo-split --source raw/ --output split/` |
| `yolo-clean` | Remove duplicates/corruption | `yolo-clean datasets/mine` |
| `yolo-merge` | Merge annotation sources | `yolo-merge --sources a/ b/ --output merged/` |
| `yolo-autolabel` | Auto-annotate images | `yolo-autolabel --model best.pt --sources imgs/` |
| `yolo-cvat pull` | Download from CVAT | `yolo-cvat pull --task 42` |
| `yolo-cvat push` | Send to CVAT for review | `yolo-cvat push --from-analysis reports/uncertain_images.txt` |
| `yolo-cvat deploy` | Deploy model to CVAT | `yolo-cvat deploy --model best.pt --name my_det` |

### Claude Code Skills

| Skill | When to Use |
|-------|-------------|
| `/setup` | New project |
| `/experiment` | Experiment loop + HP tuning |
| `/analyze` | Post-training analysis + recommendations |
| `/train` | Single managed training run |
| `/review-dataset` | Dataset quality audit |
| `/review-annotations` | AI-assisted annotation review (YOLO + Claude vision) |
| `/annotate` | Claude vision annotation correction |
| `/cvat-pull` | Pull corrected annotations from CVAT |
| `/cvat-push` | Push uncertain images to CVAT |
| `/cvat-deploy` | Deploy model to CVAT as Nuclio function |
| `/compare-models` | Compare 2+ models on same dataset |
| `/benchmark` | Profile model speed and size |
| `/explain-results` | Plain-English training report |
| `/active-learning` | Full active learning loop orchestration |

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `YOLO_WORKSPACE_PATH` | Root directory for datasets, models, experiments | Current working directory |
| `CVAT_ACCESS_TOKEN` | PAT for CVAT API authentication | *(required for CVAT features)* |

### Key Files

| File | Who Edits | Purpose |
|------|-----------|---------|
| `yolo-project.yaml` | You | Project config -- classes, defaults, CVAT settings |
| `program.md` | You | Experiment goals and constraints |
| `experiments/summary.md` | AI | Dashboard of all experiments |
| `experiments/session_*.md` | AI | Per-session before/after reports |
| `experiments/analysis.md` | AI | Training advisor recommendations |
| `experiments/active_learning_log.md` | AI | Active learning iteration history |
| `reports/uncertain_images.txt` | AI | Images flagged by `yolo-analyze` |
