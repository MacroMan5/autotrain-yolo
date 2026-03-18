# yolocc

> YOLO11/YOLO26 training toolkit — dataset validation, hyperparameter tuning (model.tune),
> experiment tracking, CVAT active learning, ONNX export. CLI tools + Claude Code skills.

Universal YOLO training pipeline with semi-autonomous experimentation.
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — but for object detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![YOLO](https://img.shields.io/badge/YOLO-11%20%7C%2026-orange.svg)]()
[![Claude Code](https://img.shields.io/badge/Claude%20Code-skills-blueviolet.svg)]()
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status: Experimental](https://img.shields.io/badge/status-experimental-yellow.svg)]()
[![Tests](https://github.com/MacroMan5/yolocc-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/MacroMan5/yolocc-toolkit/actions/workflows/test.yml)

---

## This Project Is Itself An Experiment

We're exploring a question: **Can semi-automated experimentation consistently improve object detection models?**

Some parts work today. Others are hypotheses. We're building in the open and need your help testing.

### What Works (proven engineering)

| Feature | Status |
|---|---|
| Universal YOLO training pipeline (train, finetune, validate, export) | Working |
| Experiment tracking with markdown reports | Working |
| Dataset validation, splitting, cleaning, merging | Working |
| Auto-labeling with trained models | Working |
| Claude Code skills for structured workflows | Working |

### What Needs Testing (research questions)

| Feature | Question | How to help |
|---|---|---|
| Autonomous experiment loop | Does modify-train-evaluate-iterate find better configs? | Run `/experiment` on your dataset, share results |
| Claude reviewing annotations (multimodal) | How accurately can Claude judge bounding box quality? | Test `/review-annotations`, compare to manual |
| Multi-pass annotation (YOLO + Claude) | Does low-confidence YOLO + Claude vision beat manual speed? | Benchmark against your labeling workflow |

**If you test any of these, [open an issue](https://github.com/MacroMan5/yolocc-toolkit/issues) with your findings!**

---

## How It Works

```
1. Prepare dataset + write program.md (goals, constraints, budget)
2. /experiment assesses bottlenecks, tunes HPs, swaps architectures
3. Session report: what improved, what didn't, what to try next
```

**Full walkthrough**: See [WORKFLOW.md](WORKFLOW.md) for the complete end-to-end guide — from raw images to deployed model, including CVAT active learning, troubleshooting, and performance tuning.

## Quickstart

### 1. Install

```bash
pip install -e .
```

### 2. Prepare Your Dataset

You need a YOLO-format dataset:
```
your_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 3. Initialize Project

In Claude Code:
```
/setup
```

Or manually — copy `yolo-project.example.yaml` to `yolo-project.yaml` and edit.

### 4. Write Your Experiment Program

Edit `program.md` — defines goals, constraints, and allowed actions:

```markdown
## Goal
Maximize mAP50-95.

## Hard Constraints
- Budget: 10 experiments, 50 epochs each
- Model: yolo11n
- Don't modify the dataset

## Allowed Actions
### HP Optimization (via model.tune)
- Presets: lr, augmentation, loss, optimizer, all
```

### 5. Run Experiments

In Claude Code:
```
/experiment
```

Or via CLI:
```bash
yolo-experiment baseline --budget 5
yolo-experiment tune --space lr --iterations 20 --epochs 10
yolo-experiment run --strategy learning_rate --budget 10
yolo-experiment summary
```

### 6. Review Results

```bash
cat experiments/summary.md
```

Or in Claude Code:
```
/analyze
```

## CLI Commands

| Command | Purpose |
|---|---|
| `yolo-train` | Train a model |
| `yolo-finetune` | Fine-tune with transfer learning |
| `yolo-validate` | Validate dataset integrity |
| `yolo-experiment` | Run experiments + HP tuning |
| `yolo-analyze` | Active learning analysis |
| `yolo-export` | Export to ONNX |
| `yolo-split` | Stratified train/val/test split |
| `yolo-clean` | Remove duplicates and corrupted files |
| `yolo-merge` | Merge annotation files |
| `yolo-autolabel` | Auto-annotate with trained model |
| `yolo-cvat` | CVAT integration (pull/push/deploy) |

All commands support `--help`.

## Claude Code Skills

| Skill | Purpose |
|---|---|
| `/experiment` | Experiment loop (assess → tune → report) |
| `/analyze` | Training analysis + recommendations |
| `/setup` | Project initialization wizard |
| `/review-dataset` | Dataset quality audit |
| `/train` | Managed training with reporting |
| `/review-annotations` | AI-assisted annotation review (experimental) |
| `/annotate` | Claude vision annotation correction (experimental) |
| `/cvat-pull` | Pull annotations from CVAT |
| `/cvat-push` | Push uncertain images to CVAT for review |
| `/cvat-deploy` | Deploy trained model to CVAT via Nuclio |
| `/compare-models` | Compare 2+ models side-by-side |
| `/benchmark` | Profile model speed, FPS, and size |
| `/explain-results` | Plain-English training report |
| `/active-learning` | Full active learning loop |

## CVAT Integration

yolocc integrates with [CVAT](https://github.com/cvat-ai/cvat) for the full active learning loop: annotate, train, find uncertain predictions, get human review, and retrain.

### Prerequisites

- Self-hosted CVAT with Nuclio (see [CVAT setup guide](https://github.com/MacroMan5/CVAT))
- `CVAT_ACCESS_TOKEN` environment variable (create a Personal Access Token in CVAT UI)
- Install with CVAT extras: `pip install -e ".[cvat]"`

### Active Learning Loop

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Annotate in CVAT                                  │
│        ↓                                            │
│   yolo-cvat pull         (pull annotations)         │
│        ↓                                            │
│   yolo-train / /experiment  (train model)           │
│        ↓                                            │
│   yolo-analyze           (find uncertain images)    │
│        ↓                                            │
│   yolo-cvat push         (push to CVAT for review)  │
│        ↓                                            │
│   Human reviews in CVAT → repeat                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Deploy a Trained Model to CVAT

```bash
yolo-cvat deploy --model best.pt
```

This packages your model as a Nuclio serverless function and deploys it to CVAT, enabling auto-annotation directly in the CVAT UI.

## Configuration

### `yolo-project.yaml`

The project config file controls training defaults, dataset paths, and integrations:

```yaml
project:
  name: my-project
  description: "Custom detection project"

classes:
  0: cat
  1: dog

defaults:
  base_model: yolo11n.pt        # Ultralytics model
  epochs: 100                    # Max training epochs
  imgsz: 640                    # Input resolution
  dataset: datasets/my_data     # Path to YOLO-format dataset

# Named variants for fine-tuning (optional)
variants:
  indoor:
    dataset: datasets/indoor
    epochs: 30

# CVAT integration (optional — pip install "yolocc[cvat]")
cvat:
  url: http://localhost:8080
  project_id: 1
```

### Environment Variables

| Variable | Purpose |
|---|---|
| `YOLO_WORKSPACE_PATH` | Override workspace directory (default: current directory) |
| `CVAT_ACCESS_TOKEN` | Personal access token for CVAT API |

## Ecosystem

yolocc is part of a three-repo toolkit for object detection workflows:

| Repo | Purpose |
|---|---|
| **[yolocc](https://github.com/MacroMan5/yolocc-toolkit)** | Training, experimentation, active learning |
| **[CVAT Setup](https://github.com/MacroMan5/CVAT)** | Self-hosted annotation platform with Nuclio auto-annotation |
| **[Dataset Converter](https://github.com/MacroMan5/dataset-converter)** | Convert YOLO datasets for CVAT/Roboflow import |

## What You Need

| Requirement | Why |
|---|---|
| YOLO dataset (images + labels + data.yaml) | Data to train on |
| GPU (NVIDIA, 4GB+ VRAM) | Training requires GPU |
| Python 3.10+ with torch + ultralytics | Dependencies |
| Claude Code (optional) | For guided workflows via skills |

## File Map

| File | Who | Purpose |
|---|---|---|
| `program.md` | You edit | Experiment goals + constraints |
| `yolo-project.yaml` | You edit | Project config |
| `experiments/summary.md` | Generated | Experiment dashboard |
| `experiments/session_*.md` | Generated | Session reports |
| `experiments/analysis.md` | Generated | Recommendations |

## Contributing

- **Testers**: Run experiments on your dataset, share results
- **Engineers**: Integrations (W&B, MLflow), new tools

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
