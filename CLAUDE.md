# yolocc

YOLO training toolkit — dataset tools, HP tuning, intelligent experiment automation, CVAT active learning, ONNX export.

## Key Files

| File | Who edits | Purpose |
|------|-----------|---------|
| `training-plan.md` | Human | Training goals, constraints, allowed actions |
| `yolo-project.yaml` | Human (or /setup) | Project config — classes, dataset, defaults |
| `experiments/summary.md` | AI | Dashboard of all experiments |
| `experiments/session_*.md` | AI | Per-session before/after report |
| `experiments/analysis.md` | AI | Training advisor recommendations |
| `experiments/dataset_profile.yaml` | AI | Dataset characteristics for architecture selection |
| `configs/architectures/*.yaml` | Shipped | Pre-built YOLO11 head configs (standard, P2, P2P3P4) |
| `WORKFLOW.md` | Human | Complete end-to-end workflow guide |

## Available Slash Commands

- `/experiment` — Run experiment loop (assess → tune → report)
- `/analyze` — Analyze results + write recommendations
- `/setup` — Initialize a new YOLO project (interactive wizard)
- `/review-dataset` — Audit dataset quality + stats
- `/train` — Run a single managed training with report
- `/review-annotations` — AI-assisted annotation review (YOLO + Claude vision)
- `/annotate` — Claude vision annotation correction
- `/compare-models` — Compare 2+ models side-by-side (mAP, per-class AP, speed)
- `/benchmark` — Profile model inference speed and size
- `/explain-results` — Plain-English training results report
- `/cvat-pull` — Pull annotations from CVAT
- `/cvat-push` — Push uncertain images to CVAT for review
- `/cvat-deploy` — Deploy trained model to CVAT via Nuclio
- `/active-learning` — Full active learning loop

## How Experiments Work

The `/experiment` skill runs a 3-step reasoning loop:

1. **Context**: Read `training-plan.md` (boundaries), `summary.md` (history), `dataset_profile.yaml` (characteristics)
2. **Reasoning loop** (ASSESS → ACT → LOG):
   - ASSESS: Classify bottleneck (data quality / architecture mismatch / HP not optimized)
   - ACT: Architecture swap, model.tune(), strategic experiment, or data action
   - LOG: Read report, append journal entry, decide next step
3. **Guardrails**: checkpoint backup, immutable data, architecture change gate
4. **Session report** with before/after + key learnings

## Common Commands

```bash
yolo-train --dataset path/to/dataset --name my_model
yolo-finetune --variant indoor --base models/best.pt
yolo-validate path/to/dataset
yolo-experiment run --override "lr0=0.005" --budget 10
yolo-experiment baseline --budget 5
yolo-experiment summary
yolo-experiment tune --space lr --iterations 20 --epochs 10
yolo-export --model best.pt --imgsz 640
yolo-split --source raw/ --output split/ --classes cat dog bird
yolo-analyze --model best.pt --dataset path/to/dataset
yolo-autolabel --model best.pt --source unlabeled/
yolo-clean --dataset path/to/dataset
yolo-merge --base dataset1/ --overlay dataset2/
```

## Architecture

```
src/yolocc/
├── project.py          # Config loader (yolo-project.yaml)
├── paths.py            # Workspace path resolution
├── training/           # Train, analyze, utils
├── dataset/            # Validate, split, clean, merge, autolabel
├── export/             # ONNX export
└── experiment/         # Autonomous experimentation engine
    ├── runner.py       # Execute experiments + tune (run_experiment, run_tune)
    ├── tracker.py      # Log + generate reports
    └── strategies.py   # Built-in hyperparameter sweeps (CLI)
```

## For Contributors

- Run tests: `pytest -v`
- Install dev: `pip install -e ".[dev]"`
- Lint: `ruff check src/`
