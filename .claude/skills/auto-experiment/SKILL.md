---
name: auto-experiment
description: "Run autonomous YOLO training experiments — reads training-plan.md, assesses bottlenecks, acts strategically, and delegates HP optimization to model.tune()."
---

# Auto-Experiment — Strategic Decision Loop

Autonomous experimentation via hypothesis-driven reasoning.
The agent makes strategic decisions (architecture, data, augmentation strategy).
HP optimization is delegated to `model.tune()` — don't compete with it.

## Pre-Flight

1. [ ] `yolo-project.yaml` exists — run `/setup` if not
2. [ ] `training-plan.md` exists with goals + boundaries — run `/setup` if not
   - If not found, check for `program.md` (deprecated name) and warn user to rename it
3. [ ] Dataset is valid — `yolo-validate`
4. [ ] GPU available — `python -c "import torch; print(torch.cuda.is_available())"`
5. [ ] Read `experiments/dataset_profile.yaml` — run `/review-dataset` if missing

## Context Loading

Read in order before every session:
1. `yolo-project.yaml` — classes, dataset path, model defaults
2. `training-plan.md` — goals, hard constraints, soft preferences, allowed actions, domain knowledge
3. `experiments/summary.md` — all prior journal entries (if exists)
4. `experiments/dataset_profile.yaml` — dataset characteristics for architecture reasoning
5. `experiments/analysis.md` — latest analysis (if exists)

## Baseline

If no baseline exists: `yolo-experiment baseline --budget 5 --patience 3`
After baseline, update training-plan.md Current Performance section.

## Session Start

Write a session marker in `summary.md`:
```
## Session YYYY-MM-DD HH:MM — Budget: N experiments
```
If a session marker exists with today's date and no end marker, this is a **resumed session** — count existing entries toward budget.

## Reasoning Loop

For each decision, follow all 3 steps in order:

### 1. ASSESS — What's the bottleneck?

Read dataset profile, per-class AP, training dynamics from last run.

Classify the bottleneck:
- **Data quality** → active learning (`/review-dataset`, `/cvat-push`, `/autolabel`)
- **Architecture mismatch** → swap config (`yolo-experiment run --override "model=configs/architectures/..."`)
- **HP not optimized** → `yolo-experiment tune --space <preset>`

Write a hypothesis: "I expect X because Y, so I'll do Z"

**Cold-start (first post-baseline):** Focus on dataset profile characteristics, per-class AP spread, class imbalance. Training dynamics analysis starts from experiment 2 onward.

### 2. ACT — Do the right thing

**Architecture change:**
```bash
yolo-experiment run --override "model=configs/architectures/yolo11-p2.yaml" --budget <epochs>
```
Requires dataset profile justification — cite ≥2 numbers. See `resources/architecture-guide.md`.

**HP optimization (delegate to model.tune):**
```bash
# Use presets when diagnosis maps cleanly:
yolo-experiment tune --space lr --iterations 20 --epochs 10
yolo-experiment tune --space augmentation --iterations 20 --epochs 10
yolo-experiment tune --space loss --iterations 15 --epochs 10
yolo-experiment tune --space optimizer --iterations 20 --epochs 10

# Use custom when diagnosis is specific:
yolo-experiment tune --space "lr0=0.001:0.01 momentum=0.85:0.98" --iterations 20 --epochs 10
```

**Strategic experiment** (resolution, freeze depth, augmentation strategy):
```bash
yolo-experiment run --override "<param>=<value>" --budget <epochs>
```

**Data action** (when data quality is the bottleneck):
`/review-dataset`, `/cvat-push`, `/autolabel`

### 3. LOG — Record what happened

- Read experiment report from `experiments/exp_NNN_*/report.md`
- Append simplified journal entry to `summary.md` matching schema in `resources/journal-schema.yaml`
- **Every field marked REQUIRED must be present.**
- Decide: continue this direction, try different lever, or stop

## Session End

1. Write session end marker: `## Session End — N experiments run`
2. Write `experiments/session_YYYY-MM-DD.md` with before/after + key learnings
3. Update training-plan.md Model Lineage with current best path, metrics, source experiment

## Guardrails

- **Checkpoint backup** — hook handles automatically before every run/tune
- **Immutable data** — NEVER modify original dataset files
- **Architecture change gate** — 3+ experiments on current architecture with <0.5% improvement before switching. Exception: >50% small objects with no P2 head (see `resources/architecture-guide.md`)
- **Architecture justification** — must cite ≥2 dataset profile numbers in journal
- **Log everything** — write journal entry even if experiment fails
