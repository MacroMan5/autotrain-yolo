# Skill Refactor: Reasoning-Driven Autonomous Experimentation

> Date: 2026-03-19
> Status: Approved design
> Scope: Skills-only (no Python CLI changes except `scripts/profile_dataset.py`)
> Research basis: `docs/research/llm-agents-as-ml-engineers.md`, `docs/research/LLM-driven-YOLO.md`

## Problem

The current `/experiment` skill is a scripted playbook. Its decision tree tells Claude exactly what to try and in what order:

```
No baseline? → Run baseline
Nothing tried? → Learning rate sweep
Last improved? → Fine-tune (±20%)
Augmentation untried? → Try augmentation
```

This turns Claude into a script executor. The research shows the real gains come from letting the agent **reason about why something worked** and **what to try next based on data characteristics** — not following a predetermined sequence. A landmark analysis of 10,000 LLM-guided experiments found that architectural choices explain ~75% of performance variance, while hyperparameter tuning within a fixed architecture accounts for only a fraction. The current skill locks architecture changes behind "NEVER change model architecture unless program.md allows."

## Approach

**Coordinated Skill Refactor (Approach B)**: Rewrite 2 core skills, enhance 2 others, add cross-cutting guardrails. Leave the 6 infrastructure skills untouched.

| File | Change Type | Summary |
|------|-------------|---------|
| `auto-experiment/SKILL.md` | **Rewrite** | Decision tree → reasoning loop |
| `program.md` template | **Rewrite** | Scripted phases → boundaries |
| `review-dataset/SKILL.md` | **Enhance** | Add dataset profiling for architecture decisions |
| `analyze-results/SKILL.md` | **Enhance** | Add training dynamics + experiment trajectory |
| `auto-experiment/resources/guardrails-inventory.md` | **New** | Consolidated guardrail reference |
| `auto-experiment/resources/experiment-checklist.md` | **Update** | Match new journal/hypothesis requirements |
| `setup-project/SKILL.md` | **Update** | Generate new-format program.md |
| `active-learning/SKILL.md` | **Update** | Delegate to refactored /experiment, re-profile after merge |
| `CLAUDE.md` | **Update** | "How Experiments Work" section matches reasoning loop |
| `scripts/profile_dataset.py` | **New** | Standalone dataset profiler (~50 lines) |
| `configs/architectures/*.yaml` | **New** | 3 pre-built architecture configs |
| `examples/fps_detection/program.md` | **Migrate** | Scripted phases → boundaries format |

**Not changed**: `training-loop`, `annotate`, `review-annotations`, `cvat-pull`, `cvat-push`, `cvat-deploy`, `experiment-runner` agent.

---

## Section 1: `auto-experiment/SKILL.md` Rewrite

### Current → New

The decision tree (lines 49-57) is replaced by a reasoning loop. The guardrail "NEVER change model architecture" is replaced by dataset-profile-driven architecture selection from pre-built configs.

### The Reasoning Loop

```
For each experiment:
1. DIAGNOSE  — What's limiting performance right now?
2. HYPOTHESIZE — "I expect X because Y, so I'll try Z"
3. SCOPE    — Hyperparameters, augmentation, or architecture?
4. BACKUP   — cp best.pt → best_backup.pt
5. RUN      — yolo-experiment run --override "..." --budget N
6. ANALYZE  — Full results: curves, per-class AP, convergence
7. JOURNAL  — Append structured entry to summary.md
8. DECIDE   — Continue / branch / stop
```

### Cold-Start Path (Step 1)

First post-baseline experiment has no training curves to analyze. DIAGNOSE focuses on:
- Dataset profile characteristics (scale distribution, class imbalance)
- Per-class AP spread from baseline
- Class imbalance ratio

Training dynamics analysis (convergence speed, overfitting onset, LR effectiveness) starts from experiment 2 onward.

### Architecture Selection

Agent picks from pre-built configs in `configs/architectures/` — does NOT generate YAML from scratch.

Available configs:
- `configs/architectures/yolo11.yaml` — standard P3/P4/P5 (default)
- `configs/architectures/yolo11-p2.yaml` — P2/P3/P4/P5 for small objects
- `configs/architectures/yolo11-p2p3p4.yaml` — shifted pyramid, drop P5

Selection driven by dataset profile at training resolution:
- \>30% COCO-small OR min object <10px at training imgsz → P2 4-head
- \>50% small AND <10% large → shifted P2/P3/P4
- Otherwise → standard 3-head

Scale (n/s/m/l/x) selected by class count + deployment constraints from program.md.

**No YOLO26.** YOLO11 only for v1.

### Architecture Change Gate

Architecture changes are only allowed after 3+ experiments on the current architecture show diminishing returns (<0.5% improvement).

**Exception**: If dataset profile shows >50% small objects and current architecture has no P2 head, architecture change is allowed immediately.

### Structured Journal Entry Schema

Appended to summary.md as fenced YAML after each experiment:

```yaml
# --- exp_003 ---
experiment_id: exp_003
hypothesis: "Strong augmentation will help underrepresented bird class"
reasoning: "bird AP 0.42 vs 0.80+ for others, only 47 images"
overrides: {mosaic: 1.0, mixup: 0.3}
architecture: yolo11n                    # ALWAYS present
architecture_change:                     # ONLY when model changes
  from: yolo11n
  to: yolo11s-p2
  justification: "65% small objects at imgsz=640, min 8px — profile indicates P2 head needed"
result:
  map50_95: 0.47
  delta: +0.02
  per_class_delta: {cat: -0.01, dog: +0.00, bird: +0.08}
  epochs_run: 23
  converged: true
  time_minutes: 12
learning: "Augmentation helps minority classes significantly"
verdict: improved  # improved | regressed | neutral
next_suggestion: "Try copy_paste or class-weighted loss"
```

- `architecture` is always present (agent needs to know what each experiment used)
- `architecture_change` is optional (only when model/heads/scale changes)
- Architecture change justification must cite at least two specific numbers from the dataset profile

### Session Budget Enforcement

At session start, the skill writes a session marker in summary.md:

```yaml
# === SESSION 2026-03-19T14:00 ===
# budget: 10 experiments (from program.md)
```

Before each experiment, the skill counts journal entries since the marker. Hard stop at budget — no self-extension.

### Experiment Checklist (updated)

**Before running:**
- [ ] Hypothesis written (testable, cites data)
- [ ] Overrides defined
- [ ] Budget within session limit
- [ ] `best_backup.pt` created

**After running:**
- [ ] Full journal entry appended (YAML schema)
- [ ] Compared to baseline AND current best
- [ ] Per-class regression check (threshold from program.md)
- [ ] Lightweight analysis complete

---

## Section 2: `program.md` Template Rewrite

### Current → New

Scripted phases ("Phase 1: LR sweep → Phase 2: Fine-tune → Phase 3: Validation") become boundaries: hard constraints define the walls, soft preferences guide direction, allowed actions enumerate the decision space.

### New Template

```markdown
# Experiment Program

> The agent reads this before every experiment session.
> Define the box. The agent fills it.

## Project Context

### Training Mode
- [ ] Training from scratch
- [x] Fine-tuning from pretrained model
- [ ] Transfer learning (freeze backbone)

### Model Lineage
- Base model: `yolo11n.pt`
- Architecture config: `yolo11.yaml` (standard P3/P4/P5)
- Current best: (auto-updated after each session)

### Model Intent
- [ ] Specialist (few classes, high accuracy)
- [ ] Generalist (many classes, broad coverage)
- Deployment target: (hardware, latency requirements)

### Dataset Summary
(Auto-filled by /setup or /review-dataset)
- Total images: train / val
- Classes: N — [list]
- Class balance: most / least represented
- Object scale distribution: % small / medium / large at imgsz
- Min object size at training resolution: Npx
- Avg objects per image: N
- Train/val scale divergence: (flagged if significant)

### Current Performance
(Auto-filled after baseline)
- mAP50-95:
- mAP50:
- Per-class AP50: {class: value, ...}
- Weakest class:

## Goal
Maximize mAP50-95 on the validation set.

### Secondary Goals
- (e.g., improve weakest class AP50 to > 0.60)
- (e.g., maintain inference speed < 10ms)

## Hard Constraints (agent cannot violate)
- Max experiments per session: 10
- Max epochs per experiment: 50
- Max training time per experiment: 30 minutes
- Don't delete or modify original dataset files
- Don't decrease any class AP50 by more than 0.05 vs current best model
- Minimum 3 experiments on current architecture before switching
  (exception: dataset profile shows >50% small objects with no P2 head)

## Soft Preferences (agent can override with justification)
- Start with current model variant before trying others
- Prefer augmentation approaches before architecture changes
- Prioritize weakest class improvement

## Allowed Actions
### Hyperparameters
- Any lr0, lrf, optimizer, momentum, weight_decay
- Any loss weights (box, cls, dfl)
- Any augmentation parameters

### Architecture
- Model variants: n, s, m (or specify allowed set)
- Head configs:
  - `configs/architectures/yolo11.yaml` (standard P3/P4/P5)
  - `configs/architectures/yolo11-p2.yaml` (P2/P3/P4/P5, small objects)
  - `configs/architectures/yolo11-p2p3p4.yaml` (shifted, mostly small objects)
- imgsz: 640, 1280

### Data Handling
- Can create augmented copies (NOT modify originals)
- Can adjust train/val split if justified

## Domain Knowledge
> Tell the agent things it can't learn from the dataset statistics alone.
- (e.g., "Objects are frequently occluded — erasing augmentation is relevant")
- (e.g., "Class 'smoke' is visually similar to 'fog' — confusion is the main problem")
- (e.g., "False positives are more costly than missed detections in this application")
```

### FPS Detection Example Migration

The existing `examples/fps_detection/program.md` is migrated to the new format:
- Scripted phases → soft preferences ("Prefer augmentation for small objects before architecture changes")
- Specific steps → allowed actions ("freeze: 0, 10, 15")
- Domain notes stay as Domain Knowledge section

---

## Section 3: `review-dataset/SKILL.md` Enhancement

### Current → New

Same quality audit, plus a new dataset profiling step that produces the statistics the reasoning loop needs for architecture decisions.

### New Step: Dataset Profile for Architecture Selection

Added after the existing deep analysis (step 3), before the report.

**Profiling script**: `scripts/profile_dataset.py` (~50 lines), ships with the repo.

```bash
python scripts/profile_dataset.py --labels datasets/my_data/labels/train --imgsz 640
```

Outputs structured YAML to stdout. Consistent across sessions, testable.

**Resolution adjustment**: The script samples actual image dimensions via PIL (100 images or all if <1000), computes letterbox scaling factor per image: `object_px = normalized_wh * native_size * (imgsz / max(native_w, native_h))`. Handles non-square images correctly.

**Profile output includes**:
- Object scale distribution at training imgsz: % small (<32²px) / medium (32²-96²) / large (>96²)
- Min object size (px) at training resolution
- Avg and max objects per image
- **Class-wise object size distribution** — sorted ascending by avg object size (smallest-object classes first)
- **Train/val distribution comparison** — scale distribution per split, flags significant divergence

**Profile is written to two places**:
1. `experiments/dataset_audit.md` — the full report
2. `program.md` → Dataset Summary section (auto-fills the schema)

**Architecture suggestion** (not recommendation):
> "Suggested starting point based on profile: yolo11s-p2 (65% small objects at imgsz=640, min 8px)"

Labeled as "suggested starting point" — provides data for the agent to reason about, not a directive.

**Staleness**: Explicit re-run model. After CVAT pull + merge, the `/active-learning` skill re-runs `/review-dataset` to update the profile. No hash/timestamp mechanism in v1.

---

## Section 4: `analyze-results/SKILL.md` Enhancement

### Current → New

Same metrics analysis, plus training dynamics and experiment trajectory. Split into lightweight (per-experiment) and full (on-demand) modes.

### Lightweight Analysis (after each experiment)

Runs automatically after every experiment. Covers:
- **Latest experiment dynamics**: convergence speed, overfitting onset epoch, train/val gap, loss plateau detection
- **Comparison to baseline AND current best**: deltas for all metrics
- **Per-class regression check**: flags classes that dropped beyond program.md threshold

This is what the reasoning loop's DIAGNOSE step reads (via journal entries + latest analysis).

### Full Trajectory Analysis (on-demand `/analyze` or end-of-session)

Runs when user calls `/analyze` explicitly, or automatically at session end. Covers everything in lightweight plus:
- **Cross-experiment patterns**: "augmentation changes helped bird consistently but hurt cat each time"
- **Most impactful experiment** and why
- **Diminishing returns detector** with categorized stall detection:
  - Same-category stall (e.g., all augmentation tweaks <0.5%) → "augmentation exhausted, try different lever"
  - Cross-category stall (augmentation + LR + architecture all <0.5%) → "model may be at dataset ceiling"
  - Different recommendation for each case

### results.csv Parsing Fallback

Look for results.csv in experiment directory first, then Ultralytics default output directory. If columns don't match expected names (they've changed between Ultralytics versions), fall back to metrics.yaml for final numbers only and note that training dynamics analysis is unavailable.

### Circular Dependency Prevention

The DIAGNOSE step reads:
- `analysis.md` from the **previous** experiment or last explicit `/analyze` call
- Current `summary.md` journal entries (always up to date)

It never reads analysis of the experiment it just ran before that experiment has been analyzed. This ordering is explicit in the skill.

### Analysis Report Structure (3 sections)

Written to `experiments/analysis.md`:
1. **Metrics snapshot** — same as current (per-class AP, overall, confusion matrix)
2. **Training dynamics** — convergence speed, overfitting onset, LR effectiveness, loss plateaus
3. **Experiment trajectory** — what's worked, what hasn't, categorized stall detection, what to try next

---

## Section 5: Guardrails Framework

### Three-Tier Autonomy Model

| Tier | Actions | Enforcement |
|------|---------|-------------|
| **Full autonomy** | HP changes within bounds, evaluations, standard augmentation, lightweight analysis | Agent proceeds |
| **Justification required** | Architecture changes, imgsz changes, optimizer switches | Must write justification citing ≥2 dataset profile numbers |
| **Hard stop** | Budget exceeded, data modification, model deployment, outside allowed actions | Agent stops, reports to user |

### Guardrail Inventory

Maintained in `auto-experiment/resources/guardrails-inventory.md` for reference. The agent doesn't read this file — individual skills have their own guardrails sections. This is the single source of truth for maintainability.

| # | Guardrail | Location | Tier | Recovery Path |
|---|-----------|----------|------|---------------|
| 1 | Checkpoint backup before every experiment | auto-experiment | Autonomy | N/A (preventive) |
| 2 | 3 consecutive regressions → forced diagnostic | auto-experiment | Justification | Write diagnostic → 1 more experiment from diagnostic → if 4th regression, hard stop and ask user |
| 3 | Architecture change gate (3 experiments min) | auto-experiment + program.md | Justification | Wait until gate met, or cite small-object exception |
| 4 | Architecture justification quality (≥2 profile numbers) | auto-experiment | Justification | Rewrite justification with specific stats |
| 5 | Session budget enforcement | auto-experiment | Hard stop | Write session report, stop. User can start new session. |
| 6 | Per-experiment time limit (Ultralytics `time` param) | auto-experiment + program.md | Hard stop | Training auto-stops, agent reads partial results |
| 7 | Stall detection (diminishing returns) | auto-experiment | Justification | Surfaces recommendation, doesn't force action |
| 8 | Max epochs per experiment | program.md | Hard stop | Training stops at limit |
| 9 | AP regression threshold vs current best | program.md | Justification | Agent notes regression, factors into next hypothesis |
| 10 | Allowed architecture set | program.md | Hard stop | Agent cannot use configs outside the list |
| 11 | Immutable data directory | review-dataset, active-learning | Hard stop | All modifications on copies |
| 12 | Merge validation | active-learning | Hard stop | Always yolo-validate after merge |
| 13 | Re-profile after merge | active-learning | Autonomy | Re-run /review-dataset after data changes |
| 14 | Metrics regression after new data | active-learning | Hard stop | Stop and investigate before continuing |

### The Autonomy Contract

> Guardrails define the walls. Everything inside the walls is the agent's decision space.

**The agent has full freedom over:**
- Which hyperparameters to try (within allowed set)
- What order to try things
- When to branch vs continue a direction
- How to interpret results
- What hypothesis to form
- When to call `/analyze` for full trajectory analysis

---

## New Files Created

### `scripts/profile_dataset.py`

Standalone Python script (~50 lines). Called by `/review-dataset` skill.

```
python scripts/profile_dataset.py --labels <path> --imgsz <int>
```

- Reads YOLO label files + samples image dimensions via PIL
- Computes scale distribution at training resolution with letterbox adjustment
- Reports class-wise sizes sorted ascending
- Compares train/val distributions
- Outputs structured YAML to stdout

### `configs/architectures/yolo11.yaml`

Standard YOLO11 P3/P4/P5 config. Copied from Ultralytics source with nc set to project class count.

### `configs/architectures/yolo11-p2.yaml`

YOLO11 with P2/P3/P4/P5 heads for small object detection. Based on the config in `docs/research/LLM-driven-YOLO.md`. Nearly doubles compute vs standard.

### `configs/architectures/yolo11-p2p3p4.yaml`

YOLO11 with shifted P2/P3/P4 heads (drops P5). For datasets where most objects are small and very few are large. Lighter than P2/P3/P4/P5.

---

## Minimal Updates (no design section needed)

### `setup-project/SKILL.md`
- Generates new-format program.md (Model Lineage with architecture config field, Dataset Summary with profile schema, Current Performance with defined fields)
- Calls updated `/review-dataset` which now includes profiling

### `active-learning/SKILL.md`
- Delegates training decisions to refactored `/experiment` (DRY)
- Adds re-profile step: re-run `/review-dataset` after CVAT merge to update program.md

### `CLAUDE.md`
- Update "How Experiments Work" section (steps 1-5) to match reasoning-loop design
- Update any references to the old decision tree or scripted experiment flow
