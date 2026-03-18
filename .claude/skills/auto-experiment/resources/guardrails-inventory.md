# Guardrails Inventory

Single source of truth for all experiment guardrails.

## Two-Tier Autonomy Model

| Tier | Actions | Enforcement |
|------|---------|-------------|
| **Full autonomy** | HP tune, strategic experiments, evaluations, augmentation | Agent proceeds |
| **Justification required** | Architecture changes, imgsz changes, optimizer switches | Journal entry must cite ≥2 dataset profile numbers |

## Guardrail Details

| ID | Rule | Location | Tier | Trigger | Recovery |
|----|------|----------|------|---------|----------|
| G-05 | Architecture change gate (3 experiments) | auto-experiment/SKILL.md + program.md | Justification | Agent wants architecture change | Must have 3+ experiments with <0.5% improvement first |
| G-06 | Architecture change exception | auto-experiment/SKILL.md | Full auto | >50% small objects, no P2 head | Immediate change allowed |
| G-07 | Architecture justification quality | auto-experiment/SKILL.md | Justification | Architecture change | Must cite ≥2 dataset profile numbers |
| G-08 | Immutable data | auto-experiment/SKILL.md | Hard stop | Any dataset modification | Never allowed — copies only |
| G-09 | Per-experiment time limit | auto-experiment/SKILL.md | Full auto | Always | Use Ultralytics `time` parameter |
| G-10 | AP regression observation | auto-experiment/SKILL.md | Full auto | Class AP threshold violated | Acknowledge in journal `learning` field |
| G-11 | Session resume detection | auto-experiment/SKILL.md | Full auto | Session marker with no end marker | Count existing entries toward budget |
| G-12 | Merge validation | active-learning/SKILL.md | Hard stop | After CVAT merge | Always run yolo-validate after merge |
| G-13 | Re-profile after merge | active-learning/SKILL.md | Full auto | After CVAT merge + validate | Re-run /review-dataset |
| G-14 | Metrics regression after new data | active-learning/SKILL.md | Hard stop | Metrics regress after adding data | Stop and investigate |

## Hook Enforcement

Only one hook remains: `pre_experiment_guard.py` (PreToolUse:Bash)
- Checkpoint backup: copies `models/best.pt` → `models/best_backup.pt` before every run/tune
- Budget exhausted check: blocks if `.budget_exhausted` file exists (manual safety valve)
- Session budget enforcement: counts experiments since last session marker
