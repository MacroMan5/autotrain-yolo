---
name: analyze-results
description: "Analyze YOLO training runs — compares to baseline/best, checks per-class regression, analyzes training dynamics and tune convergence, writes actionable recommendations."
---

# Analyze Results

Single-pass analysis run after each experiment or on-demand via `/analyze`.

## Data

Read:
- Latest experiment's report + `results.csv` or `tune_results.csv`
- Current best metrics from `experiments/summary.md` journal entries
- `experiments/dataset_profile.yaml`
- `program.md` (thresholds, goals, constraints)

### Results.csv Lookup Order

1. `experiments/exp_NNN_name/train/results.csv`
2. `experiments/exp_NNN_name/results.csv`
3. Fall back to `metrics.yaml` — note that training dynamics analysis is unavailable

For tune runs, also check:
- `experiments/exp_NNN_name/tune/tune_results.csv`

## Analysis

**Comparison**: Latest vs baseline AND current best (mAP50-95, per-class AP)

**Per-class regression check**: Flag any class exceeding `program.md` threshold vs current best

**Training dynamics** (if results.csv available):
- Convergence speed: epochs to reach 90% of final mAP
- Overfitting onset: epoch where val loss diverges from train loss
- Loss plateau: epochs with <0.1% change in val loss
- LR effectiveness: correlation between LR schedule phase and metric improvement

**Tune analysis** (if tune_results.csv available):
- Tune convergence: are later iterations improving or stalled?
- Best params found vs defaults — which params moved the most?
- Top-N trial spread: how tight is the performance range?

**Diminishing returns detection** (when 3+ experiments show <0.5% change):
- **Same-category stall** (all augmentation, or all LR variations): that lever is exhausted, try a different category
- **Cross-category stall** (multiple lever types tried): model may be at ceiling for this dataset/architecture
- Recommendation differs per stall type — surface this explicitly

**Recommendations**: Ordered by expected impact. Each cites evidence from the analysis.

## Output

Write `experiments/analysis.md` with findings and recommendations. Print top 3 findings + top 3 recommendations.

Findings also inform the journal entry's `learning` field and the next DIAGNOSE step in the reasoning loop.

## Important

- The reasoning loop's DIAGNOSE step reads `analysis.md` from the **previous** experiment or last explicit `/analyze`. It does NOT read analysis of the experiment that just ran before it has been analyzed.
- Keep analysis concise — focus on what changed and what to try next, not restating all historical metrics.
