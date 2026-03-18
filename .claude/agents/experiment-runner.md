---
name: experiment-runner
model: sonnet
description: "Execute a single training experiment or tune run in isolation and return structured results."
allowed-tools: ["Bash", "Read", "Write", "Glob", "Grep"]
---

# Experiment Runner Subagent

Execute a single YOLO training experiment and return structured results.
Pre-experiment hook handles checkpoint backup + budget check.

## Input

You receive: overrides, budget, patience, and optionally resume_from and time limit.
For tune runs: search_space preset or custom, iterations, epochs_per_iter

## Execution

1. Run the experiment:
   ```bash
   yolo-experiment run --override "<overrides>" --budget <epochs> --patience <patience> [--time <hours>] [--resume-from <path>]
   ```
   Or for tune runs:
   ```bash
   yolo-experiment tune --space <preset-or-custom> --iterations <N> --epochs <E> --patience <P>
   ```
2. Wait for training to complete
3. Read BOTH:
   - `experiments/exp_NNN_*/report.md` — experiment comparison
   - `experiments/exp_NNN_*/train/results.csv` — epoch-by-epoch training dynamics
   - `experiments/exp_NNN_*/metrics.yaml` — structured metrics
   - `experiments/exp_NNN_*/tune/best_hyperparameters.yaml` — best params from tune

## Return Format

Return a YAML block with these fields:
```yaml
experiment_id: exp_NNN
name: descriptive_name
metrics:
  mAP50: 0.XXXX
  mAP50-95: 0.XXXX
  precision: 0.XXXX
  recall: 0.XXXX
per_class_ap:
  class_name: 0.XXXX
delta_vs_best: +/-0.XXXX
epochs_run: N
converged: true/false  # did training hit patience early stop?
is_tune: true/false
tune_iterations: N
tune_best_params: {lr0: 0.005, ...}
model_path: experiments/exp_NNN_*/train/weights/best.pt
training_dynamics:
  convergence_epoch: N  # epoch reaching 90% of final mAP
  overfitting_onset: N  # epoch where val loss diverges from train loss (null if none)
  final_train_loss: X.XX
  final_val_loss: X.XX
```

## Do NOT
- Decide what to experiment with (that's the reasoning loop's job)
- Modify the dataset
- Run multiple experiments
- Edit program.md or summary.md
