---
name: monitor-training
description: "Set up autonomous training monitoring — creates cron jobs to track long-running training, auto-continue pipeline when training completes."
---

# Monitor Training — Autonomous Training Pipeline

Set up cron-based monitoring for long-running YOLO training sessions. Automatically detects completion, reports results, and continues to the next pipeline phase.

## When to Use

- After launching a training run that will take hours
- When you want to chain multiple pipeline phases autonomously (train → analyze → clean → retrain → tune → export)
- To monitor training progress without manual polling

## Workflow

### 1. Identify Training State

Determine what's currently running:
- Check for active GPU processes: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
- Check for results CSV: `tail -5 <training_dir>/results.csv`
- Check background task status if task ID is known

### 2. Create Monitoring Cron Job

Use `CronCreate` with a 5-minute interval to poll training status:

```
CronCreate:
  cron: "*/5 * * * *"
  recurring: true
  prompt: <monitoring prompt with next-phase instructions>
```

The monitoring prompt should include:
- **Where to check**: Path to `results.csv` and/or background task ID
- **What to report**: Current epoch, mAP50, mAP50-95, training losses
- **Completion criteria**: Task finished, early stopping triggered, or target metric reached
- **Next phase actions**: Detailed steps to execute when training completes

### 3. Monitor Prompt Template

Build the cron prompt with these sections:

```
1. CHECK STATUS
   - Read results.csv (tail -5) for latest metrics
   - Check if training process is still running
   - Report: epoch, mAP50, mAP50-95, whether still active

2. ON COMPLETION — Execute next phase:
   [Phase-specific instructions]

3. AFTER NEXT PHASE — Chain to following phase:
   - Delete current cron job (CronDelete)
   - Create new cron job for the next long-running phase

4. PLAN REFERENCE
   - Read plan file for full pipeline context
```

### 4. Pipeline Chaining Pattern

Each cron job monitors one phase and launches the next:

```
Cron 1: Monitor baseline training
  → On complete: run analysis + dataset cleanup
  → Launch retrain
  → Delete self, create Cron 2

Cron 2: Monitor retrain
  → On complete: launch model.tune()
  → Delete self, create Cron 3

Cron 3: Monitor tune
  → On complete: launch final training with best params
  → Delete self, create Cron 4

Cron 4: Monitor final training
  → On complete: export model, generate final report
  → Delete self
```

### 5. Completion Actions

When a training phase completes:

1. **Report final metrics** — epoch, per-class mAP, precision, recall
2. **Save best model** — copy `best.pt` to a named location
3. **Run per-class validation** — use a free GPU for detailed evaluation
4. **Update plan file** — mark phase complete, record metrics
5. **Update tasks** — mark current task complete, create next
6. **Launch next phase** — start the next pipeline step
7. **Create new monitor** — set up cron for the new phase

## Guidelines

- **Cron interval**: 5 minutes is good for training. Use 2 minutes for shorter operations like analysis.
- **GPU awareness**: Before running validation or analysis, check GPU memory. Use a different GPU than the one training, or wait for training to finish.
- **Error handling**: If a training crashes (OOM, data error), report the error and stop — don't auto-retry without understanding why.
- **Plan file**: Always reference the plan file for pipeline context. Update it as phases complete.
- **Session lifetime**: Cron jobs expire after 7 days or when the session ends. For multi-day pipelines, document the current state in the plan file so a new session can resume.
- **Idempotency**: The cron prompt may fire multiple times while a phase is still running. Only trigger next-phase actions on actual completion, not on intermediate checks.

## Example

```
User: "Lance le training et surveille-le automatiquement"

1. Launch training in background
2. Create cron:
   CronCreate("*/5 * * * *", "Check training at <path>/results.csv.
     If still running: report epoch and metrics.
     If complete: run validation, save model, launch next phase...")
3. Training runs autonomously
4. Cron detects completion → runs analysis → launches retrain → creates new cron
5. Pipeline continues without user intervention
```
