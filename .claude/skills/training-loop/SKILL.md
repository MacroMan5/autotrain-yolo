---
name: training-loop
description: "Run a managed YOLO training session — validates dataset, trains, analyzes, generates clean report."
---

# Training Loop

Single managed training run with automatic reporting.

## Workflow
1. Read `yolo-project.yaml`
2. Run `yolo-validate` — stop on errors
3. Check GPU availability
4. Run training: `yolo-train` or `yolo-finetune`
5. After completion, run `yolo-analyze`
6. Generate report in `experiments/`
7. Print summary + model path + recommendation
