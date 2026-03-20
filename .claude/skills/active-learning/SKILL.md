---
name: active-learning
description: "Orchestrate the full active learning loop: train, analyze, push to CVAT, wait for review, pull, merge, retrain."
---

# Active Learning Loop

Orchestrate the complete cycle of model improvement through human feedback:
train -> analyze -> push uncertain images to CVAT -> wait for human review ->
pull corrections -> merge -> retrain.

## Workflow

### 1. Check Current State
Determine where we are in the loop:

```
Has baseline run?
├── No → Run: /experiment baseline
└── Yes → Continue

Has analysis been run?
├── No → Run: /analyze
└── Yes → Continue

Has uncertain_images.txt?
├── No → Run: yolo-analyze --model <best> --dataset <ds>
└── Yes → Continue
```

### 2. Push Uncertain Images to CVAT
Use the /cvat-push skill:
```bash
yolo-cvat push --from-analysis reports/uncertain_images.txt
```

### 3. Wait for Human Review
Tell the user:
```
Images have been pushed to CVAT for review.
Please annotate/correct the images in CVAT, then tell me when you're done.

CVAT tasks: [list task IDs and URLs]
Images to review: [count]
Priority: [false negatives first, then uncertain]
```

**Do NOT proceed until the user confirms annotations are complete.**

### 4. Pull Corrected Annotations
When the user says annotations are done:
```bash
yolo-cvat pull --task <id> --output datasets/corrected_batch_N
yolo-validate datasets/corrected_batch_N
```

### 5. Merge with Existing Dataset
```bash
yolo-merge --sources <existing_labels> <corrected_labels> --output <merged_labels>
yolo-validate <merged_dataset>
```

### 5b. Re-Profile After Merge
Run `/review-dataset` to update the dataset profile in training-plan.md.
The profile may have changed (new class distribution, different object sizes).
The experiment reasoning loop needs current data.

### 6. Retrain
Delegate training decisions to the refactored `/experiment` skill.
The reasoning loop will use the updated dataset profile to decide what to try.
```bash
/experiment
```

### 7. Compare Before/After
Compare metrics from before and after the new data:
- Overall mAP50-95 change
- Per-class AP changes
- Specifically check classes that had false negatives

### 8. Write Active Learning Report
Create `experiments/active_learning_log.md`:
```markdown
## Active Learning Iteration N
- Date: YYYY-MM-DD
- Images added/corrected: X
- mAP50-95 before: X.XXXX
- mAP50-95 after: X.XXXX
- Delta: +/- X.XXXX
- Classes most improved: [list]
- Next action: [recommendation]
```

## Decision Tree

```
First iteration ever?
├── Yes → Establish baseline first, then analyze
└── No → Check if previous iteration improved metrics

Metrics improved after new data?
├── Yes → Deploy updated model to CVAT (/cvat-deploy)
│         Continue to next iteration
└── No → Investigate:
         ├── Data quality issue? → Review annotations more carefully
         ├── Overfitting to corrections? → Increase augmentation
         └── Plateaued? → Suggest dataset expansion or architecture change

More than 3 iterations with <0.5% improvement?
├── Yes → Suggest stopping: diminishing returns
└── No → Continue loop
```

## Guardrails
- NEVER skip the merge validation step — corrupted merges ruin models
- ALWAYS compare metrics before and after new data is added
- Document every iteration in `experiments/active_learning_log.md`
- If metrics regress after adding data, STOP and investigate before continuing
- Maximum recommended batch: 200 images per iteration (annotator fatigue)
- Wait for user confirmation at every human-in-the-loop step
