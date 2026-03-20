---
name: setup-project
description: "Initialize a new YOLO project — detects your dataset's starting state and routes through the right tools."
---

# Setup Project

Interactive project initialization wizard for yolocc.
Detects what you have (raw images, labeled data, complete dataset) and routes you through the right existing tools.

## Workflow

### 1. Gather Project Info
Ask the user:
- **Project name**: What are you detecting? (e.g., "wildlife_detection")
- **Target classes**: What classes do you want to detect? (e.g., "cat, dog, bird")
- **Data path**: Where are your images or data? (can be raw images, labeled data, or a complete dataset)
- **Base model**: Which YOLO model? (default: yolo11n.pt — nano for speed, yolo11s.pt for accuracy)

### 2. Detect Dataset State
Run the detection probe:
```bash
python -c "
from yolocc.dataset.validator import detect_dataset_state
from pathlib import Path
state = detect_dataset_state(Path('<data_path>'))
print(f'Structure: {state.structure}')
print(f'Images: {state.image_count}')
print(f'Labels: {state.label_count}')
print(f'Coverage: {state.label_coverage:.1%}')
print(f'Has splits: {state.has_splits}')
print(f'Has data.yaml: {state.has_data_yaml}')
print(f'Classes in labels: {state.detected_classes}')
print(f'Next steps: {state.next_steps}')
"
```

Report findings to the user, then route based on the detected structure.

### 3. Route Based on State

#### Path A — Complete Dataset (`structure = "complete"`)
The dataset is ready. Tell the user and proceed directly to **Step 4**.

#### Path B — Labeled, Not Split (`structure = "labeled_unsplit"`)
The user has images with labels but no train/val split.

1. Tell the user: "Your data has {image_count} images with labels but no train/val split. I'll split it now."
2. Run: `yolo-split --source <data_path> --output datasets/<project_name> --classes <target_classes>`
3. Update the data path to the split output directory.
4. Proceed to **Step 4**.

#### Path C — Unlabeled Images (`structure = "unlabeled"`)
Check if the user's target classes overlap with COCO's 80 pretrained classes:
```bash
python -c "
from yolocc.dataset.autolabel import get_coco_overlap
overlapping, non_overlapping = get_coco_overlap([<target_classes_as_strings>])
print(f'COCO overlap: {overlapping}')
print(f'Not in COCO: {non_overlapping}')
"
```

**If ALL target classes are in COCO:**
1. Tell the user: "All your classes ({overlapping}) are in COCO's pretrained set. I can auto-label {image_count} images using yolo11n.pt."
2. Ask: "Proceed with auto-labeling? (confidence 0.25, review threshold 0.4)"
3. If yes, run:
   ```bash
   yolo-autolabel --sources <data_path> --output datasets/<project_name> --model yolo11n.pt --review-threshold 0.4
   ```
4. Tell user: "Done. Check the `review/` folder for low-confidence predictions — correct any errors before training."
5. Update the data path. Proceed to **Step 4**.

**If SOME target classes are in COCO:**
1. Tell the user which classes overlap and which don't.
2. Offer to auto-label the overlapping classes only:
   ```bash
   yolo-autolabel --sources <data_path> --output datasets/<project_name>_partial --model yolo11n.pt --classes <overlapping_classes> --review-threshold 0.4
   ```
3. Explain: "For classes not in COCO ({non_overlapping}), you'll need to label ~50-100 images per class manually. Options:"
   - **CVAT**: `yolo-cvat push --images <data_path> --task-name <project_name>` (if CVAT is configured)
   - **Any annotation tool**: Label Studio, Roboflow, CVAT, etc.
4. Exit gracefully: "Run `/setup` again once you've labeled the remaining classes."

**If NO target classes are in COCO:**
1. Explain: "Your classes ({target_classes}) are custom — they're not in COCO's pretrained set, so auto-labeling from scratch isn't possible."
2. If the user has a previously trained model, offer: `yolo-autolabel --sources <data_path> --output datasets/<project_name> --model <their_model.pt> --review-threshold 0.4`
3. Otherwise, provide guidance:
   - "Label ~50-100 images per class manually to bootstrap"
   - Suggest CVAT or other annotation tools
4. Exit gracefully: "Run `/setup` again once you have labeled data."

#### Path E — Partial Labels (`structure = "partial_labels"`)
1. Tell the user: "You have {label_count} labels for {image_count} images ({label_coverage:.0%} coverage)."
2. If they have a trained model, offer to auto-label the rest:
   ```bash
   yolo-autolabel --sources <unlabeled_images> --model <their_model.pt> --output datasets/<project_name>_expanded --review-threshold 0.5
   ```
3. If not, suggest labeling more or using COCO pretrained if classes overlap.
4. Mention: "After initial training, `/analyze` finds weak spots and `/experiment` runs active learning loops automatically."

#### Path F — Empty Directory (`structure = "empty"`)
1. Tell the user: "No images found at {data_path}."
2. Ask them to provide a directory containing images and re-run `/setup`.

### 4. Validate & Profile
All paths converge here once a valid YOLO dataset exists.

Run: `yolo-validate <dataset_path>`

Read the output to extract:
- Number of classes and their names (from data.yaml)
- Total images (train + val)
- Class distribution
- Any validation warnings

### 5. Create yolo-project.yaml
Write `yolo-project.yaml` in the workspace root with the gathered info.

### 6. Copy Architecture Configs
Copy `configs/architectures/*.yaml` into the project workspace if not already present.
These are the pre-built configs the agent selects from during experimentation.

### 7. Run Dataset Profile
Run `/review-dataset` which includes the profiling step.
This generates `experiments/dataset_profile.yaml` and fills training-plan.md's Dataset Summary.

### 8. Offer Baseline Run
Ask: "Run a 5-epoch baseline to establish starting metrics? (recommended)"

If yes:
```bash
yolo-experiment baseline --budget 5 --patience 3
```

Read `experiments/summary.md` for baseline metrics.

### 9. Generate training-plan.md
Create `training-plan.md` using the **boundaries template** (not scripted phases):

```markdown
# <Project Name> — Training Plan

## Project Context

### Training Mode
- [ ] Training from scratch
- [ ] Fine-tuning from pretrained model
- [ ] Transfer learning (freeze backbone)

### Model Lineage
- Base model: `<model>.pt`
- Architecture config: `configs/architectures/yolo11.yaml` (standard P3/P4/P5)
- Current best: (from baseline, or "run /setup to establish baseline")
- Best backup: (none yet)

### Model Intent
- [ ] Specialist (few classes, high accuracy)
- [ ] Generalist (many classes, broad coverage)
- Deployment target: (ask user)

### Setup Path
- Starting state: <detected structure>
- Auto-labeled: yes/no (if yes, note review/ folder status)

### Dataset Summary
(Auto-filled by /review-dataset profiling step)
- Total images: train / val
- Classes: N — [list]
- Class balance: most/least represented
- Scale distribution: % small / medium / large at imgsz
- Min object size at training resolution: Npx
- Avg objects per image:

### Current Performance
(Auto-filled after baseline)
- mAP50-95:
- mAP50:
- Per-class AP50: {class: value, ...}
- Weakest class:

## Goal
(Ask user for primary metric target)

### Secondary Goals
- (from dataset analysis: e.g., improve weakest class)

## Hard Constraints (agent cannot violate)
- Max experiments per session: 10
- Max minutes per session: 120
- Max epochs per experiment: 50
- Don't delete or modify original dataset files
- Don't decrease any class AP50 by more than 0.05 vs current best model
- Minimum 3 experiments on current architecture before switching
  (exception: dataset profile shows >50% small objects with no P2 head)

## Soft Preferences (agent can override with justification)
- Start with current model variant before trying others
- Prefer augmentation approaches before architecture changes
- Prioritize weakest class improvement

## Allowed Actions
### HP Optimization (via model.tune)
- Presets: lr, augmentation, loss, optimizer, all
- Custom: any parameter with min:max range
- Agent selects preset based on diagnosis

### Tune Defaults
- Iterations per tune: 20
- Epochs per iteration: 10
- Patience: 5

### Architecture
- Model variants: n, s, m
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

Fill in what's known from steps 2, 4, 7, 8. Leave placeholders for user-provided info.

### 10. Print Summary
Tell the user what was created and suggest next steps based on their path:
- **Path A/B**: "Run `/experiment` to start autonomous experimentation"
- **Path C (auto-labeled)**: "Review the `review/` folder first, then run `/experiment`"
- **Path C/D (needs manual labels)**: "Label your data, then run `/setup` again"
- **Path E**: "Run `/experiment` — it includes active learning loops"

## Guardrails
- Never overwrite existing yolo-project.yaml without asking
- Never run yolo-autolabel without confirming with the user first
- If dataset has no images, exit with clear guidance
- All auto-labeled data should use `--review-threshold` to flag uncertain predictions
- If the user already has a yolo-project.yaml, offer to update it rather than recreate
