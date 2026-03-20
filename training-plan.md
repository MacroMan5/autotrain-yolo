# Training Plan

> Edit this file to define what to optimize and how.
> /experiment reads this before every session.

## Project Context

### Training Mode
- [ ] Training from scratch (no pretrained weights)
- [x] Fine-tuning from pretrained model
- [ ] Transfer learning (freeze backbone)

### Model Lineage
- Base model: `yolo11n.pt` (COCO pretrained, 80 classes)
- Architecture config: `configs/architectures/yolo11.yaml` (standard P3/P4/P5)
- Current best: (run /setup to establish baseline)
- Best backup: (none yet)

### Model Intent
- [ ] Specialist (specific domain, few classes, high accuracy)
- [ ] Generalist (many classes, broad coverage)
- Deployment target: (describe target hardware and latency requirements)

### Dataset Summary
(Run /review-dataset to auto-fill this section)
- Total images: ? (train: ? / val: ?)
- Classes: ?
- Class balance: ?
- Scale distribution: % small / medium / large at imgsz
- Min object size at training resolution: ?
- Avg objects per image: ?

### Current Performance
(Run /experiment baseline to fill this in)
- mAP50-95: —
- mAP50: —
- Per-class AP50: {}
- Weakest class: —

## Goal
(Set your primary metric target)

### Secondary Goals
- (e.g., improve weakest class AP50 to > 0.60)
- (e.g., maintain inference speed < 10ms)

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

### Tune Defaults
- Iterations per tune: 20
- Epochs per iteration: 10
- Patience: 5

## Domain Knowledge
> Context that can't be learned from dataset statistics alone.
- (e.g., "Objects are frequently occluded — erasing augmentation is relevant")
- (e.g., "Class 'smoke' is visually similar to 'fog' — confusion is the main problem")
- (e.g., "False positives are more costly than missed detections in this application")
