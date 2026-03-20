# FPS Detection — Training Plan

## Project Context

### Training Mode
- [x] Fine-tuning from pretrained model

### Model Lineage
- Base model: `yolo11n.pt`
- Architecture config: `configs/architectures/yolo11.yaml` (standard P3/P4/P5)
- Current best: `fps_base_v3.pt` (mAP50: 0.82, mAP50-95: 0.52)
- Best backup: (none yet)

### Model Intent
- [x] Specialist — real-time FPS player detection
- Deployment target: 640px input, <5ms inference on RTX 4070
- Critical: high recall (missing a player = death), acceptable false positives

### Dataset Summary
- Total images: 15,000 across 5 games (PUBG, CS2, Valorant, Fortnite, CoD)
- Classes: 2 — [player, head]
- Class balance: player (12,000 instances), head (8,500 instances)
- Scale distribution: (run /review-dataset to profile)
- Min object size at training resolution: (run /review-dataset to profile)
- Avg objects per image: (run /review-dataset to profile)
- Known issues: head class is smaller, harder to detect at distance

### Current Performance
- mAP50-95: 0.52
- mAP50: 0.82
- Per-class AP50: {player: 0.89, head: 0.75}
- Weakest class: head (0.75)
- Precision: 0.85
- Recall: 0.78

## Goal
Maximize recall while keeping precision > 0.80

### Secondary Goals
- Head AP50 from 0.75 to 0.85+
- Maintain inference < 5ms at 640px

## Hard Constraints (agent cannot violate)
- Max experiments per session: 15
- Max epochs per experiment: 50
- Max training time per experiment: 30 minutes
- Don't delete or modify original dataset files
- Don't decrease any class AP50 by more than 0.05 vs current best model
- Minimum 3 experiments on current architecture before switching
  (exception: dataset profile shows >50% small objects with no P2 head)

## Soft Preferences (agent can override with justification)
- Start with yolo11n (speed critical for deployment)
- Prefer augmentation approaches before architecture changes
- Prioritize head class improvement (currently weakest)
- Test on PUBG variant first (most data)

## Allowed Actions
### Hyperparameters
- Any lr0, lrf, optimizer, momentum, weight_decay
- Any loss weights (box, cls, dfl)
- Any augmentation parameters
- Freeze depths: 0, 10, 15

### Architecture
- Model variants: n only (speed constraint: <5ms inference)
- Head configs:
  - `configs/architectures/yolo11.yaml` (standard P3/P4/P5)
  - `configs/architectures/yolo11-p2.yaml` (P2/P3/P4/P5, if head class needs it)
- imgsz: 640 (deployment constraint)

### Data Handling
- Can create augmented copies (NOT modify originals)

## Domain Knowledge
> Tell the agent things it can't learn from the dataset statistics alone.
- Erasing augmentation is critical — FPS games have smoke, flash grenades, and cover that cause partial occlusion
- Mosaic helps with scale variation (near/far players at different distances)
- Don't use high mixup — blending players into backgrounds is visually unrealistic for this domain
- Head class is small and often partially occluded by helmet/equipment — P2 head may help if profile confirms small objects
- False positives near environment edges (doorframes, furniture) are less costly than missed players
