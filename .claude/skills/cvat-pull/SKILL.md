---
name: cvat-pull
description: "Pull human-corrected annotations from CVAT into local YOLO dataset for training."
---

# Pull CVAT Annotations

Pull human-corrected annotations from CVAT into your local dataset.

## Pre-Flight Checklist
- [ ] `yolo-project.yaml` has `cvat:` section with url and project_id
- [ ] `CVAT_ACCESS_TOKEN` env var is set
- [ ] CVAT instance is reachable (default: http://localhost:8080)

## Workflow

### 1. Identify What to Pull
- Ask the user which CVAT task or project to pull
- Or read the default `project_id` from `yolo-project.yaml`

### 2. Pull Annotations
```bash
yolo-cvat pull --task <TASK_ID>
# or
yolo-cvat pull --project <PROJECT_ID>
```

### 3. Validate the Downloaded Dataset
```bash
yolo-validate <output_path>
```
Check for:
- Valid data.yaml with correct class names
- Image/label count matches expectations
- No annotation errors

### 4. Compare with Existing Dataset
If the user already has a local dataset:
- Compare class distributions
- Check for new images vs corrections
- Suggest merge strategy if combining

### 5. Merge if Needed
```bash
yolo-merge --sources <existing_labels> <pulled_labels> --output <merged>
```

## Decision Tree

```
Has existing local dataset?
├── Yes → Compare distributions → Suggest yolo-merge
└── No → Set as primary dataset

Validation warnings found?
├── Yes → Flag issues, ask before training
└── No → Ready to train

Class distribution changed?
├── Significantly → Warn user, may affect model balance
└── Minor → Proceed normally
```

## Guardrails
- NEVER overwrite an existing dataset directory without user confirmation
- ALWAYS run yolo-validate after pulling
- Report class distribution changes clearly
- If merge is needed, show the user what will change before executing
