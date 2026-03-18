---
name: compare-models
description: "Compare 2+ YOLO models side-by-side on the same dataset — mAP, per-class AP, speed, size."
---

# Compare Models

Compare 2+ YOLO models on the same validation dataset. All models are evaluated with the same `data` and `imgsz` for a fair comparison.

## Inputs

- **Model paths**: 2 or more `.pt` file paths (required)
- **Dataset**: path to `data.yaml` (optional — falls back to `yolo-project.yaml` defaults.dataset)
- **imgsz**: image size (optional — falls back to `yolo-project.yaml` defaults.imgsz, then 640)

## Pre-checks

1. **Verify every model path exists** before running any validation. If any path is missing, stop and report which files were not found.
2. **Warn if >5 models** are requested — total validation time may be significant. Ask the user to confirm before proceeding.
3. **Resolve dataset and imgsz** from `yolo-project.yaml`:

```bash
python -c "
import yaml, pathlib
cfg_path = None
for p in [pathlib.Path('yolo-project.yaml')] + list(pathlib.Path('.').resolve().parents):
    candidate = p / 'yolo-project.yaml' if p.is_dir() else p
    if candidate.exists():
        cfg_path = candidate
        break
if cfg_path:
    cfg = yaml.safe_load(cfg_path.read_text())
    defaults = cfg.get('defaults', {})
    print(f\"dataset: {defaults.get('dataset', 'NOT SET')}\")
    print(f\"imgsz: {defaults.get('imgsz', 640)}\")
else:
    print('No yolo-project.yaml found')
"
```

If no dataset is provided and none is in the project config, stop and ask the user.

4. **Locate data.yaml**: The dataset path from the config points to a YOLO dataset directory. The `data.yaml` file is at `{dataset}/data.yaml`. Verify it exists.

## Validation

For each model, run validation with identical settings:

```bash
python -c "
import json, os
from ultralytics import YOLO

model_path = '<MODEL_PATH>'
data_yaml = '<DATA_YAML>'
imgsz = <IMGSZ>
device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

model = YOLO(model_path)
results = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=False)

# File size in MB
size_mb = os.path.getsize(model_path) / (1024 * 1024)

# Inference speed (ms per image, preprocess + inference + postprocess)
speed = results.speed
total_ms = speed.get('preprocess', 0) + speed.get('inference', 0) + speed.get('postprocess', 0)

# Per-class AP50 values
class_names = results.names
per_class_ap50 = {}
per_class_ap50_95 = {}
if results.box.ap_class_index is not None:
    for i, cls_idx in enumerate(results.box.ap_class_index):
        name = class_names[int(cls_idx)]
        per_class_ap50[name] = round(float(results.box.ap50[i]), 4)
        per_class_ap50_95[name] = round(float(results.box.ap[i]), 4)

out = {
    'model': model_path,
    'mAP50': round(float(results.box.map50), 4),
    'mAP50-95': round(float(results.box.map), 4),
    'precision': round(float(results.box.mp), 4),
    'recall': round(float(results.box.mr), 4),
    'speed_ms': round(total_ms, 1),
    'size_mb': round(size_mb, 1),
    'per_class_ap50': per_class_ap50,
    'per_class_ap50_95': per_class_ap50_95,
}
print(json.dumps(out))
"
```

Collect the JSON output from each run.

## Output

### 1. Overall Comparison Table

Build a markdown table from the collected results:

```
| Model | mAP50 | mAP50-95 | Precision | Recall | Speed (ms) | Size (MB) |
|-------|-------|----------|-----------|--------|------------|-----------|
| ...   | ...   | ...      | ...       | ...    | ...        | ...       |
```

Sort by mAP50-95 descending. Bold the best value in each column.

### 2. Per-Class AP Delta Table (if >1 class)

Show per-class AP50 for each model. If exactly 2 models, add a delta column. If >2 models, show raw values and bold the best per class.

```
| Class | Model A | Model B | Delta |
|-------|---------|---------|-------|
| cat   | 0.92    | 0.88    | +0.04 |
| dog   | 0.85    | 0.91    | -0.06 |
```

### 3. Recommendations

State plainly:

- **Best overall**: Model with highest mAP50-95
- **Best per-class**: For each class, which model wins (only if models disagree)
- **Speed/accuracy tradeoff**: If a smaller/faster model is within 1-2% mAP of the best, call it out as a viable deployment option
- **Notable gaps**: Any class where a model drops >5% AP compared to the best — flag it

## Important

- All models MUST be validated on the same dataset with the same imgsz. Do not mix settings.
- Do not install anything. `ultralytics` and `torch` are already available.
- Print the comparison tables directly — do not write to a file unless the user asks.
- Keep commentary factual. No speculation about why a model is better — just report the numbers and let the user decide.
