---
name: benchmark
description: "Profile YOLO model inference speed, FPS, and size across image sizes and export formats."
---

# Benchmark

Profile a YOLO model's inference performance across image sizes and devices.

## Inputs

### Model Path

1. If the user provides a model path, use it directly.
2. If not provided, look for the current best model:
   - Check `experiments/summary.md` for the latest best model path
   - Search `experiments/*/weights/best.pt` for the most recent
   - Fall back to asking the user

### Image Sizes

Default: `[320, 640, 1280]`. The user can override this list.

### Device

Detect available devices. Run on GPU (`0`) if available, CPU (`cpu`) always. If both exist, benchmark both and compare.

## Procedure

1. **Read `training-plan.md`** — check the "Deployment target" and "Secondary Goals" sections for latency/size constraints.

2. **Locate the model** using the lookup order above. Confirm the path exists before proceeding.

3. **Run benchmarks** — for each image size and each available device, run:

```python
from ultralytics import YOLO
import torch

model = YOLO(path)

# Check GPU availability
gpu_available = torch.cuda.is_available()
devices = ["cpu"]
if gpu_available:
    devices.insert(0, 0)  # GPU first

for device in devices:
    for imgsz in image_sizes:
        results = model.benchmark(imgsz=imgsz, half=False, device=device)
```

`model.benchmark()` handles warm-up internally. Do not add manual warm-up runs.

4. **Collect results** — from each benchmark call, extract:
   - Inference time (ms)
   - FPS (frames per second)
   - Model size (MB)
   - Format tested

5. **Report** — print a summary table:

```
## Benchmark Results: <model_name>

### GPU (NVIDIA <name>) / CPU

| Format   | imgsz | Inference (ms) | FPS    | Size (MB) |
|----------|-------|----------------|--------|-----------|
| PyTorch  | 320   | ...            | ...    | ...       |
| PyTorch  | 640   | ...            | ...    | ...       |
| PyTorch  | 1280  | ...            | ...    | ...       |
```

6. **Analyze against deployment constraints** — if `training-plan.md` specifies:
   - A latency target (e.g., "< 10ms"): flag any configuration that exceeds it
   - A model size limit: flag if the model exceeds it
   - A target device: highlight the relevant device results

7. **Recommend optimal imgsz** — based on the results:
   - Identify the largest imgsz that meets latency requirements
   - If no latency requirement is specified, note the speed/accuracy tradeoff (larger imgsz = better accuracy, slower inference)
   - If all sizes exceed the constraint, say so and suggest smaller model variants (n < s < m)

## Output Format

Print results directly to the conversation. Do not write to a file unless the user asks.

Structure:
1. Model info (path, parameter count, variant)
2. Results table(s) — one per device
3. Deployment check — pass/fail against `training-plan.md` constraints
4. Recommendation — optimal imgsz and format for the stated deployment target

## Important

- Do not export the model to other formats unless the user asks. `model.benchmark()` tests the PyTorch format by default.
- If `training-plan.md` has no deployment constraints, skip the constraint check and just report the numbers.
- Keep output factual. Report what the numbers show, suggest next steps if relevant.
- If the model file does not exist, stop and tell the user. Do not train a model.
