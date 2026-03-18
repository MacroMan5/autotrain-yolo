---
name: cvat-deploy
description: "Deploy a trained YOLO model as a Nuclio serverless function for CVAT auto-annotation."
---

# Deploy Model to CVAT

Deploy a trained YOLO model as a Nuclio serverless function so CVAT
users can auto-annotate images directly from the CVAT UI.

## Pre-Flight Checklist
- [ ] Trained model exists (best.pt or user-specified)
- [ ] Nuclio is running (check port 8070)
- [ ] `yolo-project.yaml` has class definitions
- [ ] Docker is running (required for Nuclio builds)

## Workflow

### 1. Identify the Model
- Use the best model from the latest experiment, or ask the user
- Check `experiments/summary.md` for the best-performing model path

### 2. Generate Nuclio Function
```bash
yolo-cvat deploy --model <path_to_model> --name <detector_name>
```
This generates:
- `serverless/<name>/function.yaml` — Nuclio config with class spec
- `serverless/<name>/main.py` — Inference handler
- `serverless/<name>/best.onnx` — Exported model

### 3. Verify Generated Files
Read the generated `function.yaml` and confirm:
- Class names match `yolo-project.yaml`
- Image name is unique
- ONNX model was exported correctly

### 4. Deploy to Nuclio
Provide the command for the user to run:
```bash
nuctl deploy --path ./serverless/<name> --platform local
```

### 5. Verify Deployment
```bash
nuctl get functions
```
Check that the function is running.

## Decision Tree

```
Model is .pt format?
├── Yes → Export to ONNX first (yolo-export)
└── Already .onnx → Copy directly

Nuclio reachable on port 8070?
├── Yes → Ready to deploy
└── No → Warn user, suggest checking Docker and CVAT stack

Function with same name exists?
├── Yes → Ask user: overwrite or use different name?
└── No → Deploy normally
```

## Guardrails
- NEVER deploy without confirming class mapping matches CVAT project labels
- ALWAYS verify the ONNX export succeeded before deploying
- Document the deployed model version and metrics in experiments/
- If deployment fails, show the Nuclio dashboard URL for debugging (http://localhost:8070)
