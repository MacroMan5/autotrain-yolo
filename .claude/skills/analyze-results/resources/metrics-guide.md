# YOLO Metrics Guide

| Metric | What | Good | Great |
|--------|------|------|-------|
| mAP50 | Detection at 50% IoU | >0.70 | >0.85 |
| mAP50-95 | Detection avg 50-95% IoU | >0.45 | >0.60 |
| Precision | % detections correct | >0.80 | >0.90 |
| Recall | % objects detected | >0.70 | >0.85 |

## Overfitting Signals
- Train loss down, val loss up = OVERFITTING
- Both decreasing = GOOD
- Both plateaued = converged

## Per-Class AP
- >0.80: Well-learned
- 0.60-0.80: Decent
- 0.40-0.60: Struggling — need more data
- <0.40: Severely underperforming

## Common Fixes
| Symptom | Fix |
|---------|-----|
| High precision, low recall | Lower conf threshold, more data |
| Low precision, high recall | More negative examples |
| One class much worse | More data for that class, focal loss |
| Small objects missed | Increase imgsz, use SAHI |
