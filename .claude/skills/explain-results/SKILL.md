---
name: explain-results
description: "Explain training results in plain English — translates metrics into actionable insights for any audience."
---

# Explain Results

Translates YOLO training metrics into plain-English explanations. Two audience modes: **non-technical** (default) and **technical**.

## Data

Find the experiment to explain:

1. If user specifies a path or experiment ID, use that
2. Otherwise, find the latest `experiments/exp_NNN_*` directory (highest NNN)

Read from the experiment directory:
- `report.md` — experiment summary with metrics and deltas
- `metrics.yaml` — structured metrics, per-class AP, overrides, config
- `train/results.csv` or `results.csv` (if available) — epoch-by-epoch training log

Also read:
- `experiments/summary.md` — for historical context (how this compares to other runs)
- `experiments/dataset_profile.yaml` — for dataset size, class counts, image counts
- Reference thresholds from `skills/analyze-results/resources/metrics-guide.md`

## Report Structure

Generate a plain-English report with these sections:

### 1. What Happened

Summarize the training run in one paragraph a project manager could understand.

Template: "We trained a [model] for [X epochs] on [Y images] across [Z classes]. The training [completed normally / stopped early at epoch N / showed signs of overfitting]. This took approximately [time if available]."

Include what changed vs baseline if this is not the baseline run (e.g., "This run increased the image size from 640 to 1280 pixels").

### 2. How Good Is It

Translate the core metrics using analogies, not jargon. Use the metrics-guide.md thresholds to provide context.

| Quality | mAP50-95 | Plain English |
|---------|----------|---------------|
| Great   | >0.60    | "The model reliably finds and precisely locates objects — ready for production use" |
| Good    | 0.45-0.60| "The model catches most objects and places boxes reasonably well — solid for many applications" |
| Fair    | 0.30-0.45| "The model finds some objects but often misses or misplaces them — needs more work" |
| Poor    | <0.30    | "The model is still learning — not ready for real use yet" |

Translate individual metrics:
- **Precision**: "When the model says it found something, it's right X% of the time" (e.g., precision=0.85 -> "correctly identifies the object 85 out of 100 times it makes a detection")
- **Recall**: "Out of all the objects actually present, the model finds X%" (e.g., recall=0.70 -> "catches 70 out of every 100 objects in the images")
- **mAP50**: "How well it draws boxes around objects (at a forgiving threshold)" — use this to explain detection quality
- **mAP50-95**: "The overall score — combines finding objects AND precisely outlining them"

If there is a baseline comparison, translate the delta: "Compared to our starting point, this run [improved/decreased] overall detection by [X points]. In practical terms, [concrete impact]."

### 3. What It's Best At

List the top-performing classes by per-class AP (from metrics.yaml `per_class_ap`).

Use plain language: "The model is strongest at detecting [class] — it correctly finds and outlines [X]% of them."

Group into tiers using metrics-guide.md per-class thresholds:
- **Nailed it** (AP > 0.80): These classes are well-learned
- **Getting there** (AP 0.60-0.80): Decent but room to improve
- **Needs work** (AP 0.40-0.60): Struggling — likely needs more training data
- **Not yet** (AP < 0.40): Severely underperforming

### 4. What It Struggles With

List the worst-performing classes and provide likely reasons:
- Few training examples for that class (check dataset_profile.yaml for class distribution)
- Small objects (if dataset profile shows small object prevalence)
- Visual similarity to other classes
- High intra-class variation

Frame constructively: "The model has the hardest time with [class] (finds only X out of 100). This is likely because [reason]. To improve this, [action]."

### 5. What to Do Next

Provide 1-3 actionable recommendations in plain language. No ML jargon.

Map common situations to advice:
| Situation | Recommendation |
|-----------|---------------|
| High precision, low recall | "The model is cautious — it's accurate when it detects something, but misses too many objects. Adding more training images would help." |
| Low precision, high recall | "The model is trigger-happy — it finds most objects but also flags things that aren't there. Adding more examples of empty backgrounds would help." |
| One class much worse than others | "The model struggles with [class]. Collecting 50-100 more labeled examples of [class] would likely improve this." |
| Small objects missed | "Small objects are being missed. Try training with larger image sizes (e.g., 1280 instead of 640) so the model can see fine details." |
| Overall metrics plateaued | "The model has stopped improving with the current setup. Consider trying a different model architecture or adding more diverse training data." |
| Overfitting detected | "The model memorized the training images instead of learning general patterns. Reduce training duration or add more variety to training images." |

### 6. Key Numbers

A summary table for stakeholders — no jargon, just the essentials.

```
| Metric              | Value  | Rating    | What It Means                          |
|---------------------|--------|-----------|----------------------------------------|
| Overall Score       | 0.XX   | Good/Great| [one-line plain explanation]            |
| Detection Accuracy  | 0.XX   | ...       | "Right X% of the time it makes a call" |
| Object Coverage     | 0.XX   | ...       | "Finds X% of all objects"              |
| Best Class          | [name] | [AP]      | "Finds X% of [class] objects"          |
| Worst Class         | [name] | [AP]      | "Finds only X% of [class] objects"     |
| Training Duration   | N eps  | ...       | "Trained for N rounds"                 |
```

## Technical Mode

When user requests `--technical` or asks for technical details, add these sections after the plain-English report:

### Per-Class AP Breakdown

Full table of all classes with AP values, sorted descending. Include delta vs baseline if available.

### Training Dynamics

If results.csv is available, analyze:
- **Convergence**: How quickly did metrics improve? At what epoch did improvement slow?
- **Overfitting signals**: Compare train loss vs val loss trends. Flag divergence.
- **Loss components**: Which losses (box, cls, dfl) are dominant?
- **Learning rate**: What schedule was used? Did it correlate with metric jumps?

### Configuration Details

List all overrides, architecture config, and key hyperparameters.

## Output

Print the report to the console.

If the user requests saving, or if the experiment has a clear ID, offer to save as `experiments/explain_<exp_id>.md`.

## Guidelines

- Keep it practical — no AI hype, no buzzwords
- Use "the model" not "the AI" or "the neural network"
- Percentages and fractions over decimal scores: "85 out of 100" not "0.85"
- Frame struggles as opportunities: "needs more examples" not "fails at"
- When in doubt, explain less and recommend more
- Always ground quality judgments in the metrics-guide.md thresholds — don't invent your own scale
- If data is missing (no results.csv, no per-class AP), say so plainly and skip that section rather than guessing
