# Contributing to yolocc

Thank you for your interest in yolocc!

## Ways to Contribute

### Report Issues & Share Feedback
Run the tools on your dataset and tell us what happened:
- Which skills or commands did you use?
- What worked well and what didn't?
- What was your dataset type? (medical, agriculture, security, etc.)

Open an issue with the label `feedback`.

### Report Bugs
Open an issue with:
- What you ran (command or skill)
- What you expected
- What happened
- Your environment (OS, Python version, GPU)

### Add Features
1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run: `pytest -v && ruff check src/`
5. Open a PR

### Add Experiment Strategies
Add new strategies to `src/yolocc/experiment/strategies.py`.

### Add Claude Code Skills
Follow the pattern in `.claude/skills/`. Each skill needs a `SKILL.md` with frontmatter.

## Development Setup

```bash
git clone https://github.com/MacroMan5/yolocc-toolkit.git
cd yolocc
pip install -e ".[dev]"
pytest -v
```

## Running Tests

```bash
# Unit tests only (fast, no GPU)
pytest -v --ignore=tests/test_integration

# With coverage report
pytest -v --cov=yolocc --cov-report=term-missing

# Integration tests (may download YOLO model)
pytest -v -m integration

# CVAT integration tests (requires running CVAT on localhost:8080)
pytest -v -m cvat
```

## Ecosystem

yolocc is part of a three-repo architecture:

| Repo | Role |
|---|---|
| **[yolocc](https://github.com/MacroMan5/yolocc-toolkit)** | Training pipeline, experiment engine, active learning, Claude Code skills |
| **[CVAT](https://github.com/MacroMan5/CVAT)** | Self-hosted annotation platform with Nuclio for serverless auto-annotation |
| **[dataset-converter](https://github.com/MacroMan5/dataset-converter)** | Convert YOLO datasets to CVAT/Roboflow import format |

The typical flow: annotate in CVAT, export and convert with dataset-converter if needed, train and experiment with yolocc, deploy the model back to CVAT for auto-annotation, push uncertain predictions for human review.

When contributing, keep in mind that CVAT integration features live in `src/yolocc/cvat/` and require the `[cvat]` extra to be installed.

## Code Style
- Python 3.10+ type hints
- Ruff for linting
- Tests in `tests/` mirroring `src/` structure
