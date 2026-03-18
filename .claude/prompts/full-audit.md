# Full Project Audit — yolocc

You are a senior code reviewer performing a comprehensive audit of the **yolocc** project. This is a pip-installable YOLO training pipeline with autonomous experimentation, CVAT integration, and Claude Code skills.

Your job is to find **real problems** — not nitpick style. Be specific: cite file paths, line numbers, and concrete evidence. If something is fine, don't mention it.

---

## PHASE 1: Code Quality & Bugs

For each source file in `src/yolocc/`, check:

### 1.1 — Logic Bugs
- Off-by-one errors, wrong comparisons, inverted conditions
- Variables used before assignment or after reassignment
- Exception handling that swallows errors silently (`except: pass`, bare `except Exception`)
- Race conditions or file handle leaks (unclosed files, missing `with` statements)
- Mutable default arguments (`def foo(x=[])`)
- Division by zero possibilities (especially in metrics calculations in `analyzer.py`, `tracker.py`)

### 1.2 — API Contract Violations
- Functions that return inconsistent types (sometimes dict, sometimes None)
- CLI commands that don't validate required arguments before proceeding
- Functions that read files without checking existence first
- YAML/JSON parsing without error handling

### 1.3 — Dead Code
- Functions/classes defined but never imported or called anywhere
- Imports that are unused
- Config options in `yolo-project.example.yaml` that no code actually reads
- CLI flags that are parsed but never used

### 1.4 — Hardcoded Values
- Hardcoded paths (especially Windows-specific `C:\Users\...` paths)
- Hardcoded URLs that should be configurable
- Magic numbers without explanation (especially in `strategies.py`, `analyzer.py`)

**Source files to audit:**
```
src/yolocc/__init__.py
src/yolocc/__main__.py
src/yolocc/project.py
src/yolocc/paths.py
src/yolocc/training/trainer.py
src/yolocc/training/analyzer.py
src/yolocc/training/utils.py
src/yolocc/dataset/validator.py
src/yolocc/dataset/splitter.py
src/yolocc/dataset/cleaner.py
src/yolocc/dataset/merger.py
src/yolocc/dataset/autolabel.py
src/yolocc/experiment/runner.py
src/yolocc/experiment/strategies.py
src/yolocc/experiment/tracker.py
src/yolocc/export/onnx.py
src/yolocc/cvat/cli.py
src/yolocc/cvat/client.py
src/yolocc/cvat/nuclio.py
src/yolocc/cvat/pull.py
src/yolocc/cvat/push.py
```

---

## PHASE 2: AI-Generated Code Smell

Look for patterns that indicate sloppy AI-generated code:

### 2.1 — Over-Engineering
- Unnecessary abstractions (base classes with only one subclass)
- Excessive type annotations on internal functions that add no clarity
- Wrapper functions that just call another function with no added value
- Overuse of `**kwargs` forwarding hiding what parameters are actually accepted
- Classes that should be plain functions

### 2.2 — Boilerplate & Redundancy
- Repeated code blocks across files that should be extracted (or not — check if extraction is actually warranted)
- Copy-paste patterns between similar modules (e.g., `splitter.py` vs `merger.py` vs `cleaner.py`)
- Docstrings that just restate the function name ("Validates the dataset" on `validate_dataset()`)
- Comments that describe what the code obviously does

### 2.3 — Placeholder / Stub Code
- Functions with `# TODO` or `# FIXME` that are actually called in production paths
- `NotImplementedError` raises in code paths that users can reach
- Features mentioned in README/WORKFLOW.md that don't actually exist in code
- Default parameter values that are clearly wrong placeholders (e.g., `epochs=1`)

### 2.4 — Defensive Overkill
- Validation that can never fail (checking `isinstance(x, str)` right after `x = str(...)`)
- Try/except blocks that are wider than necessary
- Redundant null checks on values that are guaranteed non-null by the caller
- Fallback logic for scenarios that can't happen

---

## PHASE 3: Test Audit

### 3.1 — Missing Test Coverage
Read each source module and its corresponding test file. Flag:
- Public functions/methods with no test at all
- CLI entry points that aren't tested (check all 11 entry points in pyproject.toml)
- Error paths that aren't tested (what happens with bad input? missing files? corrupt YAML?)
- Edge cases: empty datasets, single-image datasets, datasets with only one class

Cross-reference:
```
src/yolocc/project.py        ↔ tests/test_project.py
src/yolocc/paths.py          ↔ tests/test_paths.py
src/yolocc/training/trainer.py   ↔ tests/test_training/test_trainer.py
src/yolocc/training/analyzer.py  ↔ tests/test_training/test_analyzer.py
src/yolocc/training/utils.py     ↔ tests/test_training/test_utils.py
src/yolocc/dataset/validator.py  ↔ tests/test_dataset/test_validator.py
src/yolocc/dataset/splitter.py   ↔ tests/test_dataset/test_splitter.py
src/yolocc/dataset/cleaner.py    ↔ tests/test_dataset/test_cleaner.py
src/yolocc/dataset/merger.py     ↔ tests/test_dataset/test_merger.py
src/yolocc/dataset/autolabel.py  ↔ tests/test_dataset/test_autolabel.py
src/yolocc/experiment/runner.py      ↔ tests/test_experiment/test_runner.py
src/yolocc/experiment/strategies.py  ↔ tests/test_experiment/test_strategies.py
src/yolocc/experiment/tracker.py     ↔ tests/test_experiment/test_tracker.py
src/yolocc/export/onnx.py        ↔ tests/test_export/test_onnx.py
src/yolocc/cvat/client.py    ↔ tests/test_cvat/test_client.py
src/yolocc/cvat/nuclio.py    ↔ tests/test_cvat/test_nuclio.py
src/yolocc/cvat/pull.py      ↔ tests/test_cvat/test_pull.py
src/yolocc/cvat/push.py      ↔ tests/test_cvat/test_push.py
```

### 3.2 — Useless Tests
Flag tests that:
- Only test that a function exists or is callable (no real assertion)
- Mock everything including the thing being tested (testing the mock, not the code)
- Assert `True` or `is not None` on values that can never be False/None
- Duplicate another test with trivially different input
- Test implementation details instead of behavior (will break on any refactor)
- Have assertions that pass regardless of the code's behavior

### 3.3 — Test Quality Issues
- Tests that don't clean up temp files/directories
- Tests that depend on execution order
- Missing `conftest.py` fixtures that could reduce duplication
- Tests marked `@pytest.mark.skip` without explanation
- Flaky patterns: time-dependent, network-dependent, OS-dependent

### 3.4 — Integration Test Gaps
- Check `tests/test_integration/test_smoke.py` — does it actually exercise the full pipeline?
- Is there an end-to-end test: config → validate → train → analyze → export?
- Are CVAT integration tests properly gated behind `@pytest.mark.cvat`?

---

## PHASE 4: Documentation Audit

### 4.1 — Cross-Documentation Consistency
Compare these files against each other AND against the actual code:
- `README.md` — features, CLI commands, examples
- `WORKFLOW.md` — step-by-step workflows
- `CLAUDE.md` — architecture, skills, commands
- `CONTRIBUTING.md` — setup, testing instructions
- `program.md` — experiment template
- `yolo-project.example.yaml` — config options

Flag:
- CLI commands documented but don't exist in `pyproject.toml` (or vice versa)
- Config options documented but not read by `project.py`
- Workflow steps that reference commands with wrong flags/arguments
- Features claimed in README that aren't implemented
- Architecture diagram in CLAUDE.md that doesn't match actual directory structure
- Inconsistent naming (e.g., `yolo-cvat push` vs `yolo-cvat push --threshold`)

### 4.2 — Skill Documentation vs Code
For each of the 11 skills in `.claude/skills/`:
- Does the skill reference CLI commands that actually exist?
- Does the skill's pre-flight checklist match what the code actually requires?
- Are there skills that overlap significantly (doing the same thing)?
- Do skill resource files (experiment-checklist.md, metrics-guide.md) match current code behavior?

### 4.3 — Config File Consistency
- Does every field in `yolo-project.example.yaml` get read by `project.py`?
- Do the experiment presets in `configs/experiments/*.yaml` use valid parameter names?
- Does the example project in `examples/fps_detection/` actually work with current code?

### 4.4 — Cross-Repo Links
Check all URLs in documentation that point to:
- `https://github.com/MacroMan5/yolocc-toolkit`
- `https://github.com/MacroMan5/CVAT`
- `https://github.com/MacroMan5/dataset-converter`
- External docs (Ultralytics, CVAT, PyTorch)

Flag broken or incorrect links.

---

## PHASE 5: Architecture & Design Issues

### 5.1 — Dependency Issues
- Read `pyproject.toml` dependencies — are any pinned too tightly or too loosely?
- Are there circular imports between modules?
- Is `cvat-sdk` properly optional? Does importing main modules fail if cvat-sdk isn't installed?
- Are dev dependencies separated from runtime dependencies?

### 5.2 — Error Handling Patterns
- Do CLI commands give helpful error messages when config is missing?
- What happens when `yolo-project.yaml` doesn't exist?
- What happens when the dataset path doesn't exist?
- Do CVAT commands fail gracefully when CVAT is unreachable?

### 5.3 — Security
- Are there any command injection risks (user input passed to `subprocess` or `os.system`)?
- Are file paths sanitized (path traversal)?
- Are CVAT tokens logged or printed anywhere?
- Any use of `eval()` or `exec()`?

---

## PHASE 6: CI/CD Audit

Read `.github/workflows/test.yml`:
- Does CI run the full test suite?
- Are CVAT tests excluded in CI (they need a running CVAT instance)?
- Is linting (ruff) part of CI?
- Does CI test on multiple Python versions?
- Are there any CI steps that could fail silently?

---

## OUTPUT FORMAT

Organize findings by severity:

### CRITICAL — Bugs or issues that will cause failures
```
[CRITICAL] file.py:123 — Description of the bug
  Evidence: <code snippet or reasoning>
  Fix: <suggested fix>
```

### HIGH — Issues that affect reliability or correctness
```
[HIGH] file.py:45 — Description
  Evidence: ...
  Fix: ...
```

### MEDIUM — Code quality, maintainability, or documentation issues
```
[MEDIUM] file.py:78 — Description
  Evidence: ...
  Fix: ...
```

### LOW — Minor improvements, style issues
```
[LOW] file.py:99 — Description
```

### USELESS TESTS — Tests to delete or rewrite
```
[USELESS] test_file.py:TestClass::test_name — Why it's useless
  Suggestion: Delete / Rewrite to test X instead
```

### MISSING TESTS — Tests that should exist
```
[MISSING] module.py::function_name — What should be tested
  Priority: High/Medium/Low
```

### DOCUMENTATION INCONSISTENCIES
```
[DOC] file1.md vs file2.md — What's inconsistent
  file1 says: "..."
  file2 says: "..."
  Reality (code): "..."
```

---

## RULES

1. **Read the actual code** — don't guess based on file names
2. **Cite line numbers** — every finding must reference a specific location
3. **No false positives** — if you're not sure, investigate before reporting
4. **Prioritize actionable findings** — skip cosmetic issues unless they indicate deeper problems
5. **Check both directions** — code without docs AND docs without code
6. **Run `pytest --co -q` first** to see what tests actually exist and are collected
7. **Run `ruff check src/`** to catch any linting issues the author missed
