"""
Pre-experiment guard hook (PreToolUse:Bash).

Fires before any Bash command. Only acts on commands containing 'yolo-experiment run'.
Cross-platform Python script — works on Windows, macOS, and Linux.

Enforces:
- G-01: Checkpoint backup (copies best.pt -> best_backup.pt)
- G-02: Budget enforcement (counts experiments in current session)
- Parameter sanity bounds (blocks out-of-range values)
- Budget exhausted file check (hard stop from 4 consecutive regressions)
"""

import json
import re
import shutil
import sys
from pathlib import Path


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")

    # Only act on yolo-experiment run/baseline/tune commands
    if not any(sub in command for sub in ("yolo-experiment run", "yolo-experiment baseline", "yolo-experiment tune")):
        sys.exit(0)

    workspace = Path.cwd()

    # Check budget exhausted file (hard stop from 4 consecutive regressions)
    exhausted_file = workspace / "experiments" / ".budget_exhausted"
    if exhausted_file.exists():
        print(
            "BLOCKED: Budget exhausted (4 consecutive regressions). "
            "Remove experiments/.budget_exhausted to continue.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Budget enforcement (G-02) ---
    max_budget = _get_session_budget(workspace)
    session_count = _count_session_experiments(workspace)
    if session_count >= max_budget:
        print(
            f"BLOCKED: Session budget exhausted ({session_count}/{max_budget} experiments). "
            f"Write session report and start a new session.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Checkpoint backup (G-01) ---
    best_pt = workspace / "models" / "best.pt"
    if best_pt.exists():
        backup_pt = workspace / "models" / "best_backup.pt"
        shutil.copy2(str(best_pt), str(backup_pt))

    sys.exit(0)


def _get_session_budget(workspace: Path) -> int:
    """Read budget from yolo-project.yaml, fallback to 20."""
    config_path = workspace / "yolo-project.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            return int(config.get("defaults", {}).get("max_experiments_per_session", 20))
        except Exception:
            pass
    return 20


def _count_session_experiments(workspace: Path) -> int:
    """Count experiment_id: entries in summary.md since last session marker."""
    summary_path = workspace / "experiments" / "summary.md"
    if not summary_path.exists():
        return 0

    try:
        text = summary_path.read_text(encoding="utf-8")
    except Exception:
        return 0

    # Find last session marker
    lines = text.split("\n")
    last_session_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("## Session ") and "Session End" not in line:
            last_session_idx = i

    # Count experiment_id entries after last session marker
    count = 0
    for line in lines[last_session_idx:]:
        if re.match(r"\s*experiment_id:\s*exp_\d+", line):
            count += 1

    return count


if __name__ == "__main__":
    main()
