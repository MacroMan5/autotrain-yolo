"""
Experiment Tracker
==================

Logs experiments and generates AI-readable markdown reports.
Every experiment produces structured files in the experiments/ directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run."""

    experiment_id: str
    name: str
    overrides: dict[str, Any]
    metrics: dict[str, float]
    per_class_ap: dict[str, float] = field(default_factory=dict)
    epochs_run: int = 0
    epochs_max: int = 0
    is_baseline: bool = False
    model_path: Optional[str] = None
    architecture_config: Optional[str] = None
    resume_from: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_tune: bool = False
    tune_iterations: int = 0
    tune_search_space: dict[str, Any] = field(default_factory=dict)
    tune_best_params: dict[str, Any] = field(default_factory=dict)
    tune_dir: Optional[str] = None

    @property
    def primary_metric(self) -> float:
        """mAP50-95 is the primary optimization target."""
        return self.metrics.get("mAP50-95", 0.0)

    def improved_over(self, other: ExperimentResult) -> bool:
        return self.primary_metric > other.primary_metric

    def delta(self, baseline: ExperimentResult) -> dict[str, float]:
        return {
            k: round(self.metrics.get(k, 0) - baseline.metrics.get(k, 0), 4)
            for k in baseline.metrics
        }


def safe_experiment_dir_name(exp_id: str, name: str) -> str:
    """Compute a filesystem-safe directory name for an experiment."""
    safe_name = re.sub(r"[^\w\-.]", "_", name)[:50]
    return f"{exp_id}_{safe_name}" if safe_name else exp_id


class ExperimentTracker:
    """Log experiments and generate reports to experiments/ directory."""

    def __init__(self, experiments_dir: Optional[Path] = None):
        if experiments_dir is None:
            from yolocc.paths import get_experiments_root
            experiments_dir = get_experiments_root()
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[ExperimentResult] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load previously logged experiments from disk."""
        for metrics_file in sorted(self.experiments_dir.glob("exp_*/metrics.yaml")):
            try:
                with open(metrics_file) as f:
                    data = yaml.safe_load(f) or {}
                result = ExperimentResult(
                    experiment_id=data.get("experiment_id", metrics_file.parent.name),
                    name=data.get("name", ""),
                    overrides=data.get("overrides", {}),
                    metrics=data.get("metrics", {}),
                    per_class_ap=data.get("per_class_ap", {}),
                    epochs_run=data.get("epochs_run", 0),
                    epochs_max=data.get("epochs_max", 0),
                    is_baseline=data.get("is_baseline", False),
                    model_path=data.get("model_path"),
                    architecture_config=data.get("architecture_config"),
                    resume_from=data.get("resume_from"),
                    timestamp=data.get("timestamp", ""),
                    is_tune=data.get("is_tune", False),
                    tune_iterations=data.get("tune_iterations", 0),
                    tune_search_space=data.get("tune_search_space", {}),
                    tune_best_params=data.get("tune_best_params", {}),
                    tune_dir=data.get("tune_dir"),
                )
                self._results.append(result)
            except Exception as e:
                print(f"WARNING: Could not load experiment from {metrics_file}: {e}")
                continue

    def next_experiment_id(self) -> str:
        """Return the next experiment ID (exp_000, exp_001, ...)."""
        nums = []
        # Check in-memory results
        for r in self._results:
            match = re.match(r"exp_(\d+)", r.experiment_id)
            if match:
                nums.append(int(match.group(1)))
        # Also scan filesystem directories (handles corrupted/incomplete experiments)
        for d in self.experiments_dir.glob("exp_*"):
            if d.is_dir():
                match = re.match(r"exp_(\d+)", d.name)
                if match:
                    nums.append(int(match.group(1)))
        next_num = max(nums) + 1 if nums else 0
        return f"exp_{next_num:03d}"

    def _experiment_dir_name(self, result: ExperimentResult) -> str:
        """Compute a filesystem-safe directory name for an experiment."""
        return safe_experiment_dir_name(result.experiment_id, result.name)

    def log(self, result: ExperimentResult) -> Path:
        """Log an experiment result to disk."""
        dir_name = self._experiment_dir_name(result)
        exp_dir = self.experiments_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_data = {"overrides": result.overrides, "epochs_max": result.epochs_max}
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)

        # Save metrics
        metrics_data = {
            "experiment_id": result.experiment_id,
            "name": result.name,
            "overrides": result.overrides,
            "metrics": result.metrics,
            "per_class_ap": result.per_class_ap,
            "epochs_run": result.epochs_run,
            "epochs_max": result.epochs_max,
            "is_baseline": result.is_baseline,
            "model_path": result.model_path,
            "architecture_config": result.architecture_config,
            "resume_from": result.resume_from,
            "timestamp": result.timestamp,
            "is_tune": result.is_tune,
            "tune_iterations": result.tune_iterations,
            "tune_search_space": result.tune_search_space,
            "tune_best_params": result.tune_best_params,
            "tune_dir": result.tune_dir,
        }
        with open(exp_dir / "metrics.yaml", "w") as f:
            yaml.safe_dump(metrics_data, f, sort_keys=False)

        self._results.append(result)
        return exp_dir

    def get_best(self) -> Optional[ExperimentResult]:
        """Return the experiment with the highest mAP50-95."""
        if not self._results:
            return None
        return max(self._results, key=lambda r: r.primary_metric)

    def get_baseline(self) -> Optional[ExperimentResult]:
        """Return the most recent baseline experiment."""
        for r in reversed(self._results):
            if r.is_baseline:
                return r
        return None

    def get_history(self) -> list[ExperimentResult]:
        """Return all logged experiments in order."""
        return list(self._results)

    def generate_report(
        self,
        result: ExperimentResult,
        baseline: Optional[ExperimentResult] = None,
    ) -> str:
        """Generate a single experiment report in markdown."""
        if baseline is None:
            baseline = self.get_baseline()

        lines = [f"# Experiment {result.experiment_id}: {result.name}", ""]

        if result.architecture_config:
            lines.append(f"Architecture: `{result.architecture_config}`")
        if result.resume_from:
            lines.append(f"Resumed from: `{result.resume_from}`")
        if result.architecture_config or result.resume_from:
            lines.append("")

        if result.is_tune:
            lines.append(f"## Tune Summary ({result.tune_iterations} iterations)")
            if result.tune_search_space:
                space_str = ", ".join(
                    f"{k}: {v}" for k, v in result.tune_search_space.items()
                )
                lines.append(f"Search space: {space_str}")
            if result.tune_best_params:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in result.tune_best_params.items()
                )
                lines.append(f"Best params: {params_str}")
            lines.append("")

        if baseline and not result.is_baseline:
            lines.append("## Config Changes (vs baseline)")
            for k, v in result.overrides.items():
                base_v = baseline.overrides.get(k, "default")
                lines.append(f"- {k}: {base_v} -> {v}")
            lines.append("")

            lines.append("## Results")
            lines.append("| Metric | Baseline | This Run | Delta |")
            lines.append("|--------|----------|----------|-------|")
            delta = result.delta(baseline)
            for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
                bv = baseline.metrics.get(metric, 0)
                rv = result.metrics.get(metric, 0)
                dv = delta.get(metric, 0)
                sign = "+" if dv >= 0 else ""
                lines.append(f"| {metric} | {bv:.4f} | {rv:.4f} | {sign}{dv:.4f} |")
            lines.append("")

            if result.per_class_ap and baseline.per_class_ap:
                lines.append("## Per-Class Performance")
                lines.append("| Class | mAP50-95 (baseline) | mAP50-95 (this) | Delta |")
                lines.append("|-------|-----------------|-------------|-------|")
                for cls in result.per_class_ap:
                    bv = baseline.per_class_ap.get(cls, 0)
                    rv = result.per_class_ap[cls]
                    dv = rv - bv
                    sign = "+" if dv >= 0 else ""
                    lines.append(f"| {cls} | {bv:.4f} | {rv:.4f} | {sign}{dv:.4f} |")
                lines.append("")

            improved = result.improved_over(baseline)
            dm = delta.get("mAP50-95", 0)
            sign = "+" if dm >= 0 else ""
            verdict = "IMPROVED" if improved else "NO IMPROVEMENT"
            lines.append(f"## Verdict: {verdict} ({sign}{dm:.4f} mAP50-95)")
        else:
            lines.append("## Baseline Results")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in result.metrics.items():
                lines.append(f"| {metric} | {value:.4f} |")

        lines.append("")
        lines.append(f"Epochs: {result.epochs_run}/{result.epochs_max}")
        return "\n".join(lines)

    def generate_summary(self) -> str:
        """Generate a dashboard comparing all experiments."""
        lines = ["# Experiment Dashboard", ""]

        if not self._results:
            lines.append("No experiments logged yet.")
            return "\n".join(lines)

        best = self.get_best()
        baseline = self.get_baseline()

        if best:
            lines.append("## Best Configuration")
            overrides_str = ", ".join(f"{k}={v}" for k, v in best.overrides.items())
            lines.append(
                f"{overrides_str or 'baseline'} -> mAP50-95: {best.primary_metric:.4f}"
            )
            lines.append("")

        lines.append("## All Experiments")
        lines.append("| # | Change | mAP50-95 | vs Baseline | Status |")
        lines.append("|---|--------|----------|-------------|--------|")

        for r in self._results:
            change = f"tune({r.tune_iterations})" if r.is_tune else (r.name or "baseline")
            m = r.primary_metric
            if baseline and not r.is_baseline:
                delta = r.primary_metric - baseline.primary_metric
                sign = "+" if delta >= 0 else ""
                vs = f"{sign}{delta:.4f}"
            else:
                vs = "—"

            if r.is_baseline:
                status = "baseline"
            elif best and r.experiment_id == best.experiment_id:
                status = "best"
            elif baseline and r.improved_over(baseline):
                status = "improved"
            else:
                status = "rejected"

            lines.append(f"| {r.experiment_id} | {change} | {m:.4f} | {vs} | {status} |")

        return "\n".join(lines)

    def generate_session_report(
        self, session_results: list[ExperimentResult]
    ) -> str:
        """Generate a start-to-end session comparison."""
        if not session_results:
            return "# Session Report\n\nNo experiments in this session."

        date = datetime.now().strftime("%Y-%m-%d")
        best_session = max(session_results, key=lambda r: r.primary_metric)
        baseline = self.get_baseline()

        lines = [f"# Session Report — {date}", ""]
        lines.append("## Overview")
        lines.append(f"- Experiments: {len(session_results)}")

        if baseline:
            start = baseline.primary_metric
            end = best_session.primary_metric
            improvement = end - start
            pct = (improvement / start * 100) if start > 0 else 0
            lines.append(f"- Starting mAP50-95: {start:.4f}")
            lines.append(f"- Ending mAP50-95: {end:.4f}")
            lines.append(f"- Total improvement: {improvement:+.4f} ({pct:+.1f}%)")

        lines.append("")
        lines.append("## What Worked")
        worked = [r for r in session_results if baseline and r.improved_over(baseline)]
        for i, r in enumerate(worked, 1):
            delta = r.primary_metric - (baseline.primary_metric if baseline else 0)
            lines.append(f"{i}. {r.name}: {delta:+.4f}")

        lines.append("")
        lines.append("## What Didn't Work")
        didnt = [
            r
            for r in session_results
            if baseline and not r.is_baseline and not r.improved_over(baseline)
        ]
        for i, r in enumerate(didnt, 1):
            delta = r.primary_metric - (baseline.primary_metric if baseline else 0)
            lines.append(f"{i}. {r.name}: {delta:+.4f}")

        if best_session.model_path:
            lines.append("")
            lines.append("## Best Model")
            lines.append(f"Path: {best_session.model_path}")

        return "\n".join(lines)

    def save_summary(self) -> Path:
        """Write summary.md to experiments directory."""
        path = self.experiments_dir / "summary.md"
        path.write_text(self.generate_summary(), encoding="utf-8")
        return path

    def save_report(self, result: ExperimentResult, baseline: Optional[ExperimentResult] = None) -> Path:
        """Write individual experiment report.md."""
        dir_name = self._experiment_dir_name(result)
        exp_dir = self.experiments_dir / dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        report_path = exp_dir / "report.md"
        report_path.write_text(self.generate_report(result, baseline), encoding="utf-8")
        return report_path

    @property
    def experiment_count(self) -> int:
        """Number of logged experiments."""
        return len(self._results)

__all__ = [
    "ExperimentResult",
    "ExperimentTracker",
    "safe_experiment_dir_name",
]
