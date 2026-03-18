"""
Experiment Runner
=================

Core experiment loop: modify training config -> train -> evaluate -> log.
Designed for both programmatic use and CLI invocation.

The runner does NOT decide what to experiment with — that's Claude's job
(via the /experiment skill) or the user's job (via CLI). The runner
just executes a single experiment and returns structured results.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from yolocc.experiment.tracker import ExperimentResult, ExperimentTracker, safe_experiment_dir_name
from yolocc.experiment.strategies import generate_experiments, list_strategies
from yolocc.paths import resolve_workspace_path, get_experiments_root
from yolocc.project import load_project_config, get_default, warn_no_config
from yolocc.training.utils import get_device, prepare_ultralytics_data_yaml


class ExperimentRunner:
    """Execute training experiments and track results."""

    def __init__(
        self,
        project_config=None,
        experiments_dir: Optional[Path] = None,
    ):
        self.config = project_config or load_project_config()
        self.tracker = ExperimentTracker(experiments_dir=experiments_dir)

    def run_experiment(
        self,
        overrides: dict[str, Any],
        name: Optional[str] = None,
        budget_epochs: int = 50,
        patience: int = 10,
        dataset: Optional[str] = None,
        base_model: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> ExperimentResult:
        """
        Run a single training experiment with the given overrides.

        Args:
            overrides: Training parameter overrides (e.g., {"lr0": 0.005})
            name: Human-readable experiment name
            budget_epochs: Maximum epochs for this experiment
            patience: Early stopping patience
            dataset: Dataset path (default: from project config)
            base_model: Base model path (default: from project config)
            resume_from: Checkpoint path to resume/fork from (overrides base_model for weights)

        Returns:
            ExperimentResult with metrics and metadata
        """
        from ultralytics import YOLO

        exp_id = self.tracker.next_experiment_id()
        if name is None:
            name = "_".join(f"{k}={v}" for k, v in overrides.items()) or "default"

        # Resolve paths
        if dataset is None:
            dataset = get_default("dataset", config=self.config, fallback="datasets")
        dataset_path = resolve_workspace_path(dataset)
        data_yaml = dataset_path / "data.yaml"

        if base_model is None:
            base_model = get_default("base_model", config=self.config, fallback="yolo11n.pt")

        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset not found at {data_yaml}")

        # Build training params
        default_imgsz = get_default("imgsz", config=self.config, fallback=640)
        training_params = {
            "epochs": budget_epochs,
            "patience": patience,
            "imgsz": default_imgsz,
            "batch": -1,
            "device": get_device(),
            "verbose": False,
            "plots": True,
            "save": True,
            "exist_ok": True,
        }
        training_params.update(overrides)

        # Extract architecture config from overrides (model=*.yaml means arch swap)
        # Note: is_baseline is True only when overrides is empty. An architecture-only
        # override (model=...yaml) stays in overrides dict, so is_baseline=False.
        # This is correct — architecture changes are not baselines.
        architecture_config = None
        model_override = training_params.pop("model", None)
        if model_override and str(model_override).endswith((".yaml", ".yml")):
            architecture_config = str(model_override)
        elif model_override:
            base_model = str(model_override)  # treat as .pt path

        # Ensure epochs/patience aren't overridden accidentally
        if "epochs" not in overrides:
            training_params["epochs"] = budget_epochs
        if "patience" not in overrides:
            training_params["patience"] = patience

        # Output directory (use same sanitization as tracker for consistency)
        exp_output = get_experiments_root() / safe_experiment_dir_name(exp_id, name)

        print(f"\n{'='*60}")
        print(f"Experiment {exp_id}: {name}")
        print(f"{'='*60}")
        print(f"Overrides: {overrides}")
        if architecture_config:
            print(f"Architecture: {architecture_config}")
        if resume_from:
            print(f"Resume from: {resume_from}")
        print(f"Budget: {budget_epochs} epochs, patience={patience}")

        # Train — architecture swap uses YOLO(config.yaml).load(weights.pt)
        weights = resume_from or base_model
        if architecture_config:
            model = YOLO(str(resolve_workspace_path(architecture_config))).load(
                str(resolve_workspace_path(weights))
            )
        else:
            model = YOLO(str(resolve_workspace_path(weights)))

        # Remove keys that conflict with explicit train() args
        params = {k: v for k, v in training_params.items()
                  if k not in ("data", "project", "name", "model")}

        # Normalize data.yaml paths for Ultralytics
        ultra_data_yaml, temp_data_yaml = prepare_ultralytics_data_yaml(data_yaml, dataset_path)

        try:
            results = model.train(
                data=ultra_data_yaml,
                project=str(exp_output),
                name="train",
                **params,
            )
        except Exception as e:
            print(f"ERROR: Training failed for experiment {exp_id}: {e}")
            result = ExperimentResult(
                experiment_id=exp_id,
                name=name,
                overrides=overrides,
                metrics={"mAP50": 0, "mAP50-95": 0, "precision": 0, "recall": 0},
                per_class_ap={},
                epochs_run=0,
                epochs_max=budget_epochs,
                is_baseline=isinstance(overrides, dict) and len(overrides) == 0,
                model_path=None,
                architecture_config=architecture_config,
                resume_from=resume_from,
            )
            self.tracker.log(result)
            self.tracker.save_summary()
            return result
        finally:
            if temp_data_yaml is not None:
                temp_data_yaml.unlink(missing_ok=True)

        # Extract metrics
        metrics_dict = results.results_dict if hasattr(results, "results_dict") else {}
        metrics = {
            "mAP50": round(metrics_dict.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(metrics_dict.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(metrics_dict.get("metrics/precision(B)", 0), 4),
            "recall": round(metrics_dict.get("metrics/recall(B)", 0), 4),
        }

        # Per-class AP (from model validation)
        # results.maps is indexed positionally by sorted class ID order
        per_class_ap = {}
        if hasattr(results, "maps") and self.config and self.config.classes:
            try:
                sorted_class_ids = sorted(self.config.classes.keys())
                for i, ap in enumerate(results.maps):
                    if i < len(sorted_class_ids):
                        class_name = self.config.classes[sorted_class_ids[i]]
                        per_class_ap[class_name] = round(float(ap), 4)
            except (TypeError, ValueError, IndexError) as e:
                print(f"WARNING: Could not extract per-class AP: {e}")

        # How many epochs actually ran (early stopping may have cut it short)
        epochs_run = int(metrics_dict.get("epoch", budget_epochs))

        # Best model path
        best_pt = exp_output / "train" / "weights" / "best.pt"
        model_path = str(best_pt) if best_pt.exists() else None

        result = ExperimentResult(
            experiment_id=exp_id,
            name=name,
            overrides=overrides,
            metrics=metrics,
            per_class_ap=per_class_ap,
            epochs_run=epochs_run,
            epochs_max=budget_epochs,
            is_baseline=isinstance(overrides, dict) and len(overrides) == 0,
            model_path=model_path,
            architecture_config=architecture_config,
            resume_from=resume_from,
        )

        # Log and generate report
        self.tracker.log(result)
        baseline = self.tracker.get_baseline()
        self.tracker.save_report(result, baseline)
        self.tracker.save_summary()

        print(f"\nResult: mAP50-95 = {result.primary_metric:.4f}")
        if baseline and not result.is_baseline:
            delta = result.primary_metric - baseline.primary_metric
            verdict = "IMPROVED" if delta > 0 else "NO IMPROVEMENT"
            print(f"vs baseline: {delta:+.4f} ({verdict})")

        return result

    def run_baseline(self, **kwargs) -> ExperimentResult:
        """Run a baseline experiment with default settings."""
        return self.run_experiment(overrides={}, name="baseline", **kwargs)

    def compare(
        self, a: ExperimentResult, b: ExperimentResult
    ) -> dict[str, float]:
        """Compare two experiments, return deltas."""
        return a.delta(b)

    def get_baseline(self) -> Optional[ExperimentResult]:
        """Get the current baseline result."""
        return self.tracker.get_baseline()

    def run_tune(
        self,
        search_space: dict[str, tuple],
        iterations: int = 20,
        epochs_per_iter: int = 10,
        patience: int = 5,
        name: Optional[str] = None,
        dataset: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> ExperimentResult:
        """
        Run HP optimization via Ultralytics model.tune().

        The agent decides WHAT to tune and within what bounds,
        model.tune() handles the mechanical genetic-algorithm search.

        Args:
            search_space: Parameter bounds as {name: (min, max)} tuples.
                Empty dict uses Ultralytics default (~25 params).
            iterations: Number of tune iterations (each trains for epochs_per_iter)
            epochs_per_iter: Epochs per tune iteration
            patience: Early stopping patience per iteration
            name: Human-readable name
            dataset: Dataset path (default: from project config)
            base_model: Base model path (default: from project config)

        Returns:
            ExperimentResult with best config found by tune
        """
        from ultralytics import YOLO

        exp_id = self.tracker.next_experiment_id()
        if name is None:
            space_keys = "_".join(search_space.keys()) if search_space else "all"
            name = f"tune_{space_keys}_{iterations}iter"

        # Resolve paths (same as run_experiment)
        if dataset is None:
            dataset = get_default("dataset", config=self.config, fallback="datasets")
        dataset_path = resolve_workspace_path(dataset)
        data_yaml = dataset_path / "data.yaml"

        if base_model is None:
            base_model = get_default("base_model", config=self.config, fallback="yolo11n.pt")

        if not data_yaml.exists():
            raise FileNotFoundError(f"Dataset not found at {data_yaml}")

        exp_output = get_experiments_root() / safe_experiment_dir_name(exp_id, name)
        exp_output.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Tune {exp_id}: {name}")
        print(f"{'='*60}")
        print(f"Search space: {search_space or 'Ultralytics defaults'}")
        print(f"Iterations: {iterations}, epochs/iter: {epochs_per_iter}, patience: {patience}")

        model = YOLO(str(resolve_workspace_path(base_model)))

        # Normalize data.yaml paths for Ultralytics
        ultra_data_yaml, temp_data_yaml = prepare_ultralytics_data_yaml(data_yaml, dataset_path)

        tune_kwargs = {
            "data": ultra_data_yaml,
            "iterations": iterations,
            "epochs": epochs_per_iter,
            "patience": patience,
            "device": get_device(),
            "plots": True,
            "verbose": False,
        }
        if search_space:
            tune_kwargs["space"] = search_space

        try:
            model.tune(**tune_kwargs)
        except Exception as e:
            print(f"ERROR: Tune failed for {exp_id}: {e}")
            result = ExperimentResult(
                experiment_id=exp_id,
                name=name,
                overrides={},
                metrics={"mAP50": 0, "mAP50-95": 0, "precision": 0, "recall": 0},
                per_class_ap={},
                epochs_run=0,
                epochs_max=iterations * epochs_per_iter,
                is_tune=True,
                tune_iterations=iterations,
                tune_search_space={k: list(v) for k, v in search_space.items()},
            )
            self.tracker.log(result)
            self.tracker.save_summary()
            return result
        finally:
            if temp_data_yaml is not None:
                temp_data_yaml.unlink(missing_ok=True)

        # Discover tune output directory
        tune_dir = self._find_tune_dir(exp_output)
        print(f"Tune output: {tune_dir}")

        # Copy tune outputs into experiment directory if not already there
        target_tune_dir = exp_output / "tune"
        if tune_dir != target_tune_dir:
            if target_tune_dir.exists():
                shutil.rmtree(target_tune_dir)
            shutil.copytree(str(tune_dir), str(target_tune_dir))
            tune_dir = target_tune_dir

        # Read best hyperparameters
        best_hp_path = tune_dir / "best_hyperparameters.yaml"
        best_hp = {}
        if best_hp_path.exists():
            with open(best_hp_path) as f:
                best_hp = yaml.safe_load(f) or {}

        # Load best.pt FRESH and run validation for metrics + per-class AP
        best_pt = tune_dir / "weights" / "best.pt"
        metrics = {"mAP50": 0, "mAP50-95": 0, "precision": 0, "recall": 0}
        per_class_ap = {}
        model_path = None

        if best_pt.exists():
            model_path = str(best_pt)
            try:
                val_model = YOLO(str(best_pt))
                val_results = val_model.val(
                    data=ultra_data_yaml,
                    device=get_device(),
                    verbose=False,
                )
                val_dict = val_results.results_dict if hasattr(val_results, "results_dict") else {}
                metrics = {
                    "mAP50": round(val_dict.get("metrics/mAP50(B)", 0), 4),
                    "mAP50-95": round(val_dict.get("metrics/mAP50-95(B)", 0), 4),
                    "precision": round(val_dict.get("metrics/precision(B)", 0), 4),
                    "recall": round(val_dict.get("metrics/recall(B)", 0), 4),
                }

                if hasattr(val_results, "maps") and self.config and self.config.classes:
                    try:
                        sorted_class_ids = sorted(self.config.classes.keys())
                        for i, ap in enumerate(val_results.maps):
                            if i < len(sorted_class_ids):
                                class_name = self.config.classes[sorted_class_ids[i]]
                                per_class_ap[class_name] = round(float(ap), 4)
                    except (TypeError, ValueError, IndexError) as e:
                        print(f"WARNING: Could not extract per-class AP: {e}")
            except Exception as e:
                print(f"WARNING: Validation after tune failed: {e}")

        result = ExperimentResult(
            experiment_id=exp_id,
            name=name,
            overrides=best_hp,
            metrics=metrics,
            per_class_ap=per_class_ap,
            epochs_run=iterations * epochs_per_iter,
            epochs_max=iterations * epochs_per_iter,
            model_path=model_path,
            is_tune=True,
            tune_iterations=iterations,
            tune_search_space={k: list(v) for k, v in search_space.items()},
            tune_best_params=best_hp,
            tune_dir=str(tune_dir),
        )

        self.tracker.log(result)
        baseline = self.tracker.get_baseline()
        self.tracker.save_report(result, baseline)
        self.tracker.save_summary()

        print(f"\nTune result: mAP50-95 = {result.primary_metric:.4f}")
        if baseline:
            delta = result.primary_metric - baseline.primary_metric
            verdict = "IMPROVED" if delta > 0 else "NO IMPROVEMENT"
            print(f"vs baseline: {delta:+.4f} ({verdict})")
        if best_hp:
            print(f"Best params: {best_hp}")

        return result

    @staticmethod
    def _find_tune_dir(exp_output: Path) -> Path:
        """Discover model.tune() output — handles Ultralytics routing quirks."""
        # Check experiment dir first
        for candidate in sorted(exp_output.rglob("best_hyperparameters.yaml")):
            return candidate.parent
        # Fallback: Ultralytics default runs/detect/tune*
        for runs_dir in [Path("runs/detect"), Path("runs")]:
            if runs_dir.exists():
                for tune_dir in sorted(
                    runs_dir.glob("tune*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    if (tune_dir / "best_hyperparameters.yaml").exists():
                        return tune_dir
        raise FileNotFoundError("Could not find model.tune() output directory")


# --- Tune presets: map agent diagnosis to focused search spaces ---

TUNE_PRESETS: dict[str, dict[str, tuple]] = {
    "lr": {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "warmup_epochs": (0.0, 5.0),
        "warmup_momentum": (0.5, 0.95),
    },
    "augmentation": {
        "mosaic": (0.0, 1.0),
        "mixup": (0.0, 0.3),
        "erasing": (0.0, 0.6),
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),
        "degrees": (0.0, 45.0),
        "scale": (0.0, 0.9),
    },
    "loss": {
        "box": (0.02, 0.2),
        "cls": (0.2, 4.0),
        "dfl": (0.5, 2.0),
    },
    "optimizer": {
        "lr0": (1e-5, 1e-1),
        "momentum": (0.6, 0.98),
        "weight_decay": (0.0, 0.001),
    },
    "all": {},  # empty = Ultralytics default (~25 params)
}


def resolve_search_space(space_arg: Optional[str]) -> dict[str, tuple]:
    """Resolve a --space argument to a search space dict.

    Accepts:
        - Preset name: "lr", "augmentation", "loss", "optimizer", "all"
        - Custom format: "lr0=0.001:0.01 momentum=0.8:0.98"
        - None: defaults to "all" (Ultralytics defaults)
    """
    if space_arg is None or space_arg == "all":
        return {}

    if space_arg in TUNE_PRESETS:
        return TUNE_PRESETS[space_arg]

    # Parse custom key=min:max format
    space = {}
    for part in space_arg.split():
        if "=" not in part or ":" not in part:
            raise ValueError(
                f"Invalid search space format '{part}'. "
                f"Expected key=min:max or a preset name ({', '.join(TUNE_PRESETS.keys())})"
            )
        key, bounds = part.split("=", 1)
        lo, hi = bounds.split(":", 1)
        space[key] = (float(lo), float(hi))
    return space


def experiment_cli() -> None:
    """CLI entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available strategies: {', '.join(list_strategies())}

Examples:
  yolo-experiment run --override "lr0=0.005"
  yolo-experiment run --strategy learning_rate --budget 10
  yolo-experiment baseline --budget 5
  yolo-experiment summary
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run a single experiment")
    run_parser.add_argument("--override", type=str, help='Overrides as "key=value key2=value2"')
    run_parser.add_argument("--strategy", type=str, help="Use a built-in strategy")
    run_parser.add_argument("--budget", type=int, default=50, help="Max epochs (default: 50)")
    run_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    run_parser.add_argument("--dataset", type=str, help="Dataset path")
    run_parser.add_argument("--model", type=str, help="Base model path")
    run_parser.add_argument("--resume-from", type=str, help="Checkpoint to resume/fork from")
    run_parser.add_argument("--time", type=float, default=None, help="Max training time in hours")

    # 'baseline' subcommand
    base_parser = subparsers.add_parser("baseline", help="Run baseline experiment")
    base_parser.add_argument("--budget", type=int, default=50, help="Max epochs")
    base_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    base_parser.add_argument("--dataset", type=str, help="Dataset path")
    base_parser.add_argument("--model", type=str, help="Base model path")
    base_parser.add_argument("--resume-from", type=str, help="Checkpoint to resume/fork from")
    base_parser.add_argument("--time", type=float, default=None, help="Max training time in hours")

    # 'tune' subcommand
    tune_parser = subparsers.add_parser(
        "tune",
        help="Run HP optimization via model.tune()",
    )
    tune_parser.add_argument(
        "--space", type=str, default=None,
        help='Preset ("lr", "augmentation", "loss", "optimizer", "all") or custom "lr0=0.001:0.01 momentum=0.8:0.98"',
    )
    tune_parser.add_argument("--iterations", type=int, default=20, help="Tune iterations (default: 20)")
    tune_parser.add_argument("--epochs", type=int, default=10, help="Epochs per iteration (default: 10)")
    tune_parser.add_argument("--patience", type=int, default=5, help="Patience per iteration (default: 5)")
    tune_parser.add_argument("--dataset", type=str, help="Dataset path")
    tune_parser.add_argument("--model", type=str, help="Base model path")
    tune_parser.add_argument("--name", type=str, default=None, help="Experiment name")

    # 'summary' subcommand
    subparsers.add_parser("summary", help="Print experiment dashboard")

    # 'strategies' subcommand
    subparsers.add_parser("strategies", help="List available strategies")

    args = parser.parse_args()

    if args.command == "strategies":
        from yolocc.experiment.strategies import BUILT_IN_STRATEGIES
        for name, info in BUILT_IN_STRATEGIES.items():
            print(f"  {name:20s} {info['description']}")
        return

    if args.command == "summary":
        tracker = ExperimentTracker()
        print(tracker.generate_summary())
        return

    _cfg = load_project_config()
    if _cfg is None:
        warn_no_config()
    runner = ExperimentRunner(project_config=_cfg)

    if args.command == "baseline":
        runner.run_baseline(
            budget_epochs=args.budget,
            patience=args.patience,
            dataset=args.dataset,
            base_model=args.model,
            resume_from=args.resume_from,
        )
        return

    if args.command == "tune":
        try:
            search_space = resolve_search_space(args.space)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        runner.run_tune(
            search_space=search_space,
            iterations=args.iterations,
            epochs_per_iter=args.epochs,
            patience=args.patience,
            name=args.name,
            dataset=args.dataset,
            base_model=args.model,
        )
        return

    if args.command == "run":
        # Inject time limit into overrides if specified
        time_limit = getattr(args, "time", None)

        if args.strategy:
            experiments = generate_experiments(args.strategy)
            if not experiments:
                print(f"Unknown strategy: {args.strategy}")
                print(f"Available: {', '.join(list_strategies())}")
                sys.exit(1)
            total_epochs = len(experiments) * args.budget
            if total_epochs > 100:
                print(f"WARNING: Strategy '{args.strategy}' generates {len(experiments)} experiments "
                      f"x {args.budget} epochs = ~{total_epochs} total epochs")
            for exp in experiments:
                if time_limit is not None:
                    exp["overrides"]["time"] = time_limit
                runner.run_experiment(
                    overrides=exp["overrides"],
                    name=exp["name"],
                    budget_epochs=args.budget,
                    patience=args.patience,
                    dataset=args.dataset,
                    base_model=args.model,
                    resume_from=args.resume_from,
                )
        elif args.override:
            overrides = {}
            for part in args.override.split():
                if "=" not in part:
                    print(f"ERROR: Invalid override format '{part}'. Expected key=value")
                    sys.exit(1)
                k, v = part.split("=", 1)
                # Try to parse as number
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                overrides[k] = v
            if time_limit is not None:
                overrides["time"] = time_limit
            runner.run_experiment(
                overrides=overrides,
                budget_epochs=args.budget,
                patience=args.patience,
                dataset=args.dataset,
                base_model=args.model,
                resume_from=args.resume_from,
            )
        else:
            print("Provide --override or --strategy")
            sys.exit(1)


__all__ = [
    "ExperimentRunner",
    "TUNE_PRESETS",
    "resolve_search_space",
    "experiment_cli",
]
