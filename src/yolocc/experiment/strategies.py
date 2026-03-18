"""
Experiment Strategies
=====================

Built-in hyperparameter strategies for autonomous experimentation.
Each strategy defines a set of values to sweep. Claude can use these
as starting points and combine or modify them based on results.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Optional

BUILT_IN_STRATEGIES: dict[str, dict[str, Any]] = {
    "learning_rate": {
        "description": "Sweep learning rate from conservative to aggressive",
        "param": "lr0",
        "values": [0.0005, 0.001, 0.005, 0.01, 0.02],
    },
    "optimizer": {
        "description": "Compare optimizers",
        "param": "optimizer",
        "values": ["SGD", "Adam", "AdamW"],
    },
    "augmentation": {
        "description": "Sweep augmentation strengths",
        "params": {
            "mosaic": [0.0, 0.5, 1.0],
            "mixup": [0.0, 0.15, 0.3],
        },
    },
    "resolution": {
        "description": "Test different input resolutions",
        "param": "imgsz",
        "values": [320, 416, 512, 640],
    },
    "batch_size": {
        "description": "Find optimal batch size",
        "param": "batch",
        "values": [-1, 8, 16, 32],
    },
    "erasing": {
        "description": "Random erasing for occlusion robustness",
        "param": "erasing",
        "values": [0.0, 0.2, 0.4, 0.6],
    },
    "freeze": {
        "description": "Backbone freeze depth for transfer learning",
        "param": "freeze",
        "values": [0, 5, 10, 15],
    },
    "warmup": {
        "description": "Warmup epoch duration",
        "param": "warmup_epochs",
        "values": [0.0, 1.0, 3.0, 5.0],
    },
}


def list_strategies() -> list[str]:
    """Return names of all built-in strategies."""
    return list(BUILT_IN_STRATEGIES.keys())


def get_strategy(name: str) -> Optional[dict[str, Any]]:
    """Get a strategy by name, or None if not found."""
    return BUILT_IN_STRATEGIES.get(name)


def generate_experiments(
    strategy_name: str,
    *,
    custom_values: Optional[dict[str, list]] = None,
) -> list[dict[str, Any]]:
    """
    Generate a list of experiment configs from a strategy.

    Each returned item has:
      - name: human-readable experiment name
      - overrides: dict of training param overrides

    For 'custom' strategy, pass custom_values with param->values mapping.
    """
    if strategy_name == "custom" and custom_values:
        return _generate_grid(custom_values)

    strategy = BUILT_IN_STRATEGIES.get(strategy_name)
    if not strategy:
        return []

    # Single-param strategy
    if "param" in strategy:
        param = strategy["param"]
        return [
            {
                "name": f"{param}={v}",
                "overrides": {param: v},
            }
            for v in strategy["values"]
        ]

    # Multi-param strategy (grid search)
    if "params" in strategy:
        return _generate_grid(strategy["params"])

    return []


def _generate_grid(params: dict[str, list]) -> list[dict[str, Any]]:
    """Generate all combinations of parameter values."""
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]
    experiments = []

    for combo in product(*value_lists):
        overrides = dict(zip(keys, combo))
        name = "_".join(f"{k}={v}" for k, v in overrides.items())
        experiments.append({"name": name, "overrides": overrides})

    return experiments


__all__ = [
    "BUILT_IN_STRATEGIES",
    "list_strategies",
    "get_strategy",
    "generate_experiments",
]
