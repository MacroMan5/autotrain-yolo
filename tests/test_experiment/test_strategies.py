"""Tests for experiment strategies."""

import pytest
from yolocc.experiment.strategies import (
    get_strategy,
    list_strategies,
    generate_experiments,
    BUILT_IN_STRATEGIES,
)


class TestBuiltInStrategies:
    def test_all_strategies_exist(self):
        names = list_strategies()
        assert "learning_rate" in names
        assert "augmentation" in names
        assert "resolution" in names
        assert "optimizer" in names

    def test_get_strategy(self):
        lr = get_strategy("learning_rate")
        assert "values" in lr
        assert len(lr["values"]) > 0

    def test_unknown_strategy_returns_none(self):
        assert get_strategy("nonexistent") is None


class TestGenerateExperiments:
    def test_learning_rate_generates_overrides(self):
        experiments = generate_experiments("learning_rate")
        assert len(experiments) > 0
        for exp in experiments:
            assert "lr0" in exp["overrides"]
            assert isinstance(exp["name"], str)

    def test_augmentation_generates_combinations(self):
        experiments = generate_experiments("augmentation")
        assert len(experiments) > 0

    def test_custom_overrides(self):
        experiments = generate_experiments(
            "custom",
            custom_values={"lr0": [0.001, 0.01], "batch": [8, 16]}
        )
        assert len(experiments) == 4  # 2x2 combinations

    def test_resolution_strategy(self):
        experiments = generate_experiments("resolution")
        for exp in experiments:
            assert "imgsz" in exp["overrides"]
