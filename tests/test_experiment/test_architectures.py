"""
Architecture YAML verification tests.

Validates that shipped architecture configs load correctly with Ultralytics YOLO.
These tests require ultralytics installed and are marked @pytest.mark.slow.
"""

import pytest
from pathlib import Path


CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "architectures"

ARCHITECTURE_CONFIGS = [
    ("yolo11.yaml", 3),           # Standard P3/P4/P5 — 3 detection heads
    ("yolo11-p2.yaml", 4),        # P2/P3/P4/P5 — 4 detection heads
    ("yolo11-p2p3p4.yaml", 3),    # Shifted P2/P3/P4 — 3 detection heads (no P5)
]


@pytest.mark.slow
class TestArchitectureConfigs:
    """Verify shipped YOLO architecture YAML configs are valid."""

    @pytest.mark.parametrize("config_name,expected_heads", ARCHITECTURE_CONFIGS)
    def test_config_loads_and_builds(self, config_name, expected_heads):
        """Config loads with YOLO() without errors."""
        from ultralytics import YOLO

        config_path = CONFIGS_DIR / config_name
        assert config_path.exists(), f"Config not found: {config_path}"

        model = YOLO(str(config_path))
        assert model is not None
        assert model.model is not None

    @pytest.mark.parametrize("config_name,expected_heads", ARCHITECTURE_CONFIGS)
    def test_detect_head_count(self, config_name, expected_heads):
        """Detect head has the expected number of output layers."""
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect

        config_path = CONFIGS_DIR / config_name
        model = YOLO(str(config_path))

        # Find the Detect head module
        detect_head = None
        for m in model.model.modules():
            if isinstance(m, Detect):
                detect_head = m
                break

        assert detect_head is not None, f"No Detect head found in {config_name}"
        assert detect_head.nl == expected_heads, (
            f"{config_name}: expected {expected_heads} detection layers, got {detect_head.nl}"
        )

    @pytest.mark.parametrize("config_name,expected_heads", ARCHITECTURE_CONFIGS)
    def test_predict_on_dummy_tensor(self, config_name, expected_heads):
        """Model can forward a dummy tensor without errors."""
        import torch
        from ultralytics import YOLO

        config_path = CONFIGS_DIR / config_name
        model = YOLO(str(config_path))

        dummy = torch.randn(1, 3, 640, 640)
        output = model.model(dummy)
        assert output is not None
