"""Tests for project config system."""

import pytest
import yaml
from pathlib import Path
from yolocc.project import (
    load_project_config,
    get_classes,
    get_default,
    ProjectConfig,
    PROJECT_CONFIG_FILENAME,
)


@pytest.fixture
def sample_config(tmp_path):
    """Create a sample yolo-project.yaml."""
    config = {
        "project": {"name": "test_detection", "description": "Test project"},
        "classes": {0: "cat", 1: "dog", 2: "bird"},
        "defaults": {
            "base_model": "yolo11n.pt",
            "imgsz": 640,
            "epochs": 100,
            "dataset": "datasets/test",
        },
        "variants": {
            "indoor": {"dataset": "datasets/indoor", "epochs": 30},
            "outdoor": {"dataset": "datasets/outdoor", "epochs": 50},
        },
    }
    config_path = tmp_path / PROJECT_CONFIG_FILENAME
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return tmp_path


@pytest.fixture
def config_with_workspace_env(tmp_path):
    """Create config with custom workspace_env."""
    config = {
        "project": {"name": "custom_env"},
        "classes": {0: "target"},
        "defaults": {"base_model": "yolo11n.pt"},
        "workspace_env": "MY_CUSTOM_PATH",
    }
    config_path = tmp_path / PROJECT_CONFIG_FILENAME
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return tmp_path


class TestLoadProjectConfig:
    def test_loads_from_directory(self, sample_config):
        config = load_project_config(sample_config)
        assert config is not None
        assert config.name == "test_detection"

    def test_returns_none_when_not_found(self, tmp_path):
        config = load_project_config(tmp_path)
        assert config is None

    def test_searches_parent_directories(self, sample_config):
        child = sample_config / "subdir" / "deep"
        child.mkdir(parents=True)
        config = load_project_config(child)
        assert config is not None
        assert config.name == "test_detection"


class TestGetClasses:
    def test_from_project_config(self, sample_config):
        config = load_project_config(sample_config)
        classes = get_classes(config)
        assert classes == {0: "cat", 1: "dog", 2: "bird"}

    def test_from_data_yaml(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(yaml.dump({"nc": 2, "names": {0: "a", 1: "b"}}))
        classes = get_classes(None, data_yaml_path=data_yaml)
        assert classes == {0: "a", 1: "b"}

    def test_project_config_takes_priority(self, sample_config):
        data_yaml = sample_config / "data.yaml"
        data_yaml.write_text(yaml.dump({"nc": 1, "names": {0: "other"}}))
        config = load_project_config(sample_config)
        classes = get_classes(config, data_yaml_path=data_yaml)
        assert classes == {0: "cat", 1: "dog", 2: "bird"}


class TestGetDefault:
    def test_from_project_config(self, sample_config):
        config = load_project_config(sample_config)
        assert get_default("imgsz", config=config) == 640
        assert get_default("epochs", config=config) == 100

    def test_cli_overrides_config(self, sample_config):
        config = load_project_config(sample_config)
        assert get_default("imgsz", cli_value=320, config=config) == 320

    def test_fallback_when_no_config(self):
        assert get_default("imgsz", fallback=640) == 640

    def test_cli_overrides_fallback(self):
        assert get_default("imgsz", cli_value=320, fallback=640) == 320


class TestProjectConfigVariants:
    def test_get_variant(self, sample_config):
        config = load_project_config(sample_config)
        variant = config.get_variant("indoor")
        assert variant["epochs"] == 30
        assert variant["dataset"] == "datasets/indoor"

    def test_variant_not_found(self, sample_config):
        config = load_project_config(sample_config)
        assert config.get_variant("nonexistent") is None

    def test_list_variants(self, sample_config):
        config = load_project_config(sample_config)
        assert set(config.list_variants()) == {"indoor", "outdoor"}


class TestWorkspaceEnv:
    def test_custom_workspace_env(self, config_with_workspace_env):
        config = load_project_config(config_with_workspace_env)
        assert config.workspace_env == "MY_CUSTOM_PATH"

    def test_default_workspace_env(self, sample_config):
        config = load_project_config(sample_config)
        assert config.workspace_env == "YOLO_WORKSPACE_PATH"
