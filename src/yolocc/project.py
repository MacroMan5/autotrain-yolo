"""
Project Config System
=====================

Central configuration for yolocc projects.

Users create a ``yolo-project.yaml`` in their workspace root. This module
provides loading, validation, and access to the config values with a
clear priority chain: CLI arg > project config > package default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_CONFIG_FILENAME = "yolo-project.yaml"
DEFAULT_WORKSPACE_ENV = "YOLO_WORKSPACE_PATH"


@dataclass
class ProjectConfig:
    """Parsed project configuration."""

    name: str
    description: str = ""
    classes: dict[int, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)
    cvat: Optional[dict] = None  # {url, project_id, org}
    workspace_env: str = DEFAULT_WORKSPACE_ENV
    config_path: Optional[Path] = None

    def get_variant(self, name: str) -> Optional[dict[str, Any]]:
        """Get a named variant config, or None if not found."""
        return self.variants.get(name)

    def list_variants(self) -> list[str]:
        """List available variant names."""
        return list(self.variants.keys())

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_names(self) -> list[str]:
        return [self.classes[i] for i in sorted(self.classes.keys())]


def load_project_config(search_from: Optional[Path] = None) -> Optional[ProjectConfig]:
    """
    Load yolo-project.yaml, searching upward from *search_from*.

    Returns None if no config file is found.
    """
    if search_from is None:
        search_from = Path.cwd()

    search_from = Path(search_from).resolve()

    # Search current directory and parents
    current = search_from if search_from.is_dir() else search_from.parent
    while True:
        config_file = current / PROJECT_CONFIG_FILENAME
        if config_file.exists():
            return _parse_config(config_file)
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _parse_config(config_file: Path) -> ProjectConfig:
    """Parse a yolo-project.yaml file into a ProjectConfig."""
    with open(config_file, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    project = raw.get("project", {})
    classes_raw = raw.get("classes", {})

    # Normalize class keys to int
    classes = {int(k): str(v) for k, v in classes_raw.items()}

    return ProjectConfig(
        name=project.get("name", config_file.parent.name),
        description=project.get("description", ""),
        classes=classes,
        defaults=raw.get("defaults", {}),
        variants=raw.get("variants", {}),
        cvat=raw.get("cvat"),
        workspace_env=raw.get("workspace_env", DEFAULT_WORKSPACE_ENV),
        config_path=config_file,
    )


def get_classes(
    config: Optional[ProjectConfig] = None,
    *,
    data_yaml_path: Optional[Path] = None,
) -> dict[int, str]:
    """
    Get class mapping. Priority: project config > data.yaml.

    Returns dict like ``{0: 'cat', 1: 'dog'}``.
    """
    if config and config.classes:
        return config.classes

    if data_yaml_path and data_yaml_path.exists():
        with open(data_yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}

    return {}


def get_default(
    key: str,
    *,
    cli_value: Any = None,
    config: Optional[ProjectConfig] = None,
    fallback: Any = None,
) -> Any:
    """
    Resolve a setting with priority: CLI arg > project config > fallback.
    """
    if cli_value is not None:
        return cli_value
    if config and key in config.defaults:
        return config.defaults[key]
    return fallback


def warn_no_config():
    """Print a warning when no project config is found."""
    print("WARNING: No yolo-project.yaml found. Using defaults.")
    print("  Run /setup or copy yolo-project.example.yaml to configure.")


__all__ = [
    "PROJECT_CONFIG_FILENAME",
    "ProjectConfig",
    "load_project_config",
    "get_classes",
    "get_default",
    "warn_no_config",
]
