"""
Workspace Paths
===============

Centralized helpers for resolving where datasets, models, reports,
and experiments are stored.

The workspace root is controlled by an environment variable
(default: ``YOLO_WORKSPACE_PATH``). If unset, CWD is used.
The env var name can be overridden per-project via ``yolo-project.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

# Default environment variable — can be overridden by project config
_workspace_env_var = "YOLO_WORKSPACE_PATH"

PathLike = Union[str, os.PathLike[str]]


def set_workspace_env_var(name: str) -> None:
    """Override the environment variable used for workspace root."""
    global _workspace_env_var
    _workspace_env_var = name


def get_workspace_env_var() -> str:
    """Return the current workspace environment variable name."""
    return _workspace_env_var


def get_workspace_root() -> Path:
    """
    Return the root directory for all generated artifacts.

    Priority:
      1. Environment variable (default: YOLO_WORKSPACE_PATH)
      2. Current working directory
    """
    env_root = os.environ.get(_workspace_env_var)
    if env_root:
        expanded = os.path.expandvars(env_root)
        return Path(expanded).expanduser()
    return Path.cwd()


def resolve_workspace_path(path: PathLike, root: Optional[Path] = None) -> Path:
    """
    Resolve a user-provided path relative to the workspace root.

    Absolute paths are returned unchanged. Relative paths are joined
    with *root* (or ``get_workspace_root()``).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    base = root if root is not None else get_workspace_root()
    return base / p


def get_datasets_root() -> Path:
    """Return the root directory for datasets."""
    return get_workspace_root() / "datasets"


def get_models_root() -> Path:
    """Return the root directory for trained models."""
    return get_workspace_root() / "models"


def get_reports_root() -> Path:
    """Return the root directory for training reports and logs."""
    return get_workspace_root() / "reports"


def get_experiments_root() -> Path:
    """Return the root directory for experiment tracking."""
    return get_workspace_root() / "experiments"


__all__ = [
    "set_workspace_env_var",
    "get_workspace_env_var",
    "get_workspace_root",
    "resolve_workspace_path",
    "get_datasets_root",
    "get_models_root",
    "get_reports_root",
    "get_experiments_root",
]
