"""Tests for workspace path resolution."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
from yolocc.paths import (
    get_workspace_root,
    resolve_workspace_path,
    get_datasets_root,
    get_models_root,
    get_reports_root,
    get_experiments_root,
    set_workspace_env_var,
)


class TestGetWorkspaceRoot:
    def test_returns_cwd_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            root = get_workspace_root()
            assert root == Path.cwd()

    def test_returns_env_var_path(self, tmp_path):
        with patch.dict(os.environ, {"YOLO_WORKSPACE_PATH": str(tmp_path)}):
            root = get_workspace_root()
            assert root == tmp_path

    def test_custom_env_var(self, tmp_path):
        set_workspace_env_var("MY_CUSTOM_PATH")
        with patch.dict(os.environ, {"MY_CUSTOM_PATH": str(tmp_path)}):
            root = get_workspace_root()
            assert root == tmp_path
        set_workspace_env_var("YOLO_WORKSPACE_PATH")  # reset


class TestResolveWorkspacePath:
    def test_absolute_path_unchanged(self, tmp_path):
        p = tmp_path / "models"
        result = resolve_workspace_path(p)
        assert result == p

    def test_relative_path_joined(self, tmp_path):
        result = resolve_workspace_path("models/best.pt", root=tmp_path)
        assert result == tmp_path / "models" / "best.pt"


class TestSubdirectoryHelpers:
    def test_datasets_root(self, tmp_path):
        with patch.dict(os.environ, {"YOLO_WORKSPACE_PATH": str(tmp_path)}):
            assert get_datasets_root() == tmp_path / "datasets"

    def test_models_root(self, tmp_path):
        with patch.dict(os.environ, {"YOLO_WORKSPACE_PATH": str(tmp_path)}):
            assert get_models_root() == tmp_path / "models"

    def test_reports_root(self, tmp_path):
        with patch.dict(os.environ, {"YOLO_WORKSPACE_PATH": str(tmp_path)}):
            assert get_reports_root() == tmp_path / "reports"

    def test_experiments_root(self, tmp_path):
        with patch.dict(os.environ, {"YOLO_WORKSPACE_PATH": str(tmp_path)}):
            assert get_experiments_root() == tmp_path / "experiments"
