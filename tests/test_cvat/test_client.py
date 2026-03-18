"""Tests for CVAT client wrapper."""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestRequireCvat:
    def test_require_cvat_raises_when_not_installed(self):
        """require_cvat raises ImportError with install instructions."""
        with patch.dict("sys.modules", {"cvat_sdk": None}):
            # Re-import to pick up the patched module
            import importlib
            from yolocc.cvat import client as client_mod
            importlib.reload(client_mod)

            with pytest.raises(ImportError, match="cvat-sdk"):
                client_mod.require_cvat()

    def test_require_cvat_succeeds_when_installed(self):
        """require_cvat should not raise when cvat_sdk is importable."""
        mock_sdk = MagicMock()
        with patch.dict("sys.modules", {"cvat_sdk": mock_sdk}):
            import importlib
            from yolocc.cvat import client as client_mod
            importlib.reload(client_mod)

            # Should not raise
            client_mod.require_cvat()


class TestGetCvatConfig:
    def test_returns_cvat_section(self):
        from yolocc.cvat.client import get_cvat_config
        from yolocc.project import ProjectConfig

        config = ProjectConfig(
            name="test",
            classes={0: "a"},
            cvat={"url": "http://test:8080", "project_id": 1},
        )
        with patch("yolocc.cvat.client.load_project_config", return_value=config):
            result = get_cvat_config()
            assert result["url"] == "http://test:8080"
            assert result["project_id"] == 1

    def test_returns_empty_when_no_cvat_section(self):
        from yolocc.cvat.client import get_cvat_config
        from yolocc.project import ProjectConfig

        config = ProjectConfig(name="test", classes={0: "a"})
        with patch("yolocc.cvat.client.load_project_config", return_value=config):
            result = get_cvat_config()
            assert result == {}

    def test_returns_empty_when_no_config(self):
        from yolocc.cvat.client import get_cvat_config

        with patch("yolocc.cvat.client.load_project_config", return_value=None):
            result = get_cvat_config()
            assert result == {}


class TestGetClient:
    def test_raises_without_token(self):
        """get_client should raise ValueError when no token is available."""
        from yolocc.cvat.client import get_client

        mock_sdk = MagicMock()
        with patch.dict("sys.modules", {"cvat_sdk": mock_sdk, "cvat_sdk.core": mock_sdk}):
            with patch("yolocc.cvat.client.load_project_config", return_value=None):
                with patch.dict(os.environ, {}, clear=True):
                    # Remove CVAT_ACCESS_TOKEN if it exists
                    env = {k: v for k, v in os.environ.items() if k != "CVAT_ACCESS_TOKEN"}
                    with patch.dict(os.environ, env, clear=True):
                        with pytest.raises(ValueError, match="CVAT_ACCESS_TOKEN"):
                            get_client()

    def test_uses_config_url(self):
        """get_client should read URL from project config."""
        from yolocc.cvat.client import get_client
        from yolocc.project import ProjectConfig

        config = ProjectConfig(
            name="test",
            classes={0: "a"},
            cvat={"url": "http://custom:9090", "org": "myorg"},
        )

        mock_make_client = MagicMock()
        # Patch require_cvat so it doesn't try to import cvat_sdk,
        # and patch the actual make_client call inside get_client.
        mock_core = MagicMock()
        mock_core.make_client = mock_make_client

        with patch("yolocc.cvat.client.require_cvat"):
            with patch.dict("sys.modules", {"cvat_sdk.core": mock_core}):
                with patch("yolocc.cvat.client.load_project_config", return_value=config):
                    with patch.dict(os.environ, {"CVAT_ACCESS_TOKEN": "test-token"}):
                        get_client()

        mock_make_client.assert_called_once_with(
            host="http://custom:9090",
            credentials=("test-token", ""),
            organization_slug="myorg",
        )

    def test_explicit_args_override_config(self):
        """Explicit url/token args should override config values."""
        from yolocc.cvat.client import get_client
        from yolocc.project import ProjectConfig

        config = ProjectConfig(
            name="test",
            classes={0: "a"},
            cvat={"url": "http://config:8080"},
        )

        mock_make_client = MagicMock()
        mock_core = MagicMock()
        mock_core.make_client = mock_make_client

        with patch("yolocc.cvat.client.require_cvat"):
            with patch.dict("sys.modules", {"cvat_sdk.core": mock_core}):
                with patch("yolocc.cvat.client.load_project_config", return_value=config):
                    get_client(url="http://explicit:7070", token="explicit-token")

        mock_make_client.assert_called_once_with(
            host="http://explicit:7070",
            credentials=("explicit-token", ""),
            organization_slug=None,
        )
