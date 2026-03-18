"""CVAT client wrapper — reads config from yolo-project.yaml."""
from __future__ import annotations

import os
from typing import Optional, Any

from yolocc.project import load_project_config


def require_cvat():
    """Check that cvat-sdk is installed, raise helpful error if not."""
    try:
        import cvat_sdk  # noqa: F401
    except ImportError:
        raise ImportError(
            "CVAT integration requires cvat-sdk. "
            'Install with: pip install "yolocc[cvat]"'
        )


def get_client(url: Optional[str] = None, token: Optional[str] = None) -> Any:
    """
    Get an authenticated CVAT client.

    Reads from yolo-project.yaml and CVAT_ACCESS_TOKEN env var.
    Explicit args override config.
    """
    require_cvat()
    from cvat_sdk.core import make_client

    config = load_project_config()
    cvat_cfg = (config.cvat or {}) if config else {}

    _url = url or cvat_cfg.get("url", "http://localhost:8080")
    _token = token or os.environ.get("CVAT_ACCESS_TOKEN", "")
    _org = cvat_cfg.get("org", "")

    if not _token:
        raise ValueError(
            "CVAT_ACCESS_TOKEN env var is required. "
            "Create a Personal Access Token in CVAT UI -> User menu -> Settings -> Access Tokens"
        )

    client = make_client(host=_url, credentials=(_token, ""), organization_slug=_org or None)
    return client


def get_cvat_config() -> dict:
    """Get CVAT config from yolo-project.yaml."""
    config = load_project_config()
    if config and config.cvat:
        return config.cvat
    return {}
