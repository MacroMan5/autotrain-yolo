"""
yolocc doctor --preflight health check
=======================================

Verifies environment, dependencies, CVAT connectivity, and project config.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path

from yolocc.project import load_project_config


def _setup_encoding() -> None:
    """Force UTF-8 on Windows consoles that default to cp1252."""
    if sys.platform == "win32":
        import io
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        else:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )


def _symbols() -> tuple[str, str, str, str, str]:
    """Return (ok, fail, skip, arrow, dash) — unicode if supported, ASCII otherwise."""
    try:
        "\u2713".encode(sys.stdout.encoding or "ascii")
        return "\u2713", "\u2717", "~", "\u2192", "\u2500"
    except (UnicodeEncodeError, LookupError):
        return "+", "X", "~", "->", "-"


def _ok(msg: str) -> None:
    ok, *_ = _symbols()
    print(f"  [{ok}] {msg}")


def _fail(msg: str, fix: str | None = None) -> None:
    _, fail, _, arrow, _ = _symbols()
    print(f"  [{fail}] {msg}")
    if fix:
        print(f"      {arrow} {fix}")


def _check_python() -> bool:
    v = platform.python_version()
    vi = sys.version_info
    if vi >= (3, 10):
        _ok(f"Python {v}")
        return True
    _fail(f"Python {v} --requires >= 3.10", "Install Python 3.10+")
    return False


def _check_torch() -> bool:
    try:
        import torch

        ver = torch.__version__
        if torch.cuda.is_available():
            cuda_ver = torch.version.cuda
            gpu = torch.cuda.get_device_name(0)
            _ok(f"torch {ver} (CUDA {cuda_ver} -- {gpu})")
        else:
            _ok(f"torch {ver} (CPU only)")
        return True
    except ImportError:
        _fail("torch -- not installed", 'pip install torch')
        return False


def _check_ultralytics() -> bool:
    try:
        ver = importlib.metadata.version("ultralytics")
        _ok(f"ultralytics {ver}")
        return True
    except importlib.metadata.PackageNotFoundError:
        _fail("ultralytics -- not installed", "pip install ultralytics")
        return False


def _check_cvat_sdk() -> bool:
    try:
        ver = importlib.metadata.version("cvat-sdk")
        _ok(f"cvat-sdk {ver}")
        return True
    except importlib.metadata.PackageNotFoundError:
        _fail("cvat-sdk -- not installed", 'pip install "yolocc[cvat]"')
        return False


def _check_cvat_server(url: str) -> bool:
    try:
        import urllib.request
        import json

        req = urllib.request.Request(f"{url}/api/server/about", method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            version = data.get("version", "unknown")
            _ok(f"CVAT server {url} -- v{version}")
            return True
    except Exception:
        _fail(
            f"CVAT server -- unreachable at {url}",
            "Launch CVAT or check the URL in yolo-project.yaml",
        )
        return False


def _check_cvat_auth(url: str, token: str) -> bool:
    try:
        import urllib.request
        import json

        req = urllib.request.Request(f"{url}/api/users/self", method="GET")
        req.add_header("Accept", "application/json")
        req.add_header("Authorization", f"Token {token}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            username = data.get("username", "unknown")
            _ok(f"CVAT auth -- connected as \"{username}\"")
            return True
    except Exception:
        _fail("CVAT auth -- authentication failed", "Check CVAT_ACCESS_TOKEN or your token")
        return False


def _check_project_config() -> tuple[bool, dict | None]:
    config = load_project_config()
    if config:
        _ok(f"yolo-project.yaml found (project: {config.name})")
        return True, {
            "cvat_url": (config.cvat or {}).get("url", "http://localhost:8080"),
        }
    _fail("yolo-project.yaml -- not found", "Run /setup to initialize your project")
    return False, None


def _check_dataset() -> bool:
    config = load_project_config()
    if not config:
        return False

    dataset_path = config.defaults.get("dataset")
    if not dataset_path:
        _fail("Dataset -- no dataset path in yolo-project.yaml defaults", "Add defaults.dataset")
        return False

    data_yaml = Path(dataset_path)
    if not data_yaml.is_absolute():
        data_yaml = Path.cwd() / data_yaml

    if not data_yaml.exists():
        _fail(f"Dataset -- {dataset_path} not found", "Check defaults.dataset path")
        return False

    # Try to count images from data.yaml
    try:
        import yaml

        with open(data_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        names = data.get("names", {})
        nc = len(names) if isinstance(names, (dict, list)) else data.get("nc", "?")
        train_path = data.get("train", "")
        val_path = data.get("val", "")

        train_count = _count_images(data_yaml.parent / train_path) if train_path else 0
        val_count = _count_images(data_yaml.parent / val_path) if val_path else 0
        total = train_count + val_count

        _ok(f"Dataset: {total} images ({train_count} train / {val_count} val), {nc} classes")
        return True
    except Exception:
        _ok(f"Dataset: {dataset_path} exists")
        return True


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sum(1 for f in path.iterdir() if f.suffix.lower() in exts)


def doctor() -> int:
    """Run all health checks. Returns 0 if all pass, 1 if any fail."""
    _setup_encoding()
    print()
    print("yolocc doctor")
    _, _, _, _, dash = _symbols()
    print(dash * 40)

    passed = 0
    failed = 0
    total = 0

    def run(check_fn, *args):
        nonlocal passed, failed, total
        total += 1
        if check_fn(*args):
            passed += 1
        else:
            failed += 1

    # Core env
    run(_check_python)
    run(_check_torch)
    run(_check_ultralytics)

    # CVAT SDK
    has_sdk = _check_cvat_sdk()
    total += 1
    if has_sdk:
        passed += 1
    else:
        failed += 1

    # Project config (needed for CVAT URL)
    has_config, config_info = _check_project_config()
    total += 1
    if has_config:
        passed += 1
    else:
        failed += 1

    # CVAT server + auth (only if SDK is installed)
    cvat_url = (config_info or {}).get("cvat_url", "http://localhost:8080")

    if has_sdk:
        server_ok = _check_cvat_server(cvat_url)
        total += 1
        if server_ok:
            passed += 1

            # Auth check only if server is reachable
            token = os.environ.get("CVAT_ACCESS_TOKEN", "")
            total += 1
            if token:
                if _check_cvat_auth(cvat_url, token):
                    passed += 1
                else:
                    failed += 1
            else:
                _fail(
                    "CVAT auth -- CVAT_ACCESS_TOKEN not set",
                    "Export CVAT_ACCESS_TOKEN=<token> or set it in your environment",
                )
                failed += 1
        else:
            failed += 1
    else:
        _, _, skip, _, _ = _symbols()
        print(f"  [{skip}] CVAT checks skipped (no cvat-sdk)")

    # Dataset
    if has_config:
        run(_check_dataset)

    # Summary
    print()
    if failed == 0:
        print(f"All {passed} checks passed!")
    else:
        print(f"{passed}/{total} passed, {failed} failed")
    print()

    return 0 if failed == 0 else 1


def doctor_cli():
    """CLI entry point for yolocc-doctor."""
    sys.exit(doctor())


if __name__ == "__main__":
    doctor_cli()
