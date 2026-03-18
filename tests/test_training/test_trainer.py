"""Tests for trainer helper functions."""

import sys
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from yolocc.training.trainer import _sanitize_path_arg, train_cli
from yolocc.training.utils import prepare_ultralytics_data_yaml
from yolocc.project import ProjectConfig


class TestSanitizePathArg:
    def test_sanitize_path_arg_none(self):
        assert _sanitize_path_arg(None) is None

    def test_sanitize_path_arg_empty(self):
        assert _sanitize_path_arg("") is None

    def test_sanitize_path_arg_normal(self):
        assert _sanitize_path_arg("  /some/path  ") == "/some/path"

    def test_sanitize_path_arg_newlines(self):
        result = _sanitize_path_arg("C:\\Users\nfoo\\bar")
        assert "\n" not in result
        assert result == "C:\\Usersfoo\\bar"

    def test_sanitize_path_arg_tabs(self):
        result = _sanitize_path_arg("path\tto\tmodel")
        assert "\t" not in result
        assert result == "pathtomodel"

    def test_sanitize_path_arg_strips_quotes(self):
        assert _sanitize_path_arg('"my/path"') == "my/path"
        assert _sanitize_path_arg("'my/path'") == "my/path"

    def test_sanitize_path_arg_whitespace_only(self):
        assert _sanitize_path_arg("   ") is None


class TestPrepareUltralyticsDataYaml:
    def test_prepare_data_yaml_relative_path(self, tmp_path):
        """A data.yaml with 'path: .' should be rewritten to an absolute path."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.dump({"path": ".", "train": "images/train", "val": "images/val", "nc": 2}),
            encoding="utf-8",
        )

        result_path, temp_file = prepare_ultralytics_data_yaml(data_yaml, tmp_path)

        try:
            assert temp_file is not None
            # The returned path should point to a temp file
            assert Path(result_path).exists()

            with open(result_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            # path should now be absolute
            resolved = Path(cfg["path"])
            assert resolved.is_absolute()
            assert resolved == tmp_path.resolve()
        finally:
            if temp_file is not None:
                temp_file.unlink(missing_ok=True)

    def test_prepare_data_yaml_absolute_path_unchanged(self, tmp_path):
        """An absolute path in data.yaml should be left as-is (no temp file)."""
        abs_dataset = tmp_path / "my_dataset"
        abs_dataset.mkdir()

        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.dump({
                "path": str(abs_dataset),
                "train": "images/train",
                "val": "images/val",
                "nc": 3,
            }),
            encoding="utf-8",
        )

        result_path, temp_file = prepare_ultralytics_data_yaml(data_yaml, tmp_path)

        assert temp_file is None
        assert result_path == str(data_yaml)

    def test_prepare_data_yaml_missing_file(self, tmp_path):
        """When the YAML file doesn't exist, return the original path string."""
        missing = tmp_path / "nonexistent.yaml"

        result_path, temp_file = prepare_ultralytics_data_yaml(missing, tmp_path)

        assert result_path == str(missing)
        assert temp_file is None

    def test_prepare_data_yaml_no_path_key(self, tmp_path):
        """When path key is missing, it should be set to dataset_path."""
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text(
            yaml.dump({"train": "images/train", "val": "images/val", "nc": 1}),
            encoding="utf-8",
        )

        result_path, temp_file = prepare_ultralytics_data_yaml(data_yaml, tmp_path)

        try:
            assert temp_file is not None
            with open(result_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            resolved = Path(cfg["path"])
            assert resolved.is_absolute()
        finally:
            if temp_file is not None:
                temp_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yolo_dataset(tmp_path: Path) -> Path:
    """Create a minimal YOLO dataset directory with a data.yaml."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text(
        yaml.dump({
            "path": str(dataset),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: "cat"},
        }),
        encoding="utf-8",
    )
    return dataset


def _base_patches(dataset: Path):
    """Return the set of patches used by all TestTrainCliProjectDefaults tests."""
    return [
        patch("yolocc.training.trainer.load_project_config"),
        patch("yolocc.training.trainer.YOLO"),
        patch("yolocc.training.trainer.copy_model_safe", return_value=True),
        patch("yolocc.training.trainer.save_training_summary"),
        patch(
            "yolocc.training.trainer.prepare_ultralytics_data_yaml",
            return_value=(str(dataset / "data.yaml"), None),
        ),
        patch("yolocc.training.trainer.check_gpu"),
        patch("yolocc.training.trainer.get_device", return_value="cpu"),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainCliProjectDefaults:
    """Integration tests for train_cli() project-config priority chain (AC-002/003/004)."""

    def test_project_defaults_used_when_no_cli_args(self, tmp_path):
        """AC-002: epochs from project config is used when not supplied on CLI."""
        dataset = _make_yolo_dataset(tmp_path)

        project_config = ProjectConfig(
            name="test-project",
            defaults={"epochs": 200, "dataset": str(dataset)},
        )

        patches = _base_patches(dataset)
        with patches[0] as mock_load_cfg, \
             patches[1] as mock_yolo_cls, \
             patches[2], patches[3], patches[4], patches[5], patches[6]:

            mock_load_cfg.return_value = project_config
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            with patch.object(sys, "argv", ["yolo-train", "--dataset", str(dataset),
                                            "--skip-validation", "--skip-analysis"]):
                train_cli()

        mock_model.train.assert_called_once()
        actual_epochs = mock_model.train.call_args.kwargs.get("epochs")
        assert actual_epochs == 200, (
            f"Expected epochs=200 from project config, got {actual_epochs}"
        )

    def test_config_file_overrides_project_defaults(self, tmp_path):
        """AC-003: --config file value overrides project config defaults."""
        dataset = _make_yolo_dataset(tmp_path)

        project_config = ProjectConfig(
            name="test-project",
            defaults={"epochs": 200, "dataset": str(dataset)},
        )

        config_file = tmp_path / "overrides.yaml"
        config_file.write_text(yaml.dump({"epochs": 150}), encoding="utf-8")

        patches = _base_patches(dataset)
        with patches[0] as mock_load_cfg, \
             patches[1] as mock_yolo_cls, \
             patches[2], patches[3], patches[4], patches[5], patches[6]:

            mock_load_cfg.return_value = project_config
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            with patch.object(sys, "argv", [
                "yolo-train",
                "--dataset", str(dataset),
                "--config", str(config_file),
                "--skip-validation", "--skip-analysis",
            ]):
                train_cli()

        mock_model.train.assert_called_once()
        actual_epochs = mock_model.train.call_args.kwargs.get("epochs")
        assert actual_epochs == 150, (
            f"Expected epochs=150 from --config file, got {actual_epochs}"
        )

    def test_cli_arg_overrides_all(self, tmp_path):
        """AC-004: explicit --epochs CLI arg overrides both config file and project defaults."""
        dataset = _make_yolo_dataset(tmp_path)

        project_config = ProjectConfig(
            name="test-project",
            defaults={"epochs": 200, "dataset": str(dataset)},
        )

        patches = _base_patches(dataset)
        with patches[0] as mock_load_cfg, \
             patches[1] as mock_yolo_cls, \
             patches[2], patches[3], patches[4], patches[5], patches[6]:

            mock_load_cfg.return_value = project_config
            mock_model = MagicMock()
            mock_yolo_cls.return_value = mock_model

            with patch.object(sys, "argv", [
                "yolo-train",
                "--dataset", str(dataset),
                "--epochs", "50",
                "--skip-validation", "--skip-analysis",
            ]):
                train_cli()

        mock_model.train.assert_called_once()
        actual_epochs = mock_model.train.call_args.kwargs.get("epochs")
        assert actual_epochs == 50, (
            f"Expected epochs=50 from CLI arg, got {actual_epochs}"
        )


class TestNoConfigWarning:
    """FR-006, FR-007: CLIs warn but don't exit when no project config found."""

    def test_train_cli_warns_no_config(self, tmp_path, capsys):
        """AC-007: warning mentions yolo-project.yaml. AC-008: doesn't exit."""
        # Create a minimal dataset
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("train: images/train\nval: images/train\nnc: 1\nnames: {0: cat}\n")

        mock_model = MagicMock()
        mock_model.train.return_value = MagicMock(results_dict={})

        with patch("yolocc.training.trainer.load_project_config", return_value=None):
            with patch("yolocc.training.trainer.YOLO", return_value=mock_model):
                with patch("yolocc.training.trainer.copy_model_safe", return_value=True):
                    with patch("yolocc.training.trainer.save_training_summary"):
                        with patch("yolocc.training.trainer.prepare_ultralytics_data_yaml",
                                   return_value=(str(data_yaml), None)):
                            with patch("yolocc.training.trainer.check_gpu"):
                                with patch("yolocc.training.trainer.get_device", return_value="cpu"):
                                    with patch("sys.argv", ["yolo-train", "--dataset", str(tmp_path),
                                                            "--skip-validation", "--skip-analysis"]):
                                        train_cli()

        captured = capsys.readouterr()
        assert "yolo-project.yaml" in captured.out
        # AC-008: if we got here, the CLI didn't exit
