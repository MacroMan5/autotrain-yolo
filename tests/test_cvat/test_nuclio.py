"""Tests for Nuclio function generation."""
import pytest
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

from yolocc.project import ProjectConfig


class TestGenerateNuclioFunction:
    def test_missing_model_raises(self, tmp_path):
        """Should raise FileNotFoundError for non-existent model."""
        from yolocc.cvat.nuclio import generate_nuclio_function

        with patch(
            "yolocc.cvat.nuclio.resolve_workspace_path",
            side_effect=lambda p: tmp_path / "nonexistent.onnx",
        ):
            with pytest.raises(FileNotFoundError):
                generate_nuclio_function("nonexistent.onnx", output_dir=str(tmp_path / "out"))

    def test_generates_function_yaml(self, tmp_path):
        """Generate function with a fake .onnx model and verify output files."""
        from yolocc.cvat.nuclio import generate_nuclio_function

        # Create a fake ONNX model
        model_file = tmp_path / "best.onnx"
        model_file.write_bytes(b"fake onnx model")

        config = ProjectConfig(name="test_detector", classes={0: "cat", 1: "dog"})

        output_dir = tmp_path / "serverless" / "test"

        def mock_resolve(p):
            p = Path(p)
            if p.is_absolute():
                return p
            return tmp_path / p

        with patch("yolocc.cvat.nuclio.load_project_config", return_value=config):
            with patch(
                "yolocc.cvat.nuclio.resolve_workspace_path",
                side_effect=mock_resolve,
            ):
                result = generate_nuclio_function(
                    str(model_file),
                    function_name="test",
                    output_dir=str(output_dir),
                )

        assert (result / "function.yaml").exists()
        assert (result / "main.py").exists()
        assert (result / "best.onnx").exists()

    def test_function_yaml_has_class_spec(self, tmp_path):
        """Generated function.yaml should contain class names in the spec."""
        from yolocc.cvat.nuclio import generate_nuclio_function

        model_file = tmp_path / "best.onnx"
        model_file.write_bytes(b"fake onnx model")

        config = ProjectConfig(name="mymodel", classes={0: "cat", 1: "dog", 2: "bird"})

        output_dir = tmp_path / "serverless" / "mymodel"

        def mock_resolve(p):
            p = Path(p)
            if p.is_absolute():
                return p
            return tmp_path / p

        with patch("yolocc.cvat.nuclio.load_project_config", return_value=config):
            with patch(
                "yolocc.cvat.nuclio.resolve_workspace_path",
                side_effect=mock_resolve,
            ):
                result = generate_nuclio_function(
                    str(model_file),
                    function_name="mymodel",
                    output_dir=str(output_dir),
                )

        func_yaml = yaml.safe_load((result / "function.yaml").read_text())
        spec_str = func_yaml["metadata"]["annotations"]["spec"]
        assert "cat" in spec_str
        assert "dog" in spec_str
        assert "bird" in spec_str

    def test_main_py_has_handler(self):
        """Generated main.py template should contain handler and init_context."""
        from yolocc.cvat.nuclio import MAIN_PY_TEMPLATE

        assert "def handler(" in MAIN_PY_TEMPLATE
        assert "def init_context(" in MAIN_PY_TEMPLATE

    def test_function_name_defaults_to_project_name(self, tmp_path):
        """When no function_name is given, use the project name."""
        from yolocc.cvat.nuclio import generate_nuclio_function

        model_file = tmp_path / "best.onnx"
        model_file.write_bytes(b"fake onnx model")

        config = ProjectConfig(name="my-detector", classes={0: "obj"})

        def mock_resolve(p):
            p = Path(p)
            if p.is_absolute():
                return p
            return tmp_path / p

        with patch("yolocc.cvat.nuclio.load_project_config", return_value=config):
            with patch(
                "yolocc.cvat.nuclio.resolve_workspace_path",
                side_effect=mock_resolve,
            ):
                result = generate_nuclio_function(
                    str(model_file),
                    output_dir=str(tmp_path / "out"),
                )

        # The default output would use the project name
        func_yaml = yaml.safe_load((result / "function.yaml").read_text())
        # Name should be derived from config name
        assert "my-detector" in func_yaml["metadata"]["name"] or "my_detector" in func_yaml["metadata"]["name"]

    def test_copies_onnx_model(self, tmp_path):
        """Should copy the ONNX model to the function directory."""
        from yolocc.cvat.nuclio import generate_nuclio_function

        model_content = b"real onnx content here"
        model_file = tmp_path / "best.onnx"
        model_file.write_bytes(model_content)

        config = ProjectConfig(name="test", classes={0: "a"})

        def mock_resolve(p):
            p = Path(p)
            if p.is_absolute():
                return p
            return tmp_path / p

        with patch("yolocc.cvat.nuclio.load_project_config", return_value=config):
            with patch(
                "yolocc.cvat.nuclio.resolve_workspace_path",
                side_effect=mock_resolve,
            ):
                result = generate_nuclio_function(
                    str(model_file),
                    function_name="test",
                    output_dir=str(tmp_path / "out"),
                )

        copied = result / "best.onnx"
        assert copied.exists()
        assert copied.read_bytes() == model_content
