"""Tests for transformer_roofline_analyzer CLI"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base_parser after path setup
from core.base_parser import BaseModelConfigParser  # noqa: E402


def import_cli_module():
    """Import the transformer_roofline_analyzer script as a module."""
    script_path = project_root / "transformer_roofline_analyzer"
    loader = importlib.machinery.SourceFileLoader("transformer_roofline_analyzer", str(script_path))
    spec = importlib.util.spec_from_loader("transformer_roofline_analyzer", loader)
    assert spec is not None, "Failed to create module spec"
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_roofline_analyzer"] = module
    assert spec.loader is not None, "Module spec has no loader"
    spec.loader.exec_module(module)
    return module


# Import the CLI module once
cli_module = import_cli_module()


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_local_json_file(self, tmp_path):
        """Verify loading from local config.json file."""
        config_data = {"model_type": "llama", "hidden_size": 4096}
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = cli_module.load_config(str(config_file))

        assert result == config_data

    def test_raises_for_directory_path(self, tmp_path):
        """Verify error when path is directory, not file."""
        with pytest.raises(ValueError, match="not a file"):
            cli_module.load_config(str(tmp_path))

    def test_loads_nested_json(self, tmp_path):
        """Verify loading nested config.json file."""
        config_data = {
            "model_type": "llama4",
            "text_config": {
                "hidden_size": 5120,
                "num_hidden_layers": 48,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = cli_module.load_config(str(config_file))

        assert result == config_data
        assert result["text_config"]["hidden_size"] == 5120


class TestComputeRooflineMetrics:
    """Tests for compute_roofline_metrics function."""

    def test_raises_for_unknown_model_type(self):
        """Verify NotImplementedError for unknown model_type."""
        config = {"model_type": "unknown_model"}
        args = argparse.Namespace(cached_tokens=[0], input_tokens=[1], batch_size=None)

        with pytest.raises(NotImplementedError, match="No parser for model_type"):
            cli_module.compute_roofline_metrics(config, args)

    def test_raises_for_mismatched_token_lists(self, sample_llama_config):
        """Verify error when cached/input token lists have different lengths."""
        args = argparse.Namespace(cached_tokens=[0, 0], input_tokens=[1], batch_size=None)

        with pytest.raises(ValueError, match="same number of elements"):
            cli_module.compute_roofline_metrics(sample_llama_config, args)

    def test_raises_for_batch_size_not_multiple(self, sample_llama_config):
        """Verify error when batch_size not multiple of token list length."""
        args = argparse.Namespace(cached_tokens=[0, 0], input_tokens=[1, 1], batch_size=3)

        with pytest.raises(ValueError, match="multiple"):
            cli_module.compute_roofline_metrics(sample_llama_config, args)

    def test_batch_expansion(self, sample_llama_config, capsys):
        """Verify batch expansion correctly replicates token patterns."""
        # 2 token patterns, batch_size=4 should create 4 queries (2x2)
        args = argparse.Namespace(cached_tokens=[100, 200], input_tokens=[1, 1], batch_size=4)

        cli_module.compute_roofline_metrics(sample_llama_config, args)

        # Should complete without error (output checked by E2E tests)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_default_batch_size_none(self, sample_llama_config, capsys):
        """Verify batch_size=None works correctly."""
        args = argparse.Namespace(cached_tokens=[0], input_tokens=[1], batch_size=None)

        cli_module.compute_roofline_metrics(sample_llama_config, args)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_single_token_pattern(self, sample_llama_config, capsys):
        """Verify single token pattern works."""
        args = argparse.Namespace(cached_tokens=[1024], input_tokens=[1], batch_size=None)

        cli_module.compute_roofline_metrics(sample_llama_config, args)

        captured = capsys.readouterr()
        assert "Total" in captured.out


class TestParserRegistry:
    """Tests for PARSER_REGISTRY."""

    def test_llama_registered(self):
        """Verify LLaMA parser is registered."""
        assert "llama" in cli_module.PARSER_REGISTRY

    def test_llama4_registered(self):
        """Verify LLaMA-4 parser is registered."""
        assert "llama4" in cli_module.PARSER_REGISTRY

    def test_registry_values_are_parser_classes(self):
        """Verify registry contains valid parser classes."""
        for model_type, parser_cls in cli_module.PARSER_REGISTRY.items():
            assert issubclass(parser_cls, BaseModelConfigParser)

    def test_registry_has_expected_entries(self):
        """Verify registry has exactly expected entries."""
        assert len(cli_module.PARSER_REGISTRY) == 2
        assert set(cli_module.PARSER_REGISTRY.keys()) == {"llama", "llama4"}


class TestArgumentValidation:
    """Tests for command-line argument validation logic."""

    def test_valid_matching_token_lists(self, sample_llama_config, capsys):
        """Verify matching token lists work."""
        args = argparse.Namespace(cached_tokens=[0, 100], input_tokens=[1, 1], batch_size=None)

        cli_module.compute_roofline_metrics(sample_llama_config, args)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_valid_batch_size_multiple(self, sample_llama_config, capsys):
        """Verify valid batch_size works."""
        args = argparse.Namespace(cached_tokens=[0, 100], input_tokens=[1, 1], batch_size=4)

        cli_module.compute_roofline_metrics(sample_llama_config, args)

        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_empty_model_type_raises_error(self):
        """Verify empty model_type raises error."""
        config = {"model_type": ""}
        args = argparse.Namespace(cached_tokens=[0], input_tokens=[1], batch_size=None)

        with pytest.raises(NotImplementedError):
            cli_module.compute_roofline_metrics(config, args)

    def test_missing_model_type_raises_error(self):
        """Verify missing model_type raises error."""
        config = {"hidden_size": 4096}  # No model_type
        args = argparse.Namespace(cached_tokens=[0], input_tokens=[1], batch_size=None)

        with pytest.raises(NotImplementedError):
            cli_module.compute_roofline_metrics(config, args)
