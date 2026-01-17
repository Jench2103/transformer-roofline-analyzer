"""Tests for parsers/llama.py"""

from __future__ import annotations

import pytest

from transformer_roofline_analyzer.core.base_parser import BaseModelConfigParser
from transformer_roofline_analyzer.core.utils import (
    Number,
    QueryConfig,
    TransformerMode,
    torch_dtype_width,
)
from transformer_roofline_analyzer.parsers.llama import LlamaConfigParser


class TestNormalizeConfig:
    """Tests for normalize_config class method."""

    def test_adds_default_torch_dtype(self):
        """Verify default torch_dtype is added when missing."""
        config = {"model_type": "llama", "hidden_size": 4096}
        normalized = LlamaConfigParser.normalize_config(config)
        assert normalized["torch_dtype"] == "float16"

    def test_preserves_existing_torch_dtype(self):
        """Verify existing torch_dtype is not overwritten."""
        config = {"model_type": "llama", "torch_dtype": "bfloat16"}
        normalized = LlamaConfigParser.normalize_config(config)
        assert normalized["torch_dtype"] == "bfloat16"

    def test_modifies_in_place(self):
        """Verify normalize_config modifies the config in place."""
        config = {"model_type": "llama"}
        result = LlamaConfigParser.normalize_config(config)
        assert config is result
        assert "torch_dtype" in config


class TestGetLayerList:
    """Tests for get_layer_list method."""

    def test_returns_expected_layers(self, sample_llama_config, simple_query_config):
        """Verify all expected LLaMA layers are returned."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layers = parser.get_layer_list()

        expected_layers = [
            "Attn - RMSNorm",
            "Attn - QKV_Proj",
            "Attn - RoPE",
            "Attn - SDPA",
            "Attn - O_Proj",
            "Attn - ResidualAdd",
            "Ffn - RMSNorm",
            "Ffn - GateUp_Proj",
            "Ffn - ActMul",
            "Ffn - Down_Proj",
            "Ffn - ResidualAdd",
        ]
        assert layers == expected_layers

    def test_returns_list(self, sample_llama_config, simple_query_config):
        """Verify return type is list."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        assert isinstance(parser.get_layer_list(), list)

    def test_returns_11_layers(self, sample_llama_config, simple_query_config):
        """Verify exactly 11 layers are returned."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        assert len(parser.get_layer_list()) == 11


class TestGetNumBlocks:
    """Tests for get_num_blocks method."""

    def test_returns_num_hidden_layers(self, sample_llama_config, simple_query_config):
        """Verify returns num_hidden_layers from config."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        assert parser.get_num_blocks() == sample_llama_config["num_hidden_layers"]

    @pytest.mark.parametrize("num_layers", [8, 32, 48, 80])
    def test_different_layer_counts(self, num_layers, simple_query_config):
        """Verify correctly reads different layer counts."""
        config = {
            "model_type": "llama",
            "num_hidden_layers": num_layers,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "hidden_act": "silu",
            "vocab_size": 32000,
            "torch_dtype": "float16",
        }
        parser = LlamaConfigParser(config, simple_query_config)
        assert parser.get_num_blocks() == num_layers


class TestGetKvcacheSize:
    """Tests for get_kvcache_size method."""

    def test_kvcache_formula(self, sample_llama_config):
        """Verify KV-cache size calculation."""
        query_config = QueryConfig(cached_tokens=[1024], input_tokens=[1])
        parser = LlamaConfigParser(sample_llama_config, query_config)

        # KV dims = hidden_size / num_attention_heads * num_kv_heads
        # = 4096 / 32 * 8 = 1024
        # Total seq_len = 1024 + 1 = 1025
        # Per block = 1025 * 1024 * 2 (K+V) * 2 (bfloat16) = 4,198,400
        # Total = 4,198,400 * 32 blocks

        head_dim = sample_llama_config["hidden_size"] / sample_llama_config["num_attention_heads"]
        kv_dims = head_dim * sample_llama_config["num_key_value_heads"]
        seq_len = 1024 + 1
        dtype_width = torch_dtype_width(sample_llama_config["torch_dtype"])
        expected_per_block = seq_len * kv_dims * 2 * dtype_width
        expected_total = expected_per_block * sample_llama_config["num_hidden_layers"]

        kv_size = parser.get_kvcache_size()
        assert kv_size == expected_total

    def test_kvcache_scales_with_batch(self, sample_llama_config):
        """Verify KV-cache scales with batch size."""
        single_query = QueryConfig(cached_tokens=[1024], input_tokens=[1])
        batch_query = QueryConfig(cached_tokens=[1024, 1024], input_tokens=[1, 1])

        parser_single = LlamaConfigParser(sample_llama_config, single_query)
        parser_batch = LlamaConfigParser(sample_llama_config, batch_query)

        assert parser_batch.get_kvcache_size() == 2 * parser_single.get_kvcache_size()

    def test_kvcache_varies_with_sequence_length(self, sample_llama_config):
        """Verify KV-cache varies based on total sequence length."""
        short_seq = QueryConfig(cached_tokens=[100], input_tokens=[1])
        long_seq = QueryConfig(cached_tokens=[1000], input_tokens=[1])

        parser_short = LlamaConfigParser(sample_llama_config, short_seq)
        parser_long = LlamaConfigParser(sample_llama_config, long_seq)

        # Ratio should be approximately (1001 / 101)
        ratio = parser_long.get_kvcache_size() / parser_short.get_kvcache_size()
        assert ratio == pytest.approx(1001 / 101)

    def test_kvcache_returns_numeric(self, sample_llama_config, simple_query_config):
        """Verify KV-cache returns a numeric value."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        kv_size = parser.get_kvcache_size()
        assert isinstance(kv_size, (int, float))


class TestHwReqByLayers:
    """Tests for hw_req_by_layers property."""

    def test_returns_all_layers(self, sample_llama_config, simple_query_config):
        """Verify all layers have hardware requirements."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        hw_req = parser.hw_req_by_layers

        assert set(hw_req.keys()) == set(parser.get_layer_list())

    def test_caches_result(self, sample_llama_config, simple_query_config):
        """Verify hw_req_by_layers is cached (lazy init)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)

        # First access should compute
        _ = parser.hw_req_by_layers
        assert parser._hw_req_by_layers is not None

        # Subsequent access should return cached copy
        hw_req1 = parser.hw_req_by_layers
        hw_req2 = parser.hw_req_by_layers
        # Should be copies (different objects)
        assert hw_req1 is not hw_req2

    def test_all_metrics_computed(self, sample_llama_config, simple_query_config):
        """Verify all metric values are computed (not None)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        hw_req = parser.hw_req_by_layers

        for layer, metrics in hw_req.items():
            assert (
                metrics[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            ), f"Layer {layer} has None compute"

    def test_all_bandwidth_metrics_computed(self, sample_llama_config, simple_query_config):
        """Verify all bandwidth metrics are computed."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        hw_req = parser.hw_req_by_layers

        for layer, metrics in hw_req.items():
            # Weight bandwidth may be 0 for some layers but should not be None
            assert metrics[BaseModelConfigParser.METRIC_BW_WGT].value is not None
            assert metrics[BaseModelConfigParser.METRIC_BW_IPT].value is not None
            assert metrics[BaseModelConfigParser.METRIC_BW_OPT].value is not None


class TestGetExtraStorageReq:
    """Tests for get_extra_storage_req method."""

    def test_includes_embedding_table(self, sample_llama_config, simple_query_config):
        """Verify extra storage includes embedding table."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        req_names = [name for name, _ in extra_reqs]
        assert "Embedding Table" in req_names

    def test_embedding_table_formula(self, sample_llama_config, simple_query_config):
        """Verify embedding table size calculation."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        # Find embedding table
        emb_req = next(r for name, r in extra_reqs if name == "Embedding Table")

        expected_size = (
            sample_llama_config["hidden_size"]
            * sample_llama_config["vocab_size"]
            * torch_dtype_width(sample_llama_config["torch_dtype"])
        )
        assert emb_req.value == expected_size

    def test_returns_number_with_correct_unit(self, sample_llama_config, simple_query_config):
        """Verify extra storage returns Number with correct unit."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        for name, req in extra_reqs:
            assert isinstance(req, Number)
            assert req.unit == "B"


class TestVisionModeNotSupported:
    """Tests for Vision mode rejection."""

    def test_vision_mode_raises_error(self, sample_llama_config):
        """Verify Vision mode raises NotImplementedError."""
        query_config = QueryConfig(cached_tokens=[0], input_tokens=[1])
        # Manually set the mode to Vision
        query_config._t_mode = TransformerMode.Vision

        with pytest.raises(NotImplementedError):
            LlamaConfigParser(sample_llama_config, query_config)


class TestGetModelType:
    """Tests for get_model_type method (inherited from base)."""

    def test_returns_llama(self, sample_llama_config, simple_query_config):
        """Verify model_type is returned correctly."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        assert parser.get_model_type() == "llama"

    def test_returns_unknown_when_missing(self, simple_query_config):
        """Verify 'unknown' is returned when model_type is missing."""
        config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "hidden_act": "silu",
            "vocab_size": 32000,
            "torch_dtype": "float16",
        }
        parser = LlamaConfigParser(config, simple_query_config)
        assert parser.get_model_type() == "unknown"
