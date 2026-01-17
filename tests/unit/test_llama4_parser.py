"""Tests for parsers/llama4.py"""

from __future__ import annotations

from transformer_roofline_analyzer.core.base_parser import BaseModelConfigParser
from transformer_roofline_analyzer.core.utils import (
    Number,
    QueryConfig,
    torch_dtype_width,
)
from transformer_roofline_analyzer.parsers.llama4 import Llama4ConfigParser


class TestNormalizeConfig:
    """Tests for normalize_config class method."""

    def test_adds_default_torch_dtype_to_text_config(self):
        """Verify default torch_dtype is added to text_config when missing."""
        config = {"model_type": "llama4", "text_config": {"hidden_size": 5120}}
        normalized = Llama4ConfigParser.normalize_config(config)
        assert normalized["text_config"]["torch_dtype"] == "float16"

    def test_preserves_existing_torch_dtype(self):
        """Verify existing torch_dtype in text_config is not overwritten."""
        config = {"model_type": "llama4", "text_config": {"torch_dtype": "bfloat16"}}
        normalized = Llama4ConfigParser.normalize_config(config)
        assert normalized["text_config"]["torch_dtype"] == "bfloat16"

    def test_handles_missing_text_config(self):
        """Verify gracefully handles config without text_config."""
        config = {"model_type": "llama4"}
        normalized = Llama4ConfigParser.normalize_config(config)
        # Should not raise, just return config as-is
        assert "text_config" not in normalized

    def test_handles_non_dict_text_config(self):
        """Verify gracefully handles non-dict text_config."""
        config = {"model_type": "llama4", "text_config": "not_a_dict"}
        normalized = Llama4ConfigParser.normalize_config(config)
        # Should not raise, just return config as-is
        assert normalized["text_config"] == "not_a_dict"


class TestGetLayerList:
    """Tests for get_layer_list method."""

    def test_returns_moe_layers(self, sample_llama4_config, simple_query_config):
        """Verify MoE-specific layers are included."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        layers = parser.get_layer_list()

        # Should include MoE-specific layers
        assert "Ffn - Router" in layers
        assert "Ffn - RoutedExp_GateUp_Proj" in layers
        assert "Ffn - RoutedExp_ActMul" in layers
        assert "Ffn - RoutedExp_Down_Proj" in layers
        assert "Ffn - SharedExp_GateUp_Proj" in layers
        assert "Ffn - SharedExp_ActMul" in layers
        assert "Ffn - SharedExp_Down_Proj" in layers
        assert "Ffn - RoutedSharedExpAdd" in layers
        assert "Ffn - NonMoE_GateUp_Proj" in layers
        assert "Ffn - NonMoE_ActMul" in layers
        assert "Ffn - NonMoE_Down_Proj" in layers

    def test_includes_attention_layers(self, sample_llama4_config, simple_query_config):
        """Verify attention layers are included."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        layers = parser.get_layer_list()

        assert "Attn - RMSNorm" in layers
        assert "Attn - QKV_Proj" in layers
        assert "Attn - RoPE" in layers
        assert "Attn - SDPA" in layers
        assert "Attn - O_Proj" in layers
        assert "Attn - ResidualAdd" in layers

    def test_returns_19_layers(self, sample_llama4_config, simple_query_config):
        """Verify exactly 19 layers are returned for text mode."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        assert len(parser.get_layer_list()) == 19


class TestGetNumBlocks:
    """Tests for get_num_blocks method."""

    def test_returns_num_hidden_layers_from_text_config(
        self, sample_llama4_config, simple_query_config
    ):
        """Verify returns num_hidden_layers from text_config."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        assert parser.get_num_blocks() == sample_llama4_config["text_config"]["num_hidden_layers"]


class TestGetLayerNumBlocks:
    """Tests for get_layer_num_blocks method (MoE layer distinction)."""

    def test_attention_layers_return_full_block_count(
        self, sample_llama4_config, simple_query_config
    ):
        """Verify attention layers present in all blocks."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)

        assert parser.get_layer_num_blocks("Attn - RMSNorm") == parser.get_num_blocks()
        assert parser.get_layer_num_blocks("Attn - SDPA") == parser.get_num_blocks()
        assert parser.get_layer_num_blocks("Attn - QKV_Proj") == parser.get_num_blocks()
        assert parser.get_layer_num_blocks("Attn - O_Proj") == parser.get_num_blocks()
        assert parser.get_layer_num_blocks("Attn - ResidualAdd") == parser.get_num_blocks()

    def test_moe_layers_return_interleaved_count(self, sample_llama4_config, simple_query_config):
        """Verify MoE layers present in subset of blocks based on interleave step."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        text_config = sample_llama4_config["text_config"]

        expected_moe_blocks = parser.get_num_blocks() // text_config["interleave_moe_layer_step"]

        assert parser.get_layer_num_blocks("Ffn - RoutedExp_GateUp_Proj") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - RoutedExp_ActMul") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - RoutedExp_Down_Proj") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - SharedExp_GateUp_Proj") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - SharedExp_ActMul") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - SharedExp_Down_Proj") == expected_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - RoutedSharedExpAdd") == expected_moe_blocks

    def test_non_moe_ffn_layers_return_complement(self, sample_llama4_config, simple_query_config):
        """Verify NonMoE FFN layers present in blocks without MoE."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        text_config = sample_llama4_config["text_config"]

        moe_blocks = parser.get_num_blocks() // text_config["interleave_moe_layer_step"]
        non_moe_blocks = parser.get_num_blocks() - moe_blocks

        assert parser.get_layer_num_blocks("Ffn - NonMoE_GateUp_Proj") == non_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - NonMoE_ActMul") == non_moe_blocks
        assert parser.get_layer_num_blocks("Ffn - NonMoE_Down_Proj") == non_moe_blocks

    def test_rmsnorm_returns_full_count(self, sample_llama4_config, simple_query_config):
        """Verify RMSNorm layers appear in all blocks."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)

        assert parser.get_layer_num_blocks("Ffn - RMSNorm") == parser.get_num_blocks()

    def test_residual_add_returns_full_count(self, sample_llama4_config, simple_query_config):
        """Verify residual add layers appear in all blocks."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)

        assert parser.get_layer_num_blocks("Ffn - ResidualAdd") == parser.get_num_blocks()


class TestGetKvcacheSize:
    """Tests for get_kvcache_size method."""

    def test_uses_text_config_for_kv_dims(self, sample_llama4_config, simple_query_config):
        """Verify KV-cache uses head_dim and num_key_value_heads from text_config."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        kv_size = parser.get_kvcache_size()

        assert kv_size > 0
        assert isinstance(kv_size, int)

    def test_kvcache_formula(self, sample_llama4_config):
        """Verify KV-cache size calculation formula."""
        query_config = QueryConfig(cached_tokens=[1024], input_tokens=[1])
        parser = Llama4ConfigParser(sample_llama4_config, query_config)
        text_config = sample_llama4_config["text_config"]

        kv_dims = text_config["head_dim"] * text_config["num_key_value_heads"]
        seq_len = 1024 + 1
        dtype_width = torch_dtype_width(text_config["torch_dtype"])
        expected_per_block = seq_len * kv_dims * 2 * dtype_width
        expected_total = expected_per_block * text_config["num_hidden_layers"]

        kv_size = parser.get_kvcache_size()
        assert kv_size == expected_total

    def test_kvcache_scales_with_batch(self, sample_llama4_config):
        """Verify KV-cache scales with batch size."""
        single_query = QueryConfig(cached_tokens=[1024], input_tokens=[1])
        batch_query = QueryConfig(cached_tokens=[1024, 1024], input_tokens=[1, 1])

        parser_single = Llama4ConfigParser(sample_llama4_config, single_query)
        parser_batch = Llama4ConfigParser(sample_llama4_config, batch_query)

        assert parser_batch.get_kvcache_size() == 2 * parser_single.get_kvcache_size()


class TestGetExtraStorageReq:
    """Tests for get_extra_storage_req method."""

    def test_includes_additional_experts(self, sample_llama4_config, simple_query_config):
        """Verify extra storage includes additional (non-active) experts."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        req_names = [name for name, _ in extra_reqs]
        assert "Additional Experts" in req_names

    def test_includes_embedding_table(self, sample_llama4_config, simple_query_config):
        """Verify extra storage includes embedding table."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        req_names = [name for name, _ in extra_reqs]
        assert "Embedding Table" in req_names

    def test_additional_experts_formula(self, sample_llama4_config, simple_query_config):
        """Verify additional experts storage calculation."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()
        text_config = sample_llama4_config["text_config"]

        # Expert params: hidden_size * intermediate_size * 3 (gate, up, down) * dtype_width
        exp_size = (
            text_config["hidden_size"]
            * text_config["intermediate_size"]
            * torch_dtype_width(text_config["torch_dtype"])
            * 3
        )
        # Extra experts: (num_local_experts - num_experts_per_tok) * moe_blocks
        extra_exp_cnt = (text_config["num_local_experts"] - text_config["num_experts_per_tok"]) * (
            parser.get_num_blocks() // text_config["interleave_moe_layer_step"]
        )

        additional_experts_req = next(r for name, r in extra_reqs if name == "Additional Experts")
        assert additional_experts_req.value == exp_size * extra_exp_cnt

    def test_embedding_table_formula(self, sample_llama4_config, simple_query_config):
        """Verify embedding table size calculation."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()
        text_config = sample_llama4_config["text_config"]

        emb_req = next(r for name, r in extra_reqs if name == "Embedding Table")

        expected_size = (
            text_config["hidden_size"]
            * text_config["vocab_size"]
            * torch_dtype_width(text_config["torch_dtype"])
        )
        assert emb_req.value == expected_size

    def test_returns_number_with_correct_unit(self, sample_llama4_config, simple_query_config):
        """Verify extra storage returns Number with correct unit."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        extra_reqs = parser.get_extra_storage_req()

        for name, req in extra_reqs:
            assert isinstance(req, Number)
            assert req.unit == "B"


class TestHwReqByLayers:
    """Tests for hw_req_by_layers property."""

    def test_returns_all_layers(self, sample_llama4_config, simple_query_config):
        """Verify all layers have hardware requirements."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        hw_req = parser.hw_req_by_layers

        assert set(hw_req.keys()) == set(parser.get_layer_list())

    def test_caches_result(self, sample_llama4_config, simple_query_config):
        """Verify hw_req_by_layers is cached (lazy init)."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)

        # First access should compute
        _ = parser.hw_req_by_layers
        assert parser._hw_req_by_layers is not None

    def test_all_metrics_computed(self, sample_llama4_config, simple_query_config):
        """Verify all metric values are computed (not None)."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        hw_req = parser.hw_req_by_layers

        for layer, metrics in hw_req.items():
            assert (
                metrics[BaseModelConfigParser.METRIC_COMPUTE].value is not None
            ), f"Layer {layer} has None compute"


class TestCalcTotal:
    """Tests for calc_total with MoE layers."""

    def test_moe_layers_with_zero_blocks_excluded(self, simple_query_config):
        """Verify layers with 0 block count are excluded from total."""
        # Create a config where interleave_moe_layer_step > num_hidden_layers
        # This would make moe_blocks = 0
        config = {
            "model_type": "llama4",
            "text_config": {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_hidden_layers": 2,  # Very few layers
                "num_key_value_heads": 2,
                "head_dim": 128,
                "intermediate_size": 2048,
                "intermediate_size_mlp": 4096,
                "hidden_act": "silu",
                "vocab_size": 32000,
                "torch_dtype": "float16",
                "num_local_experts": 4,
                "num_experts_per_tok": 1,
                "interleave_moe_layer_step": 4,  # > num_hidden_layers, so moe_blocks = 0
            },
        }
        parser = Llama4ConfigParser(config, simple_query_config)
        total_dict = parser.calc_total()

        # MoE layers should be excluded since they have 0 blocks
        assert "Ffn - RoutedExp_GateUp_Proj" not in total_dict
        assert "Ffn - SharedExp_GateUp_Proj" not in total_dict

    def test_total_accounts_for_different_block_counts(
        self, sample_llama4_config, simple_query_config
    ):
        """Verify total correctly accounts for layers with different block counts."""
        parser = Llama4ConfigParser(sample_llama4_config, simple_query_config)
        total_dict = parser.calc_total()

        total_key = f"Total ({parser.get_num_blocks()} Blocks)"
        assert total_key in total_dict

        # Total should be non-zero
        total_compute = total_dict[total_key][BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        assert total_compute > 0
