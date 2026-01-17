"""Tests for core/base_parser.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.base_parser import BaseModelConfigParser  # noqa: E402
from core.utils import Number, QueryConfig  # noqa: E402
from parsers.llama import LlamaConfigParser  # noqa: E402


class TestNewReqDict:
    """Tests for new_req_dict method creating empty metric dictionaries."""

    def test_returns_all_required_keys(self, sample_llama_config, simple_query_config):
        """Verify all expected metric keys are present."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        req_dict = parser.new_req_dict()

        expected_keys = [
            BaseModelConfigParser.METRIC_NUM_BLOCKS,
            BaseModelConfigParser.METRIC_COMPUTE,
            BaseModelConfigParser.METRIC_BW_WGT,
            BaseModelConfigParser.METRIC_BW_IPT,
            BaseModelConfigParser.METRIC_BW_OPT,
        ]
        for key in expected_keys:
            assert key in req_dict

    def test_values_are_number_instances(self, sample_llama_config, simple_query_config):
        """Verify all values are Number instances."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        req_dict = parser.new_req_dict()

        for value in req_dict.values():
            assert isinstance(value, Number)

    def test_values_initialized_with_none(self, sample_llama_config, simple_query_config):
        """Verify Number values start as None."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        req_dict = parser.new_req_dict()

        # All metric Numbers should have None value initially
        for value in req_dict.values():
            assert value.value is None

    def test_correct_units_assigned(self, sample_llama_config, simple_query_config):
        """Verify correct units are assigned to each metric."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        req_dict = parser.new_req_dict()

        assert req_dict[BaseModelConfigParser.METRIC_COMPUTE].unit == "FLOPs"
        assert req_dict[BaseModelConfigParser.METRIC_BW_WGT].unit == "B"
        assert req_dict[BaseModelConfigParser.METRIC_BW_IPT].unit == "B"
        assert req_dict[BaseModelConfigParser.METRIC_BW_OPT].unit == "B"


class TestSetOpProjReq:
    """Tests for projection (GEMM) operation hardware requirements."""

    def test_compute_formula_correct(self, sample_llama_config, simple_query_config):
        """Verify FLOPs calculation: dim_m * dim_n * (dim_k * 2 - 1)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        # Simple case: 2x3x4 GEMM
        parser.set_op_proj_req(
            layer_entry=layer_entry, dim_m=2, dim_n=3, dim_k=4, torch_dtype="float16"
        )

        # Expected: 2 * 3 * (4 * 2 - 1) = 2 * 3 * 7 = 42 FLOPs
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == 42

    def test_weight_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify weight bandwidth: dim_k * dim_n * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_proj_req(
            layer_entry=layer_entry,
            dim_m=2,
            dim_n=3,
            dim_k=4,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: 4 * 3 * 2 = 24 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 24

    def test_input_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify input bandwidth: dim_m * dim_k * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_proj_req(
            layer_entry=layer_entry,
            dim_m=2,
            dim_n=3,
            dim_k=4,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: 2 * 4 * 2 = 16 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value == 16

    def test_output_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify output bandwidth: dim_m * dim_n * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_proj_req(
            layer_entry=layer_entry,
            dim_m=2,
            dim_n=3,
            dim_k=4,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: 2 * 3 * 2 = 12 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value == 12

    def test_accumulates_to_existing_values(self, sample_llama_config, simple_query_config):
        """Verify metrics accumulate when called multiple times."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        # First call
        parser.set_op_proj_req(layer_entry, dim_m=2, dim_n=3, dim_k=4, torch_dtype="float16")
        first_compute = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_float()

        # Second call
        parser.set_op_proj_req(layer_entry, dim_m=2, dim_n=3, dim_k=4, torch_dtype="float16")
        second_compute = layer_entry[BaseModelConfigParser.METRIC_COMPUTE].get_value_float()

        assert second_compute == first_compute * 2

    def test_different_dtypes_affect_bandwidth(self, sample_llama_config, simple_query_config):
        """Verify different dtypes produce different bandwidth values."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)

        layer_fp16 = parser.new_req_dict()
        layer_fp32 = parser.new_req_dict()

        parser.set_op_proj_req(layer_fp16, dim_m=1, dim_n=1, dim_k=1, torch_dtype="float16")
        parser.set_op_proj_req(layer_fp32, dim_m=1, dim_n=1, dim_k=1, torch_dtype="float32")

        # FP32 bandwidth should be 2x FP16
        assert (
            layer_fp32[BaseModelConfigParser.METRIC_BW_WGT].get_value_float()
            == layer_fp16[BaseModelConfigParser.METRIC_BW_WGT].get_value_float() * 2
        )

    def test_dtypes_dont_affect_flops(self, sample_llama_config, simple_query_config):
        """Verify different dtypes produce same FLOPs values."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)

        layer_fp16 = parser.new_req_dict()
        layer_fp32 = parser.new_req_dict()

        parser.set_op_proj_req(layer_fp16, dim_m=2, dim_n=3, dim_k=4, torch_dtype="float16")
        parser.set_op_proj_req(layer_fp32, dim_m=2, dim_n=3, dim_k=4, torch_dtype="float32")

        # FLOPs should be the same
        assert (
            layer_fp32[BaseModelConfigParser.METRIC_COMPUTE].value
            == layer_fp16[BaseModelConfigParser.METRIC_COMPUTE].value
        )


class TestSetOpSumReq:
    """Tests for element-wise summation/reduction operation requirements."""

    def test_compute_formula(self, sample_llama_config, simple_query_config):
        """Verify FLOPs: num_elem * (num_tensors - 1)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sum_req(
            layer_entry=layer_entry, num_elem=100, num_tensors=2, torch_dtype="float16"
        )

        # Expected: 100 * (2 - 1) = 100 FLOPs
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == 100

    def test_no_weight_bandwidth(self, sample_llama_config, simple_query_config):
        """Verify sum operation has no weight bandwidth (stays at 0)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sum_req(layer_entry, num_elem=100, num_tensors=2, torch_dtype="float16")

        # Sum has no weights, should remain at 0
        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 0

    def test_input_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify input bandwidth: num_elem * dtype_width * num_tensors."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sum_req(
            layer_entry=layer_entry,
            num_elem=100,
            num_tensors=2,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: 100 * 2 * 2 = 400 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value == 400

    def test_output_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify output bandwidth: num_elem * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sum_req(
            layer_entry=layer_entry,
            num_elem=100,
            num_tensors=2,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: 100 * 2 = 200 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value == 200

    def test_three_tensors(self, sample_llama_config, simple_query_config):
        """Verify FLOPs for summing 3 tensors."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sum_req(layer_entry, num_elem=100, num_tensors=3, torch_dtype="float16")

        # Expected: 100 * (3 - 1) = 200 FLOPs
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == 200


class TestSetOpRopeReq:
    """Tests for Rotary Position Embedding operation requirements."""

    def test_compute_formula_3_flops_per_element(self, sample_llama_config, simple_query_config):
        """Verify FLOPs: token_dims * 3 * n_tokens."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rope_req(
            layer_entry=layer_entry, token_dims=128, n_tokens=10, torch_dtype="float16"
        )

        # Expected: 128 * 3 * 10 = 3840 FLOPs
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == 3840

    def test_no_weight_bandwidth(self, sample_llama_config, simple_query_config):
        """Verify RoPE has no weight bandwidth (uses precomputed freqs)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rope_req(layer_entry, token_dims=128, n_tokens=10, torch_dtype="float16")

        # RoPE has no weights, should remain at 0
        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 0

    def test_input_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify input bandwidth: token_dims * n_tokens * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rope_req(
            layer_entry, token_dims=128, n_tokens=10, torch_dtype="float16"  # 2 bytes
        )

        # Expected: 128 * 10 * 2 = 2560 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value == 2560

    def test_input_output_bandwidth_equal(self, sample_llama_config, simple_query_config):
        """Verify input and output bandwidth are equal for in-place operation."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rope_req(layer_entry, token_dims=128, n_tokens=10, torch_dtype="float16")

        # Input and output should be equal
        assert (
            layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value
            == layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value
        )


class TestSetOpRmsnormReq:
    """Tests for RMSNorm operation requirements."""

    def test_compute_formula(self, sample_llama_config, simple_query_config):
        """Verify FLOPs: (hidden_size * 4 + 2) * n_tokens."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rmsnorm_req(
            layer_entry=layer_entry, hidden_size=1024, n_tokens=10, torch_dtype="float16"
        )

        # Expected: (1024 * 4 + 2) * 10 = 40980 FLOPs
        expected_flops = (1024 * 4 + 2) * 10
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_flops

    def test_weight_bandwidth_includes_gamma_and_epsilon(
        self, sample_llama_config, simple_query_config
    ):
        """Verify weight bandwidth: (hidden_size + 1) * dtype_width for gamma and eps."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rmsnorm_req(
            layer_entry=layer_entry,
            hidden_size=1024,
            n_tokens=10,
            torch_dtype="float16",  # 2 bytes
        )

        # Expected: (1024 + 1) * 2 = 2050 bytes
        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 2050

    def test_input_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify input bandwidth: hidden_size * n_tokens * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rmsnorm_req(
            layer_entry=layer_entry, hidden_size=1024, n_tokens=10, torch_dtype="float16"
        )

        expected_bw = 1024 * 10 * 2
        assert layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value == expected_bw

    def test_output_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify output bandwidth: hidden_size * n_tokens * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_rmsnorm_req(
            layer_entry=layer_entry, hidden_size=1024, n_tokens=10, torch_dtype="float16"
        )

        expected_bw = 1024 * 10 * 2
        assert layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value == expected_bw


class TestSetOpActmulReq:
    """Tests for fused activation+multiplication operation requirements."""

    def test_compute_formula(self, sample_llama_config, simple_query_config):
        """Verify FLOPs: (act_flops + 1) * intermediate_size + n_tokens."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_actmul_req(
            layer_entry=layer_entry,
            intermediate_size=1024,
            n_tokens=10,
            act_flops=4,  # SiLU
            torch_dtype="float16",
        )

        # Expected: (4 + 1) * 1024 + 10 = 5130 FLOPs
        expected_flops = (4 + 1) * 1024 + 10
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_flops

    def test_no_weight_bandwidth(self, sample_llama_config, simple_query_config):
        """Verify actmul has no weight bandwidth."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_actmul_req(
            layer_entry=layer_entry,
            intermediate_size=1024,
            n_tokens=10,
            act_flops=4,
            torch_dtype="float16",
        )

        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 0

    def test_input_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify input bandwidth: intermediate_size * n_tokens * 2 * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_actmul_req(
            layer_entry=layer_entry,
            intermediate_size=1024,
            n_tokens=10,
            act_flops=4,
            torch_dtype="float16",  # 2 bytes
        )

        # Input takes 2 tensors (gate and up projection outputs)
        expected_bw = 1024 * 10 * 2 * 2
        assert layer_entry[BaseModelConfigParser.METRIC_BW_IPT].value == expected_bw

    def test_output_bandwidth_formula(self, sample_llama_config, simple_query_config):
        """Verify output bandwidth: intermediate_size * n_tokens * dtype_width."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_actmul_req(
            layer_entry=layer_entry,
            intermediate_size=1024,
            n_tokens=10,
            act_flops=4,
            torch_dtype="float16",
        )

        expected_bw = 1024 * 10 * 2
        assert layer_entry[BaseModelConfigParser.METRIC_BW_OPT].value == expected_bw


class TestSetOpSdpaReq:
    """Tests for Scaled Dot-Product Attention operation requirements."""

    def test_compute_with_single_query(self, sample_llama_config):
        """Verify SDPA FLOPs for single query (no cache)."""
        query_config = QueryConfig(cached_tokens=[0], input_tokens=[1])
        parser = LlamaConfigParser(sample_llama_config, query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sdpa_req(
            layer_entry=layer_entry,
            tensor_qo_dims=4096,
            tensor_kv_dims=1024,
            torch_dtype="float16",
        )

        # qo_seq_len=1, kv_seq_len=1
        # QK^T: 1 * 1 * (4096 * 2 - 1) = 8191
        # SV: 1 * 1024 * (1 * 2 - 1) = 1024
        # Total: 9215
        expected_compute = 1 * 1 * (4096 * 2 - 1) + 1 * 1024 * (1 * 2 - 1)
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_compute

    def test_compute_with_kv_cache(self, sample_llama_config):
        """Verify SDPA FLOPs accounts for cached tokens."""
        query_config = QueryConfig(cached_tokens=[100], input_tokens=[1])
        parser = LlamaConfigParser(sample_llama_config, query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sdpa_req(
            layer_entry=layer_entry,
            tensor_qo_dims=4096,
            tensor_kv_dims=1024,
            torch_dtype="float16",
        )

        # qo_seq_len=1, kv_seq_len=101
        # QK^T: 1 * 101 * (4096 * 2 - 1)
        # SV: 1 * 1024 * (101 * 2 - 1)
        expected_qkt = 1 * 101 * (4096 * 2 - 1)
        expected_sv = 1 * 1024 * (101 * 2 - 1)
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_qkt + expected_sv

    def test_batch_sums_correctly(self, sample_llama_config):
        """Verify SDPA correctly sums metrics across batch."""
        query_config = QueryConfig(cached_tokens=[100, 200], input_tokens=[1, 1])
        parser = LlamaConfigParser(sample_llama_config, query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sdpa_req(
            layer_entry=layer_entry,
            tensor_qo_dims=4096,
            tensor_kv_dims=1024,
            torch_dtype="float16",
        )

        # Should be sum of two queries with different kv_seq_lens
        # Query 1: kv_seq_len=101
        # Query 2: kv_seq_len=201
        expected_q1 = 1 * 101 * (4096 * 2 - 1) + 1 * 1024 * (101 * 2 - 1)
        expected_q2 = 1 * 201 * (4096 * 2 - 1) + 1 * 1024 * (201 * 2 - 1)
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_q1 + expected_q2

    def test_no_weight_bandwidth(self, sample_llama_config, simple_query_config):
        """Verify SDPA has no weight bandwidth."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        parser.set_op_sdpa_req(
            layer_entry, tensor_qo_dims=4096, tensor_kv_dims=1024, torch_dtype="float16"
        )

        assert layer_entry[BaseModelConfigParser.METRIC_BW_WGT].value == 0

    def test_sets_values_not_accumulates(self, sample_llama_config, simple_query_config):
        """Verify SDPA sets values from scratch (not accumulating)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        layer_entry = parser.new_req_dict()

        # Pre-set some values
        layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value = 1000

        parser.set_op_sdpa_req(
            layer_entry, tensor_qo_dims=4096, tensor_kv_dims=1024, torch_dtype="float16"
        )

        # SDPA should overwrite, not accumulate
        expected_compute = 1 * 1 * (4096 * 2 - 1) + 1 * 1024 * (1 * 2 - 1)
        assert layer_entry[BaseModelConfigParser.METRIC_COMPUTE].value == expected_compute


class TestCalcTotal:
    """Tests for total metrics aggregation across all blocks."""

    def test_total_entry_created(self, sample_llama_config, simple_query_config):
        """Verify total entry is created with correct format."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()

        # Should have Total entry
        total_key = f"Total ({parser.get_num_blocks()} Blocks)"
        assert total_key in total_dict

    def test_total_includes_all_layers_plus_total(self, sample_llama_config, simple_query_config):
        """Verify total dict includes all layer entries plus total."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()

        # Should have all layers from get_layer_list plus Total
        assert len(total_dict) == len(parser.get_layer_list()) + 1

    def test_total_values_not_none(self, sample_llama_config, simple_query_config):
        """Verify total metrics are computed (not None)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()

        total_key = f"Total ({parser.get_num_blocks()} Blocks)"
        for metric, value in total_dict[total_key].items():
            assert value.value is not None

    def test_multiplies_by_num_blocks(self, sample_llama_config, simple_query_config):
        """Verify layer metrics are multiplied by block count."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)

        # Get the per-layer metrics
        hw_req = parser.hw_req_by_layers
        first_layer = parser.get_layer_list()[0]
        layer_compute = hw_req[first_layer][BaseModelConfigParser.METRIC_COMPUTE].get_value_int()

        # Get the total
        total_dict = parser.calc_total()
        total_key = f"Total ({parser.get_num_blocks()} Blocks)"
        total_compute = total_dict[total_key][BaseModelConfigParser.METRIC_COMPUTE].get_value_int()

        # Total should be greater than single layer * num_blocks (due to multiple layers)
        assert total_compute >= layer_compute * parser.get_num_blocks()


class TestCalcRoofline:
    """Tests for operational intensity calculation."""

    def test_adds_oi_metric(self, sample_llama_config, simple_query_config):
        """Verify calc_roofline adds Operational Intensity metric."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()
        roofline_dict = parser.calc_roofline(total_dict)

        for metrics in roofline_dict.values():
            assert BaseModelConfigParser.METRIC_OI in metrics

    def test_oi_formula(self, sample_llama_config, simple_query_config):
        """Verify OI = compute / (bw_wgt + bw_ipt + bw_opt)."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()
        roofline_dict = parser.calc_roofline(total_dict)

        total_key = f"Total ({parser.get_num_blocks()} Blocks)"
        metrics = roofline_dict[total_key]

        compute = metrics[BaseModelConfigParser.METRIC_COMPUTE].get_value_int()
        total_bw = (
            metrics[BaseModelConfigParser.METRIC_BW_WGT].get_value_int()
            + metrics[BaseModelConfigParser.METRIC_BW_IPT].get_value_int()
            + metrics[BaseModelConfigParser.METRIC_BW_OPT].get_value_int()
        )
        expected_oi = compute / total_bw

        assert metrics[BaseModelConfigParser.METRIC_OI].value == pytest.approx(expected_oi)

    def test_oi_unit_correct(self, sample_llama_config, simple_query_config):
        """Verify OI has correct unit."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        roofline_dict = parser.calc_roofline(parser.calc_total())

        for metrics in roofline_dict.values():
            assert metrics[BaseModelConfigParser.METRIC_OI].unit == "FLOPs/Bytes"

    def test_does_not_modify_original(self, sample_llama_config, simple_query_config):
        """Verify calc_roofline returns a copy, not modifying original."""
        parser = LlamaConfigParser(sample_llama_config, simple_query_config)
        total_dict = parser.calc_total()
        original_keys = set(list(total_dict.values())[0].keys())

        roofline_dict = parser.calc_roofline(total_dict)

        # Original should not have OI added
        assert BaseModelConfigParser.METRIC_OI not in original_keys
        # But roofline_dict should have it
        assert BaseModelConfigParser.METRIC_OI in list(roofline_dict.values())[0].keys()
