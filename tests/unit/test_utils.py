"""Tests for core/utils.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils import (  # noqa: E402
    Number,
    QueryConfig,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class TestTorchDtypeWidth:
    """Tests for torch_dtype_width function mapping dtype strings to byte widths."""

    # Integer types - 1 byte
    @pytest.mark.parametrize("dtype", ["uint8", "int8", "quint8", "qint8"])
    def test_1byte_integer_types(self, dtype: str):
        """Verify 1-byte integer types return width of 1."""
        assert torch_dtype_width(dtype) == 1

    # Integer types - 2 bytes
    @pytest.mark.parametrize("dtype", ["uint16", "int16", "short"])
    def test_2byte_integer_types(self, dtype: str):
        """Verify 2-byte integer types return width of 2."""
        assert torch_dtype_width(dtype) == 2

    # Integer types - 4 bytes
    @pytest.mark.parametrize("dtype", ["uint32", "int32", "int", "qint32"])
    def test_4byte_integer_types(self, dtype: str):
        """Verify 4-byte integer types return width of 4."""
        assert torch_dtype_width(dtype) == 4

    # Integer types - 8 bytes
    @pytest.mark.parametrize("dtype", ["uint64", "int64", "long"])
    def test_8byte_integer_types(self, dtype: str):
        """Verify 8-byte integer types return width of 8."""
        assert torch_dtype_width(dtype) == 8

    # Floating point - 1 byte
    @pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
    def test_1byte_float_types(self, dtype: str):
        """Verify 1-byte float types return width of 1."""
        assert torch_dtype_width(dtype) == 1

    # Floating point - 2 bytes
    @pytest.mark.parametrize("dtype", ["float16", "half", "bfloat16"])
    def test_2byte_float_types(self, dtype: str):
        """Verify 2-byte float types return width of 2."""
        assert torch_dtype_width(dtype) == 2

    # Floating point - 4 bytes
    @pytest.mark.parametrize("dtype", ["float32", "float"])
    def test_4byte_float_types(self, dtype: str):
        """Verify 4-byte float types return width of 4."""
        assert torch_dtype_width(dtype) == 4

    # Floating point - 8 bytes
    @pytest.mark.parametrize("dtype", ["float64", "double"])
    def test_8byte_float_types(self, dtype: str):
        """Verify 8-byte float types return width of 8."""
        assert torch_dtype_width(dtype) == 8

    # Error cases
    def test_unsupported_dtype_raises_error(self):
        """Verify unsupported dtype raises ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Unsupported torch data type"):
            torch_dtype_width("invalid_type")

    def test_empty_string_raises_error(self):
        """Verify empty string raises ValueError."""
        with pytest.raises(ValueError):
            torch_dtype_width("")

    def test_case_sensitivity(self):
        """Verify dtypes are case-sensitive (uppercase should fail)."""
        with pytest.raises(ValueError):
            torch_dtype_width("FLOAT16")


class TestActFlops:
    """Tests for act_flops function returning FLOPs for activation functions."""

    def test_silu_returns_4_flops(self):
        """Verify SiLU activation returns 4 FLOPs per element."""
        assert act_flops("silu") == 4

    def test_unsupported_activation_raises_error(self):
        """Verify unsupported activation raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported activation function"):
            act_flops("relu")

    def test_empty_string_raises_error(self):
        """Verify empty string raises ValueError."""
        with pytest.raises(ValueError):
            act_flops("")

    @pytest.mark.parametrize("activation", ["gelu", "tanh", "sigmoid", "softmax"])
    def test_other_common_activations_raise_error(self, activation: str):
        """Verify currently unsupported activations raise errors."""
        with pytest.raises(ValueError):
            act_flops(activation)


class TestTransformerMode:
    """Tests for TransformerMode enum."""

    def test_text_mode_exists(self):
        """Verify Text mode exists."""
        assert hasattr(TransformerMode, "Text")

    def test_vision_mode_exists(self):
        """Verify Vision mode exists."""
        assert hasattr(TransformerMode, "Vision")

    def test_enum_members_count(self):
        """Verify only expected modes exist."""
        assert len(TransformerMode) == 2

    def test_text_mode_value(self):
        """Verify Text mode has expected value."""
        assert TransformerMode.Text.value == 1

    def test_vision_mode_value(self):
        """Verify Vision mode has expected value."""
        assert TransformerMode.Vision.value == 2


class TestNumber:
    """Tests for Number class handling numerical values with units and formatting."""

    # Initialization tests
    def test_init_with_value(self):
        """Test initialization with a value."""
        num = Number("FLOPs", "!.2h", 1000.0)
        assert num.value == 1000.0
        assert num.unit == "FLOPs"
        assert num.formatter == "!.2h"

    def test_init_without_value(self):
        """Test initialization without a value (None)."""
        num = Number("B", "!.2k")
        assert num.value is None
        assert num.unit == "B"

    # __str__ tests
    def test_str_with_value(self):
        """Test string representation with a value."""
        num = Number("FLOPs", "!.2h", 1e9)
        result = str(num)
        assert "FLOPs" in result
        assert result != ""

    def test_str_without_value_returns_empty(self):
        """Test string representation returns empty string when value is None."""
        num = Number("FLOPs", "!.2h")
        assert str(num) == ""

    def test_str_with_zero_value(self):
        """Test string representation with zero value."""
        num = Number("B", "!.2k", 0)
        result = str(num)
        assert result != ""  # Zero is a valid value, should format

    # __add__ tests
    def test_add_two_numbers_with_values(self):
        """Test adding two Numbers with same unit and values."""
        num1 = Number("FLOPs", "!.2h", 100.0)
        num2 = Number("FLOPs", "!.2h", 200.0)
        result = num1 + num2
        assert result.value == 300.0
        assert result.unit == "FLOPs"

    def test_add_number_with_none_and_value(self):
        """Test adding Number with None to Number with value."""
        num1 = Number("FLOPs", "!.2h")  # None value
        num2 = Number("FLOPs", "!.2h", 200.0)
        result = num1 + num2
        assert result.value == 200.0

    def test_add_both_none_returns_none_value(self):
        """Test adding two Numbers both with None values."""
        num1 = Number("FLOPs", "!.2h")
        num2 = Number("FLOPs", "!.2h")
        result = num1 + num2
        assert result.value is None

    def test_add_different_units_raises_error(self):
        """Test adding Numbers with different units raises error."""
        num1 = Number("FLOPs", "!.2h", 100.0)
        num2 = Number("B", "!.2k", 200.0)
        with pytest.raises(NotImplementedError):
            _ = num1 + num2

    def test_add_non_number_raises_error(self):
        """Test adding non-Number type raises error."""
        num1 = Number("FLOPs", "!.2h", 100.0)
        with pytest.raises(NotImplementedError):
            _ = num1 + "not a number"  # type: ignore[operator]

    # __radd__ tests
    def test_radd_with_int(self):
        """Test reverse add with integer."""
        num = Number("FLOPs", "!.2h", 100.0)
        result = 50 + num
        assert result.value == 150.0

    def test_radd_with_float(self):
        """Test reverse add with float."""
        num = Number("FLOPs", "!.2h", 100.0)
        result = 50.5 + num
        assert result.value == 150.5

    def test_radd_with_none_value(self):
        """Test reverse add when Number value is None."""
        num = Number("FLOPs", "!.2h")  # None value
        result = 50 + num
        assert result.value == 50.0

    def test_radd_non_numeric_raises_error(self):
        """Test reverse add with non-numeric type raises error."""
        num = Number("FLOPs", "!.2h", 100.0)
        with pytest.raises(NotImplementedError):
            _ = "string" + num  # type: ignore[operator]

    # get_value_float tests
    def test_get_value_float_with_value(self):
        """Test get_value_float returns the value as float."""
        num = Number("FLOPs", "!.2h", 123.45)
        assert num.get_value_float() == 123.45

    def test_get_value_float_with_none_returns_zero(self):
        """Test get_value_float returns 0.0 when value is None."""
        num = Number("FLOPs", "!.2h")
        assert num.get_value_float() == 0.0

    def test_get_value_float_with_int_value(self):
        """Test get_value_float returns numeric when initialized with int."""
        num = Number("FLOPs", "!.2h", 100)
        assert num.get_value_float() == 100.0
        assert isinstance(num.get_value_float(), (int, float))

    # get_value_int tests
    def test_get_value_int_with_value(self):
        """Test get_value_int returns truncated integer."""
        num = Number("FLOPs", "!.2h", 123.99)
        assert num.get_value_int() == 123

    def test_get_value_int_with_none_returns_zero(self):
        """Test get_value_int returns 0 when value is None."""
        num = Number("FLOPs", "!.2h")
        assert num.get_value_int() == 0

    # Edge cases
    def test_negative_value(self):
        """Test Number handles negative values."""
        num = Number("FLOPs", "!.2h", -100.0)
        assert num.value == -100.0
        assert num.get_value_float() == -100.0

    def test_very_large_value(self):
        """Test Number handles very large values (petaflops range)."""
        num = Number("FLOPs", "!.2h", 1e15)
        assert num.value == 1e15
        assert str(num) != ""  # Should format successfully


class TestQueryConfig:
    """Tests for QueryConfig class storing token configuration."""

    def test_init_single_query(self):
        """Test initialization with single query."""
        qc = QueryConfig(cached_tokens=[1024], input_tokens=[1])
        assert qc.n_cached_tokens == [1024]
        assert qc.n_input_tokens == [1]

    def test_init_batch_queries(self):
        """Test initialization with batch of queries."""
        qc = QueryConfig(cached_tokens=[1024, 2048], input_tokens=[1, 1])
        assert qc.n_cached_tokens == [1024, 2048]
        assert qc.n_input_tokens == [1, 1]

    def test_default_transformer_mode(self):
        """Test default transformer mode is Text."""
        qc = QueryConfig(cached_tokens=[0], input_tokens=[1])
        assert qc.t_mode == TransformerMode.Text

    def test_properties_return_correct_types(self):
        """Test property return types."""
        qc = QueryConfig(cached_tokens=[0], input_tokens=[1])
        assert isinstance(qc.n_cached_tokens, list)
        assert isinstance(qc.n_input_tokens, list)
        assert isinstance(qc.t_mode, TransformerMode)

    def test_empty_lists(self):
        """Test initialization with empty lists."""
        qc = QueryConfig(cached_tokens=[], input_tokens=[])
        assert qc.n_cached_tokens == []
        assert qc.n_input_tokens == []

    def test_zero_tokens(self):
        """Test initialization with zero token counts."""
        qc = QueryConfig(cached_tokens=[0], input_tokens=[0])
        assert qc.n_cached_tokens == [0]
        assert qc.n_input_tokens == [0]
