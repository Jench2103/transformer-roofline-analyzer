from __future__ import annotations

from enum import Enum
from typing import Optional, Union, cast

from prefixed import Float


def torch_dtype_width(torch_type: str) -> int:
    """
    Reference: https://docs.pytorch.org/docs/stable/tensors.html#data-types
    """

    match torch_type:
        # Integer
        case "uint8" | "int8" | "quint8" | "qint8":
            return 1
        case "uint16" | "int16" | "short":
            return 2
        case "uint32" | "int32" | "int" | "qint32":
            return 4
        case "uint64" | "int64" | "long":
            return 8

        # Floating point
        case "float8_e4m3fn" | "float8_e5m2":
            return 1
        case "float16" | "half" | "bfloat16":
            return 2
        case "float32" | "float":
            return 4
        case "float64" | "double":
            return 8

        case _:
            raise ValueError(f"Unsupported torch data type: `{torch_type}`.")


def act_flops(act: str) -> int:
    match act:
        # https://github.com/vllm-project/vllm/blob/84ab4feb7e994ee6c692957e6d80a528af072e49/csrc/activation_kernels.cu#L35
        case "silu":
            return 4

        case _:
            raise ValueError(f"Unsupported activation function: `{act}`.")


class TransformerMode(Enum):
    Text = 1
    Vision = 2


class Number:
    """
    Represents a numerical value with associated unit and formatting for display.

    Attributes:
        value (Optional[float]): The numeric value (e.g., FLOPs, bytes). Can be None if not yet computed.
        unit (str): The unit associated with the value (e.g., "FLOPs", "B").
        formatter (str): A format string used for pretty-printing the value (e.g., '!.2h').

    Methods:
        __init__(unit: str, formatter: str, value: Optional[float] = None) -> None:
            Initializes a Number with the given unit, formatter, and optional value.

        __str__() -> str:
            Returns a formatted string representation of the number with its unit.
            Returns an empty string if the value is None.

        __add__(other: Number) -> Number:
            Adds two Number instances with the same unit. If one or both values are None,
            treats them as zero. Raises NotImplementedError if units differ.

        get_value_float() -> float:
            Returns the value as a float. If value is None, returns 0.0.

        get_value_int() -> int:
            Returns the value as an integer. If value is None, returns 0.
    """

    def __init__(self, unit: str, formatter: str, value: Optional[float] = None) -> None:
        self.value: Optional[float] = value
        self.unit: str = unit
        self.formatter: str = formatter

    def __str__(self) -> str:
        if self.value is not None:
            return format(Float(self.value), self.formatter) + self.unit
        else:
            return ""

    def __add__(self, other: Number) -> Number:
        if isinstance(other, Number) and self.unit == other.unit:
            if self.value is not None or other.value is not None:
                new_value: float = self.get_value_float() + other.get_value_float()
                return Number(unit=self.unit, formatter=self.formatter, value=new_value)
            else:
                return Number(unit=self.unit, formatter=self.formatter)
        else:
            raise NotImplementedError

    def __radd__(self, other: Union[int, float]) -> Number:
        if isinstance(other, (int, float)):
            return Number(
                unit=self.unit, formatter=self.formatter, value=self.get_value_float() + other
            )
        else:
            raise NotImplementedError

    def get_value_float(self) -> float:
        return cast(float, self.value) if self.value is not None else 0

    def get_value_int(self) -> int:
        return int(self.get_value_float())


class QueryConfig:
    """
    Stores token-related configuration for a transformer model inference query.

    Attributes:
        _t_mode (TransformerMode): The mode of transformer execution (e.g., text generation).
        _n_cached_tokens (list[int]): List of cached token counts for each batch element.
        _n_input_tokens (list[int]): List of new input token counts for each batch element.

    Properties:
        t_mode: Returns the transformer mode.
        n_cached_tokens: Returns the list of cached token counts.
        n_input_tokens: Returns the list of input token counts.
    """

    def __init__(self, cached_tokens: list[int], input_tokens: list[int]):
        self._t_mode: TransformerMode = TransformerMode.Text
        self._n_cached_tokens: list[int] = cached_tokens
        self._n_input_tokens: list[int] = input_tokens

    @property
    def t_mode(self) -> TransformerMode:
        return self._t_mode

    @property
    def n_cached_tokens(self) -> list[int]:
        return self._n_cached_tokens

    @property
    def n_input_tokens(self) -> list[int]:
        return self._n_input_tokens
