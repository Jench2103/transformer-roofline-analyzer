# Core Modules

This package provides the foundational framework for transformer model hardware analysis.

## Table of Contents

- [Overview](#overview)
- [base_parser.py - BaseModelConfigParser](#base_parserpy---basemodelconfigparser)
- [utils.py - Utilities](#utilspy---utilities)
- [Hardware Metrics Formulas](#hardware-metrics-formulas)

## Overview

The `core` package contains two modules:

| Module | Purpose |
|--------|---------|
| `base_parser.py` | Abstract base class defining the parser framework |
| `utils.py` | Utilities for numerical values, query configuration, and helpers |

## base_parser.py - BaseModelConfigParser

The `BaseModelConfigParser` is an abstract base class that defines the framework for all
model-specific parsers. It provides:

- Common methods for calculating hardware metrics (FLOPs, bandwidth, operational intensity)
- Operation-specific calculation methods (`set_op_*`)
- Lazy initialization and caching of computed metrics
- Tabulated output formatting and storage calculations

### Class Structure

```python
class BaseModelConfigParser(ABC):
    # Metric name constants
    METRIC_NUM_BLOCKS: Final[str] = "Block Count"
    METRIC_COMPUTE: Final[str] = "Compute"
    METRIC_BW_IPT: Final[str] = "Bandwidth (Input)"
    METRIC_BW_WGT: Final[str] = "Bandwidth (Weight)"
    METRIC_BW_OPT: Final[str] = "Bandwidth (Output)"
    METRIC_OI: Final[str] = "Operational Intensity"

    def __init__(self, model_config: dict, query_config: QueryConfig):
        self.model_conf: dict = model_config
        self.query_conf: QueryConfig = query_config
        self._hw_req_by_layers: Optional[dict[str, dict[str, Number]]] = None
```

### Abstract Methods (Must Implement)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_layer_list()` | `list[str]` | Return layer names in execution order |
| `get_num_blocks()` | `int` | Return total number of transformer blocks |
| `get_kvcache_size()` | `int` | Calculate KV-cache memory in bytes |
| `hw_req_by_layers` | `dict[str, dict[str, Number]]` | Property: hardware requirements per layer |

### Optional Override Methods

| Method | Default Behavior | Override When |
|--------|------------------|---------------|
| `normalize_config(config_dict)` | Returns unchanged | Setting architecture-specific defaults (e.g., `torch_dtype`) |
| `get_layer_num_blocks(layer)` | Returns `get_num_blocks()` | Layers appear in subset of blocks (MoE) |
| `get_extra_storage_req()` | Returns `[]` | Additional storage beyond weights/KV-cache |

### Operation Methods

These methods calculate and accumulate hardware metrics for common transformer operations:

#### `set_op_proj_req()` - Matrix Projection (GEMM)

```python
def set_op_proj_req(
    self,
    layer_entry: dict[str, Number],
    dim_m: int,      # Number of tokens (batch dimension)
    dim_n: int,      # Output dimension
    dim_k: int,      # Input dimension
    torch_dtype: str,
) -> None
```

**Use cases:** QKV projection, output projection, gate/up/down projections in FFN.

**Formulas:**

- Compute: `dim_m * dim_n * (dim_k * 2 - 1)` FLOPs
- Weight bandwidth: `dim_k * dim_n * dtype_width` bytes
- Input bandwidth: `dim_m * dim_k * dtype_width` bytes
- Output bandwidth: `dim_m * dim_n * dtype_width` bytes

#### `set_op_rope_req()` - Rotary Positional Embedding

```python
def set_op_rope_req(
    self,
    layer_entry: dict[str, Number],
    token_dims: int,    # head_dim * (num_q_heads + num_kv_heads)
    n_tokens: int,      # Number of tokens
    torch_dtype: str,
) -> None
```

**Formulas:**

- Compute: `token_dims * 3 * n_tokens` FLOPs (average 3 FLOPs per element)
- Input bandwidth: `token_dims * n_tokens * dtype_width` bytes
- Output bandwidth: `token_dims * n_tokens * dtype_width` bytes

#### `set_op_rmsnorm_req()` - RMS Normalization

```python
def set_op_rmsnorm_req(
    self,
    layer_entry: dict[str, Number],
    hidden_size: int,   # Model hidden dimension
    n_tokens: int,      # Number of tokens
    torch_dtype: str,
) -> None
```

**Formulas (per token, hidden_size = d):**

- Compute: `(d * 4 + 2) * n_tokens` FLOPs
  - Square input: d FLOPs
  - Sum squared: d - 1 FLOPs
  - Average + epsilon: 2 FLOPs
  - Square root: 1 FLOP
  - Normalize: d FLOPs
  - Scale by gamma: d FLOPs
- Weight bandwidth: `(hidden_size + 1) * dtype_width` bytes (gamma + epsilon)
- Input bandwidth: `hidden_size * n_tokens * dtype_width` bytes
- Output bandwidth: `hidden_size * n_tokens * dtype_width` bytes

#### `set_op_actmul_req()` - Fused Activation + Element-wise Multiply

```python
def set_op_actmul_req(
    self,
    layer_entry: dict[str, Number],
    intermediate_size: int,  # FFN intermediate dimension
    n_tokens: int,           # Number of tokens
    act_flops: int,          # FLOPs per element for activation (e.g., 4 for SiLU)
    torch_dtype: str,
) -> None
```

**Use cases:** SiLU gating in LLaMA FFN: `SiLU(gate_proj) * up_proj`

**Formulas:**

- Compute: `(act_flops + 1) * intermediate_size + n_tokens` FLOPs
- Input bandwidth: `intermediate_size * n_tokens * 2 * dtype_width` bytes (gate + up)
- Output bandwidth: `intermediate_size * n_tokens * dtype_width` bytes

#### `set_op_sum_req()` - Element-wise Summation

```python
def set_op_sum_req(
    self,
    layer_entry: dict[str, Number],
    num_elem: int,      # Number of elements per tensor
    num_tensors: int,   # Number of tensors to sum
    torch_dtype: str,
) -> None
```

**Use cases:** Residual connections.

**Formulas:**

- Compute: `num_elem * (num_tensors - 1)` FLOPs
- Input bandwidth: `num_elem * dtype_width * num_tensors` bytes
- Output bandwidth: `num_elem * dtype_width` bytes

#### `set_op_sdpa_req()` - Scaled Dot-Product Attention

```python
def set_op_sdpa_req(
    self,
    layer_entry: dict[str, Number],
    tensor_qo_dims: int,    # hidden_size (Q/O dimensions)
    tensor_kv_dims: int,    # head_dim * num_kv_heads
    torch_dtype: str,
) -> None
```

**Handles KV-cache:** Accounts for cached tokens in sequence length calculations.

**Formulas (per batch element):**

- GEMM P = QK^T: `qo_seq_len * kv_seq_len * (tensor_qo_dims * 2 - 1)` FLOPs
- GEMM O = SV: `qo_seq_len * tensor_kv_dims * (kv_seq_len * 2 - 1)` FLOPs
- Input bandwidth: Q tensor + KV tensors (including cached)
- Output bandwidth: O tensor

### Aggregation Methods

| Method | Description |
|--------|-------------|
| `new_req_dict()` | Create empty metrics dictionary for a layer |
| `calc_total()` | Aggregate layer metrics across all blocks |
| `calc_roofline()` | Calculate operational intensity for each layer |
| `print_summary()` | Print formatted table of all metrics |

## utils.py - Utilities

### Number Class

Represents numerical values with associated units and formatting for display.

```python
class Number:
    def __init__(self, unit: str, formatter: str, value: Optional[float] = None) -> None:
        self.value: Optional[float] = value
        self.unit: str = unit
        self.formatter: str = formatter

    def __str__(self) -> str:
        # Returns formatted string like "1.23 GFLOPs"

    def __add__(self, other: Number) -> Number:
        # Add two Numbers with the same unit

    def get_value_float(self) -> float:
        # Returns 0.0 if value is None

    def get_value_int(self) -> int:
        # Returns 0 if value is None
```

**Formatter strings:**

| Formatter | Description | Example Output |
|-----------|-------------|----------------|
| `"!.2h"` | Human-readable with SI prefixes | `1.23 G` |
| `"!.2k"` | Binary prefixes (KiB, MiB, etc.) | `1.23 Mi` |

**Usage example:**

```python
flops = Number("FLOPs", "!.2h", 1_234_567_890)
print(flops)  # "1.23 GFLOPs"

bytes_val = Number("B", "!.2k", 1_073_741_824)
print(bytes_val)  # "1.00 GiB"
```

### QueryConfig Class

Stores token-related configuration for inference queries.

```python
class QueryConfig:
    def __init__(self, cached_tokens: list[int], input_tokens: list[int]):
        self._t_mode: TransformerMode = TransformerMode.Text
        self._n_cached_tokens: list[int] = cached_tokens
        self._n_input_tokens: list[int] = input_tokens

    @property
    def t_mode(self) -> TransformerMode:
        # Returns transformer mode (Text or Vision)

    @property
    def n_cached_tokens(self) -> list[int]:
        # List of cached token counts per batch element

    @property
    def n_input_tokens(self) -> list[int]:
        # List of input token counts per batch element
```

### TransformerMode Enum

```python
class TransformerMode(Enum):
    Text = 1    # Text-only transformer
    Vision = 2  # Vision transformer mode
```

### Helper Functions

#### `torch_dtype_width(torch_type: str) -> int`

Maps PyTorch dtype strings to byte widths.

| Type Category | Types | Width |
|---------------|-------|-------|
| 8-bit int | `uint8`, `int8`, `quint8`, `qint8` | 1 |
| 16-bit int | `uint16`, `int16`, `short` | 2 |
| 32-bit int | `uint32`, `int32`, `int`, `qint32` | 4 |
| 64-bit int | `uint64`, `int64`, `long` | 8 |
| 8-bit float | `float8_e4m3fn`, `float8_e5m2` | 1 |
| 16-bit float | `float16`, `half`, `bfloat16` | 2 |
| 32-bit float | `float32`, `float` | 4 |
| 64-bit float | `float64`, `double` | 8 |

#### `act_flops(act: str) -> int`

Returns FLOP count per element for activation functions.

| Activation | FLOPs |
|------------|-------|
| `silu` | 4 |

## Hardware Metrics Formulas

### Operational Intensity (OI)

Operational Intensity is calculated as FLOPs per byte of memory traffic:

```text
OI = Compute / (Bandwidth_Weight + Bandwidth_Input + Bandwidth_Output)
```

Higher OI indicates compute-bound operations; lower OI indicates memory-bound operations.

### Total Model Metrics

`calc_total()` aggregates metrics across all layers and blocks:

```python
for layer in layers:
    total[metric] += layer_metrics[metric] * get_layer_num_blocks(layer)
```

This accounts for layers that appear in different numbers of blocks (e.g., MoE layers).
