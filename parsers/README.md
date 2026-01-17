# Model Parsers

This package contains model-specific parsers that implement hardware requirements analysis
for different transformer architectures.

## Table of Contents

- [Overview](#overview)
- [Parser Framework](#parser-framework)
- [Existing Parsers](#existing-parsers)
- [Adding a New Parser](#adding-a-new-parser)
- [Minimal Parser Template](#minimal-parser-template)

## Overview

Each parser extends `BaseModelConfigParser` and implements architecture-specific logic for:

- Layer structure and execution order
- KV-cache size calculations
- Hardware requirements per layer (FLOPs, bandwidth)
- Storage requirements (weights, embeddings, etc.)

## Parser Framework

### Parser Registry

The `PARSER_REGISTRY` in `transformer_roofline_analyzer` maps `model_type` strings from
HuggingFace configs to parser classes:

```python
PARSER_REGISTRY: dict[str, type[BaseModelConfigParser]] = {
    "llama": LlamaConfigParser,
    "llama4": Llama4ConfigParser,
}
```

When the CLI runs, it:

1. Loads the model config (from HuggingFace or local file)
2. Extracts `model_type` from the config
3. Looks up the parser class in `PARSER_REGISTRY`
4. Instantiates the parser with the config and query settings

### Required Methods

Every parser must implement these abstract methods:

| Method | Return Type | Description |
|--------|-------------|-------------|
| `get_layer_list()` | `list[str]` | Layer names in forward pass order |
| `get_num_blocks()` | `int` | Total transformer blocks in model |
| `get_kvcache_size()` | `int` | Total KV-cache size in bytes |
| `hw_req_by_layers` | `dict[str, dict[str, Number]]` | Hardware metrics per layer (property) |

### Optional Methods

Override these when the default behavior doesn't apply:

| Method | Default | Override When |
|--------|---------|---------------|
| `normalize_config(config_dict)` | Returns unchanged | Need to set defaults (e.g., `torch_dtype`) |
| `get_layer_num_blocks(layer)` | Returns `get_num_blocks()` | Some layers appear in fewer blocks (MoE) |
| `get_extra_storage_req()` | Returns `[]` | Need additional storage entries (embeddings, extra experts) |

## Existing Parsers

### LlamaConfigParser (llama.py)

**Supports:** LLaMA-2, LLaMA-3 architectures

**Model type:** `"llama"`

**Architectures:** `["LlamaForCausalLM"]`

**Characteristics:**

- Standard dense transformer architecture
- Serves as reference implementation for new parsers
- Simple config structure with all fields at root level

**Layer structure:**

```python
[
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
```

**Config fields used:**

- `num_hidden_layers`: Number of transformer blocks
- `hidden_size`: Model hidden dimension
- `num_attention_heads`: Number of query heads
- `num_key_value_heads`: Number of KV heads (for GQA)
- `intermediate_size`: FFN intermediate dimension
- `hidden_act`: Activation function (e.g., `"silu"`)
- `vocab_size`: Vocabulary size (for embedding table)
- `torch_dtype`: Data type (default: `"float16"`)

### Llama4ConfigParser (llama4.py)

**Supports:** LLaMA-4 architecture with Mixture of Experts

**Model type:** `"llama4"`

**Architectures:** `["Llama4ForConditionalGeneration"]`

**Characteristics:**

- Mixture of Experts (MoE) architecture
- Interleaved MoE and dense layers
- Shared experts alongside routed experts
- Nested config structure (`text_config`, `vision_config`)
- Text and Vision modes (Vision mode not yet implemented)

**Layer structure (Text mode):**

```python
[
    "Attn - RMSNorm",
    "Attn - QKV_Proj",
    "Attn - RoPE",
    "Attn - SDPA",
    "Attn - O_Proj",
    "Attn - ResidualAdd",
    "Ffn - RMSNorm",
    "Ffn - Router",
    "Ffn - RoutedExp_GateUp_Proj",
    "Ffn - RoutedExp_ActMul",
    "Ffn - RoutedExp_Down_Proj",
    "Ffn - SharedExp_GateUp_Proj",
    "Ffn - SharedExp_ActMul",
    "Ffn - SharedExp_Down_Proj",
    "Ffn - RoutedSharedExpAdd",
    "Ffn - NonMoE_GateUp_Proj",
    "Ffn - NonMoE_ActMul",
    "Ffn - NonMoE_Down_Proj",
    "Ffn - ResidualAdd",
]
```

**MoE-specific behavior:**

- `get_layer_num_blocks()` returns different counts:
  - MoE layers: `num_blocks // interleave_moe_layer_step`
  - Non-MoE layers: `num_blocks - (num_blocks // interleave_moe_layer_step)`
  - Other layers: `num_blocks`

**Config fields used (under `text_config`):**

- `num_hidden_layers`: Number of transformer blocks
- `hidden_size`: Model hidden dimension
- `head_dim`: Attention head dimension
- `num_attention_heads`: Number of query heads
- `num_key_value_heads`: Number of KV heads
- `intermediate_size`: Expert FFN intermediate dimension
- `intermediate_size_mlp`: Non-MoE FFN intermediate dimension
- `num_local_experts`: Total number of experts
- `num_experts_per_tok`: Experts activated per token
- `interleave_moe_layer_step`: MoE layer frequency
- `hidden_act`: Activation function
- `vocab_size`: Vocabulary size
- `torch_dtype`: Data type

**Extra storage requirements:**

- Additional expert weights (experts beyond `num_experts_per_tok`)
- Embedding table

## Adding a New Parser

Follow these steps to add support for a new transformer architecture.

### Step 1: Read Reference Config

**Important:** Before implementing, obtain and study a reference config file:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("model-org/model-name")
print(config.to_dict())
```

Understand:

- Config structure (flat vs nested)
- Field names and their meanings
- Default values and optional fields

### Step 2: Create Parser File

Create `parsers/newmodel.py`:

```python
from core import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class NewModelConfigParser(BaseModelConfigParser):
    """Parser for NewModel transformer architecture."""
    pass
```

### Step 3: Implement Required Methods

#### `get_layer_list()`

Return layer names in forward pass execution order:

```python
def get_layer_list(self) -> list[str]:
    return [
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
```

#### `get_num_blocks()`

Extract block count from config:

```python
def get_num_blocks(self) -> int:
    return self.model_conf["num_hidden_layers"]
```

#### `get_kvcache_size()`

Calculate total KV-cache size:

```python
def get_kvcache_size(self) -> int:
    kvcache_size_per_block: int = 0
    batch_size: int = len(self.query_conf.n_cached_tokens)

    # Calculate dimensions
    head_dim: int = self.model_conf["hidden_size"] // self.model_conf["num_attention_heads"]
    tensor_kv_dims: int = head_dim * self.model_conf["num_key_value_heads"]
    torch_dtype: str = self.model_conf["torch_dtype"]

    # Sum across batch elements
    for query_idx in range(batch_size):
        kv_seq_len: int = (
            self.query_conf.n_cached_tokens[query_idx]
            + self.query_conf.n_input_tokens[query_idx]
        )
        # K and V tensors
        kvcache_size_per_block += (
            kv_seq_len * (tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)
        )

    return kvcache_size_per_block * self.get_num_blocks()
```

#### `hw_req_by_layers` Property

Compute hardware requirements for each layer:

```python
@property
def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
    # Return cached result if available
    if self._hw_req_by_layers is not None:
        return self._hw_req_by_layers.copy()

    # Initialize empty metrics for each layer
    req_dict: dict[str, dict[str, Number]] = {
        key: self.new_req_dict() for key in self.get_layer_list()
    }

    # Calculate metrics for each layer using set_op_* methods
    self.set_op_rmsnorm_req(
        layer_entry=req_dict["Attn - RMSNorm"],
        hidden_size=self.model_conf["hidden_size"],
        n_tokens=sum(self.query_conf.n_input_tokens),
        torch_dtype=self.model_conf["torch_dtype"],
    )

    # ... continue for all layers ...

    # Cache and return
    self._hw_req_by_layers = req_dict
    return self._hw_req_by_layers.copy()
```

### Step 4: Implement Optional Methods (if needed)

#### `normalize_config()`

Set default values:

```python
@classmethod
def normalize_config(cls, config_dict: dict) -> dict:
    """Set default torch_dtype if not present."""
    if "torch_dtype" not in config_dict:
        config_dict["torch_dtype"] = "float16"
    return config_dict
```

#### `get_layer_num_blocks()`

Handle layers that appear in subset of blocks:

```python
def get_layer_num_blocks(self, layer: str) -> int:
    if "MoE" in layer:
        return self.get_num_blocks() // self.model_conf["moe_layer_step"]
    return self.get_num_blocks()
```

#### `get_extra_storage_req()`

Report additional storage requirements:

```python
def get_extra_storage_req(self) -> list[tuple[str, Number]]:
    req_list: list[tuple[str, Number]] = []

    # Embedding table
    emb_size: int = (
        self.model_conf["hidden_size"]
        * self.model_conf["vocab_size"]
        * torch_dtype_width(self.model_conf["torch_dtype"])
    )
    req_list.append(("Embedding Table", Number("B", "!.2k", emb_size)))

    return req_list
```

### Step 5: Register Parser

Add to `PARSER_REGISTRY` in `transformer_roofline_analyzer`:

```python
from parsers import NewModelConfigParser

PARSER_REGISTRY: dict[str, type[BaseModelConfigParser]] = {
    "llama": LlamaConfigParser,
    "llama4": Llama4ConfigParser,
    "newmodel": NewModelConfigParser,
}
```

### Step 6: Export from Package

Add to `parsers/__init__.py`:

```python
from .newmodel import NewModelConfigParser
```

### Step 7: Add Tests

1. **Unit tests:** Create `tests/unit/test_newmodel_parser.py`

   ```python
   import pytest
   from parsers import NewModelConfigParser
   from core import QueryConfig

   @pytest.fixture
   def sample_config():
       return {
           "model_type": "newmodel",
           "num_hidden_layers": 32,
           "hidden_size": 4096,
           # ... other fields
       }

   def test_get_num_blocks(sample_config):
       query = QueryConfig(cached_tokens=[1024], input_tokens=[1])
       parser = NewModelConfigParser(sample_config, query)
       assert parser.get_num_blocks() == 32
   ```

2. **E2E tests:** Create `tests/end-to-end/newmodel/` directory with:
   - `config.json`: Sample model config
   - `test.json`: Test configuration
   - Expected output files

## Minimal Parser Template

Copy this template to start a new parser:

```python
from core import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class NewModelConfigParser(BaseModelConfigParser):
    """
    Parser for NewModel transformer architecture.

    Supports: NewModel-7B, NewModel-13B, etc.
    """

    @classmethod
    def normalize_config(cls, config_dict: dict) -> dict:
        """Set default torch_dtype if not present."""
        if "torch_dtype" not in config_dict:
            config_dict["torch_dtype"] = "float16"
        return config_dict

    def get_layer_list(self) -> list[str]:
        """Return layer names in execution order."""
        return [
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

    def get_num_blocks(self) -> int:
        """Return number of transformer blocks."""
        return self.model_conf["num_hidden_layers"]

    def get_kvcache_size(self) -> int:
        """Calculate KV-cache memory requirements."""
        kvcache_size_per_block: int = 0
        batch_size: int = len(self.query_conf.n_cached_tokens)
        head_dim: int = (
            self.model_conf["hidden_size"] // self.model_conf["num_attention_heads"]
        )
        tensor_kv_dims: int = head_dim * self.model_conf["num_key_value_heads"]
        torch_dtype: str = self.model_conf["torch_dtype"]

        for query_idx in range(batch_size):
            kv_seq_len: int = (
                self.query_conf.n_cached_tokens[query_idx]
                + self.query_conf.n_input_tokens[query_idx]
            )
            kvcache_size_per_block += (
                kv_seq_len * (tensor_kv_dims * 2) * torch_dtype_width(torch_dtype)
            )

        return kvcache_size_per_block * self.get_num_blocks()

    def get_extra_storage_req(self) -> list[tuple[str, Number]]:
        """Return additional storage requirements."""
        emb_size: int = (
            self.model_conf["hidden_size"]
            * self.model_conf["vocab_size"]
            * torch_dtype_width(self.model_conf["torch_dtype"])
        )
        return [("Embedding Table", Number("B", "!.2k", emb_size))]

    @property
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        """Compute hardware requirements per layer."""
        if self._hw_req_by_layers is not None:
            return self._hw_req_by_layers.copy()

        req_dict: dict[str, dict[str, Number]] = {
            key: self.new_req_dict() for key in self.get_layer_list()
        }
        conf: dict = self.model_conf
        head_dim: int = conf["hidden_size"] // conf["num_attention_heads"]

        # Attention block
        self.set_op_rmsnorm_req(
            layer_entry=req_dict["Attn - RMSNorm"],
            hidden_size=conf["hidden_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Attn - QKV_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=head_dim * (conf["num_attention_heads"] + conf["num_key_value_heads"] * 2),
            dim_k=conf["hidden_size"],
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_rope_req(
            layer_entry=req_dict["Attn - RoPE"],
            token_dims=head_dim * (conf["num_attention_heads"] + conf["num_key_value_heads"]),
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_sdpa_req(
            layer_entry=req_dict["Attn - SDPA"],
            tensor_qo_dims=conf["hidden_size"],
            tensor_kv_dims=int(head_dim * conf["num_key_value_heads"]),
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Attn - O_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=conf["hidden_size"],
            dim_k=conf["hidden_size"],
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_sum_req(
            layer_entry=req_dict["Attn - ResidualAdd"],
            num_elem=sum(self.query_conf.n_input_tokens) * conf["hidden_size"],
            num_tensors=2,
            torch_dtype=conf["torch_dtype"],
        )

        # FFN block
        self.set_op_rmsnorm_req(
            layer_entry=req_dict["Ffn - RMSNorm"],
            hidden_size=conf["hidden_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Ffn - GateUp_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=conf["intermediate_size"] * 2,
            dim_k=conf["hidden_size"],
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_actmul_req(
            layer_entry=req_dict["Ffn - ActMul"],
            intermediate_size=conf["intermediate_size"],
            n_tokens=sum(self.query_conf.n_input_tokens),
            act_flops=act_flops(conf["hidden_act"]),
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_proj_req(
            layer_entry=req_dict["Ffn - Down_Proj"],
            dim_m=sum(self.query_conf.n_input_tokens),
            dim_n=conf["hidden_size"],
            dim_k=conf["intermediate_size"],
            torch_dtype=conf["torch_dtype"],
        )
        self.set_op_sum_req(
            layer_entry=req_dict["Ffn - ResidualAdd"],
            num_elem=sum(self.query_conf.n_input_tokens) * conf["hidden_size"],
            num_tensors=2,
            torch_dtype=conf["torch_dtype"],
        )

        self._hw_req_by_layers = req_dict
        return self._hw_req_by_layers.copy()
```
