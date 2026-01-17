# LLaMA-4 Model Support

This document describes how the Transformer Roofline Analyzer computes hardware requirements
for LLaMA-4 model architectures with Mixture of Experts (MoE).

## Overview

| Property | Value |
|----------|-------|
| Model Type | `llama4` |
| Architecture | `Llama4ForConditionalGeneration` |
| Parser | `Llama4ConfigParser` |
| Config Format | Nested (`text_config`, `vision_config`) |
| Key Feature | Mixture of Experts (MoE) |

**Supported Modes:**

- Text mode: Fully implemented
- Vision mode: Not yet implemented

## Architecture

LLaMA-4 uses an **interleaved MoE architecture** with:

- **MoE Layers:** Appear every `interleave_moe_layer_step` blocks
- **Dense Layers:** Fill the remaining blocks
- **Shared Expert:** Always activated alongside routed experts
- **Routed Experts:** `num_experts_per_tok` activated per token via router

### Block Distribution

With `num_hidden_layers = 48` and `interleave_moe_layer_step = 4`:

| Block Type | Count | Blocks |
|------------|-------|--------|
| MoE Blocks | 12 | 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48 |
| Dense Blocks | 36 | All others |

### MoE Block Structure

```text
Input
  │
  ├─► RMSNorm ─► QKV_Proj ─► RoPE ─► SDPA ─► O_Proj ─┐
  │                                                   │
  └───────────────────────────────────────────────────┼─► Add (Residual)
                                                      │
  ├─► RMSNorm ─► Router ─┬─► RoutedExp (×N) ─┐       │
  │                      │                    │       │
  │                      └─► SharedExp ───────┼─► Add │
  │                                           │       │
  └───────────────────────────────────────────┴───────┼─► Add (Residual)
                                                      │
                                                   Output
```

### Dense Block Structure

```text
Input
  │
  ├─► RMSNorm ─► QKV_Proj ─► RoPE ─► SDPA ─► O_Proj ─┐
  │                                                   │
  └───────────────────────────────────────────────────┼─► Add (Residual)
                                                      │
  ├─► RMSNorm ─► NonMoE_GateUp ─► ActMul ─► Down ────┐│
  │                                                   ││
  └───────────────────────────────────────────────────┴┼─► Add (Residual)
                                                       │
                                                    Output
```

## Layer Analysis

### Attention Block (All Blocks)

| Layer | Operation | Block Count |
|-------|-----------|-------------|
| `Attn - RMSNorm` | RMSNorm | All blocks |
| `Attn - QKV_Proj` | GEMM | All blocks |
| `Attn - RoPE` | Element-wise | All blocks |
| `Attn - SDPA` | Attention | All blocks |
| `Attn - O_Proj` | GEMM | All blocks |
| `Attn - ResidualAdd` | Element-wise | All blocks |

### MoE FFN Block

| Layer | Operation | Block Count |
|-------|-----------|-------------|
| `Ffn - RMSNorm` | RMSNorm | All blocks |
| `Ffn - Router` | GEMM | MoE blocks only |
| `Ffn - RoutedExp_GateUp_Proj` | GEMM | MoE blocks only |
| `Ffn - RoutedExp_ActMul` | Element-wise | MoE blocks only |
| `Ffn - RoutedExp_Down_Proj` | GEMM | MoE blocks only |
| `Ffn - SharedExp_GateUp_Proj` | GEMM | MoE blocks only |
| `Ffn - SharedExp_ActMul` | Element-wise | MoE blocks only |
| `Ffn - SharedExp_Down_Proj` | GEMM | MoE blocks only |
| `Ffn - RoutedSharedExpAdd` | Element-wise | MoE blocks only |

### Dense FFN Block

| Layer | Operation | Block Count |
|-------|-----------|-------------|
| `Ffn - NonMoE_GateUp_Proj` | GEMM | Dense blocks only |
| `Ffn - NonMoE_ActMul` | Element-wise | Dense blocks only |
| `Ffn - NonMoE_Down_Proj` | GEMM | Dense blocks only |
| `Ffn - ResidualAdd` | Element-wise | All blocks |

## Config Fields

The parser uses fields from `text_config` (nested under the root config):

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `num_hidden_layers` | int | Total transformer blocks | 48 |
| `hidden_size` | int | Model hidden dimension | 5120 |
| `head_dim` | int | Attention head dimension | 128 |
| `num_attention_heads` | int | Number of query heads | 40 |
| `num_key_value_heads` | int | Number of KV heads | 8 |
| `intermediate_size` | int | Expert FFN dimension | 8192 |
| `intermediate_size_mlp` | int | Dense FFN dimension | 16384 |
| `num_local_experts` | int | Total number of experts | 16 |
| `num_experts_per_tok` | int | Experts activated per token | 2 |
| `interleave_moe_layer_step` | int | MoE layer frequency | 4 |
| `hidden_act` | str | Activation function | `"silu"` |
| `vocab_size` | int | Vocabulary size | 202048 |
| `torch_dtype` | str | Data type | `"bfloat16"` |

## MoE Routing

### How Expert Selection Works

1. **Router** projects input to expert scores: `hidden_size → num_local_experts`
2. **Top-K Selection** chooses `num_experts_per_tok` experts per token
3. **Routed Experts** process tokens (each expert sees all tokens in simplified analysis)
4. **Shared Expert** always processes all tokens
5. **Outputs Combined** via weighted sum

### Metrics Computation for Routed Experts

The parser computes routed expert metrics by calling `set_op_*` methods
`num_experts_per_tok` times:

```python
for _ in range(text_config["num_experts_per_tok"]):
    self.set_op_proj_req(layer_entry=req_dict["Ffn - RoutedExp_GateUp_Proj"], ...)
    self.set_op_actmul_req(layer_entry=req_dict["Ffn - RoutedExp_ActMul"], ...)
    self.set_op_proj_req(layer_entry=req_dict["Ffn - RoutedExp_Down_Proj"], ...)
```

This accumulates metrics for all activated experts.

## Hardware Metrics Formulas

### Router

Projects input to expert logits:

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * num_local_experts * (hidden_size * 2 - 1)` FLOPs |
| Weight BW | `hidden_size * num_local_experts * dtype_width` bytes |
| Input BW | `n_tokens * hidden_size * dtype_width` bytes |
| Output BW | `n_tokens * num_local_experts * dtype_width` bytes |

### Routed Expert Layers

Per activated expert (multiply by `num_experts_per_tok` for total):

#### RoutedExp_GateUp_Proj

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * (intermediate_size * 2) * (hidden_size * 2 - 1)` FLOPs |
| Weight BW | `hidden_size * intermediate_size * 2 * dtype_width` bytes |
| Input BW | `n_tokens * hidden_size * dtype_width` bytes |
| Output BW | `n_tokens * intermediate_size * 2 * dtype_width` bytes |

#### RoutedExp_ActMul

| Metric | Formula |
|--------|---------|
| Compute | `(4 + 1) * intermediate_size + n_tokens` FLOPs |
| Weight BW | 0 bytes |
| Input BW | `intermediate_size * n_tokens * 2 * dtype_width` bytes |
| Output BW | `intermediate_size * n_tokens * dtype_width` bytes |

#### RoutedExp_Down_Proj

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * hidden_size * (intermediate_size * 2 - 1)` FLOPs |
| Weight BW | `intermediate_size * hidden_size * dtype_width` bytes |
| Input BW | `n_tokens * intermediate_size * dtype_width` bytes |
| Output BW | `n_tokens * hidden_size * dtype_width` bytes |

### Shared Expert Layers

Same formulas as routed experts, but computed only once (not multiplied by
`num_experts_per_tok`).

### Dense FFN Layers (NonMoE)

Uses `intermediate_size_mlp` instead of `intermediate_size`:

| Layer | Key Difference |
|-------|----------------|
| NonMoE_GateUp_Proj | Output dim: `intermediate_size_mlp * 2` |
| NonMoE_ActMul | Size: `intermediate_size_mlp` |
| NonMoE_Down_Proj | Input dim: `intermediate_size_mlp` |

## Storage Requirements

### Weight Bandwidth vs Storage

The analyzer distinguishes between:

1. **Weight Bandwidth (in metrics table):** Weights loaded during inference
   - Only accounts for activated experts (`num_experts_per_tok`)

2. **Additional Expert Storage:** Reported separately
   - Inactive expert weights that must be stored but not loaded per inference

### Storage Components

| Component | Formula |
|-----------|---------|
| Weights | Sum of weight bandwidths (activated experts only) |
| KV-Cache | `kv_seq_len * tensor_kv_dims * 2 * dtype_width * num_blocks` |
| Additional Experts | `expert_size * (num_local_experts - num_experts_per_tok) * num_moe_blocks` |
| Embedding Table | `hidden_size * vocab_size * dtype_width` |

Where:

- `expert_size = hidden_size * intermediate_size * dtype_width * 3` (gate, up, down)
- `num_moe_blocks = num_hidden_layers // interleave_moe_layer_step`

### Example Storage Breakdown

For a model with 16 experts, 2 activated per token, 12 MoE blocks:

```text
Additional Experts = expert_size * (16 - 2) * 12
                   = expert_size * 14 * 12
                   = expert_size * 168
```

## Block Count Handling

The `get_layer_num_blocks()` method returns different counts based on layer type:

```python
def get_layer_num_blocks(self, layer: str) -> int:
    if "Ffn - RoutedExp" in layer or "Ffn - SharedExp" in layer or "Ffn - RoutedShared" in layer:
        # MoE layers: only in MoE blocks
        return num_blocks // interleave_moe_layer_step
    elif "Ffn - NonMoE" in layer:
        # Dense layers: only in non-MoE blocks
        return num_blocks - (num_blocks // interleave_moe_layer_step)
    else:
        # Attention and common layers: all blocks
        return num_blocks
```

This ensures `calc_total()` correctly aggregates metrics across the model.

## Example Output

```bash
./transformer_roofline_analyzer --cached-tokens 1024 --input-tokens 1 -- meta-llama/Llama-4-Scout-17B-16E-Instruct
```

The output shows:

- Per-layer metrics with block counts (e.g., "12 / 48" for MoE layers)
- Total metrics aggregated correctly
- Storage breakdown including additional expert weights

## Related Documentation

- [Parser Implementation](../parsers/llama4.py)
- [LLaMA-2/3 Documentation](llama.md) - For comparison with dense architecture
- [Base Parser Framework](../core/README.md)
- [Adding New Models](../parsers/README.md)
