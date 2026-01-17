# LLaMA-2/3 Model Support

This document describes how the Transformer Roofline Analyzer computes hardware requirements
for LLaMA-2 and LLaMA-3 model architectures.

## Overview

| Property | Value |
|----------|-------|
| Model Type | `llama` |
| Architecture | `LlamaForCausalLM` |
| Parser | `LlamaConfigParser` |
| Config Format | Flat (all fields at root level) |

**Supported Models:**

Any HuggingFace model with `model_type: "llama"` is supported. The tool reads the model's
`config.json` and computes hardware requirements based on the configuration fields
(hidden_size, num_layers, etc.), so it works with any model size or fine-tuned variant.

## Architecture

LLaMA-2/3 uses a standard dense transformer architecture with the following characteristics:

- **Grouped Query Attention (GQA):** Fewer KV heads than query heads for efficiency
- **SwiGLU Activation:** Gated FFN with SiLU activation
- **RMSNorm:** Root Mean Square Layer Normalization (no bias)
- **Rotary Position Embeddings (RoPE):** Applied to Q and K tensors

### Transformer Block Structure

```text
Input
  │
  ├─► RMSNorm ─► QKV_Proj ─► RoPE ─► SDPA ─► O_Proj ─┐
  │                                                   │
  └───────────────────────────────────────────────────┼─► Add (Residual)
                                                      │
  ├─► RMSNorm ─► GateUp_Proj ─► ActMul ─► Down_Proj ─┐│
  │                                                   ││
  └───────────────────────────────────────────────────┴┼─► Add (Residual)
                                                       │
                                                    Output
```

## Layer Analysis

The parser analyzes the following layers in execution order:

### Attention Block

| Layer | Operation | Description |
|-------|-----------|-------------|
| `Attn - RMSNorm` | RMSNorm | Pre-attention normalization |
| `Attn - QKV_Proj` | GEMM | Query, Key, Value projections (fused) |
| `Attn - RoPE` | Element-wise | Rotary positional embeddings on Q and K |
| `Attn - SDPA` | Attention | Scaled dot-product attention with KV-cache |
| `Attn - O_Proj` | GEMM | Output projection |
| `Attn - ResidualAdd` | Element-wise | Residual connection |

### FFN Block

| Layer | Operation | Description |
|-------|-----------|-------------|
| `Ffn - RMSNorm` | RMSNorm | Pre-FFN normalization |
| `Ffn - GateUp_Proj` | GEMM | Gate and Up projections (fused) |
| `Ffn - ActMul` | Element-wise | SiLU activation with gating: `SiLU(gate) * up` |
| `Ffn - Down_Proj` | GEMM | Down projection |
| `Ffn - ResidualAdd` | Element-wise | Residual connection |

## Config Fields

The parser uses the following fields from the HuggingFace config:

| Field | Type | Description | Example (LLaMA-2-7B) |
|-------|------|-------------|----------------------|
| `num_hidden_layers` | int | Number of transformer blocks | 32 |
| `hidden_size` | int | Model hidden dimension | 4096 |
| `num_attention_heads` | int | Number of query heads | 32 |
| `num_key_value_heads` | int | Number of KV heads (GQA) | 32 (or 8 for GQA) |
| `intermediate_size` | int | FFN intermediate dimension | 11008 |
| `hidden_act` | str | Activation function | `"silu"` |
| `vocab_size` | int | Vocabulary size | 32000 |
| `torch_dtype` | str | Data type (default: `float16`) | `"bfloat16"` |

### Derived Values

- **head_dim:** `hidden_size / num_attention_heads`
- **tensor_kv_dims:** `head_dim * num_key_value_heads`

## Hardware Metrics Formulas

### Attention Layers

#### RMSNorm

For `hidden_size = d` and `n_tokens` tokens:

| Metric | Formula |
|--------|---------|
| Compute | `(d * 4 + 2) * n_tokens` FLOPs |
| Weight BW | `(d + 1) * dtype_width` bytes |
| Input BW | `d * n_tokens * dtype_width` bytes |
| Output BW | `d * n_tokens * dtype_width` bytes |

#### QKV Projection

Fused projection with output dimension `head_dim * (num_q_heads + 2 * num_kv_heads)`:

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * out_dim * (hidden_size * 2 - 1)` FLOPs |
| Weight BW | `hidden_size * out_dim * dtype_width` bytes |
| Input BW | `n_tokens * hidden_size * dtype_width` bytes |
| Output BW | `n_tokens * out_dim * dtype_width` bytes |

#### RoPE

Applies rotary embeddings to Q and K tensors:

| Metric | Formula |
|--------|---------|
| Compute | `token_dims * 3 * n_tokens` FLOPs |
| Weight BW | 0 bytes |
| Input BW | `token_dims * n_tokens * dtype_width` bytes |
| Output BW | `token_dims * n_tokens * dtype_width` bytes |

Where `token_dims = head_dim * (num_q_heads + num_kv_heads)`.

#### SDPA (Scaled Dot-Product Attention)

Accounts for KV-cache when processing cached tokens:

| Metric | Formula |
|--------|---------|
| Compute (QK^T) | `qo_seq_len * kv_seq_len * (hidden_size * 2 - 1)` FLOPs |
| Compute (SV) | `qo_seq_len * tensor_kv_dims * (kv_seq_len * 2 - 1)` FLOPs |
| Weight BW | 0 bytes |
| Input BW | Q tensor + KV tensors (including cached) |
| Output BW | O tensor |

Where:

- `qo_seq_len = n_input_tokens`
- `kv_seq_len = n_cached_tokens + n_input_tokens`

### FFN Layers

#### GateUp Projection

Fused gate and up projection with output dimension `intermediate_size * 2`:

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * (intermediate_size * 2) * (hidden_size * 2 - 1)` FLOPs |
| Weight BW | `hidden_size * intermediate_size * 2 * dtype_width` bytes |
| Input BW | `n_tokens * hidden_size * dtype_width` bytes |
| Output BW | `n_tokens * intermediate_size * 2 * dtype_width` bytes |

#### ActMul (SiLU Gating)

Fused `SiLU(gate) * up` operation:

| Metric | Formula |
|--------|---------|
| Compute | `(4 + 1) * intermediate_size + n_tokens` FLOPs |
| Weight BW | 0 bytes |
| Input BW | `intermediate_size * n_tokens * 2 * dtype_width` bytes |
| Output BW | `intermediate_size * n_tokens * dtype_width` bytes |

Note: SiLU requires 4 FLOPs per element.

#### Down Projection

| Metric | Formula |
|--------|---------|
| Compute | `n_tokens * hidden_size * (intermediate_size * 2 - 1)` FLOPs |
| Weight BW | `intermediate_size * hidden_size * dtype_width` bytes |
| Input BW | `n_tokens * intermediate_size * dtype_width` bytes |
| Output BW | `n_tokens * hidden_size * dtype_width` bytes |

## Storage Requirements

The analyzer reports total storage requirements:

| Component | Formula |
|-----------|---------|
| Weights | Sum of all weight bandwidths across layers and blocks |
| KV-Cache | `kv_seq_len * tensor_kv_dims * 2 * dtype_width * num_blocks` |
| Embedding Table | `hidden_size * vocab_size * dtype_width` |

## Example Output

```bash
./transformer_roofline_analyzer --cached-tokens 1024 --input-tokens 1 -- meta-llama/Llama-2-7b-hf
```

The output shows per-layer metrics and totals for:

- Compute (FLOPs)
- Bandwidth (Weight, Input, Output)
- Operational Intensity (FLOPs/Byte)

## Related Documentation

- [Parser Implementation](../parsers/llama.py)
- [Base Parser Framework](../core/README.md)
- [Adding New Models](../parsers/README.md)
