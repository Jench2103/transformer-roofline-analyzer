# Transformer Roofline Analyzer

**Transformer Roofline Analyzer** is a CLI tool that estimates the compute (FLOPs) and memory bandwidth requirements of each layer—and the entire model—for transformer architectures. It accepts either HuggingFace model names (e.g., `meta-llama/Llama-2-7b-hf`) or local `config.json` files. The tool is particularly useful for analyzing hardware resource demands and performance trade-offs during model inference.

## ✨ Features

- Accepts HuggingFace model names or local `config.json` files
- Parses HuggingFace-compatible configurations with the following model types:
  - `llama` ([schema](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py))
  - `llama4` ([schema](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/configuration_llama4.py))
- Reports:
  - Compute (FLOPs)
  - Bandwidth: weights, inputs, outputs
  - Operational intensity (FLOPs/Byte)
  - Minimum data storage requirement
- Layer-wise and model-level summaries.
- Supports batching and KV-cache estimation.
- Useful for performance roofline modeling.

## 🚀 Getting Started

### Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management.

#### Requirements

- Python ≥ 3.10
- Poetry ≥ 2.0.0

#### Setup

```shell
# Clone the repo
git clone https://github.com/Jench2103/transformer-roofline-analyzer.git
cd transformer-roofline-analyzer

# Install dependencies using Poetry
poetry install

# Activate the virtual environment
eval $(poetry env activate)
```

#### Updating

After pulling new changes, reinstall the package to update the CLI:

```shell
git pull
poetry install
```

### Usage

```shell
transformer-roofline-analyzer [OPTIONS] -- <model_name_or_config>
```

The tool accepts two types of input:

1. **HuggingFace Model Names**: Automatically downloads config from HuggingFace Hub
   - Example: `meta-llama/Llama-2-7b-hf`
   - Requires internet connection on first use
   - Configs are cached by the transformers library

2. **Local Config Files**: Uses existing `config.json` files
   - Example: `path/to/config.json`
   - No internet required
   - Useful for offline analysis or custom configs

#### Example: Single Query with HuggingFace Model Name

Analyze a single query with 1,048,576 cached tokens (in KV cache) and 1 input token:

```shell
transformer-roofline-analyzer --cached-tokens 1048576 --input-tokens 1 -- meta-llama/Llama-2-7b-hf
```

#### Example: Single Query with Local Config File

```shell
transformer-roofline-analyzer --cached-tokens 1048576 --input-tokens 1 -- Llama-4-Scout-17B-16E-config.json
```

Sample output:

```text
| Node                        |  Block Count  |       Compute |   Bandwidth (Weight) |   Bandwidth (Input) |   Bandwidth (Output) |   Operational Intensity |
|-----------------------------|---------------|---------------|----------------------|---------------------|----------------------|-------------------------|
| Attn - RMSNorm              |    48 / 48    |  20.48 kFLOPs |            10.00 KiB |           10.00 KiB |            10.00 KiB |     666.69 mFLOPs/Bytes |
| Attn - QKV_Proj             |    48 / 48    |  73.39 MFLOPs |            70.00 MiB |           10.00 KiB |            14.00 KiB |     999.57 mFLOPs/Bytes |
| Attn - RoPE                 |    48 / 48    |  18.43 kFLOPs |               0.00 B |           12.00 KiB |            12.00 KiB |     750.00 mFLOPs/Bytes |
| Attn - SDPA                 |    48 / 48    |  12.88 GFLOPs |               0.00 B |            4.00 GiB |            10.00 KiB |        3.00 FLOPs/Bytes |
| Attn - O_Proj               |    48 / 48    |  52.42 MFLOPs |            50.00 MiB |           10.00 KiB |            10.00 KiB |     999.51 mFLOPs/Bytes |
| Attn - ResidualAdd          |    48 / 48    |   5.12 kFLOPs |               0.00 B |           20.00 KiB |            10.00 KiB |     166.67 mFLOPs/Bytes |
| Ffn - RMSNorm               |    48 / 48    |  20.48 kFLOPs |            10.00 KiB |           10.00 KiB |            10.00 KiB |     666.69 mFLOPs/Bytes |
| Ffn - Router                |    48 / 48    | 163.82 kFLOPs |           160.00 KiB |           10.00 KiB |              32.00 B |     940.91 mFLOPs/Bytes |
| Ffn - RoutedExp_GateUp_Proj |    48 / 48    | 167.76 MFLOPs |           160.00 MiB |           10.00 KiB |            32.00 KiB |     999.65 mFLOPs/Bytes |
| Ffn - RoutedExp_ActMul      |    48 / 48    |  40.96 kFLOPs |               0.00 B |           32.00 KiB |            16.00 KiB |     833.35 mFLOPs/Bytes |
| Ffn - RoutedExp_Down_Proj   |    48 / 48    |  83.88 MFLOPs |            80.00 MiB |           16.00 KiB |            10.00 KiB |     999.62 mFLOPs/Bytes |
| Ffn - SharedExp_GateUp_Proj |    48 / 48    | 167.76 MFLOPs |           160.00 MiB |           10.00 KiB |            32.00 KiB |     999.65 mFLOPs/Bytes |
| Ffn - SharedExp_ActMul      |    48 / 48    |  40.96 kFLOPs |               0.00 B |           32.00 KiB |            16.00 KiB |     833.35 mFLOPs/Bytes |
| Ffn - SharedExp_Down_Proj   |    48 / 48    |  83.88 MFLOPs |            80.00 MiB |           16.00 KiB |            10.00 KiB |     999.62 mFLOPs/Bytes |
| Ffn - RoutedSharedExpAdd    |    48 / 48    |   5.12 kFLOPs |               0.00 B |           20.00 KiB |            10.00 KiB |     166.67 mFLOPs/Bytes |
| Ffn - ResidualAdd           |    48 / 48    |   5.12 kFLOPs |               0.00 B |           20.00 KiB |            10.00 KiB |     166.67 mFLOPs/Bytes |
|                             |               |               |                      |                     |                      |                         |
| Total (48 Blocks)           |      N/A      | 648.64 GFLOPs |            28.13 GiB |          192.01 GiB |             9.94 MiB |        2.74 FLOPs/Bytes |

Minimum Storage Requirement: (Weights) 28.13 GiB + (KV-cache) 192.00 GiB + (Additional Experts) 168.75 GiB + (Embedding Table) 1.93 GiB = 390.81 GiB
```

#### Example: Multiple Queries with Varying Tokens

Analyze two queries with different numbers of cached and input tokens:

```shell
transformer-roofline-analyzer --cached-tokens 1048576 1024 --input-tokens 1 1 -- meta-llama/Llama-4-Scout-17B-16E
```

#### Example: Batched Queries with Identical Token Counts

Analyze two queries with the same number of cached and input tokens:

```shell
transformer-roofline-analyzer --cached-tokens 1024 --input-tokens 1 --batch-size 2 -- meta-llama/Llama-4-Scout-17B-16E
```

#### Help Message

```text
usage: transformer-roofline-analyzer [-h] [--cached-tokens CACHED_TOKENS [CACHED_TOKENS ...]] [--input-tokens INPUT_TOKENS [INPUT_TOKENS ...]] [--batch-size BATCH_SIZE] model_name_or_config

positional arguments:
  model_name_or_config  HuggingFace model name (e.g., 'meta-llama/Llama-2-7b-hf') or path to local config.json file

options:
  -h, --help            show this help message and exit
  --cached-tokens CACHED_TOKENS [CACHED_TOKENS ...]
                        List of token counts already present in the KV-cache for each query in the batch (e.g., cached context tokens used during attention).
  --input-tokens INPUT_TOKENS [INPUT_TOKENS ...]
                        List of input token counts for each query in the batch (e.g., prompt or the tokens generated by the previous inference).
  --batch-size BATCH_SIZE
                        The number of queries in a batch.
```

#### Notes

- If `--batch-size` is not provided, it is inferred from the length of `--cached-tokens` and `--input-tokens`.
- Both `--cached-tokens` and `--input-tokens` support per-query customization by accepting space-separated lists.
- Both `--cached-tokens` and `--input-tokens` must have the same number of elements.
- If `--batch-size` is provided, it must be a multiple of the number of elements in `--cached-tokens` and `--input-tokens`.

## 📍 Roadmap

- [x] Support model type `llama4` for models using earlier LLaMA architectures.
- [x] Support model type `llama` for models using earlier LLaMA architectures (e.g., LLaMA-2, LLaMA-3).
- [x] Support calculating the minimum storage requirement based on model configurations and given prompt-related parameters.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or issue.

## 📝 License

This project is licensed under the [MIT License](./LICENSE).
