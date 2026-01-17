# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Transformer Roofline Analyzer is a CLI tool that estimates compute (FLOPs) and memory bandwidth requirements for transformer model architectures. It accepts either HuggingFace model names (e.g., `meta-llama/Llama-2-7b-hf`) or local `config.json` files. The tool performs roofline analysis to understand hardware resource demands and performance trade-offs during model inference.

## Development Setup

### Prerequisites

- Python >= 3.10
- Poetry >= 2.0.0

### Installation

```bash
poetry install
```

### Virtual Environment

**IMPORTANT**: Always use the Python executable from the Poetry virtual environment when running any Python program or script:

```bash
# Activate the virtual environment first
poetry shell

# Then run Python scripts or the CLI tool
python script.py
./transformer_roofline_analyzer [OPTIONS] -- <model_name_or_config>

# Alternatively, use poetry run to run commands in the virtual environment
poetry run python script.py
```

### Dependency Management

**IMPORTANT**: Always use Poetry commands for managing dependencies and the virtual environment:

```bash
# Add a new dependency
poetry add <package>

# Add a dev dependency
poetry add --group dev <package>

# Remove a dependency
poetry remove <package>

# Update dependencies
poetry update

# Show installed packages
poetry show

# Export requirements (if needed)
poetry export -f requirements.txt --output requirements.txt
```

Do NOT use `pip install` directly, as it bypasses Poetry's dependency resolution and lock file.

### Running the CLI

```bash
./transformer_roofline_analyzer [OPTIONS] -- <model_name_or_config>

# Using HuggingFace model name
./transformer_roofline_analyzer --cached-tokens 1048576 --input-tokens 1 -- meta-llama/Llama-2-7b-hf

# Using local config file
./transformer_roofline_analyzer --cached-tokens 1048576 --input-tokens 1 -- path/to/config.json

# Multiple queries
./transformer_roofline_analyzer --cached-tokens 1048576 1024 --input-tokens 1 1 -- meta-llama/Llama-2-7b-hf

# Batched queries
./transformer_roofline_analyzer --cached-tokens 1024 --input-tokens 1 --batch-size 2 -- meta-llama/Llama-2-7b-hf
```

The tool accepts HuggingFace model names (downloads config automatically, cached by transformers library) or local `config.json` files. Default `torch_dtype` is `float16` for models that don't specify it.

## Common Commands

### Testing

```bash
# Run all tests (verbose mode configured in pyproject.toml)
pytest

# Run unit tests only
pytest tests/unit/ -v

# Run end-to-end tests only
pytest tests/end-to-end/ -v

# Debug E2E tests by printing actual output
pytest tests/end-to-end/ --print-actual-output
```

For detailed test documentation, see [tests/README.md](tests/README.md) and [tests/CLAUDE.md](tests/CLAUDE.md).

### Code Quality

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Install pre-commit hooks
pre-commit install

# Format code with black
black .

# Sort imports with isort
isort .

# Lint with ruff
ruff check --fix

# Type checking with pyright
pyright
```

## Architecture

### Core Components

The codebase follows a parser-based architecture for analyzing different transformer model types:

1. **Base Parser Framework** ([core/base_parser.py](core/base_parser.py))
   - `BaseModelConfigParser`: Abstract base class defining the framework for all model parsers
   - Provides common methods for calculating hardware metrics (FLOPs, bandwidth, operational intensity)
   - Includes operation-specific methods: `set_op_proj_req()`, `set_op_rope_req()`, `set_op_rmsnorm_req()`, `set_op_actmul_req()`, `set_op_sum_req()`, `set_op_sdpa_req()`
   - Provides `normalize_config()` class method for architecture-specific config defaults
   - Handles tabulated output and storage requirement calculations

2. **Model-Specific Parsers** ([parsers/](parsers/))
   - `LlamaConfigParser` ([parsers/llama.py](parsers/llama.py)): Handles LLaMA-2/LLaMA-3 architectures
   - `Llama4ConfigParser` ([parsers/llama4.py](parsers/llama4.py)): Handles LLaMA-4 architectures with MoE support
   - Each parser implements:
     - `get_layer_list()`: Returns layer names for the model architecture
     - `get_num_blocks()`: Returns number of transformer blocks
     - `get_kvcache_size()`: Calculates KV-cache memory requirements
     - `hw_req_by_layers`: Property that lazily computes hardware requirements per layer

3. **Utilities** ([core/utils.py](core/utils.py))
   - `Number`: Custom class for handling numerical values with units and formatting
   - `QueryConfig`: Stores token-related configuration (cached tokens, input tokens)
   - `TransformerMode`: Enum for text vs. vision modes
   - `torch_dtype_width()`: Maps PyTorch dtypes to byte widths
   - `act_flops()`: Returns FLOP count for activation functions

4. **CLI Entry Point** ([transformer_roofline_analyzer](transformer_roofline_analyzer))
   - Parser registry (`PARSER_REGISTRY`) maps model types to parser classes
   - Validates command-line arguments
   - Loads HuggingFace config.json and delegates to appropriate parser

### Key Design Patterns

**Parser Registry Pattern**: The `PARSER_REGISTRY` dictionary in [transformer_roofline_analyzer](transformer_roofline_analyzer) maps `model_type` strings from HuggingFace configs to parser classes.

**Lazy Initialization**: Hardware requirements are computed once and cached in `_hw_req_by_layers`. The `hw_req_by_layers` property checks if the cache exists; if not, it computes all layer metrics and stores them.

**Layer-wise Metrics Aggregation**: Each parser computes metrics per layer, then `calc_total()` aggregates across all blocks, accounting for layers that may not appear in every block (e.g., MoE layers in LLaMA-4).

### Hardware Metrics Computed

For each layer, the parsers calculate:

- **Compute**: Total FLOPs (floating-point operations)
- **Bandwidth (Weight)**: Memory traffic for loading model weights
- **Bandwidth (Input)**: Memory traffic for reading input tensors
- **Bandwidth (Output)**: Memory traffic for writing output tensors
- **Operational Intensity**: FLOPs per byte (compute / total bandwidth)

### Testing Strategy

Tests are organized into two categories:

1. **Unit Tests** ([tests/unit/](tests/unit/))
   - Test individual functions and classes in isolation
   - Verify mathematical formulas for hardware metrics
   - Fast execution, no external dependencies

2. **End-to-End Tests** ([tests/end-to-end/](tests/end-to-end/))
   - JSON-driven test discovery using `test.json` files
   - Run the CLI and compare output against expected files
   - Support both local configs and HuggingFace model names
   - Use `pytest --print-actual-output` to debug test outputs

See [tests/README.md](tests/README.md) for detailed documentation on adding new tests.

## Code Style

The project uses:

- **black** (line length: 100) for formatting
- **isort** with black profile for import sorting
- **ruff** for linting
- **pyright** for type checking
- **markdownlint** for documentation

Pre-commit hooks enforce these styles automatically.

## Development Guidelines

### Readability and Extensibility

**IMPORTANT**: Maintain good readability and clear architecture for extensibility when adding support for new model architectures. The codebase is designed to easily accommodate new transformer variants.

### Reference Config Requirement

**IMPORTANT**: Before planning or implementing any feature that parses model configs, you **must** first read a reference config file to understand the actual structure and fields available. Reference configs can be obtained from:

- HuggingFace Hub (download via `AutoConfig.from_pretrained()`)

If you cannot find a suitable reference config for the architecture you're working with, ask the developer to provide one or point to where it can be found.

### Explicit Architecture Handling

**IMPORTANT**: For model architecture-dependent operations, always explicitly decode the architecture name from the config and take corresponding actions. HuggingFace configs provide two relevant fields:

- `architectures`: A list of architecture class names (e.g., `["LlamaForCausalLM"]`, `["Llama4ForConditionalGeneration"]`)
- `model_type`: A string identifier (e.g., `"llama"`, `"llama4"`)

**Never use a catch-all `else` clause as a fallback to an arbitrary architecture.** The `else` branch should always raise an error for unsupported architectures.

```python
# CORRECT: Explicitly handle each supported architecture, raise error for unknown
architectures = config_dict.get("architectures", [])
if "LlamaForCausalLM" in architectures:
    # Handle Llama 2/3 specific logic
    ...
elif "Llama4ForConditionalGeneration" in architectures:
    # Handle Llama 4 specific logic
    ...
else:
    raise ValueError(f"Unsupported architecture: {architectures}")

# CORRECT: Using PARSER_REGISTRY with explicit error for unknown model_type
model_type = config.get("model_type", "").lower()
parser_cls = PARSER_REGISTRY.get(model_type)
if parser_cls is None:
    raise NotImplementedError(f"No parser for model_type: {model_type}")

# INCORRECT: Using else as fallback to arbitrary architecture
if "Llama4ForConditionalGeneration" in architectures:
    # Handle Llama 4
    ...
else:
    # Assumes everything else is Llama 2/3 - DON'T DO THIS
    ...
```

This ensures:

1. New architectures are explicitly added with intentional support
2. Unsupported architectures fail clearly with informative error messages
3. No silent incorrect behavior from mismatched architecture handling

## Adding New Model Support

To support a new transformer architecture:

1. Create a new parser in `parsers/` (e.g., `parsers/newmodel.py`)
2. Extend `BaseModelConfigParser`
3. Implement required methods:
   - `get_layer_list()`: Define the sequence of operations
   - `get_num_blocks()`: Extract from model config
   - `get_kvcache_size()`: Calculate based on architecture
   - `hw_req_by_layers`: Use `set_op_*` methods to populate metrics
4. Optionally override:
   - `normalize_config()`: Set architecture-specific config defaults (e.g., default `torch_dtype`)
   - `get_layer_num_blocks()`: Return different block counts for layers appearing in subset of blocks (e.g., MoE layers)
   - `get_extra_storage_req()`: Return additional storage requirements beyond weights/KV-cache
5. Add to `PARSER_REGISTRY` in [transformer_roofline_analyzer](transformer_roofline_analyzer)
6. Export from `parsers/__init__.py`
7. Add test cases with config files and expected outputs

### Available Operation Methods

The base class provides these methods for calculating hardware metrics:

- `set_op_proj_req()`: Matrix projection (GEMM) operations
- `set_op_rope_req()`: Rotary positional embeddings
- `set_op_rmsnorm_req()`: RMSNorm layer normalization
- `set_op_actmul_req()`: Fused activation + element-wise multiplication (e.g., SiLU gating)
- `set_op_sum_req()`: Element-wise summation/reduction
- `set_op_sdpa_req()`: Scaled Dot-Product Attention (handles KV-cache)

## Important Notes

- All model config files must follow HuggingFace `config.json` schema
- The tool analyzes inference workloads, not training
- KV-cache calculations assume standard attention mechanisms
- MoE models (like LLaMA-4) require special handling for routed vs. shared experts
