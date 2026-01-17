# Contributing to Transformer Roofline Analyzer

This guide provides comprehensive documentation for developers contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Running the CLI](#running-the-cli)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Architecture Overview](#architecture-overview)
- [Adding New Model Support](#adding-new-model-support)
- [Code Style Guidelines](#code-style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Documentation Maintenance](#documentation-maintenance)

## Development Setup

### Prerequisites

- Python >= 3.10
- Poetry >= 2.0.0

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/transformer-roofline-calculator.git
cd transformer-roofline-calculator

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Virtual Environment

**Important:** Always use the Python executable from the Poetry virtual environment:

```bash
# Option 1: Activate the virtual environment first
poetry shell
python script.py
./transformer_roofline_analyzer [OPTIONS] -- <model_name_or_config>

# Option 2: Use poetry run prefix
poetry run python script.py
poetry run ./transformer_roofline_analyzer [OPTIONS] -- <model>
```

### Dependency Management

**Important:** Always use Poetry commands for managing dependencies. Do NOT use `pip install`
directly, as it bypasses Poetry's dependency resolution and lock file.

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

## Running the CLI

The CLI accepts HuggingFace model names (downloads config automatically) or local `config.json`
files. Default `torch_dtype` is `float16` for models that don't specify it.

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

# Show help
./transformer_roofline_analyzer --help
```

## Code Quality

### Pre-commit Hooks

**Important:** Always use pre-commit to run code formatting and linting. Do not run individual
tools manually.

```bash
# Run all code quality checks (formatting, linting, type checking)
pre-commit run --all-files

# Install pre-commit hooks (run once after cloning)
pre-commit install
```

### Tools Used

- **black** (line length: 100) - Code formatting
- **isort** (black profile) - Import sorting
- **ruff** - Linting
- **pyright** - Type checking
- **markdownlint** - Markdown linting

### Type Annotations

**Important:** Type hints are enforced via pyright. All functions and variables must have
type annotations.

```python
# Good: Explicit type annotations
def get_layer_count(config: dict) -> int:
    num_layers: int = config["num_hidden_layers"]
    return num_layers

# Bad: Missing type annotations
def get_layer_count(config):
    num_layers = config["num_hidden_layers"]
    return num_layers
```

## Testing

### Running Tests

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

### Test Organization

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes in isolation
   - Verify mathematical formulas for hardware metrics
   - Fast execution, no external dependencies

2. **End-to-End Tests** (`tests/end-to-end/`)
   - JSON-driven test discovery using `test.json` files
   - Run the CLI and compare output against expected files
   - Support both local configs and HuggingFace model names

For detailed test documentation, see [tests/README.md](tests/README.md).

## Architecture Overview

### Core Components

The codebase follows a parser-based architecture for analyzing transformer model types:

```text
transformer-roofline-calculator/
├── transformer_roofline_analyzer    # CLI entry point with parser registry
├── core/
│   ├── base_parser.py              # Abstract base class for parsers
│   └── utils.py                    # Number, QueryConfig, utilities
└── parsers/
    ├── llama.py                    # LLaMA-2/3 parser
    └── llama4.py                   # LLaMA-4 parser with MoE support
```

1. **Base Parser Framework** (`core/base_parser.py`)
   - `BaseModelConfigParser`: Abstract base class for all model parsers
   - Provides operation methods: `set_op_proj_req()`, `set_op_rope_req()`, `set_op_rmsnorm_req()`,
     `set_op_actmul_req()`, `set_op_sum_req()`, `set_op_sdpa_req()`
   - Handles tabulated output and storage requirement calculations

2. **Model-Specific Parsers** (`parsers/`)
   - `LlamaConfigParser`: LLaMA-2/LLaMA-3 architectures
   - `Llama4ConfigParser`: LLaMA-4 with MoE support

3. **Utilities** (`core/utils.py`)
   - `Number`: Numerical values with units and formatting
   - `QueryConfig`: Token-related configuration
   - `TransformerMode`: Text vs. vision modes
   - Helper functions: `torch_dtype_width()`, `act_flops()`

4. **CLI Entry Point** (`transformer_roofline_analyzer`)
   - `PARSER_REGISTRY`: Maps model types to parser classes
   - Loads HuggingFace configs and delegates to parsers

For detailed module documentation, see:

- [core/README.md](core/README.md) - Core module documentation
- [parsers/README.md](parsers/README.md) - Parser framework and extension guide

### Key Design Patterns

**Parser Registry Pattern:** The `PARSER_REGISTRY` dictionary maps `model_type` strings from
HuggingFace configs to parser classes.

```python
PARSER_REGISTRY = {
    "llama": LlamaConfigParser,
    "llama4": Llama4ConfigParser,
}
```

**Lazy Initialization:** Hardware requirements are computed once and cached in
`_hw_req_by_layers`. The `hw_req_by_layers` property checks if the cache exists; if not, it
computes all layer metrics and stores them.

**Layer-wise Metrics Aggregation:** Each parser computes metrics per layer, then `calc_total()`
aggregates across all blocks, accounting for layers that may not appear in every block (e.g.,
MoE layers).

### Hardware Metrics Computed

For each layer, parsers calculate:

| Metric | Description |
|--------|-------------|
| Compute | Total FLOPs (floating-point operations) |
| Bandwidth (Weight) | Memory traffic for loading model weights |
| Bandwidth (Input) | Memory traffic for reading input tensors |
| Bandwidth (Output) | Memory traffic for writing output tensors |
| Operational Intensity | FLOPs per byte (compute / total bandwidth) |

## Adding New Model Support

To support a new transformer architecture, follow these steps:

### Step 1: Create Parser File

Create a new file in `parsers/` (e.g., `parsers/newmodel.py`):

```python
from core import (
    BaseModelConfigParser,
    Number,
    TransformerMode,
    act_flops,
    torch_dtype_width,
)


class NewModelConfigParser(BaseModelConfigParser):
    pass
```

### Step 2: Implement Required Methods

```python
class NewModelConfigParser(BaseModelConfigParser):
    def get_layer_list(self) -> list[str]:
        """Return list of layer names in execution order."""
        return [
            "Attn - RMSNorm",
            "Attn - QKV_Proj",
            # ... add all layers
        ]

    def get_num_blocks(self) -> int:
        """Return number of transformer blocks."""
        return self.model_conf["num_hidden_layers"]

    def get_kvcache_size(self) -> int:
        """Calculate KV-cache memory requirements."""
        # Implementation depends on architecture
        pass

    @property
    def hw_req_by_layers(self) -> dict[str, dict[str, Number]]:
        """Compute hardware requirements per layer."""
        if self._hw_req_by_layers is not None:
            return self._hw_req_by_layers.copy()

        req_dict: dict[str, dict[str, Number]] = {
            key: self.new_req_dict() for key in self.get_layer_list()
        }

        # Use set_op_* methods to populate metrics
        self.set_op_rmsnorm_req(...)
        self.set_op_proj_req(...)
        # ... continue for all layers

        self._hw_req_by_layers = req_dict
        return self._hw_req_by_layers.copy()
```

### Step 3: Implement Optional Methods (if needed)

```python
@classmethod
def normalize_config(cls, config_dict: dict) -> dict:
    """Set architecture-specific config defaults."""
    if "torch_dtype" not in config_dict:
        config_dict["torch_dtype"] = "float16"
    return config_dict

def get_layer_num_blocks(self, layer: str) -> int:
    """Return different block counts for MoE layers."""
    if "MoE" in layer:
        return self.get_num_blocks() // 2
    return self.get_num_blocks()

def get_extra_storage_req(self) -> list[tuple[str, Number]]:
    """Return additional storage beyond weights/KV-cache."""
    return [("Embedding Table", Number("B", "!.2k", size))]
```

### Step 4: Register Parser

Add to `PARSER_REGISTRY` in `transformer_roofline_analyzer`:

```python
PARSER_REGISTRY = {
    "llama": LlamaConfigParser,
    "llama4": Llama4ConfigParser,
    "newmodel": NewModelConfigParser,  # Add new parser
}
```

### Step 5: Export from `parsers/__init__.py`

```python
from .newmodel import NewModelConfigParser
```

### Step 6: Add Tests

1. **Unit tests:** `tests/unit/test_newmodel_parser.py`
2. **E2E tests:** `tests/end-to-end/newmodel/` with `test.json` and expected output

## Code Style Guidelines

### Readability and Extensibility

Maintain good readability and clear architecture for extensibility when adding support for new
model architectures. The codebase is designed to easily accommodate new transformer variants.

### Reference Config Requirement

**Important:** Before implementing features that parse model configs, first read a reference
config file to understand the actual structure and fields available. Reference configs can be
obtained from HuggingFace Hub via `AutoConfig.from_pretrained()`.

### Explicit Architecture Handling

**Important:** For model architecture-dependent operations, always explicitly decode the
architecture name and take corresponding actions. Never use a catch-all `else` clause as a
fallback to an arbitrary architecture.

```python
# CORRECT: Explicitly handle each architecture, raise error for unknown
architectures = config_dict.get("architectures", [])
if "LlamaForCausalLM" in architectures:
    # Handle Llama 2/3 specific logic
    ...
elif "Llama4ForConditionalGeneration" in architectures:
    # Handle Llama 4 specific logic
    ...
else:
    raise ValueError(f"Unsupported architecture: {architectures}")

# INCORRECT: Using else as fallback
if "Llama4ForConditionalGeneration" in architectures:
    ...
else:
    # Assumes everything else is Llama 2/3 - DON'T DO THIS
    ...
```

## Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the code style guidelines
3. **Run pre-commit checks:** `pre-commit run --all-files`
4. **Run tests:** `pytest`
5. **Update documentation** if your changes affect APIs or behavior
6. **Submit a pull request** with a clear description of changes

### Commit Guidelines

- Use clear, descriptive commit messages
- Reference related issues when applicable
- Keep commits focused and atomic

## Documentation Maintenance

**Important:** After any code change, review if all existing documentation matches the latest
implementation. Documentation files to check:

| File | Content |
|------|---------|
| `README.md` | User-facing features and usage examples |
| `CONTRIBUTING.md` | Development setup, architecture, code style |
| `CLAUDE.md` | Claude Code-specific guidelines |
| `core/README.md` | Core module APIs and methods |
| `parsers/README.md` | Parser framework and extension guide |
| `tests/README.md` | Test structure and commands |

## Important Notes

- All model config files must follow HuggingFace `config.json` schema
- The tool analyzes inference workloads, not training
- KV-cache calculations assume standard attention mechanisms
- MoE models (like LLaMA-4) require special handling for routed vs. shared experts
