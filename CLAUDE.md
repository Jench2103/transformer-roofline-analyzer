# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Project Overview

Transformer Roofline Analyzer is a CLI tool that estimates compute (FLOPs) and memory bandwidth
requirements for transformer model architectures. It accepts HuggingFace model names or local
`config.json` files and performs roofline analysis to understand hardware resource demands.

## Quick Reference

```bash
# Activate virtual environment
poetry shell

# Run CLI
./transformer_roofline_analyzer --cached-tokens 1048576 --input-tokens 1 -- meta-llama/Llama-2-7b-hf

# Run tests
pytest
```

## Claude Code-Specific Guidelines

### IMPORTANT: Ignore Temporary Files and Cache Directories

Do not read or explore the following directories and files for efficient codebase understanding:

- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `.ruff_cache/` - Ruff linter cache
- `.mypy_cache/` - Mypy type checker cache
- `*.pyc`, `*.pyo` - Compiled Python files
- `.git/` - Git internal files
- `*.egg-info/` - Python package metadata
- `.venv/`, `venv/` - Virtual environment directories

Focus on source files (`.py`), configuration files (`pyproject.toml`, `config.json`), and
documentation (`.md`) for understanding the codebase.

### IMPORTANT: Always Read Reference Configs First

Before planning or implementing any feature that parses model configs, you **must** first read
a reference config file to understand the actual structure and fields available. Reference
configs can be obtained from:

- HuggingFace Hub (download via `AutoConfig.from_pretrained()`)

If you cannot find a suitable reference config for the architecture you're working with, ask
the developer to provide one or point to where it can be found.

### IMPORTANT: Explicit Architecture Handling

For model architecture-dependent operations, always explicitly decode the architecture name
from the config and take corresponding actions. HuggingFace configs provide two relevant fields:

- `architectures`: A list of architecture class names (e.g., `["LlamaForCausalLM"]`)
- `model_type`: A string identifier (e.g., `"llama"`, `"llama4"`)

**Never use a catch-all `else` clause as a fallback to an arbitrary architecture.** The `else`
branch should always raise an error for unsupported architectures.

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

### IMPORTANT: Readability and Extensibility

Maintain good readability and clear architecture for extensibility when adding support for new
model architectures. The codebase is designed to easily accommodate new transformer variants.

### IMPORTANT: Type Hints Required

All functions and variables must have type annotations. Type checking is enforced via pyright.

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

### IMPORTANT: Documentation Maintenance

After any code change, review if all existing documentation matches the latest implementation.
Documentation files to check:

- `README.md` - User-facing features and usage examples
- `CONTRIBUTING.md` - Development setup, architecture, code style
- `CLAUDE.md` - Claude Code-specific guidelines
- `core/README.md` - Core module APIs and methods
- `parsers/README.md` - Parser framework and extension guide
- `tests/README.md` - Test structure and commands

## Key Files Reference

| File | Purpose |
|------|---------|
| `transformer_roofline_analyzer` | CLI entry point with parser registry |
| `core/base_parser.py` | Abstract base class for all model parsers |
| `core/utils.py` | Number, QueryConfig, utilities |
| `parsers/llama.py` | LLaMA-2/3 parser (reference implementation) |
| `parsers/llama4.py` | LLaMA-4 parser with MoE support |

## For Full Documentation

See:

- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup, architecture, code style, adding
  new model support
- [core/README.md](core/README.md) - Core module documentation
- [parsers/README.md](parsers/README.md) - Parser framework and extension guide
- [tests/README.md](tests/README.md) - Testing documentation
