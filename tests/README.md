# Tests

This directory contains all tests for the Transformer Roofline Analyzer.

## Directory Structure

```text
tests/
├── conftest.py          # Root pytest configuration (shared hooks)
├── unit/                # Unit tests for individual components
│   ├── conftest.py      # Unit test fixtures
│   ├── test_utils.py    # Tests for core/utils.py
│   ├── test_base_parser.py  # Tests for core/base_parser.py
│   ├── test_llama_parser.py # Tests for parsers/llama.py
│   ├── test_llama4_parser.py # Tests for parsers/llama4.py
│   └── test_cli.py      # Tests for CLI entry point
└── end-to-end/          # End-to-end CLI tests
    ├── conftest.py      # E2E test fixtures
    ├── test_transformer_roofline.py  # Parametrized E2E tests
    ├── llama/           # Test data for LLaMA models (local configs)
    ├── llama4/          # Test data for LLaMA-4 models (local configs)
    └── llama_hf/        # Test data for HuggingFace model names
```

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit/ -v

# Run only end-to-end tests
pytest tests/end-to-end/ -v

# Run with coverage
pytest --cov=core --cov=parsers --cov-report=term-missing

# Debug E2E tests by printing actual output
pytest tests/end-to-end/ --print-actual-output
```

## Test Types

### Unit Tests (`tests/unit/`)

Unit tests verify individual functions and classes in isolation:

- **test_utils.py**: Tests for `torch_dtype_width()`, `act_flops()`, `Number`, `QueryConfig`, `TransformerMode`
- **test_base_parser.py**: Tests for base parser operations (GEMM, RoPE, RMSNorm, SDPA calculations)
- **test_llama_parser.py**: Tests for LLaMA-specific parser implementation
- **test_llama4_parser.py**: Tests for LLaMA-4 MoE parser implementation
- **test_cli.py**: Tests for CLI argument handling and config loading

### End-to-End Tests (`tests/end-to-end/`)

E2E tests run the CLI and compare output against expected results:

- Uses JSON-driven test discovery (`test.json` files)
- Supports both local config files and HuggingFace model names
- Validates complete CLI output formatting

## Adding New Tests

### Adding Unit Tests

1. Create or modify test files in `tests/unit/`
2. Use fixtures from `tests/unit/conftest.py`
3. Follow existing patterns (class-based organization, parametrized tests)

### Adding E2E Tests

1. Create a test data directory under `tests/end-to-end/` (e.g., `tests/end-to-end/newmodel/`)
2. Add config files (`config-0.json`, etc.) if testing local configs
3. Create `test.json` with test case definitions:

    ```json
    {
    "tests": [
        {
        "config": "config-0.json",
        "cached-tokens": [0],
        "input-tokens": [1],
        "output": "output-0-0.txt"
        }
    ]
    }
    ```

4. Generate expected output files by running the CLI manually or using `--print-actual-output`

## Fixtures

### Unit Test Fixtures (`tests/unit/conftest.py`)

- `sample_llama_config`: Minimal valid LLaMA config dict
- `sample_llama4_config`: Minimal valid LLaMA-4 config dict with MoE params
- `simple_query_config`: Single query (0 cached, 1 input token)
- `batch_query_config`: Batch of 2 queries
- `prefill_query_config`: Prefill scenario (2048 input tokens)

### E2E Test Fixtures (`tests/end-to-end/conftest.py`)

- `print_actual_output`: Returns True if `--print-actual-output` flag is set
