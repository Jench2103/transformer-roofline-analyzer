# CLAUDE.md - Test Suite Instructions

This file provides guidance for AI assistants working with the test suite.

## Test Structure Overview

```text
tests/
├── unit/           # Isolated component tests (fast, no external dependencies)
└── end-to-end/     # CLI integration tests (validates complete workflows)
```

## Unit Tests (`tests/unit/`)

### Purpose

Test individual functions and classes in isolation, verifying:

- Mathematical formulas for hardware metrics (FLOPs, bandwidth)
- Error handling and edge cases
- Config normalization logic
- Parser behavior

### Key Patterns

1. **Class-based organization**: Group related tests in classes

   ```python
   class TestTorchDtypeWidth:
       def test_1byte_types(self):
           ...
   ```

2. **Parametrized tests**: Use `@pytest.mark.parametrize` for multiple inputs

   ```python
   @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
   def test_2byte_float_types(self, dtype):
       assert torch_dtype_width(dtype) == 2
   ```

3. **Fixtures**: Use shared fixtures from `conftest.py`
   - `sample_llama_config` / `sample_llama4_config` for model configs
   - `simple_query_config` / `batch_query_config` for query params

### Writing New Unit Tests

1. Add tests to existing files when testing existing modules
2. Create new test files for new modules (mirror the source structure)
3. Test both happy paths and error cases
4. Verify mathematical formulas with known values

## End-to-End Tests (`tests/end-to-end/`)

### Purpose

Validate complete CLI workflows by running the tool and comparing output against expected results.

### Test Discovery

Tests are defined in `test.json` files and discovered automatically by `discover_test_cases()`:

```json
{
  "tests": [
    {
      "config": "config-0.json",        // Local config file OR HuggingFace model name
      "cached-tokens": [0],             // CLI --cached-tokens argument
      "input-tokens": [1],              // CLI --input-tokens argument
      "batch-size": null,               // Optional: CLI --batch-size argument
      "output": "output-0-0.txt"        // Expected output file
    }
  ]
}
```

### Test Data Directories

- `llama/`: LLaMA 2/3 tests with local config files
- `llama4/`: LLaMA-4 tests with local config files
- `llama_hf/`: Tests using HuggingFace model names (requires `HF_TOKEN` in CI)

### Debugging E2E Tests

Use `--print-actual-output` to see actual CLI output:

```bash
pytest tests/end-to-end/ --print-actual-output -v
```

### Adding New E2E Tests

1. Create directory: `tests/end-to-end/<model_name>/`
2. Add config files: `config-0.json`, `config-1.json`, etc.
3. Create `test.json` with test definitions
4. Generate expected outputs using `--print-actual-output` or manual CLI runs
5. Verify with `pytest tests/end-to-end/<model_name>/ -v`

## Guidelines

### When to Add Unit Tests

- New utility functions or classes
- New parser implementations
- Bug fixes (add regression test)
- Changes to calculation formulas

### When to Add E2E Tests

- New model architecture support
- Changes to CLI output format
- New CLI arguments

### Test Naming

- Unit tests: `test_<function_name>_<scenario>`
- E2E test data: `output-<config_index>-<test_index>.txt`

### Imports in Test Files

Unit tests need to add the project root to `sys.path`:

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils import ...  # noqa: E402
```
