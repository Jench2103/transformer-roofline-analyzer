"""Shared fixtures for unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.utils import QueryConfig  # noqa: E402


@pytest.fixture
def sample_llama_config() -> dict:
    """Minimal valid LLaMA config for testing."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "hidden_act": "silu",
        "vocab_size": 32000,
        "torch_dtype": "bfloat16",
    }


@pytest.fixture
def sample_llama4_config() -> dict:
    """Minimal valid LLaMA-4 config for testing."""
    return {
        "model_type": "llama4",
        "text_config": {
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "num_hidden_layers": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 8192,
            "intermediate_size_mlp": 16384,
            "hidden_act": "silu",
            "vocab_size": 202048,
            "torch_dtype": "bfloat16",
            "num_local_experts": 16,
            "num_experts_per_tok": 1,
            "interleave_moe_layer_step": 4,
        },
    }


@pytest.fixture
def simple_query_config() -> QueryConfig:
    """Single query with 0 cached, 1 input token."""
    return QueryConfig(cached_tokens=[0], input_tokens=[1])


@pytest.fixture
def batch_query_config() -> QueryConfig:
    """Batch of 2 queries with different token counts."""
    return QueryConfig(cached_tokens=[1024, 2048], input_tokens=[1, 1])


@pytest.fixture
def prefill_query_config() -> QueryConfig:
    """Prefill scenario: many input tokens."""
    return QueryConfig(cached_tokens=[0], input_tokens=[2048])
