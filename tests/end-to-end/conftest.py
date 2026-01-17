"""Fixtures for end-to-end tests."""

import pytest
from _pytest.fixtures import FixtureRequest


@pytest.fixture
def print_actual_output(request: FixtureRequest) -> bool:
    """Fixture that returns True if --print-actual-output is set, False otherwise."""
    return request.config.getoption("--print-actual-output")
