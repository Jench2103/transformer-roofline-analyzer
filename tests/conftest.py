import pytest
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest


def pytest_addoption(parser: Parser) -> None:
    """Adds a custom command-line option to pytest."""
    parser.addoption(
        "--print-actual-output",
        action="store_true",
        default=False,
        help="Print actual output without comparing to expected output.",
    )


@pytest.fixture
def print_actual_output(request: FixtureRequest) -> bool:
    """Fixture that returns True if --print-actual-output is set, False otherwise."""
    return request.config.getoption("--print-actual-output")
