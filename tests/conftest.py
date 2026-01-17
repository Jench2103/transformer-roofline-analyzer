"""Root pytest configuration for all tests."""

from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--print-actual-output",
        action="store_true",
        default=False,
        help="Print actual output without comparing to expected output (end-to-end tests only).",
    )
