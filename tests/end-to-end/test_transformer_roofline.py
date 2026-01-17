import json
import subprocess
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest


def discover_test_cases() -> list[tuple[str, str, str, str]]:
    """
    Discovers test cases from test.json files.

    Returns:
        list of (case_id, cmd_opts, model_or_config, expected_output)
    """
    test_cases: list[tuple[str, str, str, str]] = []
    test_root: Path = Path("tests")

    for test_json in test_root.rglob("test.json"):
        with open(test_json) as f:
            cases: list[dict[str, Any]] = json.load(f)

        for idx, case in enumerate(cases):
            # Support both new ModelName and legacy Config field
            if "ModelName" in case:
                model_or_config: str = case["ModelName"]
            elif "Config" in case:
                # Legacy: use local config file
                config_path: Path = test_json.parent / case["Config"]
                model_or_config: str = str(config_path)
            else:
                raise ValueError(
                    f"Test case must have either 'ModelName' or 'Config' field: {test_json}"
                )

            output_path: str = test_json.parent / case["Output"]
            test_cases.append(
                (
                    f"{test_json.parent.name}-{idx}",  # id for pytest
                    case["CommandOptions"],
                    model_or_config,
                    str(output_path),
                )
            )

    return test_cases


@pytest.mark.parametrize(
    "case_id, cmd_opts, model_or_config, expected_output_file",
    discover_test_cases(),
    ids=lambda x: x,  # Show the case_id in test output
)
def test_transformer_roofline(
    case_id: str,
    cmd_opts: dict[str, Any],
    model_or_config: str,
    expected_output_file: str,
    print_actual_output: bool,
) -> None:
    # Collect options
    cached_tokens: list[str] = ["--cached-tokens", *str(cmd_opts["cached-tokens"]).split()]
    input_tokens: list[str] = ["--input-tokens", *str(cmd_opts["input-tokens"]).split()]
    batch_size: list[str] = (
        ["--batch-size", str(cmd_opts["batch-size"])]
        if "batch-size" in cmd_opts and cmd_opts["batch-size"] != ""
        else []
    )

    # Build command
    cmd: list[str] = [
        "python3",
        "-m",
        "transformer_roofline_analyzer",
        *cached_tokens,
        *input_tokens,
        *batch_size,
        "--",
        model_or_config,
    ]

    # Run command
    result: CompletedProcess[str] = subprocess.run(cmd, capture_output=True, text=True)
    actual_output: str = result.stdout.strip()

    if print_actual_output:
        # If the flag is set, just print the actual output for this case
        print()
        print(f"--- Actual Output for {case_id} ---")
        print(actual_output)
        print("-----------------------------------")

        # No assertions, so this test will "PASS" if the command ran without error.
        # You might still want to check returncode even in this mode, or skip it.
        # For simplicity, keeping the returncode check.
        assert result.returncode == 0, f"[{case_id}] Command failed with stderr:\n{result.stderr}"

    else:
        # Normal behavior: check exit status and then compare output
        # Load expected output
        with open(expected_output_file) as f:
            expected_output: str = f.read().strip()

        # Check exit status
        assert result.returncode == 0, f"[{case_id}] Command failed with stderr:\n{result.stderr}"

        # Check output
        assert actual_output == expected_output, f"[{case_id}] Output mismatch"
