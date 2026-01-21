"""Failure taxonomy unit tests (no sandbox required)."""

from __future__ import annotations


def test_categorize_failure_timeout():
    from nmoe.rl.failures import categorize_failure

    assert categorize_failure(success=False, timed_out=True, exit_code=124) == "timeout"


def test_categorize_failure_command_not_found():
    from nmoe.rl.failures import categorize_failure

    assert categorize_failure(success=False, exit_code=127, stderr="command not found") == "command_not_found"


def test_categorize_failure_sandbox_denied():
    from nmoe.rl.failures import categorize_failure

    assert categorize_failure(success=False, exit_code=1, error="Permission denied") == "sandbox_denied"


def test_categorize_failure_nonzero_exit():
    from nmoe.rl.failures import categorize_failure

    assert categorize_failure(success=False, exit_code=2, stderr="") == "nonzero_exit"

