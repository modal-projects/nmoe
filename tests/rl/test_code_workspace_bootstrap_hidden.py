"""Tests for bootstrap commands and hidden tests in code workspace runner.

Validates:
1. Bootstrap commands run before tests and are sandboxed
2. Hidden tests are not leaked in prompts but are executed by verifier
3. Verification requires ALL commands to pass (bootstrap → test → hidden)
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_bootstrap_runs_before_tests(tmp_path: Path):
    """Bootstrap creates a file that the test reads - proves ordering."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.code_workspace import verify_workspace
    from nmoe.rl.tasks.bootstrap import BootstrapConfig

    ws = tmp_path / "ws"
    ws.mkdir()

    # Write a test that reads a file created by bootstrap
    (ws / "test_check.py").write_text(
        "def test_bootstrap_ran():\n"
        "    with open('bootstrap_marker.txt') as f:\n"
        "        assert f.read().strip() == 'BOOTSTRAPPED'\n",
        encoding="utf-8",
    )

    # Without bootstrap, test fails (no marker file)
    result = verify_workspace(
        workspace_path=ws,
        test_command="python -m pytest test_check.py -v",
        timeout_ms=30_000,
    )
    assert result is False

    # With bootstrap, test passes
    result = verify_workspace(
        workspace_path=ws,
        test_command="python -m pytest test_check.py -v",
        timeout_ms=30_000,
        bootstrap=BootstrapConfig(commands=["echo 'BOOTSTRAPPED' > bootstrap_marker.txt"]),
    )
    assert result is True


def test_bootstrap_failure_fails_verification(tmp_path: Path):
    """If bootstrap fails, verification fails (even if tests would pass)."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.code_workspace import verify_workspace
    from nmoe.rl.tasks.bootstrap import BootstrapConfig

    ws = tmp_path / "ws"
    ws.mkdir()

    # Trivial passing test
    (ws / "test_pass.py").write_text(
        "def test_always_passes():\n    assert True\n",
        encoding="utf-8",
    )

    # Bootstrap that fails
    result = verify_workspace(
        workspace_path=ws,
        test_command="python -m pytest test_pass.py -v",
        timeout_ms=30_000,
        bootstrap=BootstrapConfig(commands=["exit 1"]),
    )
    assert result is False


def test_bootstrap_is_sandboxed(tmp_path: Path):
    """Bootstrap cannot write outside the workspace (to non-/tmp paths)."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.code_workspace import verify_workspace
    from nmoe.rl.tasks.bootstrap import BootstrapConfig

    ws = tmp_path / "ws"
    ws.mkdir()

    # Trivial test
    (ws / "test_pass.py").write_text(
        "def test_always_passes():\n    assert True\n",
        encoding="utf-8",
    )

    # Try to escape to /var (not /tmp, which is always allowed)
    escape_path = Path("/var/nmoe_escape_test.txt")

    # Bootstrap tries to write outside allowed paths - should fail
    result = verify_workspace(
        workspace_path=ws,
        test_command="python -m pytest test_pass.py -v",
        timeout_ms=30_000,
        bootstrap=BootstrapConfig(commands=[f"echo 'ESCAPED' > {escape_path}"]),
    )
    # Bootstrap should fail because /var is not in allowlist
    assert result is False


def test_hidden_tests_not_in_prompt(tmp_path: Path):
    """Hidden test command is not exposed in task prompt."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.agentic import CodeEditTask
    from nmoe.rl.tasks.bootstrap import BootstrapConfig

    task = CodeEditTask(
        task_id="test_hidden",
        issue_description="Fix the bug",
        repo_path=str(tmp_path),
        test_command="python -m pytest tests/public/",
        hidden_test_command="python -m pytest tests/hidden/",
    )

    prompt = task.to_prompt()

    # Public test command may appear in prompt (via issue description or hints)
    # but hidden_test_command must NOT appear
    assert "tests/hidden" not in prompt
    assert "hidden_test_command" not in prompt


def test_hidden_tests_executed_by_verifier(tmp_path: Path):
    """Verifier runs hidden tests and requires them to pass."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.agentic import CodeEditTask

    ws = tmp_path / "ws"
    ws.mkdir()

    # Public test passes
    (ws / "test_public.py").write_text(
        "def test_public():\n    assert True\n",
        encoding="utf-8",
    )

    # Hidden test fails
    (ws / "test_hidden.py").write_text(
        "def test_hidden():\n    assert False, 'Hidden test fails'\n",
        encoding="utf-8",
    )

    task = CodeEditTask(
        task_id="test_verify_hidden",
        issue_description="Fix the bug",
        repo_path=str(ws),
        test_command="python -m pytest test_public.py -v",
        hidden_test_command="python -m pytest test_hidden.py -v",
    )

    # Verification fails because hidden test fails
    assert task.verify(None) is False

    # Fix the hidden test
    (ws / "test_hidden.py").write_text(
        "def test_hidden():\n    assert True\n",
        encoding="utf-8",
    )

    # Now verification passes
    assert task.verify(None) is True


def test_verification_requires_all_stages(tmp_path: Path):
    """Verification requires bootstrap + test + hidden to all pass."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.agentic import CodeEditTask
    from nmoe.rl.tasks.bootstrap import BootstrapConfig

    ws = tmp_path / "ws"
    ws.mkdir()

    # Both tests pass
    (ws / "test_public.py").write_text(
        "def test_public():\n"
        "    with open('setup_done.txt') as f:\n"
        "        assert f.read().strip() == 'OK'\n",
        encoding="utf-8",
    )
    (ws / "test_hidden.py").write_text(
        "def test_hidden():\n    assert True\n",
        encoding="utf-8",
    )

    task = CodeEditTask(
        task_id="test_all_stages",
        issue_description="Fix the bug",
        repo_path=str(ws),
        test_command="python -m pytest test_public.py -v",
        hidden_test_command="python -m pytest test_hidden.py -v",
        bootstrap=BootstrapConfig(commands=["echo 'OK' > setup_done.txt"]),
    )

    # All stages pass
    assert task.verify(None) is True
