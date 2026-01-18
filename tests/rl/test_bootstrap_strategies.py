"""Tests for bootstrap strategies and provenance in code workspace runner.

Validates:
1. Bootstrap strategies install deps correctly
2. Provenance is recorded and loadable
3. Tests requiring editable install work with the strategy
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_bootstrap_strategy_none_succeeds(tmp_path: Path):
    """NONE strategy succeeds without running anything."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
    from nmoe.rl.tasks.code_workspace import run_bootstrap_strategy

    ws = tmp_path / "ws"
    ws.mkdir()

    config = BootstrapConfig(strategy=BootstrapStrategy.NONE)
    result = run_bootstrap_strategy(ws, config)

    assert result.success is True
    assert result.strategy == "none"
    assert result.commands_run == []
    assert result.duration_ms == 0


def test_bootstrap_pip_editable_requires_setup_py(tmp_path: Path):
    """PIP_EDITABLE strategy fails gracefully without setup.py/pyproject.toml."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
    from nmoe.rl.tasks.code_workspace import run_bootstrap_strategy

    ws = tmp_path / "ws"
    ws.mkdir()
    # No setup.py or pyproject.toml

    config = BootstrapConfig(strategy=BootstrapStrategy.PIP_EDITABLE)
    result = run_bootstrap_strategy(ws, config)

    # Should fail (no installable package)
    assert result.success is False
    assert "pip install" in result.commands_run[0]


def test_bootstrap_pyproject_editable_requires_pyproject(tmp_path: Path):
    """PYPROJECT_EDITABLE strategy fails if pyproject.toml missing."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
    from nmoe.rl.tasks.code_workspace import run_bootstrap_strategy

    ws = tmp_path / "ws"
    ws.mkdir()

    config = BootstrapConfig(strategy=BootstrapStrategy.PYPROJECT_EDITABLE)
    result = run_bootstrap_strategy(ws, config)

    assert result.success is False
    assert "pyproject.toml not found" in result.error


def test_bootstrap_uv_sync_requires_uvlock(tmp_path: Path):
    """UV_SYNC strategy fails if uv.lock missing."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
    from nmoe.rl.tasks.code_workspace import run_bootstrap_strategy

    ws = tmp_path / "ws"
    ws.mkdir()

    config = BootstrapConfig(strategy=BootstrapStrategy.UV_SYNC)
    result = run_bootstrap_strategy(ws, config)

    assert result.success is False
    assert "uv.lock not found" in result.error


def test_detect_bootstrap_strategy(tmp_path: Path):
    """Auto-detection picks correct strategy based on files."""
    from nmoe.rl.tasks.code_workspace import (
        BootstrapStrategy,
        detect_bootstrap_strategy,
    )

    ws = tmp_path / "ws"
    ws.mkdir()

    # Empty workspace -> NONE
    assert detect_bootstrap_strategy(ws) == BootstrapStrategy.NONE

    # Add setup.py -> PIP_EDITABLE
    (ws / "setup.py").write_text("from setuptools import setup\nsetup()")
    assert detect_bootstrap_strategy(ws) == BootstrapStrategy.PIP_EDITABLE

    # Add requirements.txt -> REQUIREMENTS_TXT (higher priority)
    (ws / "requirements.txt").write_text("pytest\n")
    assert detect_bootstrap_strategy(ws) == BootstrapStrategy.REQUIREMENTS_TXT

    # Add pyproject.toml -> PYPROJECT_EDITABLE (higher priority)
    (ws / "pyproject.toml").write_text("[project]\nname='test'\n")
    assert detect_bootstrap_strategy(ws) == BootstrapStrategy.PYPROJECT_EDITABLE

    # Add uv.lock -> UV_SYNC (highest priority)
    (ws / "uv.lock").write_text("version = 1\n")
    assert detect_bootstrap_strategy(ws) == BootstrapStrategy.UV_SYNC


def test_provenance_save_and_load(tmp_path: Path):
    """Provenance can be saved to and loaded from workspace."""
    from nmoe.rl.tasks.code_workspace import (
        BootstrapResult,
        WorkspaceProvenance,
    )

    ws = tmp_path / "ws"
    ws.mkdir()

    bootstrap = BootstrapResult(
        success=True,
        strategy="pip_editable",
        commands_run=["pip install -e ."],
        exit_codes=[0],
        duration_ms=1234,
    )

    provenance = WorkspaceProvenance(
        workspace_id="ws_abc123",
        base_sha="aaa",
        target_sha="bbb",
        repo_id="test_repo",
        created_at="2025-01-01T00:00:00Z",
        bootstrap=bootstrap,
        sandbox_config={"working_dir": str(ws)},
        test_command="pytest",
        hidden_test_command="pytest --hidden",
    )

    provenance.save(ws)
    assert (ws / ".nmoe_workspace.json").exists()

    loaded = WorkspaceProvenance.load(ws)
    assert loaded is not None
    assert loaded.workspace_id == "ws_abc123"
    assert loaded.base_sha == "aaa"
    assert loaded.bootstrap is not None
    assert loaded.bootstrap.strategy == "pip_editable"
    assert loaded.test_command == "pytest"


def test_editable_install_required_for_package_import(tmp_path: Path):
    """Tests that require package import fail without editable install.

    Uses src/ layout where the package is NOT on sys.path by default,
    proving that editable install is necessary for the import to work.

    Note: This test requires pip to be available in the sandbox.
    """
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.code_workspace import (
        run_bootstrap_strategy,
        run_in_workspace,
    )
    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy

    ws = tmp_path / "ws"
    ws.mkdir()

    # First check if pip is available in the sandbox
    pip_check = run_in_workspace(
        workspace_path=ws,
        command="python -m pip --version",
        timeout_ms=10_000,
    )
    if not pip_check.success:
        pytest.skip("pip not available in sandbox")

    # Create a src/ layout package (NOT importable from cwd)
    src = ws / "src"
    src.mkdir()
    pkg = src / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VERSION = '1.0.0'\n")
    (pkg / "compute.py").write_text(
        "def double(x):\n"
        "    return x * 2\n"
    )

    # Create pyproject.toml with src layout
    (ws / "pyproject.toml").write_text(
        "[project]\n"
        "name = 'mypackage'\n"
        "version = '1.0.0'\n"
        "\n"
        "[tool.setuptools.packages.find]\n"
        "where = ['src']\n"
        "\n"
        "[build-system]\n"
        "requires = ['setuptools']\n"
        "build-backend = 'setuptools.build_meta'\n"
    )

    # Create tests/ directory with test that imports the package
    tests = ws / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "test_mypackage.py").write_text(
        "from mypackage.compute import double\n"
        "\n"
        "def test_double():\n"
        "    assert double(21) == 42\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    test_double()\n"
        "    print('OK')\n"
    )

    # Without bootstrap: import fails (src/ not on sys.path)
    result_no_bootstrap = run_in_workspace(
        workspace_path=ws,
        command="python -m pytest tests/test_mypackage.py -v",
        timeout_ms=30_000,
    )
    assert result_no_bootstrap.success is False
    # Should fail with import error
    assert "ModuleNotFoundError" in result_no_bootstrap.stdout or "No module" in result_no_bootstrap.stdout or result_no_bootstrap.exit_code != 0

    # With editable install: import succeeds
    bootstrap_config = BootstrapConfig(
        strategy=BootstrapStrategy.PYPROJECT_EDITABLE,
        timeout_ms=120_000,
        allow_network=True,
    )
    bootstrap_result = run_bootstrap_strategy(ws, bootstrap_config)
    assert bootstrap_result.success is True, f"Bootstrap failed: {bootstrap_result.error}"

    result_with_bootstrap = run_in_workspace(
        workspace_path=ws,
        command="python -m pytest tests/test_mypackage.py -v",
        timeout_ms=30_000,
    )
    assert result_with_bootstrap.success is True, f"Tests failed after bootstrap: {result_with_bootstrap.stderr}"


def test_setup_workspace_with_provenance(tmp_path: Path):
    """setup_workspace_with_provenance runs bootstrap and saves provenance."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
    from nmoe.rl.tasks.code_workspace import WorkspaceProvenance, setup_workspace_with_provenance

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("# Test\n")

    success, provenance = setup_workspace_with_provenance(
        workspace_path=ws,
        base_sha="abc123",
        target_sha="def456",
        repo_id="test_repo",
        test_command="pytest",
        hidden_test_command="pytest --hidden",
        bootstrap_config=BootstrapConfig(strategy=BootstrapStrategy.NONE),
    )

    assert success is True
    assert provenance is not None
    assert provenance.base_sha == "abc123"
    assert provenance.target_sha == "def456"
    assert provenance.bootstrap is not None
    assert provenance.bootstrap.success is True

    # Verify provenance was saved
    loaded = WorkspaceProvenance.load(ws)
    assert loaded is not None
    assert loaded.workspace_id == provenance.workspace_id
