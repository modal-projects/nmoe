"""Hermetic code-workspace utilities for code self-play.

Single purpose:
- Materialize isolated workspaces for code episodes.
- Run verification commands inside a strict sandbox rooted at that workspace.
- Bootstrap dependencies via explicit strategies (pip_editable, uv_sync, etc.).
- Record provenance for reproducibility and debugging.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapResult, BootstrapStrategy


@dataclass
class WorkspaceProvenance:
    """Recorded provenance for a workspace setup.

    Written to .nmoe_workspace.json for reproducibility and debugging.
    """
    workspace_id: str
    base_sha: str
    target_sha: str
    repo_id: str
    created_at: str                       # ISO timestamp
    bootstrap: BootstrapResult | None
    sandbox_config: dict[str, Any]        # CodexConfig as dict
    test_command: str
    hidden_test_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, workspace_path: Path) -> None:
        """Write provenance to workspace."""
        path = workspace_path / ".nmoe_workspace.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, workspace_path: Path) -> "WorkspaceProvenance | None":
        """Load provenance from workspace."""
        path = workspace_path / ".nmoe_workspace.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                d = json.load(f)
            # Reconstruct BootstrapResult if present
            if d.get("bootstrap"):
                d["bootstrap"] = BootstrapResult(**d["bootstrap"])
            return cls(**d)
        except Exception:
            return None


@dataclass(frozen=True)
class WorkspaceRun:
    success: bool
    stdout: str
    stderr: str
    exit_code: int


def _q(p: str | Path) -> str:
    return shlex.quote(str(p))


def materialize_from_git_archive(
    *,
    executor,
    repo_path: Path,
    sha: str,
    workspace_path: Path,
) -> bool:
    """Materialize `workspace_path` as a clean snapshot of `repo_path@sha`.

    Uses `git archive | tar -x` instead of git worktrees for sandbox compatibility.
    """
    from nmoe.rl.tools.codex import CodexExecutor

    if not isinstance(executor, CodexExecutor):
        raise TypeError("executor must be a CodexExecutor")
    repo = repo_path.resolve()
    ws = workspace_path.resolve()
    cmd = f"cd {_q(repo)} && git archive {shlex.quote(sha)} | tar -x -C {_q(ws)}"
    return bool(executor.exec_bash(cmd).success)


def run_in_workspace(
    *,
    workspace_path: Path,
    command: str,
    timeout_ms: int = 120_000,
    allow_network: bool = False,
    extra_env: dict[str, str] | None = None,
) -> WorkspaceRun:
    """Run a bash command inside a strict sandbox rooted at `workspace_path`."""
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    ws = workspace_path.resolve()
    cfg = CodexConfig(
        timeout_ms=int(timeout_ms),
        working_dir=str(ws),
        allow_network=bool(allow_network),
        read_write_paths=[str(ws)],
    )
    executor = CodexExecutor(cfg)

    # Avoid same-second .pyc staleness:
    # - delete any repo-local __pycache__ so Python can't reuse old bytecode
    # - redirect new bytecode to a unique /tmp directory (in case something writes)
    env = {"PYTHONPYCACHEPREFIX": f"/tmp/nmoe_pycache_{uuid4().hex}"}
    if extra_env:
        env.update(extra_env)

    script = "\n".join([
        "set -e",
        "set -o pipefail",
        "find . -type d -name __pycache__ -prune -exec rm -rf {} +",
        command,
    ])
    res = executor.exec_bash(script, env=env)
    return WorkspaceRun(
        success=bool(res.success),
        stdout=res.stdout or "",
        stderr=res.stderr or "",
        exit_code=int(res.exit_code),
    )


def verify_workspace(
    *,
    workspace_path: Path,
    test_command: str,
    timeout_ms: int = 120_000,
    bootstrap: BootstrapConfig | None = None,
    hidden_test_command: str | None = None,
) -> bool:
    """Run verification in a hermetic workspace.

    Args:
        workspace_path: Root of the isolated workspace.
        test_command: Primary test command (shown to agent, used for training signal).
        timeout_ms: Per-command timeout.
        bootstrap: Optional bootstrap configuration. If provided, runs once before
            tests. Failures here fail verification.
        hidden_test_command: Optional eval-only test command (never shown to agent).
            If provided, verification requires BOTH test_command AND hidden_test_command to pass.

    Returns:
        True iff all commands succeed (bootstrap → test_command → hidden_test_command).
    """
    if not test_command:
        return False

    # Bootstrap (if any)
    if bootstrap is not None:
        result = run_bootstrap_strategy(workspace_path, bootstrap)
        if not result.success:
            return False

    # Primary test command
    out = run_in_workspace(workspace_path=workspace_path, command=test_command, timeout_ms=timeout_ms)
    if not out.success:
        return False

    # Hidden tests (eval-only, if provided)
    if hidden_test_command:
        out = run_in_workspace(workspace_path=workspace_path, command=hidden_test_command, timeout_ms=timeout_ms)
        if not out.success:
            return False

    return True


def is_writable_dir(path: Path) -> bool:
    """Best-effort check for a writable directory."""
    if not path.is_dir():
        return False
    try:
        _ = next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return False
    return os.access(path, os.R_OK | os.W_OK | os.X_OK)


# =============================================================================
# Bootstrap Strategy Implementations
# =============================================================================

def _expand_bootstrap_commands(ws: Path, config: BootstrapConfig) -> tuple[BootstrapStrategy, list[str], str]:
    """Expand BootstrapConfig into an executable command list.

    Returns:
        (resolved_strategy, commands, error_message)
    """
    if config.commands:
        return config.strategy, list(config.commands), ""

    strategy = config.strategy
    if strategy == BootstrapStrategy.AUTO:
        strategy = detect_bootstrap_strategy(ws)

    if strategy == BootstrapStrategy.NONE:
        return strategy, [], ""

    if strategy == BootstrapStrategy.PIP_EDITABLE:
        extras_str = ",".join(config.extras) if config.extras else ""
        if extras_str:
            return strategy, [f"python -m pip install -e '.[{extras_str}]'"], ""
        return strategy, ["python -m pip install -e ."], ""

    if strategy == BootstrapStrategy.PYPROJECT_EDITABLE:
        pyproject = ws / "pyproject.toml"
        if not pyproject.exists():
            return strategy, [], "pyproject.toml not found"
        extras_str = ",".join(config.extras) if config.extras else ""
        if extras_str:
            return strategy, [f"python -m pip install -e '.[{extras_str}]'"], ""
        return strategy, ["python -m pip install -e ."], ""

    if strategy == BootstrapStrategy.UV_SYNC:
        if not (ws / "uv.lock").exists():
            return strategy, [], "uv.lock not found"
        return strategy, ["uv sync --frozen"], ""

    if strategy == BootstrapStrategy.REQUIREMENTS_TXT:
        req_file = ws / config.requirements_file
        if not req_file.exists():
            return strategy, [], f"{config.requirements_file} not found"
        return strategy, [f"python -m pip install -r {config.requirements_file}"], ""

    return strategy, [], f"Unknown strategy: {strategy}"


def run_bootstrap_strategy(
    workspace_path: Path,
    config: BootstrapConfig,
) -> BootstrapResult:
    """Run the configured bootstrap strategy in the workspace.

    Args:
        workspace_path: Root of the isolated workspace.
        config: Bootstrap configuration.

    Returns:
        BootstrapResult with success status and provenance.
    """
    ws = workspace_path.resolve()
    resolved, commands, err = _expand_bootstrap_commands(ws, config)
    exit_codes: list[int] = []
    start_ms = int(time.time() * 1000)

    if err:
        return BootstrapResult(
            success=False,
            strategy=resolved.value,
            commands_run=commands,
            exit_codes=[],
            duration_ms=int(time.time() * 1000) - start_ms,
            error=err,
        )

    if not commands:
        return BootstrapResult(
            success=True,
            strategy=resolved.value,
            commands_run=[],
            exit_codes=[],
            duration_ms=0,
        )

    # Execute commands
    for cmd in commands:
        out = run_in_workspace(
            workspace_path=ws,
            command=cmd,
            timeout_ms=config.timeout_ms,
            allow_network=config.allow_network,
        )
        exit_codes.append(out.exit_code)
        if not out.success:
            return BootstrapResult(
                success=False,
                strategy=resolved.value,
                commands_run=commands,
                exit_codes=exit_codes,
                duration_ms=int(time.time() * 1000) - start_ms,
                error=out.stderr[:500] if out.stderr else f"exit code {out.exit_code}",
            )

    return BootstrapResult(
        success=True,
        strategy=resolved.value,
        commands_run=commands,
        exit_codes=exit_codes,
        duration_ms=int(time.time() * 1000) - start_ms,
    )


def detect_bootstrap_strategy(workspace_path: Path) -> BootstrapStrategy:
    """Auto-detect the best bootstrap strategy for a workspace.

    Detection order (first match wins):
    1. uv.lock exists → UV_SYNC
    2. pyproject.toml exists → PYPROJECT_EDITABLE
    3. requirements.txt exists → REQUIREMENTS_TXT
    4. setup.py exists → PIP_EDITABLE
    5. Otherwise → NONE
    """
    ws = workspace_path.resolve()

    if (ws / "uv.lock").exists():
        return BootstrapStrategy.UV_SYNC
    if (ws / "pyproject.toml").exists():
        return BootstrapStrategy.PYPROJECT_EDITABLE
    if (ws / "requirements.txt").exists():
        return BootstrapStrategy.REQUIREMENTS_TXT
    if (ws / "setup.py").exists():
        return BootstrapStrategy.PIP_EDITABLE

    return BootstrapStrategy.NONE


def setup_workspace_with_provenance(
    *,
    workspace_path: Path,
    base_sha: str,
    target_sha: str,
    repo_id: str,
    test_command: str,
    hidden_test_command: str = "",
    bootstrap_config: BootstrapConfig | None = None,
    auto_detect_bootstrap: bool = False,
) -> tuple[bool, WorkspaceProvenance | None]:
    """Setup workspace with bootstrap and record provenance.

    Args:
        workspace_path: Root of the isolated workspace (already materialized).
        base_sha: Base commit SHA.
        target_sha: Target commit SHA.
        repo_id: Repository identifier.
        test_command: Test command for verification.
        hidden_test_command: Hidden test command (eval-only).
        bootstrap_config: Explicit bootstrap config, or None for auto-detect.
        auto_detect_bootstrap: If True and no config, auto-detect strategy.

    Returns:
        (success, provenance) tuple.
    """
    ws = workspace_path.resolve()
    workspace_id = f"ws_{uuid4().hex[:12]}"
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Determine bootstrap config
    if bootstrap_config is None:
        bootstrap_config = BootstrapConfig(strategy=BootstrapStrategy.AUTO if auto_detect_bootstrap else BootstrapStrategy.NONE)

    # Run bootstrap
    bootstrap_result = run_bootstrap_strategy(ws, bootstrap_config)

    # Create provenance
    provenance = WorkspaceProvenance(
        workspace_id=workspace_id,
        base_sha=base_sha,
        target_sha=target_sha,
        repo_id=repo_id,
        created_at=created_at,
        bootstrap=bootstrap_result,
        sandbox_config={
            "working_dir": str(ws),
            "read_write_paths": [str(ws)],
            "allow_network": bootstrap_config.allow_network,
        },
        test_command=test_command,
        hidden_test_command=hidden_test_command,
    )

    # Save provenance (mandatory for reproducibility and debugging).
    provenance.save(ws)

    return bootstrap_result.success, provenance
