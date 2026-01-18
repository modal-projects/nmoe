"""Regression tests for hermetic code-workspace verification.

Focus: ensure verify runs against *source*, not stale repo-local bytecode.
We force a deterministic stale-`__pycache__` scenario by:
1) generating a .pyc for a module returning 0
2) editing the source to return 1 but restoring the old mtime (and same size)
3) ensuring our sandboxed verify still passes by clearing __pycache__ first
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_codeedit_verify_clears_repo_local_pycache(tmp_path: Path):
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tasks.agentic import CodeEditTask

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "tests").mkdir(parents=True, exist_ok=True)

    # v0: correct implementation returns 0; tests expect 0.
    a0 = "def f():\n    return 0\n"
    (ws / "a.py").write_text(a0, encoding="utf-8")
    (ws / "tests" / "test_a.py").write_text(
        "import a\n\ndef test_f():\n    assert a.f() == 0\n",
        encoding="utf-8",
    )

    # Run once in-process to generate repo-local __pycache__/a.*.pyc.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ws), check=True)

    # Record mtime after initial compile.
    st0 = os.stat(ws / "a.py")

    # v1: change implementation to return 1 but keep file size and restore mtime.
    a1 = "def f():\n    return 1\n"
    assert len(a1.encode("utf-8")) == len(a0.encode("utf-8"))
    (ws / "a.py").write_text(a1, encoding="utf-8")
    os.utime(ws / "a.py", (st0.st_atime, st0.st_mtime))

    # Update tests to expect 1.
    (ws / "tests" / "test_a.py").write_text(
        "import a\n\ndef test_f():\n    assert a.f() == 1\n",
        encoding="utf-8",
    )

    # If Python reuses stale __pycache__/a.*.pyc, this would fail.
    task = CodeEditTask(
        task_id="t",
        issue_description="x",
        repo_path=str(ws),
        test_command="python3 -m pytest -q",
    )
    assert task.verify(None) is True

