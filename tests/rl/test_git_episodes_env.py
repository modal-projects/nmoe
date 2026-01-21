"""Functional kill test for code self-play git episodes.

This validates the *environment contract* for code self-play:
1) Mine an Option-B episode from git history (tests + impl changes)
2) Materialize an isolated workspace at base_sha + new tests
3) Verification oracle is "tests pass" in that workspace
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_git_commit_task_pool_option_b_workspace_and_verify(tmp_path: Path):
    import pytest

    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tasks.git_episodes import GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    _git(repo, "init")
    _git(repo, "config", "user.email", "nmoe@example.com")
    _git(repo, "config", "user.name", "nmoe")

    # Base commit: impl only, failing relative to later tests.
    (repo / "a.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-m", "base: add a.f()")

    # Child commit: add tests + fix impl (Option B).
    (repo / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_a.py").write_text(
        "import a\n\n"
        "def test_f():\n"
        "    assert a.f() == 1\n\n"
        "if __name__ == \"__main__\":\n"
        "    test_f()\n",
        encoding="utf-8",
    )
    _git(repo, "add", "a.py", "tests/__init__.py", "tests/test_a.py")
    _git(repo, "commit", "-m", "fix: make a.f() pass tests")

    executor = CodexExecutor(CodexConfig(timeout_ms=60_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        test_command="python -m tests.test_a",
        max_commits=10,
        verify_option_b=True,
        workspaces_dir=tmp_path / "workspaces",
    )

    stats = pool.scan()
    assert stats["option_b"] == 1

    # Capture expected SHAs for provenance assertions.
    assert len(pool.episodes) == 1
    ep = pool.episodes[0]

    task = pool.sample(1)[0]
    ws = Path(task.repo_path)
    assert ws.exists()
    assert (ws / "tests" / "test_a.py").exists()

    prov_path = ws / ".nmoe_workspace.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["base_sha"] == ep.base_sha
    assert prov["target_sha"] == ep.target_sha
    assert prov["repo_id"] == pool.repo_id
    assert prov["test_command"] == "python -m tests.test_a"

    # Workspace starts from the *base* commit (+ new tests), so verification fails.
    assert task.verify(None) is False

    # Apply the fix (what the agent is supposed to do) and verify passes.
    (ws / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert task.verify(None) is True
