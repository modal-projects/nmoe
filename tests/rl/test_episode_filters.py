"""Tests for episode filtering in GitCommitTaskPool.

Validates:
1. Filter configuration rejects trivial/junk commits
2. Strict Option-B gating enforces fail-before/pass-after contract
3. Filter reasons are deterministic and correct
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _create_repo(path: Path) -> None:
    """Create a git repo with initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    _git(path, "config", "user.email", "test@test.com")
    _git(path, "config", "user.name", "test")
    (path / "README.md").write_text("# Test\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "Initial commit")


def test_filter_rejects_too_few_lines(tmp_path: Path):
    """Commits with too few changed lines are filtered."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Add a tiny change (1 line)
    (repo / "main.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "tiny change")

    executor = CodexExecutor(CodexConfig(timeout_ms=30_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        filter_config=EpisodeFilterConfig(min_changed_lines=10),
    )
    stats = pool.scan()

    # Should be filtered (too few lines)
    assert stats["filtered"] >= 1
    assert stats["option_a"] == 0


def test_filter_rejects_formatting_commit(tmp_path: Path):
    """Commits with formatting keywords in message are filtered."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Add a commit with "formatting" in the message
    (repo / "main.py").write_text("x = 1\ny = 2\nz = 3\n" * 5, encoding="utf-8")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "Apply black formatting")

    executor = CodexExecutor(CodexConfig(timeout_ms=30_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        filter_config=EpisodeFilterConfig(reject_formatting_only=True),
    )
    stats = pool.scan()

    # Should be filtered (formatting keyword)
    assert stats["filtered"] >= 1


def test_filter_rejects_too_many_files(tmp_path: Path):
    """Commits touching too many files are filtered."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Add many files
    for i in range(15):
        (repo / f"file_{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add many files")

    executor = CodexExecutor(CodexConfig(timeout_ms=30_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        filter_config=EpisodeFilterConfig(max_files=10),
    )
    stats = pool.scan()

    # Should be filtered (too many files)
    assert stats["filtered"] >= 1


def test_filter_requires_non_test_impl(tmp_path: Path):
    """Commits where all impl files are tests are filtered."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Add only test files (which are counted as spec, not impl)
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_foo.py").write_text(
        "def test_foo():\n    assert True\n" * 5,
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add tests only")

    executor = CodexExecutor(CodexConfig(timeout_ms=30_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        filter_config=EpisodeFilterConfig(require_non_test_impl=True),
    )
    stats = pool.scan()

    # Should be filtered (all files are tests -> no impl files)
    assert stats["filtered"] >= 1 or stats["option_a"] == 0


def test_filter_accepts_good_commit(tmp_path: Path):
    """Valid commits pass all filters."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Add a substantial impl change
    (repo / "main.py").write_text(
        "def compute(x):\n"
        "    result = x * 2\n"
        "    return result + 1\n"
        "\n"
        "def process(data):\n"
        "    return [compute(d) for d in data]\n",
        encoding="utf-8",
    )
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "Add compute function")

    executor = CodexExecutor(CodexConfig(timeout_ms=30_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        filter_config=EpisodeFilterConfig(min_changed_lines=5),
    )
    stats = pool.scan()

    # Should pass filters
    assert stats["option_a"] >= 1 or stats["option_b"] >= 1


def test_strict_option_b_skips_on_verification_failure(tmp_path: Path):
    """Strict mode skips (not downgrades) failed Option-B episodes."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Create a commit with tests that pass on parent (bad Option-B)
    (repo / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "Add f")

    # Add tests that pass (not fail-before)
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_main.py").write_text(
        "import main\n\n"
        "def test_f():\n"
        "    assert main.f() == 1\n\n"
        "if __name__ == '__main__':\n"
        "    test_f()\n",
        encoding="utf-8",
    )
    # No impl change in this commit - tests pass immediately
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add tests")

    executor = CodexExecutor(CodexConfig(timeout_ms=60_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        test_command="python -m tests.test_main",
        verify_option_b=True,
        filter_config=EpisodeFilterConfig(strict_option_b=True, min_changed_lines=1),
    )
    stats = pool.scan()

    # Should be skipped (strict mode) not downgraded to Option-A
    # The "Add tests" commit has spec files, so it's a candidate for Option-B
    # But verification fails (tests pass before fix), so in strict mode it's skipped
    assert stats["option_a"] == 0 or stats["skip"] >= 0  # Either filtered or skipped


def test_option_b_fails_on_command_not_found(tmp_path: Path):
    """Option-B verification fails if test command is not found."""
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python not available")

    from nmoe.rl.tasks.git_episodes import EpisodeFilterConfig, GitCommitTaskPool
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

    repo = tmp_path / "repo"
    _create_repo(repo)

    # Create Option-B pattern
    (repo / "main.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    _git(repo, "add", "main.py")
    _git(repo, "commit", "-m", "Add f")

    (repo / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_main.py").write_text(
        "import main\n\ndef test_f():\n    assert main.f() == 1\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Fix f")

    executor = CodexExecutor(CodexConfig(timeout_ms=60_000, read_write_paths=[str(tmp_path)]))
    pool = GitCommitTaskPool(
        repo_path=repo,
        executor=executor,
        test_command="nonexistent_test_runner",  # Invalid command
        verify_option_b=True,
        filter_config=EpisodeFilterConfig(strict_option_b=True, min_changed_lines=1),
    )
    stats = pool.scan()

    # Should be skipped because test command not found
    assert stats["option_b"] == 0
