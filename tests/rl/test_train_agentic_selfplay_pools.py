"""Smoke tests for train_agentic self-play pool builders.

These tests ensure our 3 self-play environments (git / agents / proof) are
wired correctly through build_selfplay_pool(), including minimal deterministic
contracts and sandbox allowlists for the git environment.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_build_selfplay_pool_agents_smoke_and_determinism(tmp_path: Path):
    from nmoe.rl.train_agentic import build_selfplay_pool
    from nmoe.rl.tasks.agents import AgentSelfPlayTaskPool, MultiToolGCDTask

    pool1 = build_selfplay_pool(task_type="agents", workspaces_dir=str(tmp_path / "ws1"), seed=123)
    pool2 = build_selfplay_pool(task_type="agents", workspaces_dir=str(tmp_path / "ws2"), seed=123)

    assert isinstance(pool1, AgentSelfPlayTaskPool)
    assert isinstance(pool2, AgentSelfPlayTaskPool)

    t1 = pool1.sample(1)[0]
    t2 = pool2.sample(1)[0]
    assert isinstance(t1, MultiToolGCDTask)
    assert isinstance(t2, MultiToolGCDTask)
    assert t1.gold_gcd == t2.gold_gcd

    p = Path(t1.inputs_path)
    assert p.exists()
    assert p.is_file()
    assert (tmp_path / "ws1" / "agent_env") in p.parents

    a_s, b_s = p.read_text(encoding="utf-8").splitlines()[:2]
    assert str(math.gcd(int(a_s), int(b_s))) == t1.gold_gcd


def test_build_selfplay_pool_proof_and_meta_from_jsonl(tmp_path: Path):
    from nmoe.rl.train_agentic import build_selfplay_pool
    from nmoe.rl.tasks.proof import ProofMetaVerifierTask, ProofTaskPool, ProofVerifierTask

    verifier_path = tmp_path / "proof_verifier.jsonl"
    verifier_path.write_text(
        "\n".join([
            json.dumps({"problem": "P1", "proof": "Y1", "score": 0.5}),
            json.dumps({"problem": "P2", "proof": "Y2", "score": 1}),
            json.dumps({"problem": 123, "proof": "bad", "score": 0}),  # filtered
            "",  # ignored
        ]) + "\n",
        encoding="utf-8",
    )

    meta_path = tmp_path / "proof_meta.jsonl"
    meta_path.write_text(
        "\n".join([
            json.dumps({"problem": "P", "proof": "Y", "verifier_response": "V", "meta_score": 0}),
            json.dumps({"problem": "P", "proof": "Y", "verifier_response": "V", "meta_score": 1}),
        ]) + "\n",
        encoding="utf-8",
    )

    pool = build_selfplay_pool(task_type="proof", proof_dataset=str(verifier_path))
    assert isinstance(pool, ProofTaskPool)
    assert len(pool) == 2
    tasks = pool.sample(2)
    assert all(isinstance(t, ProofVerifierTask) for t in tasks)
    p0 = tasks[0].to_prompt()
    assert "<|start|>" in p0
    assert "Return ONLY the overall score" in p0

    meta_pool = build_selfplay_pool(task_type="proof_meta", proof_dataset=str(meta_path))
    assert isinstance(meta_pool, ProofTaskPool)
    assert len(meta_pool) == 2
    meta_tasks = meta_pool.sample(2)
    assert all(isinstance(t, ProofMetaVerifierTask) for t in meta_tasks)
    assert "Verifier response:" in meta_tasks[0].to_prompt()


def test_build_selfplay_pool_git_smoke(tmp_path: Path):
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.train_agentic import build_selfplay_pool
    from nmoe.rl.tasks.git_episodes import GitCommitTaskPool

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    _git(repo, "init")
    _git(repo, "config", "user.email", "nmoe@example.com")
    _git(repo, "config", "user.name", "nmoe")

    (repo / "a.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-m", "base: add a.f()")

    # Option B child commit: add tests + fix impl (tests pass on child, fail on base+tests).
    (repo / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_a.py").write_text(
        "import a\n\n"
        "def test_f():\n"
        "    assert a.f() == 1\n",
        encoding="utf-8",
    )
    _git(repo, "add", "a.py", "tests/test_a.py")
    _git(repo, "commit", "-m", "fix: make a.f() pass tests")

    workspaces = tmp_path / "workspaces"
    pool = build_selfplay_pool(task_type="git", repo_paths=[str(repo)], workspaces_dir=str(workspaces), seed=0)
    assert isinstance(pool, GitCommitTaskPool)

    # Sandbox allowlist must include the scanner repo and the workspace root.
    rw = set(pool.executor.config.read_write_paths)
    assert str(repo.resolve()) in rw
    assert str(workspaces.resolve()) in rw

    # Materialize workspace at base_sha + tests; verify fails until we apply the fix.
    task = pool.sample(1)[0]
    ws = Path(task.repo_path)
    assert workspaces.resolve() in ws.resolve().parents
    assert (ws / "tests" / "test_a.py").exists()
    assert task.verify(None) is False
    (ws / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    assert task.verify(None) is True
