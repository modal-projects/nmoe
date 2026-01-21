from __future__ import annotations

import json
from pathlib import Path


def test_environment_from_toml_builds_selfplay_agents(tmp_path: Path):
    from nmoe.rl.environment import Environment
    from nmoe.rl.tasks.agents import AgentSelfPlayTaskPool

    cfg = tmp_path / "env.toml"
    cfg.write_text(
        "\n".join(
            [
                'env_id = "agents_gcd"',
                'format_type = "harmony"',
                "",
                "[task_pool]",
                'type = "selfplay"',
                'task_type = "agents"',
                f'workspaces_dir = "{(tmp_path / "ws").as_posix()}"',
                "seed = 123",
                "",
                "[tools]",
                'executor_type = "codex_python"',
                "allow_network = false",
                f'cwd = "{(tmp_path / "ws").as_posix()}"',
                f'allowed_paths = ["{(tmp_path / "ws").as_posix()}"]',
                "timeout_default_ms = 1000",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = Environment.from_toml(cfg)
    assert env.env_id == "agents_gcd"
    assert env.format_type == "harmony"
    assert isinstance(env.task_pool, AgentSelfPlayTaskPool)
    assert env.tool_config is not None
    ex = env.sample(1)[0]
    assert getattr(ex, "task_type", "") == "agent_gcd"


def test_environment_from_toml_builds_selfplay_proof(tmp_path: Path):
    from nmoe.rl.environment import Environment
    from nmoe.rl.tasks.proof import ProofTaskPool, ProofVerifierTask

    ds = tmp_path / "proof.jsonl"
    ds.write_text(
        "\n".join(
            [
                json.dumps({"problem": "P1", "proof": "Y1", "score": 1}),
                json.dumps({"problem": "P2", "proof": "Y2", "score": 0.5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = tmp_path / "env.toml"
    cfg.write_text(
        "\n".join(
            [
                'env_id = "proof_verifier"',
                "",
                "[task_pool]",
                'type = "selfplay"',
                'task_type = "proof"',
                f'proof_dataset = "{ds.as_posix()}"',
                "seed = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = Environment.from_toml(cfg)
    assert isinstance(env.task_pool, ProofTaskPool)
    t = env.sample(1)[0]
    assert isinstance(t, ProofVerifierTask)
    p = t.to_prompt()
    assert "<|start|>" in p
    assert "Return ONLY the overall score" in p
