from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_environment_mixture_pool_deterministic(tmp_path: Path):
    pytest.importorskip("datasets")

    from nmoe.rl.environment import Environment

    gsm_path = tmp_path / "gsm.jsonl"
    math_path = tmp_path / "math.jsonl"
    _write_jsonl(gsm_path, [{"question": "1+1=?", "answer": "x #### 2"}])
    _write_jsonl(math_path, [{"problem": "2+2=?", "solution": "\\boxed{4}"}])

    toml_path = tmp_path / "env.toml"
    toml_path.write_text(
        "\n".join(
            [
                'env_id = "mix"',
                'format_type = "harmony"',
                "",
                "[task_pool]",
                'type = "mixture"',
                "seed = 0",
                "",
                "[[task_pool.sources]]",
                'name = "gsm8k"',
                'type = "hf"',
                'dataset = "json"',
                f'data_files = "{gsm_path}"',
                'split = "train"',
                'task_type = "gsm8k"',
                'problem_field = "question"',
                'answer_field = "answer"',
                'gold_extractor = "gsm8k_hash"',
                "max_examples = 10",
                "weight = 0.3",
                "",
                "[[task_pool.sources]]",
                'name = "math"',
                'type = "hf"',
                'dataset = "json"',
                f'data_files = "{math_path}"',
                'split = "train"',
                'task_type = "math"',
                'problem_field = "problem"',
                'answer_field = "solution"',
                'gold_extractor = "boxed"',
                "max_examples = 10",
                "weight = 0.7",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = Environment.from_toml(toml_path)
    a = env.sample(20, seed=123)
    b = env.sample(20, seed=123)
    assert [t.task_type for t in a] == [t.task_type for t in b]

    types = {t.task_type for t in a}
    assert "gsm8k" in types
    assert "math" in types

