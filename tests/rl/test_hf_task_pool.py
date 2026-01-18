from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_taskpool_from_hf_dataset_math_json(tmp_path: Path):
    pytest.importorskip("datasets")

    from nmoe.rl.rewards_harmony import CHANNELS, harmony_message
    from nmoe.rl.tasks import TaskPool

    ds_path = tmp_path / "math.jsonl"
    _write_jsonl(ds_path, [{"problem": "2+2=?", "answer": "4"}])

    pool = TaskPool.from_hf_dataset(
        dataset="json",
        data_files=str(ds_path),
        split="train",
        task_type="math",
        problem_field="problem",
        answer_field="answer",
        gold_extractor="raw",
        max_examples=10,
        seed=0,
    )
    assert len(pool) == 1

    task = pool.sample(1, seed=123)[0]
    resp = harmony_message(role="assistant", channel=CHANNELS["final"], content="4")
    ans = task.extract_answer(resp)
    assert task.verify(ans)


def test_taskpool_from_hf_dataset_gsm8k_hash_json(tmp_path: Path):
    pytest.importorskip("datasets")

    from nmoe.rl.rewards_harmony import CHANNELS, harmony_message
    from nmoe.rl.tasks import TaskPool

    ds_path = tmp_path / "gsm8k.jsonl"
    _write_jsonl(ds_path, [{"question": "1+1=?", "answer": "solution #### 2"}])

    pool = TaskPool.from_hf_dataset(
        dataset="json",
        data_files=str(ds_path),
        split="train",
        task_type="gsm8k",
        problem_field="question",
        answer_field="answer",
        gold_extractor="gsm8k_hash",
        max_examples=10,
        seed=0,
    )
    assert len(pool) == 1

    task = pool.sample(1, seed=123)[0]
    resp = harmony_message(role="assistant", channel=CHANNELS["final"], content="2")
    ans = task.extract_answer(resp)
    assert task.verify(ans)


def test_environment_hf_pool_from_toml(tmp_path: Path):
    pytest.importorskip("datasets")

    from nmoe.rl.environment import Environment

    ds_path = tmp_path / "math.jsonl"
    _write_jsonl(ds_path, [{"problem": "3+3=?", "answer": "6"}])

    toml_path = tmp_path / "env.toml"
    toml_path.write_text(
        "\n".join(
            [
                'env_id = "hf-json"',
                'format_type = "harmony"',
                "",
                "[task_pool]",
                'type = "hf"',
                'dataset = "json"',
                f'data_files = "{ds_path}"',
                'split = "train"',
                'task_type = "math"',
                'problem_field = "problem"',
                'answer_field = "answer"',
                'gold_extractor = "raw"',
                "max_examples = 10",
                "seed = 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = Environment.from_toml(toml_path)
    tasks = env.sample(1, seed=123)
    assert len(tasks) == 1
    assert tasks[0].has_ground_truth

