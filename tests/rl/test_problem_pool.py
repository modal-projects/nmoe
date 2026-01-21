from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_problem_pool_from_deepseek_math_v2_inputs_deterministic(tmp_path: Path):
    from nmoe.rl.tasks.problem_pool import ProblemPool

    p = tmp_path / "inputs.json"
    p.write_text(
        json.dumps(
            [
                {"problem_idx": "A", "question": "Q1"},
                {"problem_idx": "B", "question": "Q2"},
            ]
        ),
        encoding="utf-8",
    )
    pool0 = ProblemPool.from_deepseek_math_v2_inputs(p, max_examples=2, seed=123)
    pool1 = ProblemPool.from_deepseek_math_v2_inputs(p, max_examples=2, seed=123)
    assert pool0.sample(5) == pool1.sample(5)
    assert len(pool0) == 2


def test_iter_hf_problem_texts_field_selection():
    from nmoe.rl.tasks.problem_pool import iter_hf_problem_texts

    ds = [
        {"question": "Q1"},
        {"problem": "P2"},
        {"text": "  "},
        "not a dict",
    ]

    def _fake_load_dataset(_name, split, streaming):
        _ = (split, streaming)
        return list(ds)

    orig = None
    try:
        from datasets import load_dataset as orig  # type: ignore
    except Exception:
        orig = None

    try:
        import datasets  # type: ignore

        datasets.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
        out = list(iter_hf_problem_texts("x", split="train", streaming=False))
    finally:
        if orig is not None:
            datasets.load_dataset = orig  # type: ignore[attr-defined]

    assert out == ["Q1", "P2"]


def test_problem_pool_from_hf_smoke_external():
    if os.environ.get("NMOE_RUN_HF_PROBLEM_SMOKE") != "1":
        pytest.skip("set NMOE_RUN_HF_PROBLEM_SMOKE=1 to run HuggingFace dataset smoke test")

    from nmoe.rl.tasks.problem_pool import ProblemPool

    dataset = os.environ.get("NMOE_HF_PROBLEM_DATASET", "openai/gsm8k")
    split = os.environ.get("NMOE_HF_PROBLEM_SPLIT", "train[:16]")
    pool = ProblemPool.from_hf(dataset, split=split, max_examples=16, seed=0)
    assert len(pool) > 0
    assert any(p.strip() for p in pool.sample(4))

