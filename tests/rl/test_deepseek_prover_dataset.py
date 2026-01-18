from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_iter_deepseek_prover_jsonl_parses(tmp_path: Path):
    from nmoe.rl.tasks.deepseek_prover import iter_deepseek_prover_jsonl

    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join(
            [
                '{"name":"ex1","split":"valid","informal_prefix":"/-- hi -/","formal_statement":"theorem t : True := by trivial"}',
                '{"name":"ex2","informal_prefix":"x"}',
                "not json",
                '{"name":""}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    ex = list(iter_deepseek_prover_jsonl(p))
    assert [e.name for e in ex] == ["ex1", "ex2"]
    assert ex[0].split == "valid"
    assert ex[0].informal_prefix is not None
    assert ex[0].formal_statement is not None


def test_prover_problem_text_formats_fields(tmp_path: Path):
    from nmoe.rl.tasks.deepseek_prover import DeepSeekProverExample, prover_problem_text

    ex = DeepSeekProverExample(
        name="n",
        split="train",
        informal_prefix="informal",
        formal_statement="formal",
        goal=None,
        header=None,
        raw={"name": "n"},
    )
    s = prover_problem_text(ex, include_informal=True, include_formal=True)
    assert "Informal statement:" in s
    assert "informal" in s
    assert "Formal statement:" in s
    assert "formal" in s


def test_deepseek_prover_dataset_smoke_external():
    path = os.environ.get("NMOE_DEEPSEEK_PROVER_JSONL")
    if not path:
        pytest.skip("set NMOE_DEEPSEEK_PROVER_JSONL to run this smoke test")

    from nmoe.rl.tasks.deepseek_prover import iter_deepseek_prover_jsonl, prover_problem_text

    p = Path(path)
    assert p.exists()
    it = iter_deepseek_prover_jsonl(p)
    ex = next(it)
    s = prover_problem_text(ex)
    assert isinstance(s, str) and s.strip()


def test_sample_prover_problems_deterministic(tmp_path: Path):
    from nmoe.rl.tasks.deepseek_prover import sample_prover_problems

    p = tmp_path / "ds.jsonl"
    p.write_text('{"name":"a","informal_prefix":"ia"}\n{"name":"b","informal_prefix":"ib"}\n', encoding="utf-8")
    s0 = sample_prover_problems(p, n=5, seed=123)
    s1 = sample_prover_problems(p, n=5, seed=123)
    assert s0 == s1
    assert len(s0) == 5
