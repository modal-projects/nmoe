from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_iter_deepseek_math_v2_inputs_json_parses(tmp_path: Path):
    from nmoe.rl.tasks.deepseek_math_v2 import iter_deepseek_math_v2_inputs_json

    p = tmp_path / "inputs.json"
    p.write_text(
        json.dumps(
            [
                {"id": 1, "contest": "IMO2025", "problem_idx": "IMO2025-1", "question": "Q1", "answer": "A1"},
                {"problem_idx": "x", "question": "  "},
                "not a dict",
            ]
        ),
        encoding="utf-8",
    )
    rows = list(iter_deepseek_math_v2_inputs_json(p))
    assert len(rows) == 1
    assert rows[0].problem_idx == "IMO2025-1"
    assert rows[0].contest == "IMO2025"
    assert rows[0].question == "Q1"
    assert rows[0].answer == "A1"


def test_iter_deepseek_math_v2_outputs_jsonl_parses(tmp_path: Path):
    from nmoe.rl.tasks.deepseek_math_v2 import iter_deepseek_math_v2_outputs_jsonl

    p = tmp_path / "outputs.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "problem_idx": "PB-1",
                        "question": "Q",
                        "model_prediction": {"proof": "P", "average_automatic_rating": 0.5, "human_rating": 7},
                    }
                ),
                "not json",
                json.dumps({"problem_idx": "PB-2", "question": "Q2", "model_prediction": {"proof": ""}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = list(iter_deepseek_math_v2_outputs_jsonl(p))
    assert len(rows) == 1
    assert rows[0].problem_idx == "PB-1"
    assert rows[0].proof == "P"
    assert rows[0].average_automatic_rating == 0.5
    assert rows[0].human_rating == 7


def test_deepseek_math_v2_repo_smoke_external():
    repo_dir = os.environ.get("NMOE_DEEPSEEK_MATHV2_DIR")
    if not repo_dir:
        pytest.skip("set NMOE_DEEPSEEK_MATHV2_DIR to a DeepSeek-Math-V2 checkout to run this smoke test")

    from nmoe.rl.tasks.deepseek_math_v2 import iter_deepseek_math_v2_inputs_json, iter_deepseek_math_v2_outputs_jsonl

    root = Path(repo_dir)
    inputs = root / "inputs" / "IMO2025.json"
    outputs = root / "outputs" / "IMO-ProofBench-Advanced.jsonl"
    assert inputs.exists()
    assert outputs.exists()

    p0 = next(iter_deepseek_math_v2_inputs_json(inputs))
    assert p0.question.strip()
    pred0 = next(iter_deepseek_math_v2_outputs_jsonl(outputs))
    assert pred0.question.strip()
    assert pred0.proof.strip()

