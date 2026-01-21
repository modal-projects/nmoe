"""Dataset adapters for DeepSeek-Math-V2.

The public DeepSeek-Math-V2 repo includes small evaluation corpora:
  - `inputs/*.json`: contest problem statements (question + optional answer).
  - `outputs/*.jsonl`: model predictions for IMO-ProofBench / competitions.

These files are useful as *problem sources* for Math-V2-style self-play
(generator ↔ verifier ↔ meta-verifier). They are not verifier-labeled training
data, so we keep this adapter narrowly focused on loading and formatting
problems and (optionally) reading the bundled predictions/ratings for analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class DeepSeekMathV2Problem:
    problem_idx: str
    contest: str | None
    question: str
    answer: str | None
    raw: dict[str, Any]


def iter_deepseek_math_v2_inputs_json(path: str | Path) -> Iterator[DeepSeekMathV2Problem]:
    """Iterate `inputs/*.json` problems from the DeepSeek-Math-V2 repo."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return iter(())

    def _iter() -> Iterator[DeepSeekMathV2Problem]:
        for row in obj:
            if not isinstance(row, dict):
                continue
            question = row.get("question")
            problem_idx = row.get("problem_idx")
            if not isinstance(question, str) or not question.strip():
                continue
            if not isinstance(problem_idx, str) or not problem_idx.strip():
                continue
            contest = row.get("contest") if isinstance(row.get("contest"), str) else None
            answer = row.get("answer") if isinstance(row.get("answer"), str) else None
            yield DeepSeekMathV2Problem(
                problem_idx=problem_idx,
                contest=contest,
                question=question,
                answer=answer,
                raw=row,
            )

    return _iter()


@dataclass(frozen=True)
class DeepSeekMathV2Prediction:
    problem_idx: str
    question: str
    proof: str
    average_automatic_rating: float | None
    human_rating: int | None
    raw: dict[str, Any]


def iter_deepseek_math_v2_outputs_jsonl(path: str | Path) -> Iterator[DeepSeekMathV2Prediction]:
    """Iterate `outputs/*.jsonl` predictions from the DeepSeek-Math-V2 repo."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            question = obj.get("question")
            problem_idx = obj.get("problem_idx")
            mp = obj.get("model_prediction") if isinstance(obj.get("model_prediction"), dict) else {}
            proof = mp.get("proof")
            if not all(isinstance(x, str) and x.strip() for x in [question, problem_idx, proof]):
                continue
            aar = mp.get("average_automatic_rating")
            if not isinstance(aar, (int, float)):
                aar = None
            hr = mp.get("human_rating")
            if not isinstance(hr, int):
                hr = None
            yield DeepSeekMathV2Prediction(
                problem_idx=problem_idx,
                question=question,
                proof=proof,
                average_automatic_rating=float(aar) if aar is not None else None,
                human_rating=hr,
                raw=obj,
            )


def math_v2_problem_text(problem: DeepSeekMathV2Problem) -> str:
    """Format a problem into a single text prompt payload (no chat wrapper)."""
    header = f"[{problem.problem_idx}]"
    if problem.contest:
        header = f"{header} {problem.contest}"
    return header + "\n\n" + problem.question.strip()


def sample_math_v2_problems(
    path: str | Path,
    *,
    n: int,
    seed: int = 0,
) -> list[str]:
    """Sample formatted problem strings from a DeepSeek-Math-V2 inputs JSON file."""
    import random

    rng = random.Random(int(seed))
    problems = list(iter_deepseek_math_v2_inputs_json(path))
    if not problems:
        return []
    chosen = rng.choices(problems, k=int(n))
    return [math_v2_problem_text(p) for p in chosen]

