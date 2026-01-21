"""Problem pools for self-play (string problems, no ground-truth verification).

These pools are used to source *problems* for capability loops like Math‑V2
self-play. They are intentionally not `Task` objects (no verifiable oracle).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from nmoe.rl.tasks.deepseek_math_v2 import iter_deepseek_math_v2_inputs_json, math_v2_problem_text


def _iter_hf_text_field(
    row: dict,
    *,
    fields: Sequence[str],
) -> str | None:
    for f in fields:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def iter_hf_problem_texts(
    dataset: str,
    *,
    split: str,
    fields: Sequence[str] = ("problem", "question", "prompt", "text"),
    streaming: bool = False,
) -> Iterator[str]:
    """Iterate problem text strings from a HuggingFace dataset split."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(str(dataset), split=str(split), streaming=bool(streaming))
    for row in ds:
        if not isinstance(row, dict):
            continue
        t = _iter_hf_text_field(row, fields=fields)
        if t is not None:
            yield t


@dataclass
class ProblemPool:
    """Deterministic sampling pool over problem strings."""

    problems: list[str]
    seed: int = 0

    def __len__(self) -> int:
        return len(self.problems)

    def sample(self, n: int) -> list[str]:
        if not self.problems or int(n) <= 0:
            return []
        rng = random.Random(int(self.seed))
        return rng.choices(self.problems, k=int(n))

    @classmethod
    def from_hf(
        cls,
        dataset: str,
        *,
        split: str,
        max_examples: int,
        seed: int = 0,
        fields: Sequence[str] = ("problem", "question", "prompt", "text"),
        streaming: bool = False,
    ) -> "ProblemPool":
        it = iter_hf_problem_texts(dataset, split=split, fields=fields, streaming=streaming)
        out: list[str] = []
        for i, t in enumerate(it):
            if i >= int(max_examples):
                break
            out.append(t)
        return cls(problems=out, seed=int(seed))

    @classmethod
    def from_deepseek_math_v2_inputs(
        cls,
        path: str | Path,
        *,
        max_examples: int,
        seed: int = 0,
    ) -> "ProblemPool":
        out: list[str] = []
        for i, p in enumerate(iter_deepseek_math_v2_inputs_json(path)):
            if i >= int(max_examples):
                break
            out.append(math_v2_problem_text(p))
        return cls(problems=out, seed=int(seed))

