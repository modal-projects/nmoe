"""Dataset adapters for DeepSeek-Prover.

These datasets provide theorem/proof *problems* (not verifier labels). We keep the
loader minimal and schema-tolerant so it can be used as a source of problems for
Math-V2-style self-play (generator ↔ verifier ↔ meta-verifier).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class DeepSeekProverExample:
    """One DeepSeek-Prover dataset row."""

    name: str
    split: str | None
    informal_prefix: str | None
    formal_statement: str | None
    goal: str | None
    header: str | None
    raw: dict[str, Any]


def iter_deepseek_prover_jsonl(path: str | Path) -> Iterator[DeepSeekProverExample]:
    """Iterate DeepSeek-Prover JSONL examples.

    Known schema keys:
      - name: str
      - split: str
      - informal_prefix: str (natural language statement; may include Lean comments)
      - formal_statement: str (Lean theorem statement)
      - goal: str
      - header: str

    Unknown keys are preserved in `raw`.
    """
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
            name = obj.get("name")
            if not isinstance(name, str) or not name:
                continue

            def _s(k: str) -> str | None:
                v = obj.get(k)
                return v if isinstance(v, str) and v else None

            yield DeepSeekProverExample(
                name=name,
                split=_s("split"),
                informal_prefix=_s("informal_prefix"),
                formal_statement=_s("formal_statement"),
                goal=_s("goal"),
                header=_s("header"),
                raw=obj,
            )


def prover_problem_text(
    ex: DeepSeekProverExample,
    *,
    include_informal: bool = True,
    include_formal: bool = True,
) -> str:
    """Format a DeepSeek-Prover example into a single 'problem' string."""
    parts: list[str] = []
    if include_informal and ex.informal_prefix:
        parts.append("Informal statement:\n" + ex.informal_prefix.strip())
    if include_formal and ex.formal_statement:
        parts.append("Formal statement:\n" + ex.formal_statement.strip())
    if ex.goal:
        parts.append("Goal:\n" + ex.goal.strip())
    if not parts:
        return ex.name
    return "\n\n".join(parts)


def sample_prover_problems(
    path: str | Path,
    *,
    n: int,
    seed: int = 0,
    include_informal: bool = True,
    include_formal: bool = True,
) -> list[str]:
    """Sample formatted problem strings from a DeepSeek-Prover JSONL file."""
    import random

    rng = random.Random(int(seed))
    examples = list(iter_deepseek_prover_jsonl(path))
    if not examples:
        return []
    chosen = rng.choices(examples, k=int(n))
    return [
        prover_problem_text(ex, include_informal=include_informal, include_formal=include_formal) for ex in chosen
    ]
