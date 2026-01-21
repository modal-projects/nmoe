"""Process Reward Model (PRM) utilities.

Implements a minimal "Let's Verify Step by Step" style scoring interface:
- Build per-step prompts with explicit step boundaries.
- Compute P(step is correct) from next-token probabilities for a configured label token.
- Aggregate a full-solution score as the product of per-step correctness probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PRMLabelSpec:
    """Binary PRM label token spec (single-token strings)."""

    correct: str = " 1"
    incorrect: str = " 0"


def resolve_prm_label_token_ids(enc, spec: PRMLabelSpec) -> tuple[int, int]:
    """Resolve label strings to token ids (must each be exactly one token).

    This is intentionally strict: if the tokenizer does not represent the labels
    as single tokens, the PRM next-token scoring contract is ambiguous.
    """
    if not hasattr(enc, "encode"):
        raise TypeError("tokenizer must implement encode()")
    tok_ok = enc.encode(spec.correct)
    tok_bad = enc.encode(spec.incorrect)
    if not (isinstance(tok_ok, list) and isinstance(tok_bad, list)):
        raise TypeError("tokenizer.encode() must return list[int]")
    if len(tok_ok) != 1 or len(tok_bad) != 1:
        raise ValueError(
            "PRM labels must be single tokens under the tokenizer. "
            f"Got correct={spec.correct!r}->{tok_ok}, incorrect={spec.incorrect!r}->{tok_bad}."
        )
    ok_id, bad_id = int(tok_ok[0]), int(tok_bad[0])
    if ok_id == bad_id:
        raise ValueError("PRM correct/incorrect label token ids must be distinct")
    return ok_id, bad_id


def format_prm_step_prompt(problem: str, steps: Sequence[str], *, step_idx: int) -> str:
    """Prompt prefix for PRM next-token scoring at a given step boundary."""
    if step_idx < 0 or step_idx > len(steps):
        raise ValueError(f"step_idx out of range (got {step_idx}, steps={len(steps)})")
    lines: list[str] = [
        "Problem:",
        str(problem).strip(),
        "",
        "Solution:",
    ]
    for i, s in enumerate(steps[:step_idx], start=1):
        s = str(s).strip()
        if not s:
            continue
        lines.append(f"Step {i}: {s}")
    lines.extend(["", "Step correctness:"])
    return "\n".join(lines)


@torch.inference_mode()
def prm_step_correct_probs(
    model,
    *,
    enc,
    problem: str,
    steps: Sequence[str],
    correct_token_id: int,
    device: torch.device,
) -> list[float]:
    """Compute P(correct) at each step boundary via next-token probability."""
    if not isinstance(correct_token_id, int):
        raise TypeError("correct_token_id must be int")
    probs: list[float] = []
    for i in range(1, len(steps) + 1):
        prefix = format_prm_step_prompt(problem, steps, step_idx=i)
        ids = enc.encode(prefix)
        if not ids:
            raise ValueError("empty prompt_ids after encoding")
        toks = torch.tensor([ids], device=device, dtype=torch.long)
        logits = model(toks)[:, -1, :].squeeze(0)
        p = torch.softmax(logits, dim=-1)[int(correct_token_id)]
        probs.append(float(p.detach().item()))
    return probs


def prm_solution_score_product(step_correct_probs: Sequence[float]) -> float:
    """Aggregate PRM step correctness probabilities into a solution score."""
    s = 1.0
    for p in step_correct_probs:
        fp = float(p)
        if not (0.0 <= fp <= 1.0) or math.isnan(fp):
            raise ValueError(f"invalid step probability: {p!r}")
        s *= fp
    return float(s)


def prm_solution_log_score(step_correct_probs: Sequence[float]) -> float:
    """Log-space sum of log(step_prob), stable for long solutions."""
    total = 0.0
    for p in step_correct_probs:
        fp = float(p)
        if fp <= 0.0:
            return float("-inf")
        if fp > 1.0 or math.isnan(fp):
            raise ValueError(f"invalid step probability: {p!r}")
        total += math.log(fp)
    return float(total)

