from __future__ import annotations

import math

import pytest
import torch


class _MapTokenizer:
    def __init__(self):
        self._v: dict[str, int] = {}

    def encode(self, text: str):
        if text not in self._v:
            self._v[text] = len(self._v)
        return [self._v[text]]


class _PrefixProbModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int, next_probs: dict[int, float], correct_id: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.next_probs = dict(next_probs)
        self.correct_id = int(correct_id)

    def forward(self, tokens):
        # tokens: [B,T]
        b, t = tokens.shape
        logits = torch.full((b, t, self.vocab_size), float("-inf"), device=tokens.device, dtype=torch.float32)
        last = int(tokens[0, -1].item())
        p = float(self.next_probs.get(last, 0.5))
        # binary dist: p on correct_id, 1-p on some other id (0)
        other_id = 0 if self.correct_id != 0 else 1
        logits[:, -1, self.correct_id] = math.log(max(p, 1e-9))
        logits[:, -1, other_id] = math.log(max(1.0 - p, 1e-9))
        return logits


def test_prm_solution_score_product_and_log():
    from nmoe.rl.prm import prm_solution_log_score, prm_solution_score_product

    probs = [0.5, 0.25, 1.0]
    assert prm_solution_score_product(probs) == pytest.approx(0.125)
    assert prm_solution_log_score(probs) == pytest.approx(math.log(0.125))


def test_prm_step_correct_probs_uses_step_boundaries():
    from nmoe.rl.prm import PRMLabelSpec, prm_solution_score_product, prm_step_correct_probs, resolve_prm_label_token_ids

    enc = _MapTokenizer()
    # Single-token labels under this tokenizer by construction.
    ok_id, _bad_id = resolve_prm_label_token_ids(enc, PRMLabelSpec(correct="OK", incorrect="BAD"))

    problem = "P"
    steps = ["s1", "s2", "s3"]
    # Build the exact prefix ids we expect the scorer to encode, and map each to a p(correct).
    from nmoe.rl.prm import format_prm_step_prompt

    pmap: dict[int, float] = {}
    for i, p in enumerate([0.9, 0.5, 0.1], start=1):
        pid = enc.encode(format_prm_step_prompt(problem, steps, step_idx=i))[0]
        pmap[pid] = p

    model = _PrefixProbModel(vocab_size=128, next_probs=pmap, correct_id=ok_id).eval()
    probs = prm_step_correct_probs(
        model,
        enc=enc,
        problem=problem,
        steps=steps,
        correct_token_id=ok_id,
        device=torch.device("cpu"),
    )
    assert probs == pytest.approx([0.9, 0.5, 0.1])
    assert prm_solution_score_product(probs) == pytest.approx(0.045)
