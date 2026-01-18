from __future__ import annotations

import math

import pytest
import torch


class _ToyTokenizer:
    def __init__(self):
        self.n_vocab = 16

    def decode(self, token_ids):
        # Minimal: join ints as chars for testability.
        return "".join(chr(ord("a") + int(t)) for t in token_ids if 0 <= int(t) < 26)

    def encode(self, text: str, allowed_special=None):  # noqa: ARG002
        # Provide Harmony specials as single tokens; anything else returns a dummy id.
        if text == "<|start|>":
            return [1]
        if text == "<|end|>":
            return [2]
        if text == "<|message|>":
            return [3]
        if text == "<|channel|>":
            return [4]
        if text == "<|call|>":
            return [5]
        if text == "<|return|>":
            return [6]
        return [7]


def test_local_rollout_engine_shapes_and_stop_reason_cpu():
    from nmoe.rl.rollout_engine import LocalRolloutEngine, RolloutRequest, StopReason

    class _M(torch.nn.Module):
        def forward(self, tokens):
            b, t = tokens.shape
            vocab = 16
            logits = torch.full((b, t, vocab), -10.0, device=tokens.device, dtype=torch.float32)
            # Always make token 9 extremely likely.
            logits[:, -1, 9] = 10.0
            return logits

    enc = _ToyTokenizer()
    engine = LocalRolloutEngine(model=_M(), enc=enc, device="cpu")
    req = RolloutRequest(prompt_tokens=[1, 2, 3], n=2, max_new_tokens=4, eos_token_id=9, temperature=1.0, top_p=1.0)
    outs = engine.generate(req)
    assert len(outs) == 2
    for s in outs:
        assert s.prompt_len == 3
        assert s.completion_len >= 1
        assert len(s.tokens) == s.prompt_len + s.completion_len
        assert len(s.logprobs) == s.completion_len
        assert s.stop_reason == StopReason.EOS


def test_sample_top_p_logprob_matches_distribution():
    from nmoe.rl.rollout_engine import LocalRolloutEngine, RolloutRequest

    torch.manual_seed(0)

    class _M(torch.nn.Module):
        def forward(self, tokens):
            b, t = tokens.shape
            vocab = 16
            logits = torch.full((b, t, vocab), -20.0, device=tokens.device, dtype=torch.float32)
            # Two-token distribution at last step.
            logits[:, -1, 0] = 0.0  # exp(0)=1
            logits[:, -1, 1] = math.log(3.0)  # exp=3
            return logits

    enc = _ToyTokenizer()
    engine = LocalRolloutEngine(model=_M(), enc=enc, device="cpu")
    req = RolloutRequest(prompt_tokens=[1], n=1, max_new_tokens=1, eos_token_id=99, temperature=1.0, top_p=1.0)
    s = engine.generate(req)[0]
    assert s.completion_len == 1
    tok = s.tokens[-1]
    lp = s.logprobs[0]
    # probs: p0=1/4, p1=3/4
    expected = math.log(0.25) if tok == 0 else math.log(0.75)
    assert abs(lp - expected) < 1e-4


def test_top_p_sampling_scores_full_softmax():
    from nmoe.rl.rollout_engine import LocalRolloutEngine, RolloutRequest

    torch.manual_seed(0)

    class _M(torch.nn.Module):
        def forward(self, tokens):
            b, t = tokens.shape
            vocab = 16
            logits = torch.full((b, t, vocab), -20.0, device=tokens.device, dtype=torch.float32)
            # Make token 0 and 1 plausible; token 0 slightly higher.
            logits[:, -1, 0] = 0.0
            logits[:, -1, 1] = -0.1
            return logits

    enc = _ToyTokenizer()
    engine = LocalRolloutEngine(model=_M(), enc=enc, device="cpu")
    # With top_p=0.5, nucleus should keep only token 0 (since p0 > 0.5).
    req = RolloutRequest(prompt_tokens=[1], n=1, max_new_tokens=1, eos_token_id=99, temperature=1.0, top_p=0.5)
    s = engine.generate(req)[0]
    assert s.completion_len == 1
    assert s.tokens[-1] == 0

    # Returned logprob must be the full-softmax logprob, not the nucleus-renormalized logprob.
    expected = float(torch.log_softmax(torch.tensor([0.0, -0.1]), dim=-1)[0].item())
    assert abs(s.logprobs[0] - expected) < 1e-5


def test_require_harmony_tokenizer_rejects_missing_specials():
    from nmoe.rl.rollout_engine import require_harmony_tokenizer

    class _Bad:
        def encode(self, _t, allowed_special=None):  # noqa: ARG002
            return [1, 2]  # multi-token -> invalid

    with pytest.raises(ValueError):
        require_harmony_tokenizer(_Bad())


def test_logp_mean_from_logprobs_constant_norm():
    from nmoe.rl.rollout_engine import logp_mean_from_logprobs

    # 3 tokens of logp, constant normalize by 10.
    v = logp_mean_from_logprobs([-1.0, -2.0, -3.0], completion_len=3, max_length=10)
    assert abs(v - (-6.0 / 10.0)) < 1e-9

    # Variable normalize by completion length.
    v2 = logp_mean_from_logprobs([-1.0, -2.0, -3.0], completion_len=3, max_length=None)
    assert abs(v2 - (-6.0 / 3.0)) < 1e-9
