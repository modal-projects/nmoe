"""E2E smoke test for Math-V2 self-play loop (generator↔verifier↔meta).

This is Harmony-only (no custom tags or \\boxed parsing).
"""

from __future__ import annotations

import pytest


class _GreedyTokenizer:
    def __init__(self, vocab: dict[str, int], *, prompt_token: int):
        self.vocab = dict(vocab)
        self._inv = {v: k for k, v in self.vocab.items()}
        self._tok_texts = sorted(self.vocab.keys(), key=len, reverse=True)
        self._prompt_token = int(prompt_token)
        self.n_vocab = max(self.vocab.values()) + 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text in self.vocab:
            return [self.vocab[text]]
        # Collapse unknown prompts to a single token.
        return [self._prompt_token]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # Treat eos token id 0 as a special token that decodes to "" (HF-like).
        out = []
        for t in token_ids:
            if int(t) == 0:
                continue
            out.append(self._inv.get(t, "<prompt>"))
        return "".join(out)


def test_mathv2_selfplay_loop_produces_scores_and_records():
    import torch

    from nmoe.rl.mathv2_selfplay import MathV2SelfPlayConfig, MathV2SelfPlayRunner
    from nmoe.rl.rollout_engine import LocalRolloutEngine

    eos = 0
    proof_text = (
        "<|start|>assistant<|channel|>analysis<|message|>p<|end|>"
        "<|start|>assistant<|channel|>final<|message|>proof<|end|>"
    )
    verifier_text = (
        "<|start|>assistant<|channel|>analysis<|message|>v<|end|>"
        "<|start|>assistant<|channel|>final<|message|>1<|end|>"
    )
    meta_text = (
        "<|start|>assistant<|channel|>analysis<|message|>m<|end|>"
        "<|start|>assistant<|channel|>final<|message|>1<|end|>"
    )

    vocab = {
        proof_text: 1,
        verifier_text: 2,
        meta_text: 3,
        "<eos>": eos,
    }
    enc = _GreedyTokenizer(vocab, prompt_token=999)

    def _fixed_model(out_token: int):
        class _M(torch.nn.Module):
            def forward(self, tokens):
                b, t = tokens.shape
                vocab_size = max(vocab.values()) + 1
                logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)
                # Emit out_token then eos.
                idx = t - 1  # includes prompt
                next_id = out_token if idx == 0 else eos
                logits[:, -1, next_id] = 0.0
                return logits

        return _M()

    runner = MathV2SelfPlayRunner(
        generator_engine=LocalRolloutEngine(model=_fixed_model(vocab[proof_text]), enc=enc, device="cpu"),
        verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[verifier_text]), enc=enc, device="cpu"),
        meta_verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[meta_text]), enc=enc, device="cpu"),
        enc=enc,
        config=MathV2SelfPlayConfig(
            max_new_tokens_proof=4,
            max_new_tokens_verifier=8,
            max_new_tokens_meta=8,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=eos,
        ),
    )

    sample = runner.run_one("P")
    assert sample.proof.text == proof_text
    assert sample.verifier.score == "1"
    assert sample.meta.score == "1"
    assert sample.accepted is True

    # Token-exact transcript invariants (no tools => empty tool_events).
    sample.proof.record.assert_token_exact(enc)
    sample.verifier.record.assert_token_exact(enc)
    sample.meta.record.assert_token_exact(enc)


def test_mathv2_selfplay_emit_writes_only_accepted(tmp_path):
    import torch

    from nmoe.rl.mathv2_selfplay import MathV2SelfPlayConfig, MathV2SelfPlayEmitConfig, MathV2SelfPlayRunner
    from nmoe.rl.rollout_engine import LocalRolloutEngine

    eos = 0
    proof_text = "<|start|>assistant<|channel|>final<|message|>proof<|end|>"
    verifier_text = "<|start|>assistant<|channel|>final<|message|>1<|end|>"
    meta_text = "<|start|>assistant<|channel|>final<|message|>1<|end|>"

    vocab = {proof_text: 1, verifier_text: 2, meta_text: 3, "<eos>": eos}
    enc = _GreedyTokenizer(vocab, prompt_token=999)

    def _fixed_model(out_token: int):
        class _M(torch.nn.Module):
            def forward(self, tokens):
                b, t = tokens.shape
                vocab_size = max(vocab.values()) + 1
                logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)
                idx = t - 1
                next_id = out_token if idx == 0 else eos
                logits[:, -1, next_id] = 0.0
                return logits

        return _M()

    runner = MathV2SelfPlayRunner(
        generator_engine=LocalRolloutEngine(model=_fixed_model(vocab[proof_text]), enc=enc, device="cpu"),
        verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[verifier_text]), enc=enc, device="cpu"),
        meta_verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[meta_text]), enc=enc, device="cpu"),
        enc=enc,
        config=MathV2SelfPlayConfig(
            max_new_tokens_proof=4,
            max_new_tokens_verifier=8,
            max_new_tokens_meta=8,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=eos,
        ),
    )

    accepted = runner.run_one("P")
    rejected = field_replace_sample(accepted, accepted=False)

    out = runner.emit([accepted, rejected], cfg=MathV2SelfPlayEmitConfig(out_dir=tmp_path / "out"))
    v_lines = (out["proof_verifier_train"]).read_text(encoding="utf-8").strip().splitlines()
    m_lines = (out["proof_meta_verifier_train"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(v_lines) == 1
    assert len(m_lines) == 1
    counts = (out["counts"]).read_text(encoding="utf-8")
    assert '"proof_verifier_train": 1' in counts


def test_mathv2_selfplay_emit_writes_replay_bundles(tmp_path):
    import torch

    from nmoe.rl.mathv2_selfplay import MathV2SelfPlayConfig, MathV2SelfPlayEmitConfig, MathV2SelfPlayRunner
    from nmoe.rl.rollout_engine import LocalRolloutEngine

    eos = 0
    proof_text = "<|start|>assistant<|channel|>final<|message|>proof<|end|>"
    verifier_text = "<|start|>assistant<|channel|>final<|message|>1<|end|>"
    meta_text = "<|start|>assistant<|channel|>final<|message|>1<|end|>"

    vocab = {proof_text: 1, verifier_text: 2, meta_text: 3, "<eos>": eos}
    enc = _GreedyTokenizer(vocab, prompt_token=999)

    def _fixed_model(out_token: int):
        class _M(torch.nn.Module):
            def forward(self, tokens):
                b, t = tokens.shape
                vocab_size = max(vocab.values()) + 1
                logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)
                idx = t - 1
                next_id = out_token if idx == 0 else eos
                logits[:, -1, next_id] = 0.0
                return logits

        return _M()

    runner = MathV2SelfPlayRunner(
        generator_engine=LocalRolloutEngine(model=_fixed_model(vocab[proof_text]), enc=enc, device="cpu"),
        verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[verifier_text]), enc=enc, device="cpu"),
        meta_verifier_engine=LocalRolloutEngine(model=_fixed_model(vocab[meta_text]), enc=enc, device="cpu"),
        enc=enc,
        config=MathV2SelfPlayConfig(
            max_new_tokens_proof=4,
            max_new_tokens_verifier=8,
            max_new_tokens_meta=8,
            temperature=1.0,
            top_p=1.0,
            eos_token_id=eos,
        ),
    )
    accepted = runner.run_one("P")

    replay_dir = tmp_path / "replay"
    runner.emit(
        [accepted],
        cfg=MathV2SelfPlayEmitConfig(
            out_dir=tmp_path / "out",
            replay_dir=replay_dir,
            replay_sample_every=1,
            run_id="r",
            seed=0,
            rank=0,
        ),
    )

    base = replay_dir / "r" / "rank_000" / "step_00000000"
    assert (base / "mathv2_proof_s00" / "trajectory_record.json").exists()
    assert (base / "mathv2_verifier_s00" / "trajectory_record.json").exists()
    assert (base / "mathv2_meta_s00" / "trajectory_record.json").exists()


def field_replace_sample(sample, **kwargs):
    from dataclasses import replace

    return replace(sample, **kwargs)
