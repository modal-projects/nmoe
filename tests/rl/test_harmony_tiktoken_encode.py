from __future__ import annotations

import pytest


def test_harmony_encode_allows_harmony_special_tokens():
    tiktoken = pytest.importorskip("tiktoken")

    from nmoe.rl.rewards_harmony import HARMONY_TOKENS, harmony_encode

    enc = tiktoken.get_encoding("o200k_harmony")

    # Baseline tiktoken behavior: Harmony specials are disallowed unless explicitly allowed.
    with pytest.raises(Exception):
        enc.encode(HARMONY_TOKENS["start"])

    ids = harmony_encode(enc, HARMONY_TOKENS["start"])
    assert isinstance(ids, list) and len(ids) == 1

    text = (
        "<|start|>assistant<|channel|>analysis<|message|>x<|end|>"
        "<|start|>assistant<|channel|>final<|message|>0<|end|>"
    )
    ids2 = harmony_encode(enc, text)
    assert isinstance(ids2, list) and len(ids2) > 1

