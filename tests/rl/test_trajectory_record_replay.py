"""Token-exact trajectory record replay tests.

Correctness trap: if any stage reconstructs prompts from text (retokenization),
logprobs silently drift. The contract we enforce is byte-for-byte equivalence
of the rollout transcript (token IDs) plus tool outputs.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest


class _GreedyTokenizer:
    """Tokenizer stub that encodes by longest-string match over a fixed vocab."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self._inv = {v: k for k, v in self.vocab.items()}
        self._tok_texts = sorted(self.vocab.keys(), key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        out: list[int] = []
        i = 0
        while i < len(text):
            for t in self._tok_texts:
                if text.startswith(t, i):
                    out.append(self.vocab[t])
                    i += len(t)
                    break
            else:
                raise KeyError(f"cannot encode at pos={i}: {text[i:i+20]!r}")
        return out

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._inv[t] for t in token_ids)


def test_trajectory_record_token_exact_and_tool_replay(tmp_path: Path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("generate_turn_async uses CUDA tensors; requires a GPU")

    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tools import AsyncToolExecutor, ToolConfig
    from nmoe.rl.turns import generate_turn_async

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "sentinel.txt").write_text("OK", encoding="utf-8")  # no trailing newline

    eos = 0
    cmd = "cat sentinel.txt"
    final_msg = "<think>ok</think><answer>ok</answer>"

    vocab = {
        "<|call|>": 1,
        "<|end|>": 2,
        "<|return|>": 3,
        "bash": 4,
        "\n": 5,
        cmd: 6,
        "OK": 7,
        final_msg: 8,
        "<eos>": eos,
    }
    enc = _GreedyTokenizer(vocab)

    stage0 = [vocab["<|call|>"], vocab["bash"], vocab["\n"], vocab[cmd], vocab["<|end|>"]]
    stage1 = [vocab[final_msg], eos]
    return_id = vocab["<|return|>"]
    return_block_len = 5  # <|return|>\nOK\n<|end|>
    prompt_len = 1

    class _Model(torch.nn.Module):
        def forward(self, tokens):
            b, t = tokens.shape
            vocab_size = max(vocab.values()) + 1
            logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)

            ids = tokens[0].tolist()
            if return_id not in ids:
                idx = len(ids) - prompt_len
                next_id = stage0[idx] if idx < len(stage0) else eos
            else:
                r0 = ids.index(return_id)
                after = r0 + return_block_len
                idx = max(0, len(ids) - after)
                next_id = stage1[idx] if idx < len(stage1) else eos

            logits[:, -1, next_id] = 0.0
            return logits

    executor = AsyncToolExecutor(
        ToolConfig(
            executor_type="codex_python",
            allow_network=False,
            allowed_paths=[str(ws)],
            cwd=str(ws),
        )
    )
    try:
        turn = asyncio.run(
            generate_turn_async(
                _Model(),
                enc=enc,
                prompt_ids=[123],
                tool_executor=executor,
                max_new_tokens=128,
                max_tool_rounds=2,
                eos_token_id=eos,
                temperature=1.0,
                top_p=1.0,
            )
        )
        record = turn.record
        assert record is not None
        record.assert_token_exact(enc)
        assert record.tool_events[0].result.failure_category == "ok"

        # Online replay: re-execute tools and require output equivalence.
        asyncio.run(record.replay_tools(tokenizer=enc, tool_executor=executor))

        # JSON round-trip must preserve invariants.
        rec2 = type(record).from_json(record.to_json())
        rec2.assert_token_exact(enc)

        # Kill test: any token drift breaks invariants.
        bad_tokens = list(record.tokens)
        bad_tokens[record.tool_events[0].call_span[0]] = vocab["OK"]  # corrupt <|call|>
        bad = replace(record, tokens=bad_tokens)
        with pytest.raises(ValueError):
            bad.assert_token_exact(enc)
    finally:
        executor.close()
