"""End-to-end turn -> tools -> rewards smoke tests.

This is a functional "kill test": it exercises the intended control flow:
generate -> parse tool calls (token-native) -> execute (codex_python sandbox) ->
inject returns -> continue generation -> compute_all_rewards().
"""

from __future__ import annotations

import re

import pytest


class _GreedyTokenizer:
    """Tokenizer stub that encodes by longest-string match over a fixed vocab.

    Required because generate_turn_async() inserts tool returns via
    enc.encode("<|return|>\\n...\\n<|end|>").
    """

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


class _AnswerTask:
    def __init__(self, gold: str):
        self._gold = gold

    def extract_answer(self, response: str) -> str | None:
        m = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def verify(self, answer: str | None) -> bool:
        return answer == self._gold


class TestTurnRewardsE2E:
    def test_multi_tool_turn_flows_into_compute_all_rewards(self, tmp_path):
        import asyncio
        import torch

        from nmoe.rl.rewards_gdpo import TrajectoryContext, compute_all_rewards
        from nmoe.rl.turns import generate_turn_async
        from nmoe.rl.tools import AsyncToolExecutor, ToolConfig

        if not torch.cuda.is_available():
            pytest.skip("generate_turn_async uses CUDA tensors; requires a GPU")

        pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

        eos = 0
        py_out = "PY=4"
        sh_out = "SH=ok"

        py_code = f"import sys; sys.stdout.write({py_out!r})"
        sh_cmd = f"echo -n {sh_out!r}"

        # Final message is r1zero format so compute_all_rewards can validate structure + task correctness.
        final_msg = f"<think>I used {py_out} and {sh_out}.</think><answer>42</answer>"

        vocab = {
            "<|call|>": 1,
            "<|end|>": 2,
            "<|return|>": 3,
            "python": 4,
            "bash": 5,
            "\n": 6,
            py_code: 7,
            sh_cmd: 8,
            py_out: 9,
            sh_out: 10,
            "pre": 11,
            "mid": 12,
            final_msg: 13,
            "<eos>": eos,
        }
        enc = _GreedyTokenizer(vocab)

        stage0 = [
            vocab["pre"],
            vocab["<|call|>"],
            vocab["python"],
            vocab["\n"],
            vocab[py_code],
            vocab["<|end|>"],
        ]
        stage1 = [
            vocab["mid"],
            vocab[py_out],  # ensures tool_output_used for the python call
            vocab["<|call|>"],
            vocab["bash"],
            vocab["\n"],
            vocab[sh_cmd],
            vocab["<|end|>"],
        ]
        stage2 = [
            vocab[final_msg],
            eos,
        ]
        return_id = vocab["<|return|>"]
        return_block_len = 5  # <|return|>\n{out}\n<|end|>
        prompt_len = 1  # prompt_ids=[123]

        class _ToolAwareModel(torch.nn.Module):
            def forward(self, tokens):
                b, t = tokens.shape
                vocab_size = max(vocab.values()) + 1
                logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)

                ids = tokens[0].tolist()
                return_idxs = [i for i, tok in enumerate(ids) if tok == return_id]

                if len(return_idxs) == 0:
                    idx = len(ids) - prompt_len
                    next_id = stage0[idx] if idx < len(stage0) else eos
                elif len(return_idxs) == 1:
                    after = return_idxs[0] + return_block_len
                    idx = max(0, len(ids) - after)
                    next_id = stage1[idx] if idx < len(stage1) else eos
                else:
                    after = return_idxs[1] + return_block_len
                    idx = max(0, len(ids) - after)
                    next_id = stage2[idx] if idx < len(stage2) else eos

                logits[:, -1, next_id] = 0.0
                return logits

        executor = AsyncToolExecutor(
            ToolConfig(
                executor_type="codex_python",
                allow_network=False,
                allowed_paths=[str(tmp_path)],
            )
        )
        try:
            turn = asyncio.run(
                generate_turn_async(
                    _ToolAwareModel(),
                    enc=enc,
                    prompt_ids=[123],
                    tool_executor=executor,
                    max_new_tokens=256,
                    max_tool_rounds=4,
                    eos_token_id=eos,
                    temperature=1.0,
                    top_p=1.0,
                )
            )
        finally:
            executor.close()

        assert [c.type.value for c in turn.tool_calls] == ["python", "bash"]
        assert turn.tool_results[0].success and py_out in turn.tool_results[0].output
        assert turn.tool_results[1].success and sh_out in turn.tool_results[1].output
        assert turn.final_response is not None

        ctx = TrajectoryContext(
            response_text=turn.final_response,
            tool_sites=turn.to_tool_sites(),
            task=_AnswerTask(gold="42"),
            reasoning_tokens=turn.reasoning_tokens,
            format_type="r1zero",
        )
        signals = compute_all_rewards(ctx)

        # Structure (r1zero -> analysis/final channels)
        assert signals.has_reasoning == 1.0
        assert signals.has_final_response == 1.0
        assert signals.chan_analysis_nonempty == 1.0
        assert signals.chan_final_nonempty == 1.0

        # Tools: both executed and both outputs used downstream.
        assert signals.has_tool_use == 1.0
        assert signals.tool_executed == 1.0
        assert signals.tool_output_used == 1.0
        assert signals.python_runs == 1.0
        assert signals.bash_succeeds == 1.0

        # Task correctness from <answer>42</answer>
        assert signals.answer_correct == 1.0

