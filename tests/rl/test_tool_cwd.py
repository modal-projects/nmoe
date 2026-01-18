"""Kill tests for tool execution working directory (cwd) semantics."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest


def test_toolcall_inherits_executor_cwd_for_relative_paths(tmp_path: Path):
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tools import AsyncToolExecutor, ToolCall, ToolConfig, ToolType

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    sentinel = f"sentinel_{uuid4().hex}.txt"
    out = f"out_{uuid4().hex}.txt"
    (ws / sentinel).write_text("SENTINEL\n", encoding="utf-8")

    executor = AsyncToolExecutor(
        ToolConfig(
            executor_type="codex_python",
            allow_network=False,
            allowed_paths=[str(ws)],
            cwd=str(ws),
        )
    )
    try:
        # ToolCall.cwd is intentionally left unspecified; it must inherit ToolConfig.cwd.
        call = ToolCall(
            type=ToolType.BASH,
            command=f"pwd && printf OK > {out} && cat {sentinel} && cat {out}",
            timeout_ms=30_000,
        )
        result = asyncio.run(executor.execute_one(call))
    finally:
        executor.close()

    assert result.success
    lines = [ln.strip() for ln in result.output.splitlines() if ln.strip()]
    assert lines[0] == str(ws)
    assert lines[-2] == "SENTINEL"
    assert lines[-1] == "OK"
    assert (ws / out).read_text(encoding="utf-8") == "OK"


def test_toolcall_cannot_write_outside_allowed_paths(tmp_path: Path):
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tools import AsyncToolExecutor, ToolCall, ToolConfig, ToolType

    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    # Landlock always allows writes under /tmp (see codex sandbox rules), so this
    # must target a non-/tmp location to validate the allowlist.
    escape = Path(__file__).resolve().parents[2] / f"escape_{uuid4().hex}.txt"
    if escape.exists():
        escape.unlink()

    executor = AsyncToolExecutor(
        ToolConfig(
            executor_type="codex_python",
            allow_network=False,
            allowed_paths=[str(ws)],
            cwd=str(ws),
        )
    )
    try:
        call = ToolCall(
            type=ToolType.BASH,
            command=f"printf NO > {escape!s}",
            timeout_ms=30_000,
        )
        result = asyncio.run(executor.execute_one(call))
    finally:
        executor.close()

    assert result.success is False
    assert not escape.exists()


class _GreedyTokenizer:
    """Tokenizer stub that greedily matches known token strings.

    encode() falls back to a single prompt token for unknown prompt text.
    """

    def __init__(self, vocab: dict[str, int], *, prompt_token: int):
        self.vocab = dict(vocab)
        self._inv = {v: k for k, v in self.vocab.items()}
        self._tok_texts = sorted(self.vocab.keys(), key=len, reverse=True)
        self._prompt_token = int(prompt_token)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        if text in self.vocab:
            return [self.vocab[text]]

        out: list[int] = []
        i = 0
        while i < len(text):
            for t in self._tok_texts:
                if text.startswith(t, i):
                    out.append(self.vocab[t])
                    i += len(t)
                    break
            else:
                # Unknown prompt text: collapse to a single token.
                return [self._prompt_token]
        return out

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._inv.get(t, "<prompt>") for t in token_ids)

    @property
    def eos_token_id(self) -> int:
        return 0


def test_generate_batch_multi_turn_scopes_cwd_to_task_workspace(tmp_path: Path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("generate_turn_async uses CUDA tensors; requires a GPU")

    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.tools import AsyncToolExecutor, ToolConfig
    from nmoe.rl.train_agentic import AgenticTrainConfig, generate_batch_multi_turn
    from nmoe.rl.tasks.agentic import CodeEditTask

    ws = tmp_path / "workspace"
    other = tmp_path / "other"
    ws.mkdir(parents=True, exist_ok=True)
    other.mkdir(parents=True, exist_ok=True)

    ws_s = str(ws)
    other_s = str(other)

    # Token vocab for a single bash tool call: <|call|>bash\n{cmd}<|end|>
    #
    # Use a command that prints cwd without a trailing newline, so the tool return
    # encodes to a fixed-length token pattern: <|return|>\n{cwd}\n<|end|>.
    pycwd_cmd = 'printf %s "$PWD"'
    eos = 0
    vocab = {
        "<|call|>": 1,
        "<|end|>": 2,
        "<|return|>": 3,
        "bash": 4,
        "\n": 5,
        pycwd_cmd: 6,
        ws_s: 7,
        other_s: 8,
        "<think>ok</think><answer>ok</answer>": 9,
        "<eos>": eos,
    }
    prompt_token = 123
    enc = _GreedyTokenizer(vocab, prompt_token=prompt_token)

    stage0 = [vocab["<|call|>"], vocab["bash"], vocab["\n"], vocab[pycwd_cmd], vocab["<|end|>"]]
    stage1 = [vocab["<think>ok</think><answer>ok</answer>"], eos]

    return_id = vocab["<|return|>"]
    return_block_len = 5  # <|return|>\n{cwd}\n<|end|>
    prompt_len = 1  # enc.encode(prompt) falls back to [prompt_token]

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

    # Base executor is intentionally configured to the wrong cwd.
    base_executor = AsyncToolExecutor(
        ToolConfig(
            executor_type="codex_python",
            allow_network=False,
            allowed_paths=[str(other)],
            cwd=str(other),
        )
    )
    try:
        task = CodeEditTask(
            task_id="t",
            issue_description="x",
            repo_path=str(ws),
            test_command="pwd",
        )
        cfg = AgenticTrainConfig(
            mode="multi_turn",
            format_type="r1zero",
            enable_tools=True,
            group_size=1,
            max_new_tokens=64,
            max_tool_rounds=2,
        )

        turns, _rewards = generate_batch_multi_turn(
            _Model(),
            enc,
            [task],
            cfg,
            base_executor,
            device=torch.device("cuda"),
        )
    finally:
        base_executor.close()

    assert len(turns) == 1
    turn = turns[0]
    assert len(turn.tool_calls) == 1
    assert len(turn.tool_results) == 1
    assert turn.tool_results[0].success
    assert turn.tool_results[0].output.strip() == ws_s
