"""Tests for tool parsing + sandboxed execution.

These tests are intended to validate RL_DESIGN.md + TODO.md claims:
- Token-native tool parsing (no regex-only reliance)
- Sandboxed tool execution blocks network + blocks FS writes outside allowed paths
"""

from __future__ import annotations

import os
import platform
import textwrap
from pathlib import Path

import pytest


def test_codex_python_execute_python_tests_enforces_timeout():
    pytest.importorskip("codex_python", reason="codex_python module not built/available")

    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import codex_python
        try:
            codex_python.execute_python_tests("while True: pass\\n", "pass\\n", 200)
            print("unexpected_success")
        except Exception as e:
            print(type(e).__name__)
    """).strip()

    # Run in a subprocess so a timeout bug can't hang the whole test process.
    p = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=2.0,
    )
    assert "Timeout" in (p.stdout + p.stderr)


class _DummyTokenizer:
    """Minimal tokenizer stub for TokenLevelParser tests.

    This models the RL_DESIGN expectation that tool markers are special tokens.
    """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self._inv = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text in self.vocab:
            return [self.vocab[text]]
        raise KeyError(f"unknown token text: {text!r}")

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        parts: list[str] = []
        for tid in token_ids:
            if tid not in self._inv:
                raise KeyError(f"unknown token id: {tid}")
            parts.append(self._inv[tid])
        return "".join(parts)


class _GreedyTokenizer(_DummyTokenizer):
    """Tokenizer stub that can encode concatenated marker strings.

    Needed for generate_turn_async() integration tests because tool returns are
    inserted via enc.encode("<|return|>\\n...\\n<|end|>").
    """

    def __init__(self, vocab: dict[str, int]):
        super().__init__(vocab)
        # Longest-first matching avoids prefix ambiguities.
        self._tok_texts = sorted(self.vocab.keys(), key=len, reverse=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
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


class TestTokenLevelParserSemantics:
    def test_parse_single_call_and_mask(self):
        from nmoe.rl.tools.parsers import TokenLevelParser
        from nmoe.rl.tools import ToolType

        tok = _DummyTokenizer(
            {
                "<|call|>": 1,
                "<|end|>": 2,
                "python": 3,
                "\n": 4,
                "print('hi')": 5,
            }
        )
        parser = TokenLevelParser(tok)
        token_ids = [1, 3, 4, 5, 2]

        calls = parser.parse(token_ids)
        assert len(calls) == 1
        parsed = calls[0]
        assert parsed.call.type == ToolType.PYTHON
        assert parsed.call.code == "print('hi')"
        assert parsed.full_span == (0, 5)

        mask = parser.get_content_mask(token_ids, mask_markers=True)
        assert mask == [True, False, False, True, True]

    def test_parse_truncated_call_no_end_marker(self):
        from nmoe.rl.tools.parsers import TokenLevelParser
        from nmoe.rl.tools import ToolType

        tok = _DummyTokenizer(
            {
                "<|call|>": 10,
                "bash": 11,
                "\n": 12,
                "echo hello": 13,
            }
        )
        parser = TokenLevelParser(tok)
        token_ids = [10, 11, 12, 13]

        calls = parser.parse(token_ids)
        assert len(calls) == 1
        parsed = calls[0]
        assert parsed.call.type == ToolType.BASH
        assert parsed.call.command == "echo hello"
        assert parsed.end_span is None
        assert parsed.full_span == (0, 4)

    def test_multiple_calls(self):
        from nmoe.rl.tools.parsers import TokenLevelParser
        from nmoe.rl.tools import ToolType

        tok = _DummyTokenizer(
            {
                "<|call|>": 1,
                "<|end|>": 2,
                "python": 3,
                "bash": 4,
                "\n": 5,
                "print('a')": 6,
                "echo b": 7,
            }
        )
        parser = TokenLevelParser(tok)
        token_ids = [1, 3, 5, 6, 2, 1, 4, 5, 7, 2]
        calls = parser.parse(token_ids)

        assert [c.call.type for c in calls] == [ToolType.PYTHON, ToolType.BASH]
        assert calls[0].full_span == (0, 5)
        assert calls[1].full_span == (5, 10)


class TestTurnsTokenParsing:
    def test_parse_tool_call_from_tokens_returns_end_pos(self):
        from nmoe.rl.turns import _parse_tool_call_from_tokens
        from nmoe.rl.tools import ToolType

        tok = _DummyTokenizer(
            {
                "<|call|>": 1,
                "<|end|>": 2,
                "python": 3,
                "\n": 4,
                "print('hi')": 5,
            }
        )
        call, end_pos = _parse_tool_call_from_tokens([1, 3, 4, 5, 2], tok)
        assert call is not None
        assert call.type == ToolType.PYTHON
        assert call.code == "print('hi')"
        assert end_pos == 5

    def test_generate_turn_async_executes_single_tool_call(self, tmp_path: Path):
        import asyncio
        import torch

        from nmoe.rl.turns import generate_turn_async
        from nmoe.rl.rewards_tools import compute_all_tool_rewards
        from nmoe.rl.tools import AsyncToolExecutor, ToolConfig, ToolType

        if not torch.cuda.is_available():
            pytest.skip("generate_turn_async uses CUDA tensors; requires a GPU")

        eos = 0
        code = "import sys; sys.stdout.write('VALUE=4')"
        vocab = {
            "<|call|>": 1,
            "<|end|>": 2,
            "<|return|>": 3,
            "python": 4,
            "\n": 5,
            code: 6,
            "final": 7,
            "VALUE=4": 8,
            "<eos>": eos,
        }
        enc = _GreedyTokenizer(vocab)

        # This model is reactive to tool returns: it only emits VALUE=4 after
        # <|return|>...VALUE=4...<|end|> is present in its context.
        pre = [
            vocab["final"],  # reasoning token before the tool call (becomes messages[0])
            vocab["<|call|>"],
            vocab["python"],
            vocab["\n"],
            vocab[code],
            vocab["<|end|>"],
        ]
        post = [
            vocab["final"],  # post-tool reasoning (becomes messages[1])
            vocab["VALUE=4"],
            eos,
        ]
        return_block_len = 5  # <|return|>\nVALUE=4\n<|end|>

        class _ToolAwareFixedNext(torch.nn.Module):
            def forward(self, tokens):
                b, t = tokens.shape
                vocab_size = max(vocab.values()) + 1
                logits = torch.full((b, t, vocab_size), -1e9, device=tokens.device, dtype=torch.float32)

                ids = tokens[0].tolist()
                prompt_len = 1  # prompt_ids=[123] in this test

                if vocab["<|return|>"] not in ids:
                    idx = len(ids) - prompt_len
                    next_id = pre[idx] if idx < len(pre) else eos
                else:
                    r0 = ids.index(vocab["<|return|>"])
                    idx = max(0, len(ids) - (r0 + return_block_len))
                    next_id = post[idx] if idx < len(post) else eos

                logits[:, -1, next_id] = 0.0
                return logits

        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")
        _ = codex
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
                    _ToolAwareFixedNext(),
                    enc=enc,
                    prompt_ids=[123],
                    tool_executor=executor,
                    max_new_tokens=64,
                    max_tool_rounds=2,
                    eos_token_id=eos,
                    temperature=1.0,
                    top_p=1.0,
                )
            )
        finally:
            executor.close()

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].type == ToolType.PYTHON
        assert turn.tool_calls[0].code == code
        assert len(turn.tool_results) == 1
        assert turn.tool_results[0].success
        assert "VALUE=4" in turn.tool_results[0].output
        assert turn.final_response is not None
        assert "VALUE=4" in turn.final_response

        # Reward path: tool output is detected as used in subsequent reasoning.
        tool_sites = turn.to_tool_sites()
        rewards = compute_all_tool_rewards(tool_sites)
        assert rewards["tool_output_used"] == 1.0


class TestCodexSandboxEnforcement:
    def test_async_tool_executor_scatter_gather_order(self, tmp_path: Path):
        import asyncio

        pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

        from nmoe.rl.tools import AsyncToolExecutor, ToolCall, ToolConfig, ToolType

        executor = AsyncToolExecutor(
            ToolConfig(
                executor_type="codex_python",
                allow_network=False,
                allowed_paths=[str(tmp_path)],
            )
        )
        try:
            calls = [
                ToolCall(type=ToolType.PYTHON, code="print('A')"),
                ToolCall(type=ToolType.PYTHON, code="print('B')"),
            ]

            async def _run():
                call_ids = executor.scatter(calls)
                return await executor.gather(call_ids, timeout=30.0)

            results = asyncio.run(_run())
            assert "A" in results[0].output
            assert "B" in results[1].output
        finally:
            executor.close()

    def test_exec_python_smoke(self):
        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")
        executor = codex.CodexExecutor()
        result = executor.exec_python("print(2 + 2)")
        assert result.success
        assert "4" in result.stdout

    def test_exec_bash_smoke(self):
        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")
        executor = codex.CodexExecutor()
        result = executor.exec_bash("echo hello")
        assert result.success
        assert "hello" in result.stdout

    def test_exec_tests_pass_and_fail(self):
        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")
        executor = codex.CodexExecutor()

        ok = executor.exec_tests("def add(a, b): return a + b", "assert add(1, 2) == 3")
        assert ok.success

        bad = executor.exec_tests("def add(a, b): return a - b", "assert add(1, 2) == 3")
        assert not bad.success

    def test_network_toggle_blocks_inet_sockets(self, tmp_path: Path):
        if platform.system().lower() != "linux":
            pytest.skip("Sandbox enforcement is only supported on Linux.")

        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

        code_inet = textwrap.dedent(
            """
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.close()
            print("inet_ok")
            """
        )
        code_unix = textwrap.dedent(
            """
            import socket
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.close()
            print("unix_ok")
            """
        )

        allow = codex.CodexExecutor(codex.CodexConfig(working_dir=str(tmp_path), allow_network=True))
        allow_res = allow.exec_python(code_inet)
        if not allow_res.success and ("Operation not permitted" in allow_res.stderr or "EPERM" in allow_res.stderr):
            pytest.skip("Host blocks socket() even without seccomp; cannot validate allow_network toggle.")
        assert allow_res.success, f"Expected INET socket allowed when allow_network=True, got stderr={allow_res.stderr!r}"
        assert "inet_ok" in allow_res.stdout

        deny = codex.CodexExecutor(codex.CodexConfig(working_dir=str(tmp_path), allow_network=False))
        deny_unix = deny.exec_python(code_unix)
        assert deny_unix.success, f"Expected AF_UNIX socket allowed, got stderr={deny_unix.stderr!r}"
        assert "unix_ok" in deny_unix.stdout

        deny_inet = deny.exec_python(code_inet)
        assert not deny_inet.success, "Expected INET socket blocked when allow_network=False"
        assert deny_inet.exit_code != 0
        assert (
            "Operation not permitted" in deny_inet.stderr
            or "EPERM" in deny_inet.stderr
            or "PermissionError" in deny_inet.stderr
            or "Errno 1" in deny_inet.stderr
        ), f"expected EPERM-style failure, got stderr={deny_inet.stderr!r}"

    def test_fs_write_blocked_outside_allowed_paths(self, tmp_path: Path):
        if platform.system().lower() != "linux":
            pytest.skip("Sandbox enforcement is only supported on Linux.")

        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

        # Allow writes only in cwd (/tmp/...), not in the repo workspace.
        cfg = codex.CodexConfig(working_dir=str(tmp_path), allow_network=False)
        executor = codex.CodexExecutor(cfg)

        ok_code = "open('ok.txt', 'w').write('ok')\nprint('wrote_ok')"
        ok = executor.exec_python(ok_code)
        assert ok.success
        assert (tmp_path / "ok.txt").exists()

        repo_root = Path(__file__).resolve().parents[2]
        forbidden_path = repo_root / ".tmp_sandbox_write_test.txt"

        forbid_code = f"from pathlib import Path; Path({str(forbidden_path)!r}).write_text('nope')"
        bad = executor.exec_python(forbid_code)

        # Clean up if sandbox allowed the write (this is a failure case).
        if forbidden_path.exists():
            forbidden_path.unlink(missing_ok=True)
        assert not bad.success, "Expected Landlock to block writes outside allowed paths"
        assert bad.exit_code != 0
        assert (
            "Operation not permitted" in bad.stderr
            or "EPERM" in bad.stderr
            or "PermissionError" in bad.stderr
            or "Errno 1" in bad.stderr
            or "Read-only file system" in bad.stderr
        ), f"expected permission-style failure, got stderr={bad.stderr!r}"

    def test_fs_allowlist_via_read_write_paths(self, tmp_path: Path):
        if platform.system().lower() != "linux":
            pytest.skip("Sandbox enforcement is only supported on Linux.")

        codex = pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

        repo_root = Path(__file__).resolve().parents[2]
        allow_path = repo_root / ".tmp_sandbox_allowlist_test.txt"
        allow_path.unlink(missing_ok=True)

        cfg = codex.CodexConfig(
            working_dir=str(tmp_path),
            allow_network=False,
            read_write_paths=[str(repo_root)],
        )
        executor = codex.CodexExecutor(cfg)

        code = f"from pathlib import Path; Path({str(allow_path)!r}).write_text('ok'); print('wrote')"
        res = executor.exec_python(code)

        try:
            assert res.success, f"Expected allowlisted path write to succeed, got stderr={res.stderr!r}"
            assert allow_path.exists()
        finally:
            allow_path.unlink(missing_ok=True)


class TestToolTypes:
    def test_tool_type_and_call_coercion(self):
        from nmoe.rl.tools import ToolCall, ToolType

        call = ToolCall(type="python", call_id="1", code="print(1)")
        assert call.type == ToolType.PYTHON
