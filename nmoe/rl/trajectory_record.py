"""Token-exact trajectory records and replay invariants.

The core correctness contract for agentic RL is **token-exactness**:
- Training-time tokens MUST exactly match rollout-time tokens.
- Tool-call spans and tool-return spans must be anchored in token indices.

This module provides:
- A minimal schema (`TrajectoryRecord`) storing token IDs + tool events.
- A replay verifier that enforces byte-for-byte transcript equivalence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from nmoe.rl.tools import ToolCall, ToolResult, ToolType
from nmoe.rl.tools.parsers import TokenLevelParser
from nmoe.rl.failures import categorize_failure


def _tool_type_str(tool_type: ToolType | str) -> str:
    if isinstance(tool_type, ToolType):
        return tool_type.value
    return str(tool_type)


@dataclass(frozen=True)
class ToolCallRecord:
    """JSON-friendly ToolCall snapshot."""

    type: str
    call_id: str = ""
    command: str = ""
    code: str = ""
    path: str = ""
    query: str = ""
    content: str = ""
    timeout_ms: int = 0
    cwd: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_call(cls, call: ToolCall) -> "ToolCallRecord":
        return cls(
            type=_tool_type_str(call.type),
            call_id=call.call_id,
            command=call.command,
            code=call.code,
            path=call.path,
            query=call.query,
            content=call.content,
            timeout_ms=int(call.timeout_ms),
            cwd=call.cwd,
            arguments=dict(call.arguments or {}),
        )

    def to_call(self) -> ToolCall:
        return ToolCall(
            type=self.type,
            call_id=self.call_id,
            command=self.command,
            code=self.code,
            path=self.path,
            query=self.query,
            content=self.content,
            timeout_ms=int(self.timeout_ms or 0),
            cwd=self.cwd,
            arguments=dict(self.arguments or {}),
        )


@dataclass(frozen=True)
class ToolResultRecord:
    """JSON-friendly ToolResult snapshot."""

    success: bool
    output: str = ""
    error: str | None = None
    exit_code: int = 0
    timed_out: bool = False
    failure_category: str = "ok"

    @classmethod
    def from_result(cls, result: ToolResult) -> "ToolResultRecord":
        return cls(
            success=bool(result.success),
            output=result.output or "",
            error=result.error,
            exit_code=int(result.exit_code),
            timed_out=bool(result.timed_out),
            failure_category=categorize_failure(
                success=bool(result.success),
                timed_out=bool(result.timed_out),
                exit_code=int(result.exit_code),
                error=result.error,
                stderr=result.error,
                stdout=result.output,
            ),
        )


@dataclass(frozen=True)
class ToolEventRecord:
    """A tool call + tool return anchored to token spans."""

    call: ToolCallRecord
    result: ToolResultRecord
    # Token spans are half-open: [start, end)
    call_span: tuple[int, int]
    return_span: tuple[int, int]

    def _assert_spans_valid(self, n_tokens: int) -> None:
        cs, ce = self.call_span
        rs, re = self.return_span
        if not (0 <= cs <= ce <= n_tokens):
            raise ValueError(f"invalid call_span {self.call_span} (n_tokens={n_tokens})")
        if not (0 <= rs <= re <= n_tokens):
            raise ValueError(f"invalid return_span {self.return_span} (n_tokens={n_tokens})")
        if ce > rs:
            raise ValueError(f"call_span overlaps return_span: call={self.call_span} return={self.return_span}")


@dataclass(frozen=True)
class TrajectoryRecord:
    """A token-exact trajectory transcript.

    This schema is intentionally minimal: everything needed to replay and
    validate the transcript is here, without any text retokenization.
    """

    version: int = 1
    prompt_tokens: list[int] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    tool_events: list[ToolEventRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "prompt_tokens": list(self.prompt_tokens),
            "tokens": list(self.tokens),
            "tool_events": [
                {
                    "call": {
                        "type": e.call.type,
                        "call_id": e.call.call_id,
                        "command": e.call.command,
                        "code": e.call.code,
                        "path": e.call.path,
                        "query": e.call.query,
                        "content": e.call.content,
                        "timeout_ms": int(e.call.timeout_ms),
                        "cwd": e.call.cwd,
                        "arguments": dict(e.call.arguments or {}),
                    },
                    "result": {
                        "success": bool(e.result.success),
                        "output": e.result.output,
                        "error": e.result.error,
                        "exit_code": int(e.result.exit_code),
                        "timed_out": bool(e.result.timed_out),
                        "failure_category": e.result.failure_category,
                    },
                    "call_span": [int(e.call_span[0]), int(e.call_span[1])],
                    "return_span": [int(e.return_span[0]), int(e.return_span[1])],
                }
                for e in self.tool_events
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryRecord":
        events: list[ToolEventRecord] = []
        for e in d.get("tool_events", []):
            call_d = e.get("call", {})
            res_d = e.get("result", {})
            events.append(
                ToolEventRecord(
                    call=ToolCallRecord(
                        type=str(call_d.get("type", "")),
                        call_id=str(call_d.get("call_id", "")),
                        command=str(call_d.get("command", "")),
                        code=str(call_d.get("code", "")),
                        path=str(call_d.get("path", "")),
                        query=str(call_d.get("query", "")),
                        content=str(call_d.get("content", "")),
                        timeout_ms=int(call_d.get("timeout_ms", 0) or 0),
                        cwd=str(call_d.get("cwd", "")),
                        arguments=dict(call_d.get("arguments", {}) or {}),
                    ),
                    result=ToolResultRecord(
                        success=bool(res_d.get("success", False)),
                        output=str(res_d.get("output", "")),
                        error=res_d.get("error", None),
                        exit_code=int(res_d.get("exit_code", 0) or 0),
                        timed_out=bool(res_d.get("timed_out", False)),
                        failure_category=str(res_d.get("failure_category", "ok")),
                    ),
                    call_span=(int(e["call_span"][0]), int(e["call_span"][1])),
                    return_span=(int(e["return_span"][0]), int(e["return_span"][1])),
                )
            )
        return cls(
            version=int(d.get("version", 1)),
            prompt_tokens=list(d.get("prompt_tokens", [])),
            tokens=list(d.get("tokens", [])),
            tool_events=events,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "TrajectoryRecord":
        return cls.from_dict(json.loads(s))

    def assert_token_exact(self, tokenizer) -> None:
        """Fail-fast invariant checks (no silent drift)."""
        if not self.tokens[: len(self.prompt_tokens)] == self.prompt_tokens:
            raise ValueError("prompt_tokens must equal tokens[:prompt_len]")

        n = len(self.tokens)
        last_end = len(self.prompt_tokens)
        parser = TokenLevelParser(tokenizer)

        for idx, ev in enumerate(self.tool_events):
            ev._assert_spans_valid(n)
            cs, ce = ev.call_span
            rs, re = ev.return_span
            if cs < last_end:
                raise ValueError(f"tool event {idx} out of order (cs={cs} < last_end={last_end})")

            # Tool call slice must parse to exactly one complete call.
            call_slice = self.tokens[cs:ce]
            parsed = parser.parse(call_slice)
            if len(parsed) != 1 or parsed[0].end_span is None:
                raise ValueError(f"tool call slice did not parse as one complete call (idx={idx})")
            if parsed[0].full_span != (0, len(call_slice)):
                raise ValueError(f"tool call slice span mismatch (idx={idx})")

            parsed_call = parsed[0].call
            rec_call = ev.call.to_call()
            if _tool_type_str(parsed_call.type) != _tool_type_str(rec_call.type):
                raise ValueError(f"tool type mismatch (idx={idx})")

            if _tool_type_str(rec_call.type) == "python":
                if parsed_call.code.strip() != rec_call.code.strip():
                    raise ValueError(f"python code mismatch (idx={idx})")
            elif _tool_type_str(rec_call.type) == "bash":
                if parsed_call.command.strip() != rec_call.command.strip():
                    raise ValueError(f"bash command mismatch (idx={idx})")
            elif _tool_type_str(rec_call.type) == "read":
                if parsed_call.path.strip() != rec_call.path.strip():
                    raise ValueError(f"read path mismatch (idx={idx})")

            # Tool return slice must equal canonical encoding for the recorded result.
            ret_slice = self.tokens[rs:re]
            expected_ret = _encode_tool_return_tokens(ev.result, tokenizer)
            if ret_slice != expected_ret:
                raise ValueError(f"tool return tokens mismatch (idx={idx})")

            last_end = re

    async def replay_tools(self, *, tokenizer, tool_executor) -> None:
        """Re-execute tools and require byte-for-byte equivalence on outputs.

        This is only meaningful for deterministic tool calls (tests ensure that).
        """
        self.assert_token_exact(tokenizer)

        for idx, ev in enumerate(self.tool_events):
            call = ev.call.to_call()
            res = await tool_executor.execute_one(call)

            if bool(res.success) != bool(ev.result.success):
                raise AssertionError(f"tool replay success mismatch (idx={idx})")
            if (res.output or "") != (ev.result.output or ""):
                raise AssertionError(f"tool replay output mismatch (idx={idx})")
            if int(res.exit_code) != int(ev.result.exit_code):
                raise AssertionError(f"tool replay exit_code mismatch (idx={idx})")
            if bool(res.timed_out) != bool(ev.result.timed_out):
                raise AssertionError(f"tool replay timed_out mismatch (idx={idx})")


def _encode_tool_return_tokens(result: ToolResultRecord, tokenizer) -> list[int]:
    if result.success:
        text = f"<|return|>\n{result.output}\n<|end|>"
    else:
        error_msg = result.error or f"Exit code: {result.exit_code}"
        text = f"<|return|>Error: {error_msg}<|end|>"
    return tokenizer.encode(text, add_special_tokens=False)


def record_from_turn(*, prompt_tokens: Sequence[int], tokens: Sequence[int], tool_events: Sequence[ToolEventRecord]) -> TrajectoryRecord:
    return TrajectoryRecord(
        prompt_tokens=list(prompt_tokens),
        tokens=list(tokens),
        tool_events=list(tool_events),
    )
