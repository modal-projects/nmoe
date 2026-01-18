"""Token-level tool call parsing for RL training.

Parses tool calls directly from token IDs rather than decoded text,
which is more robust and enables precise token-level reward attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nmoe.rl.tools import ToolCall, ToolType


@dataclass
class TokenSpan:
    """A span of tokens with position info."""
    start: int  # Start index in sequence
    end: int    # End index (exclusive)
    tokens: list[int]

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ParsedToolCall:
    """A tool call parsed from tokens with position info."""
    call: ToolCall
    call_span: TokenSpan      # Span of the <|call|> marker
    content_span: TokenSpan   # Span of tool content
    end_span: TokenSpan | None  # Span of <|end|> marker (None if truncated)

    @property
    def full_span(self) -> tuple[int, int]:
        """Return (start, end) covering entire tool call."""
        end = self.end_span.end if self.end_span else self.content_span.end
        return (self.call_span.start, end)


class TokenLevelParser:
    """Token-level parser for tool calls.

    Instead of regex over decoded text, this parser works directly with
    token IDs, finding special marker tokens and extracting content spans.

    Usage:
        parser = TokenLevelParser(tokenizer)
        calls = parser.parse(token_ids)
        for call in calls:
            print(f"Tool at {call.call_span.start}: {call.call.type}")
    """

    def __init__(self, tokenizer):
        """Initialize parser with tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer with encode/decode methods
        """
        self.tokenizer = tokenizer

        # Find special token IDs for tool markers
        # Try common patterns used by different models
        self._call_token_ids = self._find_marker_tokens([
            "<|call|>", "<call>", "[CALL]", "<tool_call>",
            "<|tool_call|>", "```python", "```bash",
        ])
        self._end_token_ids = self._find_marker_tokens([
            "<|end|>", "</call>", "[/CALL]", "</tool_call>",
            "<|/tool_call|>", "```",
        ])

        # Tool type tokens
        self._python_tokens = self._find_marker_tokens(["python", "Python", "PYTHON"])
        self._bash_tokens = self._find_marker_tokens(["bash", "Bash", "BASH", "shell"])
        self._read_tokens = self._find_marker_tokens(["read", "Read", "READ", "cat"])

    def _find_marker_tokens(self, markers: list[str]) -> set[int]:
        """Find token IDs for marker strings."""
        token_ids = set()
        for marker in markers:
            try:
                # Try encoding as special token first
                ids = self.tokenizer.encode(marker, add_special_tokens=False)
                if len(ids) == 1:
                    token_ids.add(ids[0])
                # Also check vocab directly
                if hasattr(self.tokenizer, 'vocab'):
                    if marker in self.tokenizer.vocab:
                        token_ids.add(self.tokenizer.vocab[marker])
            except Exception:
                pass
        return token_ids

    def parse(self, token_ids: Sequence[int]) -> list[ParsedToolCall]:
        """Parse tool calls from token sequence.

        Args:
            token_ids: Sequence of token IDs

        Returns:
            List of ParsedToolCall with position info
        """
        calls = []
        i = 0
        n = len(token_ids)

        while i < n:
            # Look for call marker
            if token_ids[i] in self._call_token_ids:
                call_start = i
                i += 1

                # Skip whitespace/newline tokens
                while i < n and self._is_whitespace_token(token_ids[i]):
                    i += 1

                # Determine tool type from next tokens
                tool_type = self._detect_tool_type(token_ids, i)
                if tool_type is None:
                    i += 1
                    continue

                # Skip tool type token
                i += 1

                # Skip whitespace/newline
                while i < n and self._is_whitespace_token(token_ids[i]):
                    i += 1

                content_start = i

                # Find end marker or end of sequence
                end_start = None
                while i < n:
                    if token_ids[i] in self._end_token_ids:
                        end_start = i
                        break
                    i += 1

                content_end = end_start if end_start else i

                # Extract content tokens
                content_tokens = list(token_ids[content_start:content_end])

                # Decode content
                content_text = self.tokenizer.decode(content_tokens, skip_special_tokens=False)

                # Create ToolCall
                call = self._create_tool_call(tool_type, content_text.strip())

                # Create spans
                call_span = TokenSpan(
                    start=call_start,
                    end=call_start + 1,
                    tokens=[token_ids[call_start]]
                )
                content_span = TokenSpan(
                    start=content_start,
                    end=content_end,
                    tokens=content_tokens
                )
                end_span = None
                if end_start is not None:
                    end_span = TokenSpan(
                        start=end_start,
                        end=end_start + 1,
                        tokens=[token_ids[end_start]]
                    )
                    i = end_start + 1

                calls.append(ParsedToolCall(
                    call=call,
                    call_span=call_span,
                    content_span=content_span,
                    end_span=end_span,
                ))
            else:
                i += 1

        return calls

    def _is_whitespace_token(self, token_id: int) -> bool:
        """Check if token is whitespace/newline."""
        try:
            text = self.tokenizer.decode([token_id])
            return text.strip() == ""
        except Exception:
            return False

    def _detect_tool_type(self, token_ids: Sequence[int], pos: int) -> ToolType | None:
        """Detect tool type from tokens at position."""
        if pos >= len(token_ids):
            return None

        token_id = token_ids[pos]

        if token_id in self._python_tokens:
            return ToolType.PYTHON
        if token_id in self._bash_tokens:
            return ToolType.BASH
        if token_id in self._read_tokens:
            return ToolType.READ

        # Fallback: decode and check text
        try:
            text = self.tokenizer.decode([token_id]).lower().strip()
            if "python" in text:
                return ToolType.PYTHON
            if "bash" in text or "shell" in text:
                return ToolType.BASH
            if "read" in text or "cat" in text:
                return ToolType.READ
        except Exception:
            pass

        return None

    def _create_tool_call(self, tool_type: ToolType, content: str) -> ToolCall:
        """Create ToolCall from type and content."""
        if tool_type == ToolType.PYTHON:
            return ToolCall(type=tool_type, code=content)
        elif tool_type == ToolType.BASH:
            return ToolCall(type=tool_type, command=content)
        elif tool_type == ToolType.READ:
            return ToolCall(type=tool_type, path=content)
        else:
            return ToolCall(type=tool_type, command=content)

    def find_tool_boundaries(self, token_ids: Sequence[int]) -> list[tuple[int, int]]:
        """Find (start, end) boundaries of all tool calls.

        Useful for masking tool call regions in loss computation.
        """
        calls = self.parse(token_ids)
        return [call.full_span for call in calls]

    def get_content_mask(self, token_ids: Sequence[int], mask_markers: bool = True) -> list[bool]:
        """Get mask indicating which tokens are tool content.

        Args:
            token_ids: Token sequence
            mask_markers: If True, also mask the call/end markers

        Returns:
            Boolean mask where True = tool content
        """
        n = len(token_ids)
        mask = [False] * n

        for call in self.parse(token_ids):
            # Mask content
            for i in range(call.content_span.start, call.content_span.end):
                if i < n:
                    mask[i] = True

            # Optionally mask markers
            if mask_markers:
                for i in range(call.call_span.start, call.call_span.end):
                    if i < n:
                        mask[i] = True
                if call.end_span:
                    for i in range(call.end_span.start, call.end_span.end):
                        if i < n:
                            mask[i] = True

        return mask


def parse_tool_calls_from_tokens(
    token_ids: Sequence[int],
    tokenizer,
) -> list[ParsedToolCall]:
    """Convenience function to parse tool calls from tokens.

    Args:
        token_ids: Sequence of token IDs
        tokenizer: HuggingFace tokenizer

    Returns:
        List of parsed tool calls with position info
    """
    parser = TokenLevelParser(tokenizer)
    return parser.parse(token_ids)


def get_tool_content_mask(
    token_ids: Sequence[int],
    tokenizer,
    include_markers: bool = True,
) -> list[bool]:
    """Get boolean mask for tool content tokens.

    Args:
        token_ids: Token sequence
        tokenizer: HuggingFace tokenizer
        include_markers: Include call/end markers in mask

    Returns:
        Boolean mask where True = tool content
    """
    parser = TokenLevelParser(tokenizer)
    return parser.get_content_mask(token_ids, mask_markers=include_markers)
