"""Harmony format rewards for OpenAI-style channel-based responses.

Harmony format uses special tokens for structured output:
- <|start|> / <|end|> - Message boundaries
- <|message|> - Separates header from content
- <|channel|> - Specifies output channel

Channels:
- analysis: Chain-of-thought (hidden from user)
- final: User-facing answer
- commentary: Tool calls/preambles

This module provides token-level parsing and binary reward signals
for GDPO multi-reward training.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Harmony Token Constants
# =============================================================================

# Text representations of special tokens
HARMONY_TOKENS = {
    "start": "<|start|>",
    "end": "<|end|>",
    "message": "<|message|>",
    "channel": "<|channel|>",
    "call": "<|call|>",
    "return": "<|return|>",
}


def harmony_allowed_special() -> set[str]:
    """Special tokens allowed in Harmony strings for tiktoken.encode()."""
    return set(HARMONY_TOKENS.values())


def harmony_encode(tokenizer: Any, text: str) -> list[int]:
    """Encode Harmony-bearing text with strict allowlist of Harmony specials.

    tiktoken treats Harmony markers like "<|start|>" as special tokens and will
    raise unless allowed_special is provided. We keep this strict: only Harmony
    specials are allowed.
    """
    if not hasattr(tokenizer, "encode"):
        raise TypeError("tokenizer must implement encode()")
    allowed = harmony_allowed_special()
    try:
        return tokenizer.encode(text, allowed_special=allowed)
    except TypeError:
        return tokenizer.encode(text)

# Common channel names
CHANNELS = {
    "analysis": "analysis",  # CoT/reasoning (hidden)
    "final": "final",  # User-facing answer
    "commentary": "commentary",  # Tool calls
}


def harmony_message(*, role: str, channel: str, content: str) -> str:
    """Construct a single Harmony message block (start..end)."""
    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    if not isinstance(channel, str) or not channel:
        raise ValueError("channel must be a non-empty string")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    return (
        f"{HARMONY_TOKENS['start']}{role}"
        f"{HARMONY_TOKENS['channel']}{channel}"
        f"{HARMONY_TOKENS['message']}{content}"
        f"{HARMONY_TOKENS['end']}"
    )


@dataclass
class HarmonyMessage:
    """A single Harmony message (one start/end block)."""
    role: str = ""  # e.g., "assistant"
    channel: str = ""  # e.g., "analysis", "final"
    content: str = ""
    raw_tokens: list[int] = field(default_factory=list)


@dataclass
class ParsedHarmonyResponse:
    """Parsed Harmony format response."""

    messages: list[HarmonyMessage] = field(default_factory=list)
    raw_text: str = ""

    # Structure validation
    has_start: bool = False
    has_end: bool = False
    has_message: bool = False
    has_channel: bool = False
    properly_nested: bool = False

    # Channel presence
    channels_present: set[str] = field(default_factory=set)

    # Errors encountered during parsing
    parse_errors: list[str] = field(default_factory=list)

    def get_channel(self, channel: str) -> str | None:
        """Get content of a specific channel."""
        for msg in self.messages:
            if msg.channel == channel:
                return msg.content
        return None

    def get_channel_tokens(self, channel: str) -> list[int]:
        """Get raw tokens for a specific channel."""
        for msg in self.messages:
            if msg.channel == channel:
                return msg.raw_tokens
        return []

    @property
    def has_analysis(self) -> bool:
        return "analysis" in self.channels_present

    @property
    def has_final(self) -> bool:
        return "final" in self.channels_present

    @property
    def analysis_content(self) -> str:
        return self.get_channel("analysis") or ""

    @property
    def final_content(self) -> str:
        return self.get_channel("final") or ""


# =============================================================================
# Text-Based Parsing (Fallback)
# =============================================================================

def parse_harmony_text(text: str) -> ParsedHarmonyResponse:
    """Parse Harmony format from text (fallback for when we don't have tokens).

    Args:
        text: Response text to parse

    Returns:
        ParsedHarmonyResponse with structure analysis
    """
    result = ParsedHarmonyResponse(raw_text=text)

    # Check for special tokens in text
    result.has_start = HARMONY_TOKENS["start"] in text
    result.has_end = HARMONY_TOKENS["end"] in text
    result.has_message = HARMONY_TOKENS["message"] in text
    result.has_channel = HARMONY_TOKENS["channel"] in text

    if not result.has_start:
        result.parse_errors.append("Missing <|start|> token")
        return result

    # Parse messages using regex
    # Pattern: <|start|>role<|channel|>channel_name<|message|>content<|end|>
    pattern = (
        r"<\|start\|>\s*(\w+)\s*"  # role
        r"<\|channel\|>\s*(\w+)\s*"  # channel
        r"<\|message\|>"  # message separator
        r"(.*?)"  # content (non-greedy)
        r"<\|end\|>"  # end
    )

    matches = re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    match_count = 0

    for match in matches:
        match_count += 1
        role, channel, content = match.groups()

        msg = HarmonyMessage(
            role=role.strip().lower(),
            channel=channel.strip().lower(),
            content=content.strip(),
        )
        result.messages.append(msg)
        result.channels_present.add(msg.channel)

    if match_count == 0:
        result.parse_errors.append("No valid message blocks found")

    # Check nesting
    result.properly_nested = _check_nesting(text)

    return result


def _check_nesting(text: str) -> bool:
    """Check if Harmony tokens are properly nested.

    Valid: <|start|>...<|end|><|start|>...<|end|>
    Invalid: <|start|><|start|>...<|end|> (nested)
    Invalid: <|end|>...<|start|> (wrong order)
    """
    start_token = HARMONY_TOKENS["start"]
    end_token = HARMONY_TOKENS["end"]

    depth = 0
    pos = 0

    while pos < len(text):
        # Find next token
        start_pos = text.find(start_token, pos)
        end_pos = text.find(end_token, pos)

        if start_pos == -1 and end_pos == -1:
            break

        if start_pos != -1 and (end_pos == -1 or start_pos < end_pos):
            # Found start token
            depth += 1
            if depth > 1:
                return False  # Nested starts
            pos = start_pos + len(start_token)
        else:
            # Found end token
            depth -= 1
            if depth < 0:
                return False  # End before start
            pos = end_pos + len(end_token)

    return depth == 0  # All starts have matching ends


# =============================================================================
# Token-Based Parsing (Preferred)
# =============================================================================

class HarmonyTokenizer:
    """Token-level Harmony parser.

    Uses tokenizer to identify special token IDs for precise parsing.
    """

    def __init__(self, tokenizer):
        """Initialize with a tokenizer.

        Args:
            tokenizer: Tokenizer with encode() method (e.g., tiktoken)
        """
        self.tokenizer = tokenizer

        # Get token IDs for special tokens.
        #
        # tiktoken treats these as "special" and will raise unless allowed_special is set.
        self.token_ids = {}
        for name, text in HARMONY_TOKENS.items():
            try:
                try:
                    ids = tokenizer.encode(text, allowed_special={text})
                except TypeError:
                    ids = tokenizer.encode(text)
                if len(ids) == 1:
                    self.token_ids[name] = ids[0]
                else:
                    # Multi-token special token - store as tuple
                    self.token_ids[name] = tuple(ids)
            except Exception:
                pass  # Token not in vocab

    def parse(self, tokens: list[int]) -> ParsedHarmonyResponse:
        """Parse Harmony format from token IDs.

        Args:
            tokens: List of token IDs

        Returns:
            ParsedHarmonyResponse with structure analysis
        """
        # Decode for text analysis
        try:
            text = self.tokenizer.decode(tokens)
        except Exception:
            text = ""

        result = ParsedHarmonyResponse(raw_text=text)

        # Check for special tokens
        start_id = self.token_ids.get("start")
        end_id = self.token_ids.get("end")
        message_id = self.token_ids.get("message")
        channel_id = self.token_ids.get("channel")

        if start_id and self._contains_token(tokens, start_id):
            result.has_start = True
        if end_id and self._contains_token(tokens, end_id):
            result.has_end = True
        if message_id and self._contains_token(tokens, message_id):
            result.has_message = True
        if channel_id and self._contains_token(tokens, channel_id):
            result.has_channel = True

        # Parse message blocks
        messages = self._parse_message_blocks(tokens)
        result.messages = messages
        result.channels_present = {m.channel for m in messages if m.channel}

        # Check nesting
        result.properly_nested = self._check_token_nesting(tokens)

        return result

    def _contains_token(self, tokens: list[int], token_id: int | tuple) -> bool:
        """Check if token(s) present in sequence."""
        if isinstance(token_id, int):
            return token_id in tokens
        else:
            # Multi-token: check for subsequence
            for i in range(len(tokens) - len(token_id) + 1):
                if tuple(tokens[i:i + len(token_id)]) == token_id:
                    return True
            return False

    def _parse_message_blocks(self, tokens: list[int]) -> list[HarmonyMessage]:
        """Parse individual message blocks from tokens."""
        # Fallback to text parsing for now
        # Full token-level parsing would require more complex state machine
        try:
            text = self.tokenizer.decode(tokens)
            parsed = parse_harmony_text(text)
            return parsed.messages
        except Exception:
            return []

    def _check_token_nesting(self, tokens: list[int]) -> bool:
        """Check token nesting at token level."""
        start_id = self.token_ids.get("start")
        end_id = self.token_ids.get("end")

        if not start_id or not end_id:
            # Fallback to text
            try:
                text = self.tokenizer.decode(tokens)
                return _check_nesting(text)
            except Exception:
                return False

        depth = 0
        for tok in tokens:
            if tok == start_id or (isinstance(start_id, tuple) and tok == start_id[0]):
                depth += 1
                if depth > 1:
                    return False
            elif tok == end_id or (isinstance(end_id, tuple) and tok == end_id[0]):
                depth -= 1
                if depth < 0:
                    return False

        return depth == 0


# =============================================================================
# Harmony Reward Signals
# =============================================================================

def harmony_structure_rewards(parsed: ParsedHarmonyResponse) -> dict[str, float]:
    """Compute binary structure rewards for Harmony format.

    All rewards are binary (0.0 or 1.0) for RLVR.

    Args:
        parsed: Parsed Harmony response

    Returns:
        Dict of reward name -> value
    """
    return {
        "struct_has_start": 1.0 if parsed.has_start else 0.0,
        "struct_has_end": 1.0 if parsed.has_end else 0.0,
        "struct_has_message": 1.0 if parsed.has_message else 0.0,
        "struct_has_channel": 1.0 if parsed.has_channel else 0.0,
        "struct_proper_nesting": 1.0 if parsed.properly_nested else 0.0,
    }


def harmony_channel_rewards(parsed: ParsedHarmonyResponse) -> dict[str, float]:
    """Compute binary channel presence rewards.

    Args:
        parsed: Parsed Harmony response

    Returns:
        Dict of reward name -> value
    """
    return {
        "chan_has_analysis": 1.0 if parsed.has_analysis else 0.0,
        "chan_has_final": 1.0 if parsed.has_final else 0.0,
        "chan_analysis_nonempty": 1.0 if len(parsed.analysis_content) > 10 else 0.0,
        "chan_final_nonempty": 1.0 if len(parsed.final_content) > 0 else 0.0,
    }


def compute_harmony_rewards(
    text: str,
    tokenizer: Any | None = None,
    tokens: list[int] | None = None,
) -> dict[str, float]:
    """Compute all Harmony format rewards.

    Args:
        text: Response text (used if tokens not provided)
        tokenizer: Optional tokenizer for token-level parsing
        tokens: Optional token IDs for precise parsing

    Returns:
        Dict of all reward signals
    """
    # Parse response
    if tokens is not None and tokenizer is not None:
        parser = HarmonyTokenizer(tokenizer)
        parsed = parser.parse(tokens)
    else:
        parsed = parse_harmony_text(text)

    # Collect all rewards
    rewards = {}
    rewards.update(harmony_structure_rewards(parsed))
    rewards.update(harmony_channel_rewards(parsed))

    return rewards


# =============================================================================
# R1-Zero Compatibility (Think/Answer format)
# =============================================================================

def parse_r1zero_format(text: str) -> ParsedHarmonyResponse:
    """Parse R1-Zero <think>/<answer> format into Harmony-compatible structure.

    Maps:
    - <think> content -> analysis channel
    - <answer> content -> final channel

    Args:
        text: Response text

    Returns:
        ParsedHarmonyResponse (for unified reward computation)
    """
    result = ParsedHarmonyResponse(raw_text=text)

    # Extract think tag
    think_match = re.search(
        r"<think>(.*?)</think>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Extract answer tag
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if think_match:
        result.messages.append(HarmonyMessage(
            role="assistant",
            channel="analysis",
            content=think_match.group(1).strip(),
        ))
        result.channels_present.add("analysis")
        result.has_start = True
        result.has_message = True
        result.has_channel = True

    if answer_match:
        result.messages.append(HarmonyMessage(
            role="assistant",
            channel="final",
            content=answer_match.group(1).strip(),
        ))
        result.channels_present.add("final")
        result.has_end = True

    # Check nesting (think must come before answer)
    if think_match and answer_match:
        result.properly_nested = think_match.end() <= answer_match.start()
    elif think_match or answer_match:
        result.properly_nested = True  # Partial format is ok

    return result


def compute_r1zero_rewards(text: str) -> dict[str, float]:
    """Compute rewards for R1-Zero <think>/<answer> format.

    Args:
        text: Response text

    Returns:
        Dict of reward signals (compatible with Harmony reward names)
    """
    parsed = parse_r1zero_format(text)
    return {
        **harmony_structure_rewards(parsed),
        **harmony_channel_rewards(parsed),
    }
