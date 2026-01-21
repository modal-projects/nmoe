"""Multi-turn generation with tool execution for agentic RL.

Supports:
- Interleaved reasoning and tool calls
- Async tool execution (scatter/gather)
- Turn-level trajectory tracking
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from nmoe.rl.tools import AsyncToolExecutor, ToolCall, ToolConfig, ToolResult, ToolType
from nmoe.rl.rewards_tools import ToolCallSite
from nmoe.rl.trajectory_record import ToolCallRecord, ToolEventRecord, ToolResultRecord, TrajectoryRecord, record_from_turn


@dataclass
class AgentMessage:
    """A reasoning segment from the agent."""
    text: str = ""
    tokens: list[int] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)


@dataclass
class AgentTurn:
    """A complete agent turn with interleaved reasoning and tools.

    A turn consists of:
    1. Zero or more reasoning segments (agent_messages)
    2. Zero or more tool calls with results
    3. A final response (last message after all tools)
    """

    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_events: list[ToolEventRecord] = field(default_factory=list)

    # Full token sequence for the turn
    tokens: list[int] = field(default_factory=list)

    # Prompt that started this turn
    prompt_tokens: list[int] = field(default_factory=list)

    # Metadata
    max_tokens: int = 0
    format_type: str = "harmony"
    record: TrajectoryRecord | None = None

    @property
    def reasoning_tokens(self) -> int:
        """Total tokens in reasoning/analysis segments."""
        return sum(m.token_count for m in self.messages)

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens (excluding prompt)."""
        return len(self.tokens) - len(self.prompt_tokens)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def final_response(self) -> str | None:
        """Last message text (the final answer)."""
        if self.messages:
            return self.messages[-1].text
        return None

    @property
    def full_text(self) -> str:
        """Full response text including tool outputs."""
        parts = []
        msg_idx = 0
        tool_idx = 0

        # Interleave messages and tool results based on turn structure
        # For simplicity, assume messages and tool calls alternate
        while msg_idx < len(self.messages) or tool_idx < len(self.tool_results):
            if msg_idx < len(self.messages):
                parts.append(self.messages[msg_idx].text)
                msg_idx += 1

            if tool_idx < len(self.tool_results):
                result = self.tool_results[tool_idx]
                parts.append(f"[Tool Output: {result.output[:200]}...]")
                tool_idx += 1

        return "\n".join(parts)

    def to_tool_sites(self) -> list[ToolCallSite]:
        """Convert to ToolCallSite list for reward computation."""
        sites = []
        for i, (call, result) in enumerate(zip(self.tool_calls, self.tool_results)):
            # Get subsequent text (messages after this tool call)
            subsequent = ""
            if i + 1 < len(self.messages):
                subsequent = " ".join(m.text for m in self.messages[i + 1:])

            sites.append(ToolCallSite(
                call=call,
                result=result,
                position=self.tool_events[i].call_span[0] if i < len(self.tool_events) else i,
                subsequent_text=subsequent,
            ))
        return sites


# =============================================================================
# Generation with Tool Execution
# =============================================================================

def _parse_tool_call_from_tokens(
    token_ids: Sequence[int],
    tokenizer,
) -> tuple[ToolCall | None, int | None]:
    """Parse a tool call directly from token IDs (preferred method).

    Uses token-level parsing for robustness and precise position tracking.

    Args:
        token_ids: Token sequence to parse
        tokenizer: Tokenizer with decode method

    Returns:
        Tuple of (ToolCall or None, end_position or None)
    """
    try:
        from nmoe.rl.tools.parsers import TokenLevelParser
        parser = TokenLevelParser(tokenizer)
        calls = parser.parse(token_ids)
        if calls:
            # Only treat as a valid tool call if it's complete (has an end marker).
            if calls[0].end_span is None:
                return None, None
            return calls[0].call, calls[0].full_span[1]
    except ImportError:
        pass

    # Fallback to text-based parsing
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    call = _parse_tool_call_from_text(text)
    return call, None


def _parse_tool_call_from_text(text: str) -> ToolCall | None:
    """Parse a tool call from generated text (fallback method).

    NOTE: Token-level parsing via _parse_tool_call_from_tokens is preferred
    as it's more robust and provides position information.

    Looks for patterns like:
    - <|call|>python\ncode here<|end|>
    - <|call|>bash\ncommand here<|end|>

    Args:
        text: Text that may contain a tool call

    Returns:
        ToolCall if found, None otherwise
    """
    import re

    # Pattern for tool calls
    pattern = r"<\|call\|>\s*(\w+)\s*\n(.*?)(?:<\|end\|>|$)"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)

    if not match:
        return None

    tool_type = match.group(1).lower()
    content = match.group(2).strip()

    if tool_type == "python":
        return ToolCall(type=ToolType.PYTHON, code=content)
    elif tool_type == "bash":
        return ToolCall(type=ToolType.BASH, command=content)
    elif tool_type == "read":
        return ToolCall(type=ToolType.READ, path=content)
    elif tool_type == "search":
        return ToolCall(type=ToolType.SEARCH, query=content)
    else:
        return ToolCall(type=tool_type, arguments={"content": content})


def _encode_tool_return(result: ToolResult, enc) -> list[int]:
    """Encode tool result as tokens to insert into generation.

    Args:
        result: Tool execution result
        enc: Tokenizer with encode() method

    Returns:
        Token IDs representing the tool return
    """
    if result.success:
        return_text = f"<|return|>\n{result.output}\n<|end|>"
    else:
        error_msg = result.error or f"Exit code: {result.exit_code}"
        return_text = f"<|return|>Error: {error_msg}<|end|>"

    from nmoe.rl.rewards_harmony import harmony_encode

    return harmony_encode(enc, return_text)


async def generate_turn_async(
    model,
    *,
    enc,
    prompt_ids: Sequence[int],
    tool_executor: AsyncToolExecutor,
    max_new_tokens: int = 2048,
    max_tool_rounds: int = 10,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> AgentTurn:
    """Generate a complete agent turn with async tool execution.

    Args:
        model: Language model with forward pass
        enc: Tokenizer with encode/decode
        prompt_ids: Input token IDs
        tool_executor: Async tool executor
        max_new_tokens: Maximum tokens to generate
        max_tool_rounds: Maximum tool call rounds
        eos_token_id: End of sequence token
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold

    Returns:
        AgentTurn with full trajectory
    """
    from nmoe.rl.rollout import _sample_top_p

    turn = AgentTurn(
        prompt_tokens=list(prompt_ids),
        max_tokens=max_new_tokens,
    )

    # Current token sequence
    tokens = list(prompt_ids)
    current_text = ""
    tool_round = 0
    tool_events: list[ToolEventRecord] = []

    model.eval()
    with torch.inference_mode():
        while len(tokens) - len(prompt_ids) < max_new_tokens:
            if tool_round >= max_tool_rounds:
                break

            # Generate until we hit a stopping condition
            segment_tokens = []
            for _ in range(max_new_tokens - (len(tokens) - len(prompt_ids))):
                toks = torch.tensor(tokens + segment_tokens, device="cuda", dtype=torch.long)[None, :]
                logits = model(toks)[:, -1, :].squeeze(0)
                next_id = _sample_top_p(logits, temperature=temperature, top_p=top_p)
                segment_tokens.append(next_id)

                # Check for EOS
                if next_id == eos_token_id:
                    break

                # Tool calls are detected token-native (RL_DESIGN.md).
                tool_call, _ = _parse_tool_call_from_tokens(segment_tokens, enc)
                if tool_call is not None:
                    break

            tokens.extend(segment_tokens)
            segment_text = enc.decode(segment_tokens)
            current_text += segment_text

            # Check if we have a tool call
            tool_call, _ = _parse_tool_call_from_tokens(segment_tokens, enc)
            if tool_call is not None:
                tool_round += 1

                # Save reasoning before tool call
                call_start = None
                call_full_span = None
                try:
                    from nmoe.rl.tools.parsers import TokenLevelParser
                    parser = TokenLevelParser(enc)
                    parsed = parser.parse(segment_tokens)
                    if parsed and parsed[0].end_span is not None:
                        call_start = parsed[0].call_span.start
                        call_full_span = parsed[0].full_span
                except Exception:
                    call_start = None

                reasoning_tokens = segment_tokens[:call_start] if call_start is not None else []
                reasoning_text = enc.decode(reasoning_tokens).strip() if reasoning_tokens else ""
                if reasoning_text.strip():
                    turn.messages.append(AgentMessage(
                        text=reasoning_text,
                        tokens=list(reasoning_tokens),
                    ))

                # Execute tool (blocking for this call, async dispatch)
                call_ids = tool_executor.scatter([tool_call])
                results = await tool_executor.gather(call_ids, timeout=30.0)
                result = results[0]

                turn.tool_calls.append(tool_call)
                turn.tool_results.append(result)

                # Insert tool return into token sequence
                return_tokens = _encode_tool_return(result, enc)
                return_start = len(tokens)
                tokens.extend(return_tokens)
                return_end = len(tokens)
                current_text += enc.decode(return_tokens)

                # Record token spans for replay invariants.
                base = len(tokens) - len(segment_tokens) - len(return_tokens)
                if call_full_span is None:
                    # Fallback: treat the entire segment as the call span.
                    call_span_abs = (base, base + len(segment_tokens))
                else:
                    call_span_abs = (base + call_full_span[0], base + call_full_span[1])
                tool_events.append(
                    ToolEventRecord(
                        call=ToolCallRecord.from_call(tool_call),
                        result=ToolResultRecord.from_result(result),
                        call_span=call_span_abs,
                        return_span=(return_start, return_end),
                    )
                )

                continue

            # Check for EOS
            if segment_tokens and segment_tokens[-1] == eos_token_id:
                # Save final message
                if segment_text.strip():
                    turn.messages.append(AgentMessage(
                        text=segment_text.strip(),
                        tokens=segment_tokens,
                    ))
                break

    turn.tokens = tokens
    turn.tool_events = tool_events
    turn.record = record_from_turn(prompt_tokens=turn.prompt_tokens, tokens=turn.tokens, tool_events=tool_events)
    return turn


def generate_turn_sync(
    model,
    *,
    enc,
    prompt_ids: Sequence[int],
    tool_executor: AsyncToolExecutor | None = None,
    max_new_tokens: int = 2048,
    max_tool_rounds: int = 10,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> AgentTurn:
    """Synchronous wrapper for generate_turn_async.

    Args:
        Same as generate_turn_async

    Returns:
        AgentTurn with full trajectory
    """
    if tool_executor is None:
        # No tools - use simple generation
        from nmoe.rl.rollout import generate_one

        traj = generate_one(
            model,
            enc=enc,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )

        # Convert to AgentTurn
        turn = AgentTurn(
            prompt_tokens=list(prompt_ids),
            tokens=traj.tokens,
            max_tokens=max_new_tokens,
        )
        turn.messages.append(AgentMessage(
            text=traj.completion_text,
            tokens=traj.tokens[traj.prompt_len:],
        ))
        return turn

    # Run async in event loop.
    #
    # Python 3.12 no longer creates a default loop for the main thread, so
    # asyncio.get_event_loop() can raise. Prefer the running-loop API.
    coro = generate_turn_async(
        model,
        enc=enc,
        prompt_ids=prompt_ids,
        tool_executor=tool_executor,
        max_new_tokens=max_new_tokens,
        max_tool_rounds=max_tool_rounds,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    try:
        import nest_asyncio
    except ImportError as e:
        raise RuntimeError(
            "generate_turn_sync called from an async context; "
            "call generate_turn_async instead (or install nest_asyncio)."
        ) from e
    nest_asyncio.apply()
    return loop.run_until_complete(coro)


# =============================================================================
# Batch Generation
# =============================================================================

async def generate_batch_async(
    model,
    *,
    enc,
    prompts: list[Sequence[int]],
    tool_executor: AsyncToolExecutor | None = None,
    max_new_tokens: int = 2048,
    max_tool_rounds: int = 10,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[AgentTurn]:
    """Generate turns for a batch of prompts.

    Note: Currently generates sequentially. For true parallel generation,
    would need batched model inference.

    Args:
        model: Language model
        enc: Tokenizer
        prompts: List of prompt token sequences
        tool_executor: Optional tool executor
        max_new_tokens: Max tokens per turn
        max_tool_rounds: Max tool rounds per turn
        eos_token_id: EOS token
        temperature: Sampling temperature
        top_p: Nucleus sampling

    Returns:
        List of AgentTurn, one per prompt
    """
    turns = []
    for prompt_ids in prompts:
        if tool_executor is not None:
            turn = await generate_turn_async(
                model,
                enc=enc,
                prompt_ids=prompt_ids,
                tool_executor=tool_executor,
                max_new_tokens=max_new_tokens,
                max_tool_rounds=max_tool_rounds,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            turn = generate_turn_sync(
                model,
                enc=enc,
                prompt_ids=prompt_ids,
                tool_executor=None,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
        turns.append(turn)
    return turns


def generate_batch_sync(
    model,
    *,
    enc,
    prompts: list[Sequence[int]],
    tool_executor: AsyncToolExecutor | None = None,
    max_new_tokens: int = 2048,
    max_tool_rounds: int = 10,
    eos_token_id: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[AgentTurn]:
    """Synchronous batch generation.

    Args:
        Same as generate_batch_async

    Returns:
        List of AgentTurn
    """
    return asyncio.run(generate_batch_async(
        model,
        enc=enc,
        prompts=prompts,
        tool_executor=tool_executor,
        max_new_tokens=max_new_tokens,
        max_tool_rounds=max_tool_rounds,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    ))


# =============================================================================
# Log-Probability Computation for Turns
# =============================================================================

def turn_completion_nll_mean(
    model,
    *,
    turns: list[AgentTurn],
    pad_id: int,
    device: torch.device,
    normalize_by_length: bool = True,
    max_length: int | None = None,
) -> torch.Tensor:
    """Compute NLL for turn completions (for GRPO).

    Wraps the existing completion_nll_mean for AgentTurn objects.

    Args:
        model: Language model
        turns: List of AgentTurn objects
        pad_id: Padding token ID
        device: Target device
        normalize_by_length: Normalize by sequence length
        max_length: Constant length normalization (Dr.GRPO)

    Returns:
        [N] tensor of per-turn NLL
    """
    from nmoe.rl.rollout import completion_nll_mean

    seqs = [t.tokens for t in turns]
    prompt_lens = [len(t.prompt_tokens) for t in turns]
    completion_lens = [t.completion_tokens for t in turns]

    return completion_nll_mean(
        model,
        seqs=seqs,
        prompt_lens=prompt_lens,
        completion_lens=completion_lens,
        pad_id=pad_id,
        device=device,
        normalize_by_length=normalize_by_length,
        max_length=max_length,
    )
