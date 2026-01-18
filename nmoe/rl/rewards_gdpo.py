"""Unified GDPO reward aggregation for agentic RL training.

Combines all reward signals (structure, tools, accuracy, efficiency)
into a single interface for GDPO multi-reward training.

All rewards are binary or normalized for RLVR compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from nmoe.rl.rewards_harmony import (
    compute_harmony_rewards,
    compute_r1zero_rewards,
    parse_harmony_text,
)
from nmoe.rl.rewards_tools import (
    ToolCallSite,
    compute_all_tool_rewards,
)
from nmoe.rl.curriculum import (
    ThinkingBudgetConfig,
    thinking_budget_reward,
    get_thinking_budget,
)


@dataclass
class RewardSignals:
    """Complete set of reward signals for a trajectory.

    All signals are floats in [0, 1] or [-1, 1] for GDPO compatibility.
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Structure Rewards (format correctness)
    # ═══════════════════════════════════════════════════════════════════════
    has_reasoning: float = 0.0  # Has reasoning/analysis content
    has_final_response: float = 0.0  # Has final answer

    # Harmony-specific structure
    struct_has_start: float = 0.0
    struct_has_end: float = 0.0
    struct_has_message: float = 0.0
    struct_has_channel: float = 0.0
    struct_proper_nesting: float = 0.0

    # Channel presence
    chan_has_analysis: float = 0.0
    chan_has_final: float = 0.0
    chan_analysis_nonempty: float = 0.0
    chan_final_nonempty: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Tool Rewards (execution quality)
    # ═══════════════════════════════════════════════════════════════════════
    has_tool_use: float = 0.0  # Made at least one tool call
    tool_syntax_valid: float = 0.0  # Tool calls well-formed
    tool_executed: float = 0.0  # Tools ran successfully
    tool_no_error: float = 0.0  # No errors in execution
    tool_no_timeout: float = 0.0  # No timeouts
    tool_has_output: float = 0.0  # Tools produced output
    tool_output_used: float = 0.0  # Output referenced in reasoning

    # Code-specific
    python_compiles: float = 0.0
    python_runs: float = 0.0
    python_correct: float = 0.0
    bash_valid: float = 0.0
    bash_succeeds: float = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # Task Rewards (correctness)
    # ═══════════════════════════════════════════════════════════════════════
    answer_correct: float = 0.0  # Final answer is correct
    task_complete: float = 0.0  # Task objective achieved

    # ═══════════════════════════════════════════════════════════════════════
    # Efficiency Rewards (thinking budget)
    # ═══════════════════════════════════════════════════════════════════════
    thinking_efficiency: float = 0.0  # Correct with fewer reasoning tokens
    tool_efficiency: float = 0.0  # Correct with fewer tool calls

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for GDPO."""
        return {
            # Structure
            "has_reasoning": self.has_reasoning,
            "has_final_response": self.has_final_response,
            "struct_has_start": self.struct_has_start,
            "struct_has_end": self.struct_has_end,
            "struct_has_message": self.struct_has_message,
            "struct_has_channel": self.struct_has_channel,
            "struct_proper_nesting": self.struct_proper_nesting,
            "chan_has_analysis": self.chan_has_analysis,
            "chan_has_final": self.chan_has_final,
            "chan_analysis_nonempty": self.chan_analysis_nonempty,
            "chan_final_nonempty": self.chan_final_nonempty,
            # Tools
            "has_tool_use": self.has_tool_use,
            "tool_syntax_valid": self.tool_syntax_valid,
            "tool_executed": self.tool_executed,
            "tool_no_error": self.tool_no_error,
            "tool_no_timeout": self.tool_no_timeout,
            "tool_has_output": self.tool_has_output,
            "tool_output_used": self.tool_output_used,
            "python_compiles": self.python_compiles,
            "python_runs": self.python_runs,
            "python_correct": self.python_correct,
            "bash_valid": self.bash_valid,
            "bash_succeeds": self.bash_succeeds,
            # Task
            "answer_correct": self.answer_correct,
            "task_complete": self.task_complete,
            # Efficiency
            "thinking_efficiency": self.thinking_efficiency,
            "tool_efficiency": self.tool_efficiency,
        }


@dataclass
class TrajectoryContext:
    """Context for computing rewards on a trajectory."""

    # Generated content
    response_text: str = ""
    tokens: list[int] = field(default_factory=list)

    # Tool execution
    tool_sites: list[ToolCallSite] = field(default_factory=list)

    # Task info
    task: Any = None  # Task object with verify() method
    extracted_answer: str | None = None

    # Token counts
    reasoning_tokens: int = 0
    total_tokens: int = 0

    # Format type
    format_type: str = "harmony"  # "harmony" or "r1zero"

    # Tokenizer (optional, for token-level parsing)
    tokenizer: Any = None


def compute_all_rewards(
    ctx: TrajectoryContext,
    thinking_config: ThinkingBudgetConfig | None = None,
    total_examples: int = 0,
) -> RewardSignals:
    """Compute all GDPO reward signals for a trajectory.

    Args:
        ctx: Trajectory context with response, tools, task
        thinking_config: Optional thinking budget config
        total_examples: Total examples seen (for curriculum)

    Returns:
        RewardSignals with all computed values
    """
    signals = RewardSignals()

    # ─────────────────────────────────────────────────────────────────────
    # 1. Structure/Format Rewards
    # ─────────────────────────────────────────────────────────────────────
    if ctx.format_type == "harmony":
        format_rewards = compute_harmony_rewards(
            ctx.response_text,
            tokenizer=ctx.tokenizer,
            tokens=ctx.tokens if ctx.tokens else None,
        )
    else:  # r1zero
        format_rewards = compute_r1zero_rewards(ctx.response_text)

    # Map format rewards to signals
    signals.struct_has_start = format_rewards.get("struct_has_start", 0.0)
    signals.struct_has_end = format_rewards.get("struct_has_end", 0.0)
    signals.struct_has_message = format_rewards.get("struct_has_message", 0.0)
    signals.struct_has_channel = format_rewards.get("struct_has_channel", 0.0)
    signals.struct_proper_nesting = format_rewards.get("struct_proper_nesting", 0.0)
    signals.chan_has_analysis = format_rewards.get("chan_has_analysis", 0.0)
    signals.chan_has_final = format_rewards.get("chan_has_final", 0.0)
    signals.chan_analysis_nonempty = format_rewards.get("chan_analysis_nonempty", 0.0)
    signals.chan_final_nonempty = format_rewards.get("chan_final_nonempty", 0.0)

    # High-level structure
    signals.has_reasoning = signals.chan_has_analysis
    signals.has_final_response = signals.chan_has_final

    # ─────────────────────────────────────────────────────────────────────
    # 2. Tool Rewards
    # ─────────────────────────────────────────────────────────────────────
    tool_rewards = compute_all_tool_rewards(ctx.tool_sites)

    signals.has_tool_use = tool_rewards.get("has_tool_use", 0.0)
    signals.tool_syntax_valid = tool_rewards.get("tool_syntax_valid", 0.0)
    signals.tool_executed = tool_rewards.get("tool_executed", 0.0)
    signals.tool_no_error = tool_rewards.get("tool_no_error", 0.0)
    signals.tool_no_timeout = tool_rewards.get("tool_no_timeout", 0.0)
    signals.tool_has_output = tool_rewards.get("tool_has_output", 0.0)
    signals.tool_output_used = tool_rewards.get("tool_output_used", 0.0)
    signals.python_compiles = tool_rewards.get("python_compiles", 0.0)
    signals.python_runs = tool_rewards.get("python_runs", 0.0)
    signals.python_correct = tool_rewards.get("python_correct", 0.0)
    signals.bash_valid = tool_rewards.get("bash_valid", 0.0)
    signals.bash_succeeds = tool_rewards.get("bash_succeeds", 0.0)

    # ─────────────────────────────────────────────────────────────────────
    # 3. Task/Accuracy Rewards
    # ─────────────────────────────────────────────────────────────────────
    if ctx.task is not None and hasattr(ctx.task, "verify"):
        # Extract answer if not already done
        if ctx.extracted_answer is None and hasattr(ctx.task, "extract_answer"):
            ctx.extracted_answer = ctx.task.extract_answer(ctx.response_text)

        # Verify
        is_correct = ctx.task.verify(ctx.extracted_answer)
        signals.answer_correct = 1.0 if is_correct else 0.0
        signals.task_complete = signals.answer_correct

    # ─────────────────────────────────────────────────────────────────────
    # 4. Efficiency Rewards
    # ─────────────────────────────────────────────────────────────────────
    if thinking_config is not None:
        signals.thinking_efficiency = thinking_budget_reward(
            ctx.reasoning_tokens,
            signals.answer_correct > 0,
            thinking_config,
            total_examples,
        )

    # Tool efficiency: correct with fewer tool calls
    if signals.answer_correct > 0 and len(ctx.tool_sites) > 0:
        max_tools = 10  # Budget
        efficiency = 1.0 - (len(ctx.tool_sites) / max_tools)
        signals.tool_efficiency = max(0.0, min(1.0, efficiency))

    return signals


def batch_rewards_to_tensors(
    rewards_list: list[RewardSignals],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert batch of RewardSignals to GDPO-compatible tensors.

    Args:
        rewards_list: List of RewardSignals, one per trajectory
        device: Target device for tensors

    Returns:
        Dict of reward name -> [batch_size] tensor
    """
    if not rewards_list:
        return {}

    # Get all keys from first element
    keys = list(rewards_list[0].to_dict().keys())

    result = {}
    for key in keys:
        values = [r.to_dict()[key] for r in rewards_list]
        result[key] = torch.tensor(values, device=device, dtype=torch.float32)

    return result


def reshape_for_gdpo(
    rewards_dict: dict[str, torch.Tensor],
    group_size: int,
) -> dict[str, torch.Tensor]:
    """Reshape flat batch tensors to [B, G] for GDPO.

    Args:
        rewards_dict: Dict of reward name -> [batch_size] tensor
        group_size: Number of samples per prompt (G)

    Returns:
        Dict of reward name -> [B, G] tensor
    """
    result = {}
    for key, tensor in rewards_dict.items():
        batch_size = tensor.shape[0]
        if batch_size % group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} not divisible by group_size {group_size}"
            )
        n_prompts = batch_size // group_size
        result[key] = tensor.view(n_prompts, group_size)

    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_trajectory_rewards(
    response_text: str,
    task: Any | None = None,
    tool_sites: list[ToolCallSite] | None = None,
    format_type: str = "harmony",
    reasoning_tokens: int = 0,
    thinking_config: ThinkingBudgetConfig | None = None,
    total_examples: int = 0,
    tokenizer: Any = None,
    tokens: list[int] | None = None,
) -> RewardSignals:
    """Convenience function to compute rewards for a single trajectory.

    Args:
        response_text: Generated response
        task: Optional task for verification
        tool_sites: Optional tool call sites
        format_type: "harmony" or "r1zero"
        reasoning_tokens: Number of reasoning tokens
        thinking_config: Optional thinking budget config
        total_examples: Total examples seen
        tokenizer: Optional tokenizer for token-level parsing
        tokens: Optional token IDs

    Returns:
        RewardSignals
    """
    ctx = TrajectoryContext(
        response_text=response_text,
        tokens=tokens or [],
        tool_sites=tool_sites or [],
        task=task,
        reasoning_tokens=reasoning_tokens,
        format_type=format_type,
        tokenizer=tokenizer,
    )

    return compute_all_rewards(ctx, thinking_config, total_examples)


def filter_active_rewards(
    rewards_dict: dict[str, torch.Tensor],
    weights: dict[str, float],
    min_weight: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Filter rewards to only those with non-zero weights.

    Reduces computation by excluding unused reward signals.

    Args:
        rewards_dict: Dict of all rewards
        weights: Current curriculum weights
        min_weight: Minimum absolute weight to include

    Returns:
        Filtered rewards dict
    """
    return {
        k: v for k, v in rewards_dict.items()
        if abs(weights.get(k, 0.0)) >= min_weight
    }
