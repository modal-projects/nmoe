"""Tests for unified GDPO reward aggregation (rewards_gdpo.py)."""

from __future__ import annotations

import re

import pytest

from nmoe.rl.rewards_gdpo import RewardSignals, TrajectoryContext, batch_rewards_to_tensors, reshape_for_gdpo
from nmoe.rl.rewards_tools import ToolCallSite
from nmoe.rl.tools import ToolCall, ToolResult, ToolType


class _DummyTask:
    def __init__(self, gold: str):
        self._gold = gold

    def to_prompt(self) -> str:
        return "dummy"

    def extract_answer(self, response: str) -> str | None:
        m = re.search(r"<answer>(.*?)</answer>", response, flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    def verify(self, answer: str | None) -> bool:
        return answer == self._gold


class TestComputeAllRewards:
    def test_r1zero_format_sets_structure_and_accuracy(self):
        from nmoe.rl.rewards_gdpo import compute_all_rewards

        task = _DummyTask(gold="42")
        text = "<think>reasoning</think><answer>42</answer>"
        ctx = TrajectoryContext(
            response_text=text,
            format_type="r1zero",
            task=task,
            reasoning_tokens=5,
        )
        rewards = compute_all_rewards(ctx)
        assert isinstance(rewards, RewardSignals)
        assert rewards.has_reasoning == 1.0
        assert rewards.has_final_response == 1.0
        assert rewards.answer_correct == 1.0
        assert rewards.task_complete == 1.0

    def test_harmony_format_parsing_from_text(self):
        from nmoe.rl.rewards_gdpo import compute_all_rewards

        text = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "this is sufficiently long analysis"
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>ok<|end|>"
        )
        ctx = TrajectoryContext(
            response_text=text,
            format_type="harmony",
        )
        rewards = compute_all_rewards(ctx)
        assert rewards.struct_has_start == 1.0
        assert rewards.struct_has_end == 1.0
        assert rewards.struct_proper_nesting == 1.0
        assert rewards.chan_has_analysis == 1.0
        assert rewards.chan_has_final == 1.0
        assert rewards.chan_analysis_nonempty == 1.0
        assert rewards.chan_final_nonempty == 1.0

    def test_tool_rewards_flow_into_signals(self):
        from nmoe.rl.rewards_gdpo import compute_all_rewards

        call = ToolCall(type=ToolType.PYTHON, call_id="1", code="print('x')")
        result = ToolResult(call_id="1", success=True, output="VALUE=123", compiled=True, executed=True)
        site = ToolCallSite(call=call, result=result, subsequent_text="I used VALUE=123 in my reasoning")

        ctx = TrajectoryContext(
            response_text="<think>x</think><answer>42</answer>",
            format_type="r1zero",
            task=_DummyTask(gold="42"),
            tool_sites=[site],
            reasoning_tokens=10,
        )
        rewards = compute_all_rewards(ctx)
        assert rewards.has_tool_use == 1.0
        assert rewards.tool_executed == 1.0
        assert rewards.tool_output_used == 1.0
        assert rewards.python_compiles == 1.0
        assert rewards.python_runs == 1.0
        assert rewards.python_correct == 1.0

    def test_thinking_efficiency_only_if_correct(self):
        from nmoe.rl.rewards_gdpo import compute_all_rewards
        from nmoe.rl.curriculum import ThinkingBudgetConfig

        thinking = ThinkingBudgetConfig(
            max_reasoning_tokens=10,
            penalty_per_token=0.01,
            efficiency_bonus=0.1,
            budget_schedule=[(0, 10)],
        )

        ctx_bad = TrajectoryContext(
            response_text="<think>...</think><answer>no</answer>",
            format_type="r1zero",
            task=_DummyTask(gold="yes"),
            reasoning_tokens=5,
        )
        bad = compute_all_rewards(ctx_bad, thinking_config=thinking, total_examples=0)
        assert bad.answer_correct == 0.0
        assert bad.thinking_efficiency == 0.0

        ctx_good = TrajectoryContext(
            response_text="<think>...</think><answer>yes</answer>",
            format_type="r1zero",
            task=_DummyTask(gold="yes"),
            reasoning_tokens=5,
        )
        good = compute_all_rewards(ctx_good, thinking_config=thinking, total_examples=0)
        assert good.answer_correct == 1.0
        assert good.thinking_efficiency > 0.0


class TestBatchHelpers:
    def test_batch_rewards_to_tensors_and_reshape(self, device):
        rewards = [
            RewardSignals(answer_correct=1.0, tool_executed=0.0),
            RewardSignals(answer_correct=0.0, tool_executed=1.0),
            RewardSignals(answer_correct=1.0, tool_executed=1.0),
            RewardSignals(answer_correct=0.0, tool_executed=0.0),
        ]
        t = batch_rewards_to_tensors(rewards, device=device)
        assert set(t.keys()) >= {"answer_correct", "tool_executed"}

        reshaped = reshape_for_gdpo(t, group_size=2)
        assert reshaped["answer_correct"].shape == (2, 2)
        assert reshaped["tool_executed"].shape == (2, 2)

    def test_reshape_for_gdpo_requires_divisible_batch(self, device):
        rewards = [RewardSignals(answer_correct=1.0)]
        t = batch_rewards_to_tensors(rewards, device=device)
        with pytest.raises(ValueError):
            reshape_for_gdpo(t, group_size=2)

