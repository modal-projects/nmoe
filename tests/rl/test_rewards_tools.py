"""Tests for tool reward signals (rewards_tools.py)."""

from __future__ import annotations

from nmoe.rl.rewards_tools import ToolCallSite
from nmoe.rl.tools import ToolCall, ToolResult, ToolType


class TestToolRewards:
    def test_compute_all_tool_rewards_no_tools_is_zero(self):
        from nmoe.rl.rewards_tools import compute_all_tool_rewards

        rewards = compute_all_tool_rewards([])
        assert rewards["has_tool_use"] == 0.0
        assert rewards["tool_executed"] == 0.0
        assert rewards["tool_output_used"] == 0.0

    def test_output_usage_short_output_requires_exact_match(self):
        from nmoe.rl.rewards_tools import compute_output_usage_reward

        result = ToolResult(call_id="1", success=True, output="abc")
        assert compute_output_usage_reward(result, subsequent_text="...abc...", min_overlap_chars=20) == 1.0
        assert compute_output_usage_reward(result, subsequent_text="...ab...", min_overlap_chars=20) == 0.0

    def test_output_usage_long_output_uses_window_overlap(self):
        from nmoe.rl.rewards_tools import compute_output_usage_reward

        out = "0123456789ABCDEFGHIJ"  # len=20
        result = ToolResult(call_id="1", success=True, output=out)
        assert compute_output_usage_reward(result, subsequent_text=f"xxx{out}yyy", min_overlap_chars=10) == 1.0
        assert compute_output_usage_reward(result, subsequent_text="no overlap", min_overlap_chars=10) == 0.0

    def test_tool_rewards_python_success_and_usage(self):
        from nmoe.rl.rewards_tools import compute_all_tool_rewards

        call = ToolCall(type=ToolType.PYTHON, call_id="1", code="print('hi')")
        result = ToolResult(
            call_id="1",
            success=True,
            output="hello world",
            error=None,
            exit_code=0,
            timed_out=False,
            compiled=True,
            executed=True,
        )
        site = ToolCallSite(call=call, result=result, subsequent_text="I saw: hello world")
        rewards = compute_all_tool_rewards([site])

        assert rewards["has_tool_use"] == 1.0
        assert rewards["tool_syntax_valid"] == 1.0
        assert rewards["tool_executed"] == 1.0
        assert rewards["tool_no_error"] == 1.0
        assert rewards["tool_no_timeout"] == 1.0
        assert rewards["tool_has_output"] == 1.0
        assert rewards["tool_output_used"] == 1.0

        assert rewards["python_compiles"] == 1.0
        assert rewards["python_runs"] == 1.0
        assert rewards["python_correct"] == 1.0

    def test_tool_rewards_bash_failure(self):
        from nmoe.rl.rewards_tools import compute_all_tool_rewards

        call = ToolCall(type=ToolType.BASH, call_id="1", command="false")
        result = ToolResult(
            call_id="1",
            success=False,
            output="",
            error="nonzero",
            exit_code=1,
            timed_out=False,
        )
        site = ToolCallSite(call=call, result=result, subsequent_text="")
        rewards = compute_all_tool_rewards([site])

        assert rewards["has_tool_use"] == 1.0
        assert rewards["bash_valid"] == 1.0
        assert rewards["bash_succeeds"] == 0.0

