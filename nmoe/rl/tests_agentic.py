"""Comprehensive tests for agentic RL modules.

Tests cover:
1. Curriculum (StageType, WSD, thinking budget)
2. Harmony format parsing and rewards
3. R1-Zero format compatibility
4. Tool types and executor
5. Tool rewards computation
6. GDPO rewards aggregation
7. AgentTurn and multi-turn generation
8. Task base class and TaskPool
9. Math and code task implementations
10. Integration tests
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


# =============================================================================
# 1. CURRICULUM TESTS
# =============================================================================

def test_stage_type_enum():
    """Test StageType enum string compatibility."""
    from nmoe.rl.curriculum import StageType

    # Enum values
    assert StageType.WARMUP.value == "warmup"
    assert StageType.SUSTAIN.value == "sustain"
    assert StageType.DECAY.value == "decay"

    # String conversion
    assert str(StageType.WARMUP) == "warmup"
    assert str(StageType.SUSTAIN) == "sustain"

    # String comparison (key feature of str, Enum)
    assert StageType.WARMUP == "warmup"
    assert StageType.SUSTAIN == "sustain"
    assert "warmup" == StageType.WARMUP

    print("test_stage_type_enum: PASS")


def test_curriculum_stage_normalization():
    """Test CurriculumStage normalizes enum to string."""
    from nmoe.rl.curriculum import CurriculumStage, StageType

    # Create with enum
    stage = CurriculumStage(
        name=StageType.WARMUP,
        examples_count=100,
        weights={"accuracy": 1.0},
        transitions_to=StageType.SUSTAIN,
    )

    # Should be normalized to strings
    assert stage.name == "warmup"
    assert stage.transitions_to == "sustain"
    assert isinstance(stage.name, str)

    # Create with string (should also work)
    stage2 = CurriculumStage(
        name="custom_stage",
        examples_count=50,
        weights={"custom": 0.5},
    )
    assert stage2.name == "custom_stage"

    print("test_curriculum_stage_normalization: PASS")


def test_wsd_curriculum_transitions():
    """Test WSD curriculum stage transitions."""
    from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum, StageType

    stages = [
        CurriculumStage(name=StageType.WARMUP, examples_count=10, weights={"a": 1.0}, transitions_to=StageType.SUSTAIN),
        CurriculumStage(name=StageType.SUSTAIN, examples_count=20, weights={"a": 0.5}, transitions_to=StageType.DECAY),
        CurriculumStage(name=StageType.DECAY, examples_count=100, weights={"a": 0.1}),
    ]

    curriculum = WSDCurriculum(stages=stages)

    # Initial state
    assert curriculum.current_stage == "warmup"
    assert curriculum.examples_in_stage == 0

    # Step through warmup
    for i in range(10):
        result = curriculum.step()
        if i < 9:
            assert result is None
        else:
            assert result == "sustain"

    assert curriculum.current_stage == "sustain"
    assert curriculum.examples_in_stage == 0

    # Step through sustain
    for i in range(20):
        result = curriculum.step()
        if i < 19:
            assert result is None
        else:
            assert result == "decay"

    assert curriculum.current_stage == "decay"

    # Decay is terminal
    for _ in range(10):
        result = curriculum.step()
        assert result is None

    assert curriculum.current_stage == "decay"

    print("test_wsd_curriculum_transitions: PASS")


def test_curriculum_weight_decay():
    """Test within-stage weight decay."""
    from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum

    stages = [
        CurriculumStage(
            name="test",
            examples_count=1000,
            weights={"format": 1.0, "accuracy": 2.0},
            weight_decay_rate=0.1,  # 10% decay per example
        ),
    ]

    curriculum = WSDCurriculum(stages=stages)

    # Initial weights
    w0 = curriculum.get_weights()
    assert w0["format"] == 1.0
    assert w0["accuracy"] == 2.0

    # After 1 example
    curriculum.step()
    w1 = curriculum.get_weights()
    assert abs(w1["format"] - 0.9) < 0.001  # 1.0 * 0.9^1
    assert abs(w1["accuracy"] - 1.8) < 0.001  # 2.0 * 0.9^1

    # After 5 more examples (6 total)
    for _ in range(5):
        curriculum.step()
    w6 = curriculum.get_weights()
    expected_format = 1.0 * (0.9 ** 6)
    assert abs(w6["format"] - expected_format) < 0.001

    print("test_curriculum_weight_decay: PASS")


def test_thinking_budget_config():
    """Test thinking budget reward computation."""
    from nmoe.rl.curriculum import ThinkingBudgetConfig, thinking_budget_reward, get_thinking_budget

    config = ThinkingBudgetConfig(
        max_reasoning_tokens=2048,
        penalty_per_token=0.001,
        efficiency_bonus=0.5,
        budget_schedule=[(0, 4096), (100, 2048), (500, 1024)],
    )

    # Budget schedule
    assert get_thinking_budget(config, 0) == 4096
    assert get_thinking_budget(config, 50) == 4096
    assert get_thinking_budget(config, 100) == 2048
    assert get_thinking_budget(config, 300) == 2048
    assert get_thinking_budget(config, 500) == 1024
    assert get_thinking_budget(config, 1000) == 1024

    # Efficiency reward (correct answer)
    # Under budget - should get bonus
    reward = thinking_budget_reward(1000, True, config, 0)  # budget=4096
    assert reward > 0  # Under budget bonus

    # At budget - should get small bonus
    reward = thinking_budget_reward(4096, True, config, 0)
    assert abs(reward) < 0.01  # Near zero

    # Over budget - should get penalty
    reward = thinking_budget_reward(5000, True, config, 0)
    assert reward < 0  # Penalty

    # Wrong answer - no efficiency reward
    reward = thinking_budget_reward(1000, False, config, 0)
    assert reward == 0.0

    print("test_thinking_budget_config: PASS")


def test_rolling_metrics():
    """Test rolling metrics for adaptive curriculum."""
    from nmoe.rl.curriculum import RollingMetrics

    metrics = RollingMetrics(window=5)

    # Add some values
    for i in range(10):
        metrics.add({"accuracy": i / 10.0, "loss": 1.0 - i / 10.0})

    # Should only keep last 5
    assert metrics.count("accuracy") == 5
    assert metrics.count("loss") == 5

    # Mean of last 5 (0.5, 0.6, 0.7, 0.8, 0.9)
    expected_mean = (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5
    assert abs(metrics.mean("accuracy") - expected_mean) < 0.001

    # Unknown key returns default
    assert metrics.mean("unknown") == 0.0
    assert metrics.mean("unknown", default=0.5) == 0.5

    print("test_rolling_metrics: PASS")


# =============================================================================
# 2. HARMONY FORMAT TESTS
# =============================================================================

def test_harmony_token_constants():
    """Test Harmony token constants."""
    from nmoe.rl.rewards_harmony import HARMONY_TOKENS, CHANNELS

    assert HARMONY_TOKENS["start"] == "<|start|>"
    assert HARMONY_TOKENS["end"] == "<|end|>"
    assert HARMONY_TOKENS["message"] == "<|message|>"
    assert HARMONY_TOKENS["channel"] == "<|channel|>"
    assert HARMONY_TOKENS["call"] == "<|call|>"
    assert HARMONY_TOKENS["return"] == "<|return|>"

    assert CHANNELS["analysis"] == "analysis"
    assert CHANNELS["final"] == "final"
    assert CHANNELS["commentary"] == "commentary"

    print("test_harmony_token_constants: PASS")


def test_parse_harmony_text_valid():
    """Test parsing valid Harmony format."""
    from nmoe.rl.rewards_harmony import parse_harmony_text

    text = """<|start|>assistant<|channel|>analysis<|message|>
Let me think about this step by step.
First, I need to analyze the problem.
<|end|><|start|>assistant<|channel|>final<|message|>
The answer is 42.
<|end|>"""

    parsed = parse_harmony_text(text)

    assert parsed.has_start
    assert parsed.has_end
    assert parsed.has_message
    assert parsed.has_channel
    assert parsed.properly_nested

    assert len(parsed.messages) == 2
    assert parsed.messages[0].channel == "analysis"
    assert parsed.messages[1].channel == "final"
    assert "think about this" in parsed.messages[0].content
    assert "42" in parsed.messages[1].content

    assert parsed.has_analysis
    assert parsed.has_final
    assert len(parsed.analysis_content) > 10
    assert len(parsed.final_content) > 0

    print("test_parse_harmony_text_valid: PASS")


def test_parse_harmony_text_missing_tags():
    """Test parsing Harmony with missing tags."""
    from nmoe.rl.rewards_harmony import parse_harmony_text

    # Missing start
    text1 = "assistant<|channel|>final<|message|>answer<|end|>"
    parsed1 = parse_harmony_text(text1)
    assert not parsed1.has_start
    assert any("Missing" in err and "start" in err for err in parsed1.parse_errors)

    # Missing end
    text2 = "<|start|>assistant<|channel|>final<|message|>answer"
    parsed2 = parse_harmony_text(text2)
    assert parsed2.has_start
    assert not parsed2.has_end

    # Missing channel
    text3 = "<|start|>assistant<|message|>answer<|end|>"
    parsed3 = parse_harmony_text(text3)
    assert not parsed3.has_channel

    print("test_parse_harmony_text_missing_tags: PASS")


def test_harmony_nesting_validation():
    """Test Harmony nesting validation."""
    from nmoe.rl.rewards_harmony import _check_nesting

    # Valid nesting
    assert _check_nesting("<|start|>a<|end|><|start|>b<|end|>") == True
    assert _check_nesting("<|start|>content<|end|>") == True

    # Invalid: nested starts
    assert _check_nesting("<|start|><|start|>a<|end|><|end|>") == False

    # Invalid: end before start
    assert _check_nesting("<|end|><|start|>a<|end|>") == False

    # Invalid: unclosed
    assert _check_nesting("<|start|>a") == False

    print("test_harmony_nesting_validation: PASS")


def test_harmony_structure_rewards():
    """Test Harmony structure reward computation."""
    from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_structure_rewards

    # Valid format
    text = "<|start|>assistant<|channel|>final<|message|>answer<|end|>"
    parsed = parse_harmony_text(text)
    rewards = harmony_structure_rewards(parsed)

    assert rewards["struct_has_start"] == 1.0
    assert rewards["struct_has_end"] == 1.0
    assert rewards["struct_has_message"] == 1.0
    assert rewards["struct_has_channel"] == 1.0
    assert rewards["struct_proper_nesting"] == 1.0

    # Invalid format
    bad_text = "just plain text"
    bad_parsed = parse_harmony_text(bad_text)
    bad_rewards = harmony_structure_rewards(bad_parsed)

    assert bad_rewards["struct_has_start"] == 0.0
    assert bad_rewards["struct_has_end"] == 0.0

    print("test_harmony_structure_rewards: PASS")


def test_harmony_channel_rewards():
    """Test Harmony channel reward computation."""
    from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_channel_rewards

    # Full format with both channels
    text = """<|start|>assistant<|channel|>analysis<|message|>
This is my reasoning process with enough content.
<|end|><|start|>assistant<|channel|>final<|message|>
42
<|end|>"""

    parsed = parse_harmony_text(text)
    rewards = harmony_channel_rewards(parsed)

    assert rewards["chan_has_analysis"] == 1.0
    assert rewards["chan_has_final"] == 1.0
    assert rewards["chan_analysis_nonempty"] == 1.0  # >10 chars
    assert rewards["chan_final_nonempty"] == 1.0

    # Only final channel
    text2 = "<|start|>assistant<|channel|>final<|message|>42<|end|>"
    parsed2 = parse_harmony_text(text2)
    rewards2 = harmony_channel_rewards(parsed2)

    assert rewards2["chan_has_analysis"] == 0.0
    assert rewards2["chan_has_final"] == 1.0

    print("test_harmony_channel_rewards: PASS")


def test_compute_harmony_rewards():
    """Test unified Harmony reward computation."""
    from nmoe.rl.rewards_harmony import compute_harmony_rewards

    text = """<|start|>assistant<|channel|>analysis<|message|>
Let me analyze this problem carefully.
<|end|><|start|>assistant<|channel|>final<|message|>
The answer is 42.
<|end|>"""

    rewards = compute_harmony_rewards(text)

    # Structure
    assert rewards["struct_has_start"] == 1.0
    assert rewards["struct_has_end"] == 1.0
    assert rewards["struct_proper_nesting"] == 1.0

    # Channels
    assert rewards["chan_has_analysis"] == 1.0
    assert rewards["chan_has_final"] == 1.0

    print("test_compute_harmony_rewards: PASS")


# =============================================================================
# 3. R1-ZERO FORMAT TESTS
# =============================================================================

def test_r1zero_format_parsing():
    """Test R1-Zero <think>/<answer> format parsing."""
    from nmoe.rl.rewards_harmony import parse_r1zero_format

    text = "<think>Let me solve this step by step.</think><answer>42</answer>"

    parsed = parse_r1zero_format(text)

    assert len(parsed.messages) == 2
    assert parsed.messages[0].channel == "analysis"
    assert parsed.messages[1].channel == "final"
    assert "solve this" in parsed.messages[0].content
    assert "42" in parsed.messages[1].content

    assert parsed.has_analysis
    assert parsed.has_final
    assert parsed.properly_nested

    print("test_r1zero_format_parsing: PASS")


def test_r1zero_partial_format():
    """Test R1-Zero with partial format (only think or only answer)."""
    from nmoe.rl.rewards_harmony import parse_r1zero_format

    # Only think
    text1 = "<think>Thinking about it...</think>"
    parsed1 = parse_r1zero_format(text1)
    assert parsed1.has_analysis
    assert not parsed1.has_final

    # Only answer
    text2 = "<answer>42</answer>"
    parsed2 = parse_r1zero_format(text2)
    assert not parsed2.has_analysis
    assert parsed2.has_final

    # Wrong order (answer before think)
    text3 = "<answer>42</answer><think>oops</think>"
    parsed3 = parse_r1zero_format(text3)
    assert not parsed3.properly_nested  # Wrong order

    print("test_r1zero_partial_format: PASS")


def test_compute_r1zero_rewards():
    """Test R1-Zero reward computation (Harmony-compatible)."""
    from nmoe.rl.rewards_harmony import compute_r1zero_rewards

    text = "<think>Let me work through this problem.</think><answer>The answer is 42.</answer>"

    rewards = compute_r1zero_rewards(text)

    # Structure (mapped from R1-Zero)
    assert rewards["struct_has_start"] == 1.0  # has think tag
    assert rewards["struct_has_end"] == 1.0  # has answer tag
    assert rewards["struct_proper_nesting"] == 1.0

    # Channels (mapped)
    assert rewards["chan_has_analysis"] == 1.0
    assert rewards["chan_has_final"] == 1.0
    assert rewards["chan_analysis_nonempty"] == 1.0

    print("test_compute_r1zero_rewards: PASS")


# =============================================================================
# 4. TOOL TYPES AND EXECUTOR TESTS
# =============================================================================

def test_tool_type_enum():
    """Test ToolType enum."""
    from nmoe.rl.tools import ToolType

    assert ToolType.PYTHON.value == "python"
    assert ToolType.BASH.value == "bash"
    assert ToolType.READ.value == "read"
    assert ToolType.SEARCH.value == "search"
    assert ToolType.EDIT.value == "edit"

    print("test_tool_type_enum: PASS")


def test_tool_call_dataclass():
    """Test ToolCall dataclass."""
    from nmoe.rl.tools import ToolCall, ToolType

    # Python call
    python_call = ToolCall(type=ToolType.PYTHON, code="print(2+2)")
    assert python_call.type == ToolType.PYTHON
    assert python_call.code == "print(2+2)"
    assert python_call.call_id == ""  # Default

    # Bash call
    bash_call = ToolCall(type=ToolType.BASH, command="ls -la")
    assert bash_call.type == ToolType.BASH
    assert bash_call.command == "ls -la"

    # Read call
    read_call = ToolCall(type=ToolType.READ, path="/tmp/test.txt")
    assert read_call.type == ToolType.READ
    assert read_call.path == "/tmp/test.txt"

    # With custom timeout
    call_with_timeout = ToolCall(type=ToolType.PYTHON, code="x=1", timeout_ms=5000)
    assert call_with_timeout.timeout_ms == 5000

    print("test_tool_call_dataclass: PASS")


def test_tool_result_dataclass():
    """Test ToolResult dataclass."""
    from nmoe.rl.tools import ToolResult

    # Success result
    success = ToolResult(
        call_id="test_001",
        success=True,
        output="4\n",
        exit_code=0,
        execution_time_ms=10.5,
    )
    assert success.success
    assert success.output == "4\n"
    assert success.error is None
    assert not success.timed_out

    # Error result
    error = ToolResult(
        call_id="test_002",
        success=False,
        error="SyntaxError: invalid syntax",
        exit_code=1,
    )
    assert not error.success
    assert error.error == "SyntaxError: invalid syntax"

    # Timeout result
    timeout = ToolResult.from_timeout("test_003")
    assert not timeout.success
    assert timeout.timed_out
    assert timeout.exit_code == 124

    # Error factory
    err = ToolResult.from_error("test_004", "File not found")
    assert not err.success
    assert err.error == "File not found"

    print("test_tool_result_dataclass: PASS")


def test_tool_config():
    """Test ToolConfig dataclass."""
    from nmoe.rl.tools import ToolConfig

    # Default config
    default = ToolConfig()
    assert default.executor_type == "subprocess"  # Default is subprocess (codex-rs)
    assert default.timeout_default_ms == 30000
    assert default.python_workers == 4

    # Custom config with native executor
    custom = ToolConfig(
        executor_type="native",
        timeout_default_ms=60000,
        python_workers=8,
        codex_binary="/usr/local/bin/codex",
    )
    assert custom.executor_type == "native"
    assert custom.codex_binary == "/usr/local/bin/codex"

    print("test_tool_config: PASS")


def test_native_python_execution():
    """Test native Python execution."""
    from nmoe.rl.tools.executor import _execute_python_native

    # Simple code
    result = _execute_python_native("print(2 + 2)", timeout=5.0)
    assert result["success"]
    assert "4" in result["output"]
    assert result["exit_code"] == 0
    assert result["compiled"]
    assert result["executed"]

    # Syntax error
    result2 = _execute_python_native("print(", timeout=5.0)
    assert not result2["success"]
    assert result2["exit_code"] != 0

    # Runtime error
    result3 = _execute_python_native("raise ValueError('test')", timeout=5.0)
    assert not result3["success"]
    assert result3["exit_code"] != 0

    print("test_native_python_execution: PASS")


def test_native_bash_execution():
    """Test native bash execution."""
    from nmoe.rl.tools.executor import _execute_bash_native

    # Simple command
    result = _execute_bash_native("echo hello", timeout=5.0)
    assert result["success"]
    assert "hello" in result["output"]
    assert result["exit_code"] == 0

    # Command that fails
    result2 = _execute_bash_native("exit 1", timeout=5.0)
    assert not result2["success"]
    assert result2["exit_code"] == 1

    # Invalid command
    result3 = _execute_bash_native("nonexistent_command_xyz", timeout=5.0)
    assert not result3["success"]

    print("test_native_bash_execution: PASS")


def test_native_file_read():
    """Test native file read."""
    from nmoe.rl.tools.executor import _read_file_native

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test content\nline 2")
        temp_path = f.name

    try:
        result = _read_file_native(temp_path)
        assert result["success"]
        assert "test content" in result["output"]
        assert "line 2" in result["output"]

        # Non-existent file
        result2 = _read_file_native("/nonexistent/path/file.txt")
        assert not result2["success"]
        assert result2["error"] is not None
    finally:
        Path(temp_path).unlink()

    print("test_native_file_read: PASS")


def test_async_tool_executor_scatter_gather():
    """Test AsyncToolExecutor scatter/gather pattern."""
    from nmoe.rl.tools import ToolCall, ToolType, ToolConfig
    from nmoe.rl.tools.executor import AsyncToolExecutor

    async def run_test():
        executor = AsyncToolExecutor(ToolConfig(executor_type="native"))

        # Create multiple calls
        calls = [
            ToolCall(type=ToolType.PYTHON, code="print(1)"),
            ToolCall(type=ToolType.PYTHON, code="print(2)"),
            ToolCall(type=ToolType.BASH, command="echo 3"),
        ]

        # Scatter
        call_ids = executor.scatter(calls)
        assert len(call_ids) == 3
        assert all(len(cid) > 0 for cid in call_ids)

        # Gather
        results = await executor.gather(call_ids, timeout=10.0)
        assert len(results) == 3

        # Check results
        outputs = [r.output.strip() for r in results]
        assert "1" in outputs
        assert "2" in outputs
        assert "3" in outputs

        executor.close()

    asyncio.run(run_test())
    print("test_async_tool_executor_scatter_gather: PASS")


def test_async_tool_executor_execute_one():
    """Test AsyncToolExecutor execute_one convenience method."""
    from nmoe.rl.tools import ToolCall, ToolType, ToolConfig
    from nmoe.rl.tools.executor import AsyncToolExecutor

    async def run_test():
        executor = AsyncToolExecutor(ToolConfig(executor_type="native"))

        call = ToolCall(type=ToolType.PYTHON, code="print('hello world')")
        result = await executor.execute_one(call)

        assert result.success
        assert "hello world" in result.output

        executor.close()

    asyncio.run(run_test())
    print("test_async_tool_executor_execute_one: PASS")


# =============================================================================
# 5. TOOL REWARDS TESTS
# =============================================================================

def test_tool_reward_signals():
    """Test ToolRewardSignals dataclass."""
    from nmoe.rl.rewards_tools import ToolRewardSignals

    signals = ToolRewardSignals(
        syntax_valid=1.0,
        executed=1.0,
        no_error=1.0,
        no_timeout=1.0,
        has_output=1.0,
        python_compiles=1.0,
        python_runs=1.0,
    )

    assert signals.syntax_valid == 1.0
    assert signals.executed == 1.0
    assert signals.python_compiles == 1.0

    print("test_tool_reward_signals: PASS")


def test_compute_tool_call_rewards():
    """Test compute_tool_call_rewards function."""
    from nmoe.rl.tools import ToolCall, ToolResult, ToolType
    from nmoe.rl.rewards_tools import compute_tool_call_rewards

    # Successful Python call
    call = ToolCall(type=ToolType.PYTHON, code="print(1)")
    result = ToolResult(
        call_id="test",
        success=True,
        output="1\n",
        exit_code=0,
        compiled=True,
        executed=True,
    )

    signals = compute_tool_call_rewards(call, result)

    assert signals.syntax_valid == 1.0
    assert signals.executed == 1.0
    assert signals.no_error == 1.0
    assert signals.no_timeout == 1.0
    assert signals.has_output == 1.0
    assert signals.python_compiles == 1.0
    assert signals.python_runs == 1.0

    # Failed bash call
    bash_call = ToolCall(type=ToolType.BASH, command="exit 1")
    bash_result = ToolResult(
        call_id="test2",
        success=False,
        error="Command failed",
        exit_code=1,
    )

    bash_signals = compute_tool_call_rewards(bash_call, bash_result)

    assert bash_signals.syntax_valid == 1.0  # Call was well-formed
    assert bash_signals.executed == 0.0  # Did not succeed
    assert bash_signals.no_error == 0.0  # Had error
    assert bash_signals.bash_valid == 1.0
    assert bash_signals.bash_succeeds == 0.0

    print("test_compute_tool_call_rewards: PASS")


def test_aggregate_tool_rewards():
    """Test aggregating rewards across multiple tool calls."""
    from nmoe.rl.rewards_tools import ToolRewardSignals, aggregate_tool_rewards

    signals = [
        ToolRewardSignals(syntax_valid=1.0, executed=1.0, no_error=1.0),
        ToolRewardSignals(syntax_valid=1.0, executed=0.0, no_error=0.0),
        ToolRewardSignals(syntax_valid=1.0, executed=1.0, no_error=1.0),
    ]

    aggregated = aggregate_tool_rewards(signals)

    assert aggregated["tool_syntax_valid"] == 1.0  # All valid
    assert abs(aggregated["tool_executed"] - 2/3) < 0.001  # 2 of 3
    assert abs(aggregated["tool_no_error"] - 2/3) < 0.001

    # Empty list
    empty_agg = aggregate_tool_rewards([])
    assert empty_agg["tool_syntax_valid"] == 0.0

    print("test_aggregate_tool_rewards: PASS")


def test_compute_output_usage_reward():
    """Test output usage reward."""
    from nmoe.rl.tools import ToolResult
    from nmoe.rl.rewards_tools import compute_output_usage_reward

    # Short output (< min_overlap_chars) - checks if entire output appears
    result = ToolResult(
        call_id="test",
        success=True,
        output="42",  # Short output
    )

    # Output used (exact match)
    subsequent = "The answer is 42, which I found from the output."
    reward = compute_output_usage_reward(result, subsequent)
    assert reward == 1.0

    # Output not used
    subsequent2 = "I think the answer is 100."
    reward2 = compute_output_usage_reward(result, subsequent2)
    assert reward2 == 0.0

    # Longer output - checks sliding window
    long_result = ToolResult(
        call_id="test2",
        success=True,
        output="This is a longer output with detailed information about the result.",
    )
    # Subsequent contains a 20+ char substring from output
    subsequent3 = "Based on the detailed information about the result, I conclude..."
    reward3 = compute_output_usage_reward(long_result, subsequent3)
    assert reward3 == 1.0

    # Empty output
    empty_result = ToolResult(call_id="test", success=True, output="")
    reward4 = compute_output_usage_reward(empty_result, "any text")
    assert reward4 == 0.0

    print("test_compute_output_usage_reward: PASS")


def test_check_python_syntax():
    """Test Python syntax checking."""
    from nmoe.rl.rewards_tools import check_python_syntax

    assert check_python_syntax("print('hello')") == True
    assert check_python_syntax("x = 1 + 2") == True
    assert check_python_syntax("def foo():\n    return 1") == True

    assert check_python_syntax("print(") == False
    assert check_python_syntax("def foo(") == False
    assert check_python_syntax("x = ") == False

    print("test_check_python_syntax: PASS")


# =============================================================================
# 6. GDPO REWARDS AGGREGATION TESTS
# =============================================================================

def test_reward_signals_dataclass():
    """Test RewardSignals dataclass."""
    from nmoe.rl.rewards_gdpo import RewardSignals

    signals = RewardSignals(
        has_reasoning=1.0,
        has_final_response=1.0,
        struct_proper_nesting=1.0,
        answer_correct=1.0,
    )

    assert signals.has_reasoning == 1.0
    assert signals.answer_correct == 1.0

    # to_dict
    d = signals.to_dict()
    assert len(d) == 27  # All fields
    assert d["has_reasoning"] == 1.0
    assert d["answer_correct"] == 1.0
    assert d["thinking_efficiency"] == 0.0  # Default

    print("test_reward_signals_dataclass: PASS")


def test_batch_rewards_to_tensors():
    """Test converting batch of RewardSignals to tensors."""
    import torch
    from nmoe.rl.rewards_gdpo import RewardSignals, batch_rewards_to_tensors

    rewards = [
        RewardSignals(has_reasoning=1.0, answer_correct=1.0),
        RewardSignals(has_reasoning=1.0, answer_correct=0.0),
        RewardSignals(has_reasoning=0.0, answer_correct=1.0),
    ]

    tensors = batch_rewards_to_tensors(rewards, device="cpu")

    assert "has_reasoning" in tensors
    assert "answer_correct" in tensors
    assert tensors["has_reasoning"].shape == (3,)
    assert tensors["answer_correct"].shape == (3,)

    assert tensors["has_reasoning"].tolist() == [1.0, 1.0, 0.0]
    assert tensors["answer_correct"].tolist() == [1.0, 0.0, 1.0]

    print("test_batch_rewards_to_tensors: PASS")


def test_reshape_for_gdpo():
    """Test reshaping for GDPO [B, G] format."""
    import torch
    from nmoe.rl.rewards_gdpo import reshape_for_gdpo

    # 4 samples, group_size=2 -> [2, 2]
    rewards_dict = {
        "accuracy": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "format": torch.tensor([1.0, 1.0, 0.0, 1.0]),
    }

    reshaped = reshape_for_gdpo(rewards_dict, group_size=2)

    assert reshaped["accuracy"].shape == (2, 2)
    assert reshaped["format"].shape == (2, 2)

    # First prompt: [1.0, 0.0], Second prompt: [1.0, 1.0]
    assert reshaped["accuracy"][0].tolist() == [1.0, 0.0]
    assert reshaped["accuracy"][1].tolist() == [1.0, 1.0]

    print("test_reshape_for_gdpo: PASS")


def test_compute_trajectory_rewards():
    """Test compute_trajectory_rewards convenience function."""
    from nmoe.rl.rewards_gdpo import compute_trajectory_rewards

    text = """<|start|>assistant<|channel|>analysis<|message|>
Let me analyze this problem step by step.
<|end|><|start|>assistant<|channel|>final<|message|>
The answer is 42.
<|end|>"""

    signals = compute_trajectory_rewards(
        response_text=text,
        format_type="harmony",
    )

    # Structure rewards
    assert signals.struct_has_start == 1.0
    assert signals.struct_has_end == 1.0
    assert signals.struct_proper_nesting == 1.0

    # Channel rewards
    assert signals.chan_has_analysis == 1.0
    assert signals.chan_has_final == 1.0

    print("test_compute_trajectory_rewards: PASS")


# =============================================================================
# 7. AGENT TURN TESTS
# =============================================================================

def test_agent_message_dataclass():
    """Test AgentMessage dataclass."""
    from nmoe.rl.turns import AgentMessage

    msg = AgentMessage(text="Hello world", tokens=[1, 2, 3, 4, 5])

    assert msg.text == "Hello world"
    assert msg.tokens == [1, 2, 3, 4, 5]
    assert msg.token_count == 5

    # Empty message
    empty = AgentMessage()
    assert empty.text == ""
    assert empty.token_count == 0

    print("test_agent_message_dataclass: PASS")


def test_agent_turn_dataclass():
    """Test AgentTurn dataclass."""
    from nmoe.rl.turns import AgentTurn, AgentMessage

    turn = AgentTurn(
        messages=[
            AgentMessage(text="Let me think...", tokens=[1, 2, 3]),
            AgentMessage(text="The answer is 42", tokens=[4, 5, 6, 7]),
        ],
        tokens=[0, 1, 2, 3, 4, 5, 6, 7],  # Full sequence
        prompt_tokens=[0],  # Just the prompt
    )

    assert turn.reasoning_tokens == 7  # 3 + 4
    assert turn.completion_tokens == 7  # 8 - 1
    assert turn.num_tool_calls == 0
    assert turn.final_response == "The answer is 42"

    print("test_agent_turn_dataclass: PASS")


def test_agent_turn_with_tools():
    """Test AgentTurn with tool calls."""
    from nmoe.rl.turns import AgentTurn, AgentMessage
    from nmoe.rl.tools import ToolCall, ToolResult, ToolType

    turn = AgentTurn(
        messages=[
            AgentMessage(text="Let me check the files"),
            AgentMessage(text="Based on the output, the answer is 42"),
        ],
        tool_calls=[
            ToolCall(type=ToolType.BASH, command="ls"),
        ],
        tool_results=[
            ToolResult(call_id="1", success=True, output="file1.txt\nfile2.txt"),
        ],
        tokens=list(range(20)),
        prompt_tokens=list(range(5)),
    )

    assert turn.num_tool_calls == 1
    assert turn.tool_calls[0].command == "ls"
    assert turn.tool_results[0].success

    # to_tool_sites
    sites = turn.to_tool_sites()
    assert len(sites) == 1
    assert sites[0].call.command == "ls"
    assert sites[0].result.success

    print("test_agent_turn_with_tools: PASS")


def test_parse_tool_call_from_text():
    """Test tool call parsing from generated text."""
    from nmoe.rl.turns import _parse_tool_call_from_text
    from nmoe.rl.tools import ToolType

    # Python call
    text1 = "<|call|>python\nprint(2 + 2)\n<|end|>"
    call1 = _parse_tool_call_from_text(text1)
    assert call1 is not None
    assert call1.type == ToolType.PYTHON
    assert "print(2 + 2)" in call1.code

    # Bash call
    text2 = "<|call|>bash\nls -la\n<|end|>"
    call2 = _parse_tool_call_from_text(text2)
    assert call2 is not None
    assert call2.type == ToolType.BASH
    assert "ls -la" in call2.command

    # No tool call
    text3 = "Just regular text"
    call3 = _parse_tool_call_from_text(text3)
    assert call3 is None

    print("test_parse_tool_call_from_text: PASS")


# =============================================================================
# 8. TASK TESTS
# =============================================================================

def test_task_pool_sampling():
    """Test TaskPool sampling."""
    from nmoe.rl.tasks import Task, TaskPool

    class DummyTask(Task):
        def __init__(self, question: str, answer: str, task_type: str = "math"):
            self.question = question
            self.answer = answer
            self.task_type = task_type

        def to_prompt(self) -> str:
            return f"Q: {self.question}"

        def extract_answer(self, response: str) -> str | None:
            return response.strip()

        def verify(self, answer: str | None) -> bool:
            return answer == self.answer

    tasks = [
        DummyTask("1+1", "2", "easy"),
        DummyTask("2+2", "4", "easy"),
        DummyTask("10*10", "100", "medium"),
    ]

    pool = TaskPool(tasks)
    assert len(pool) == 3

    # Sample (uniform)
    sampled = pool.sample(2)
    assert len(sampled) == 2
    assert all(isinstance(t, DummyTask) for t in sampled)

    # Sample with weights by task_type
    weighted_pool = TaskPool(tasks, weights={"easy": 0.9, "medium": 0.1})
    samples = [weighted_pool.sample(1)[0] for _ in range(100)]
    # "easy" tasks should appear more often (2 easy tasks vs 1 medium)
    easy_count = sum(1 for s in samples if s.task_type == "easy")
    assert easy_count > 70  # Probabilistic, but should be true given 0.9 weight

    print("test_task_pool_sampling: PASS")


def test_task_pool_access():
    """Test TaskPool access methods."""
    from nmoe.rl.tasks import Task, TaskPool

    class DummyTask(Task):
        def __init__(self, id: int):
            self.id = id
            self.task_type = "dummy"

        def to_prompt(self) -> str:
            return f"Task {self.id}"

        def extract_answer(self, response: str) -> str | None:
            return response

        def verify(self, answer: str | None) -> bool:
            return True

    tasks = [DummyTask(i) for i in range(5)]
    pool = TaskPool(tasks)

    # Access underlying tasks list
    assert len(pool.tasks) == 5
    assert [t.id for t in pool.tasks] == [0, 1, 2, 3, 4]

    # Shuffle and verify
    pool.shuffle()
    assert len(pool.tasks) == 5  # Same count after shuffle

    print("test_task_pool_access: PASS")


# =============================================================================
# 9. MATH AND CODE TASK TESTS
# =============================================================================

def test_gsm8k_task():
    """Test GSM8K math task."""
    from nmoe.rl.tasks.math import GSM8KTask

    task = GSM8KTask(
        question="If John has 5 apples and Mary gives him 3 more, how many apples does John have?",
        gold_answer="8",
    )

    # Prompt
    prompt = task.to_prompt()
    assert "5 apples" in prompt
    assert "3 more" in prompt

    # Extract answer from response with <answer> tags
    response1 = "<think>5 + 3 = 8</think><answer>8</answer>"
    extracted1 = task.extract_answer(response1)
    assert extracted1 == "8"

    # Extract from boxed format
    response2 = "The answer is \\boxed{8}"
    extracted2 = task.extract_answer(response2)
    assert extracted2 == "8"

    # Verify
    assert task.verify("8") == True
    assert task.verify("8.0") == True  # Normalized
    assert task.verify("9") == False

    print("test_gsm8k_task: PASS")


def test_math_task():
    """Test MATH competition task."""
    from nmoe.rl.tasks.math import MATHTask

    task = MATHTask(
        problem="What is 2^10?",
        gold_answer="1024",
        level=1,
        subject="algebra",
    )

    # Verify different formats
    assert task.verify("1024") == True
    assert task.verify("1,024") == True  # With comma
    assert task.verify("1024.0") == True
    assert task.verify("1025") == False

    print("test_math_task: PASS")


def test_normalize_number():
    """Test number normalization for math tasks."""
    from nmoe.rl.tasks.math import normalize_number

    assert normalize_number("42") == "42"
    assert normalize_number("42.0") == "42"
    assert normalize_number("42.00") == "42"
    assert normalize_number("1,234") == "1234"
    assert normalize_number("1,234.56") == "1234.56"
    assert normalize_number("42%") == "42"
    assert normalize_number("  42  ") == "42"

    # Fractions
    assert normalize_number("1/2") == "0.5"
    assert normalize_number("3/4") == "0.75"

    print("test_normalize_number: PASS")


def test_extract_boxed():
    """Test extracting boxed answers."""
    from nmoe.rl.tasks.math import extract_boxed

    assert extract_boxed("The answer is \\boxed{42}") == "42"
    assert extract_boxed("\\boxed{x^2 + 1}") == "x^2 + 1"
    assert extract_boxed("No boxed answer here") is None
    assert extract_boxed("Multiple \\boxed{1} and \\boxed{2}") == "2"  # Last one

    print("test_extract_boxed: PASS")


def test_humaneval_task():
    """Test HumanEval code task."""
    from nmoe.rl.tasks.code import HumanEvalTask

    task = HumanEvalTask(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n    ",
        test_code="assert add(1, 2) == 3\nassert add(-1, 1) == 0",
        entry_point="add",
        canonical_solution="return a + b",
    )

    # Prompt
    prompt = task.to_prompt()
    assert "def add(a, b)" in prompt

    # Extract code from response
    response = "<think>Simple addition</think><answer>\ndef add(a, b):\n    return a + b\n</answer>"
    code = task.extract_answer(response)
    assert "return a + b" in code

    # Verify (this actually runs the code!)
    assert task.verify("def add(a, b):\n    return a + b") == True
    assert task.verify("def add(a, b):\n    return a - b") == False  # Wrong

    print("test_humaneval_task: PASS")


def test_mbpp_task():
    """Test MBPP code task."""
    from nmoe.rl.tasks.code import MBPPTask

    task = MBPPTask(
        task_id=1,
        description="Write a function to find the maximum of two numbers.",
        test_code="assert max_of_two(1, 2) == 2\nassert max_of_two(5, 3) == 5",
        code="def max_of_two(a, b):\n    return max(a, b)",
    )

    # Verify correct solution
    correct = "def max_of_two(a, b):\n    return a if a > b else b"
    assert task.verify(correct) == True

    # Verify incorrect solution
    incorrect = "def max_of_two(a, b):\n    return a"
    assert task.verify(incorrect) == False

    print("test_mbpp_task: PASS")


# =============================================================================
# 10. INTEGRATION TESTS
# =============================================================================

def test_full_reward_computation_pipeline():
    """Test full reward computation from response to GDPO tensors."""
    import torch
    from nmoe.rl.rewards_gdpo import (
        RewardSignals,
        TrajectoryContext,
        compute_all_rewards,
        batch_rewards_to_tensors,
        reshape_for_gdpo,
    )

    # Simulate batch of 4 responses (2 prompts x 2 samples)
    responses = [
        # Prompt 1, Sample 1: Good format, correct
        """<|start|>assistant<|channel|>analysis<|message|>
Let me solve 2+2 = 4.
<|end|><|start|>assistant<|channel|>final<|message|>
4
<|end|>""",
        # Prompt 1, Sample 2: Bad format
        "The answer is 4",
        # Prompt 2, Sample 1: Good format, wrong answer
        """<|start|>assistant<|channel|>analysis<|message|>
Thinking...
<|end|><|start|>assistant<|channel|>final<|message|>
5
<|end|>""",
        # Prompt 2, Sample 2: Good format, correct
        """<|start|>assistant<|channel|>analysis<|message|>
2+2=4
<|end|><|start|>assistant<|channel|>final<|message|>
4
<|end|>""",
    ]

    # Mock task that expects "4"
    class MockTask:
        def extract_answer(self, text):
            import re
            match = re.search(r"<answer>(.*?)</answer>|<\|channel\|>final<\|message\|>\s*(\d+)", text, re.DOTALL)
            if match:
                return (match.group(1) or match.group(2)).strip()
            # Fallback: last number
            numbers = re.findall(r"\d+", text)
            return numbers[-1] if numbers else None

        def verify(self, answer):
            return answer == "4"

    task = MockTask()

    # Compute rewards for each
    rewards_list = []
    for resp in responses:
        ctx = TrajectoryContext(
            response_text=resp,
            task=task,
            format_type="harmony",
        )
        signals = compute_all_rewards(ctx)
        rewards_list.append(signals)

    # Convert to tensors
    tensors = batch_rewards_to_tensors(rewards_list)

    assert tensors["struct_proper_nesting"].tolist() == [1.0, 0.0, 1.0, 1.0]
    # Note: answer_correct depends on extract_answer working correctly

    # Reshape for GDPO
    reshaped = reshape_for_gdpo(tensors, group_size=2)

    assert reshaped["struct_proper_nesting"].shape == (2, 2)

    print("test_full_reward_computation_pipeline: PASS")


def test_curriculum_with_rewards():
    """Test curriculum weight adjustment with real rewards."""
    import torch
    from nmoe.rl.curriculum import WSDCurriculum, CurriculumStage, StageType

    # Simple curriculum
    stages = [
        CurriculumStage(
            name=StageType.WARMUP,
            examples_count=5,
            weights={"format": 1.0, "accuracy": 0.5},
            transitions_to=StageType.SUSTAIN,
        ),
        CurriculumStage(
            name=StageType.SUSTAIN,
            examples_count=10,
            weights={"format": 0.3, "accuracy": 1.0},
        ),
    ]

    curriculum = WSDCurriculum(stages=stages, adaptive=True)

    # Simulate training with metrics
    for i in range(15):
        weights = curriculum.get_weights()

        # Simulate batch metrics
        batch_metrics = {
            "struct_success": 0.8 + i * 0.01,  # Improving format
            "accuracy": 0.2 + i * 0.02,  # Improving accuracy
        }

        # Step curriculum
        transition = curriculum.step(batch_metrics)
        if transition:
            print(f"  Transitioned to {transition} at step {i}")

        # Adapt weights
        adapted = curriculum.adapt_weights(weights)

    assert curriculum.current_stage == "sustain"
    print("test_curriculum_with_rewards: PASS")


def test_tool_execution_with_rewards():
    """Test tool execution and reward computation together."""
    import asyncio
    from nmoe.rl.tools import ToolCall, ToolType, ToolConfig
    from nmoe.rl.tools.executor import AsyncToolExecutor
    from nmoe.rl.rewards_tools import compute_tool_call_rewards, aggregate_tool_rewards

    async def run_test():
        executor = AsyncToolExecutor(ToolConfig(executor_type="native"))

        # Execute some tools
        calls = [
            ToolCall(type=ToolType.PYTHON, code="print(42)"),
            ToolCall(type=ToolType.PYTHON, code="print(1/0)"),  # Will fail
            ToolCall(type=ToolType.BASH, command="echo hello"),
        ]

        call_ids = executor.scatter(calls)
        results = await executor.gather(call_ids, timeout=10.0)

        # Compute rewards
        signals = []
        for call, result in zip(calls, results):
            sig = compute_tool_call_rewards(call, result)
            signals.append(sig)

        # Aggregate
        aggregated = aggregate_tool_rewards(signals)

        # Should have 2/3 success (first python and bash succeed)
        assert aggregated["tool_syntax_valid"] == 1.0  # All parsed
        assert abs(aggregated["tool_executed"] - 2/3) < 0.1  # 2 of 3 succeeded

        executor.close()

    asyncio.run(run_test())
    print("test_tool_execution_with_rewards: PASS")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Agentic RL Module Tests")
    print("=" * 60)
    print()

    # 1. Curriculum tests
    print("--- 1. Curriculum Tests ---")
    test_stage_type_enum()
    test_curriculum_stage_normalization()
    test_wsd_curriculum_transitions()
    test_curriculum_weight_decay()
    test_thinking_budget_config()
    test_rolling_metrics()
    print()

    # 2. Harmony format tests
    print("--- 2. Harmony Format Tests ---")
    test_harmony_token_constants()
    test_parse_harmony_text_valid()
    test_parse_harmony_text_missing_tags()
    test_harmony_nesting_validation()
    test_harmony_structure_rewards()
    test_harmony_channel_rewards()
    test_compute_harmony_rewards()
    print()

    # 3. R1-Zero format tests
    print("--- 3. R1-Zero Format Tests ---")
    test_r1zero_format_parsing()
    test_r1zero_partial_format()
    test_compute_r1zero_rewards()
    print()

    # 4. Tool types and executor tests
    print("--- 4. Tool Types and Executor Tests ---")
    test_tool_type_enum()
    test_tool_call_dataclass()
    test_tool_result_dataclass()
    test_tool_config()
    test_native_python_execution()
    test_native_bash_execution()
    test_native_file_read()
    test_async_tool_executor_scatter_gather()
    test_async_tool_executor_execute_one()
    print()

    # 5. Tool rewards tests
    print("--- 5. Tool Rewards Tests ---")
    test_tool_reward_signals()
    test_compute_tool_call_rewards()
    test_aggregate_tool_rewards()
    test_compute_output_usage_reward()
    test_check_python_syntax()
    print()

    # 6. GDPO rewards tests
    print("--- 6. GDPO Rewards Tests ---")
    test_reward_signals_dataclass()
    test_batch_rewards_to_tensors()
    test_reshape_for_gdpo()
    test_compute_trajectory_rewards()
    print()

    # 7. Agent turn tests
    print("--- 7. Agent Turn Tests ---")
    test_agent_message_dataclass()
    test_agent_turn_dataclass()
    test_agent_turn_with_tools()
    test_parse_tool_call_from_text()
    print()

    # 8. Task tests
    print("--- 8. Task Tests ---")
    test_task_pool_sampling()
    test_task_pool_access()
    print()

    # 9. Math and code task tests
    print("--- 9. Math and Code Task Tests ---")
    test_gsm8k_task()
    test_math_task()
    test_normalize_number()
    test_extract_boxed()
    test_humaneval_task()
    test_mbpp_task()
    print()

    # 10. Integration tests
    print("--- 10. Integration Tests ---")
    test_full_reward_computation_pipeline()
    test_curriculum_with_rewards()
    test_tool_execution_with_rewards()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
