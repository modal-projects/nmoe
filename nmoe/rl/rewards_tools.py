"""Tool-specific rewards for GDPO multi-reward training.

Provides binary verifiable signals for tool execution quality:
- Syntax validity (does it parse?)
- Execution success (did it run?)
- Output quality (was it useful?)
- Correctness (was it right?)

All rewards are binary (0.0 or 1.0) for RLVR compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nmoe.rl.tools import ToolCall, ToolResult, ToolType
from nmoe.rl.tools.codex import CodexConfig, CodexExecutor


@dataclass
class ToolRewardSignals:
    """Detailed reward signals for a single tool call."""

    # Syntax level
    syntax_valid: float = 0.0  # Tool call is well-formed

    # Execution level
    executed: float = 0.0  # Tool ran (didn't crash/timeout)
    no_error: float = 0.0  # No stderr/exception
    no_timeout: float = 0.0  # Completed within timeout

    # Output level
    has_output: float = 0.0  # Non-empty output
    output_used: float = 0.0  # Output referenced in subsequent reasoning

    # Code-specific (Python)
    python_compiles: float = 0.0  # Syntax check passed
    python_runs: float = 0.0  # Executed without runtime error
    python_returns: float = 0.0  # Returned expected type
    python_correct: float = 0.0  # Output matches expected

    # Code-specific (Bash)
    bash_valid: float = 0.0  # Valid command syntax
    bash_succeeds: float = 0.0  # Exit code 0


def compute_tool_call_rewards(
    call: ToolCall,
    result: ToolResult,
) -> ToolRewardSignals:
    """Compute reward signals for a single tool call.

    Args:
        call: The tool call that was made
        result: The execution result

    Returns:
        ToolRewardSignals with binary values
    """
    signals = ToolRewardSignals()

    # Syntax: the call was well-formed (we have a result)
    signals.syntax_valid = 1.0

    # Execution
    signals.executed = 1.0 if result.success or result.exit_code == 0 else 0.0
    signals.no_error = 1.0 if result.error is None else 0.0
    signals.no_timeout = 1.0 if not result.timed_out else 0.0

    # Output
    signals.has_output = 1.0 if result.output and len(result.output.strip()) > 0 else 0.0

    # Tool-type specific
    tool_type = call.type if isinstance(call.type, str) else call.type.value

    if tool_type == "python":
        signals.python_compiles = 1.0 if result.compiled else 0.0
        signals.python_runs = 1.0 if result.executed else 0.0
        # python_returns and python_correct need task context
        signals.python_returns = signals.python_runs
        signals.python_correct = 1.0 if result.success else 0.0

    elif tool_type == "bash":
        signals.bash_valid = 1.0  # If we got here, it parsed
        signals.bash_succeeds = 1.0 if result.exit_code == 0 else 0.0

    return signals


def aggregate_tool_rewards(
    signals_list: list[ToolRewardSignals],
) -> dict[str, float]:
    """Aggregate rewards across multiple tool calls.

    Takes the mean of each signal across all tools.

    Args:
        signals_list: List of ToolRewardSignals from each call

    Returns:
        Dict of aggregated reward signals
    """
    if not signals_list:
        return {
            "tool_syntax_valid": 0.0,
            "tool_executed": 0.0,
            "tool_no_error": 0.0,
            "tool_no_timeout": 0.0,
            "tool_has_output": 0.0,
            "python_compiles": 0.0,
            "python_runs": 0.0,
            "python_correct": 0.0,
            "bash_valid": 0.0,
            "bash_succeeds": 0.0,
        }

    n = len(signals_list)

    return {
        "tool_syntax_valid": sum(s.syntax_valid for s in signals_list) / n,
        "tool_executed": sum(s.executed for s in signals_list) / n,
        "tool_no_error": sum(s.no_error for s in signals_list) / n,
        "tool_no_timeout": sum(s.no_timeout for s in signals_list) / n,
        "tool_has_output": sum(s.has_output for s in signals_list) / n,
        "python_compiles": sum(s.python_compiles for s in signals_list) / n,
        "python_runs": sum(s.python_runs for s in signals_list) / n,
        "python_correct": sum(s.python_correct for s in signals_list) / n,
        "bash_valid": sum(s.bash_valid for s in signals_list) / n,
        "bash_succeeds": sum(s.bash_succeeds for s in signals_list) / n,
    }


def compute_output_usage_reward(
    result: ToolResult,
    subsequent_text: str,
    min_overlap_chars: int = 20,
) -> float:
    """Check if tool output was used in subsequent reasoning.

    Args:
        result: Tool execution result
        subsequent_text: Text generated after the tool call
        min_overlap_chars: Minimum characters to consider "used"

    Returns:
        1.0 if output was used, 0.0 otherwise
    """
    if not result.output or not subsequent_text:
        return 0.0

    output = result.output.strip()
    if len(output) < min_overlap_chars:
        # Short output - check if entire output appears
        return 1.0 if output in subsequent_text else 0.0

    # Check if substantial portion of output appears in subsequent text
    # Use sliding window to find overlap
    window_size = min(min_overlap_chars, len(output))

    for i in range(len(output) - window_size + 1):
        window = output[i:i + window_size]
        if window in subsequent_text:
            return 1.0

    return 0.0


# =============================================================================
# High-Level Reward Computation
# =============================================================================

@dataclass
class ToolCallSite:
    """A tool call with its position and result."""
    call: ToolCall
    result: ToolResult
    position: int = 0  # Token position in generation
    subsequent_text: str = ""  # Text generated after this call


def compute_all_tool_rewards(
    tool_sites: list[ToolCallSite],
) -> dict[str, float]:
    """Compute all tool reward signals for a trajectory.

    Args:
        tool_sites: List of tool call sites with results

    Returns:
        Dict of all tool reward signals
    """
    if not tool_sites:
        # No tool calls - return zeros
        return {
            "has_tool_use": 0.0,
            "tool_syntax_valid": 0.0,
            "tool_executed": 0.0,
            "tool_no_error": 0.0,
            "tool_no_timeout": 0.0,
            "tool_has_output": 0.0,
            "tool_output_used": 0.0,
            "python_compiles": 0.0,
            "python_runs": 0.0,
            "python_correct": 0.0,
            "bash_valid": 0.0,
            "bash_succeeds": 0.0,
        }

    # Compute per-call rewards
    signals_list: list[ToolRewardSignals] = []
    output_usage: list[float] = []
    python_signals: list[ToolRewardSignals] = []
    bash_signals: list[ToolRewardSignals] = []

    for site in tool_sites:
        signals = compute_tool_call_rewards(site.call, site.result)
        signals_list.append(signals)

        tool_type = site.call.type if isinstance(site.call.type, str) else site.call.type.value
        if tool_type == "python":
            python_signals.append(signals)
        elif tool_type == "bash":
            bash_signals.append(signals)

        # Check output usage
        usage = compute_output_usage_reward(
            site.result,
            site.subsequent_text,
        )
        output_usage.append(usage)

    # Aggregate
    rewards = aggregate_tool_rewards(signals_list)
    rewards["has_tool_use"] = 1.0
    rewards["tool_output_used"] = sum(output_usage) / len(output_usage)

    # Tool-type specific aggregation: RL_DESIGN.md defines these as means over
    # calls of that specific tool type (not diluted by other tools in the turn).
    if python_signals:
        n = len(python_signals)
        rewards["python_compiles"] = sum(s.python_compiles for s in python_signals) / n
        rewards["python_runs"] = sum(s.python_runs for s in python_signals) / n
        rewards["python_correct"] = sum(s.python_correct for s in python_signals) / n
    else:
        rewards["python_compiles"] = 0.0
        rewards["python_runs"] = 0.0
        rewards["python_correct"] = 0.0

    if bash_signals:
        n = len(bash_signals)
        rewards["bash_valid"] = sum(s.bash_valid for s in bash_signals) / n
        rewards["bash_succeeds"] = sum(s.bash_succeeds for s in bash_signals) / n
    else:
        rewards["bash_valid"] = 0.0
        rewards["bash_succeeds"] = 0.0

    return rewards


# =============================================================================
# Python Code Verification
# =============================================================================

def check_python_syntax(code: str) -> bool:
    """Check if Python code has valid syntax.

    Args:
        code: Python code to check

    Returns:
        True if syntax is valid
    """
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def verify_python_with_tests(
    code: str,
    test_code: str,
    timeout: float = 10.0,
) -> tuple[bool, dict[str, float]]:
    """Verify Python code by running tests.

    Uses CodexExecutor for sandboxed execution (Landlock + Seccomp).

    Args:
        code: Generated code
        test_code: Test code (assertions)
        timeout: Execution timeout

    Returns:
        Tuple of (all_tests_pass, detailed_rewards)
    """
    rewards = {
        "python_compiles": 0.0,
        "python_runs": 0.0,
        "python_correct": 0.0,
    }

    # Check syntax
    if not check_python_syntax(code):
        return False, rewards
    rewards["python_compiles"] = 1.0

    try:
        config = CodexConfig(timeout_ms=int(timeout * 1000))
        executor = CodexExecutor(config)
        result = executor.exec_tests(code, test_code)

        rewards["python_runs"] = 1.0
        if result.success:
            rewards["python_correct"] = 1.0
            return True, rewards
        else:
            return False, rewards

    except Exception:
        return False, rewards


# =============================================================================
# Bash Command Verification
# =============================================================================

def check_bash_syntax(command: str) -> bool:
    """Check if bash command has valid syntax.

    Uses CodexExecutor for sandboxed execution.

    Args:
        command: Bash command to check

    Returns:
        True if syntax is valid
    """
    try:
        config = CodexConfig(timeout_ms=5000)
        executor = CodexExecutor(config)
        # Use bash -n to check syntax without executing
        result = executor.exec_bash(f"bash -n -c {repr(command)}")
        return result.success
    except Exception:
        return False


def is_safe_bash_command(command: str) -> bool:
    """Check if bash command is likely safe (heuristic).

    Args:
        command: Bash command to check

    Returns:
        True if command appears safe
    """
    # Dangerous patterns
    dangerous = [
        "rm -rf /",
        "rm -rf ~",
        "> /dev/sda",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",  # Fork bomb
        "wget http",  # Downloads
        "curl http",
    ]

    command_lower = command.lower()
    for pattern in dangerous:
        if pattern in command_lower:
            return False

    return True
