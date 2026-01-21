"""Code tasks for RLVR training.

Supports HumanEval and MBPP with executable test verification.
Includes Pass@k estimation for code generation evaluation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterator

from nmoe.rl.tasks import Task
from nmoe.rl.rewards_harmony import CHANNELS, harmony_message, parse_harmony_text
from nmoe.rl.tools.codex import CodexConfig, CodexExecutor


# =============================================================================
# Pass@k Estimation (from slime/Codex)
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k using combinatorial formula.

    Computes the probability that at least one of k samples passes,
    given n total samples with c correct.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    This is the unbiased estimator from the Codex paper.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to consider (k <= n)

    Returns:
        Estimated pass@k probability in [0, 1]
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # Use log to avoid overflow: log(C(n,k)) = sum(log(n-i) - log(k-i))
    # pass@k = 1 - C(n-c, k) / C(n, k)
    #        = 1 - prod((n-c-i)/(n-i) for i in range(k))

    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def compute_pass_at_k(
    results: list[bool],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute pass@k for multiple k values from sample results.

    Args:
        results: List of boolean results (True = pass, False = fail)
        k_values: List of k values to compute (default: [1, 2, 4, 8])

    Returns:
        Dict mapping "pass@k" to probability
    """
    if k_values is None:
        k_values = [1, 2, 4, 8]

    n = len(results)
    c = sum(results)

    output = {}
    for k in k_values:
        if k <= n:
            output[f"pass@{k}"] = pass_at_k(n, c, k)
        else:
            output[f"pass@{k}"] = 0.0

    return output


def aggregate_pass_at_k(
    all_results: list[list[bool]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Aggregate pass@k across multiple problems.

    Args:
        all_results: List of result lists, one per problem
        k_values: List of k values to compute

    Returns:
        Dict mapping "pass@k" to mean probability across problems
    """
    if not all_results:
        return {}

    if k_values is None:
        k_values = [1, 2, 4, 8]

    # Compute per-problem pass@k
    per_problem = [compute_pass_at_k(results, k_values) for results in all_results]

    # Average across problems
    output = {}
    for k in k_values:
        key = f"pass@{k}"
        values = [p.get(key, 0.0) for p in per_problem if key in p]
        if values:
            output[key] = sum(values) / len(values)

    return output


def estimate_pass_at_k_from_rewards(
    rewards: list[float],
    threshold: float = 0.5,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Estimate pass@k from reward values.

    Useful when rewards are continuous (0-1) rather than binary.

    Args:
        rewards: List of reward values
        threshold: Threshold to consider as "pass"
        k_values: List of k values to compute

    Returns:
        Dict mapping "pass@k" to probability
    """
    results = [r >= threshold for r in rewards]
    return compute_pass_at_k(results, k_values)


def extract_code_block(text: str, language: str = "python") -> str | None:
    """Extract code from markdown code block or <answer> tags.

    Args:
        text: Text containing code
        language: Expected language (for markdown blocks)

    Returns:
        Extracted code, or None if not found
    """
    # Try <answer> tags first
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        text = answer_match.group(1)

    # Try markdown code block
    pattern = rf"```{language}\s*\n(.*?)\n```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code block
    pattern = r"```\s*\n(.*?)\n```"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find function definition directly
    if "def " in text:
        # Find the code starting from first def
        lines = text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("def "):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines).strip()

    return None


def run_python_tests(
    code: str,
    test_code: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Execute Python code with tests.

    Uses CodexExecutor for sandboxed execution (Landlock + Seccomp).

    Args:
        code: The code to test (function definitions)
        test_code: Test code to run (should raise on failure)
        timeout: Execution timeout in seconds

    Returns:
        Tuple of (success, output/error message)
    """
    try:
        config = CodexConfig(timeout_ms=int(timeout * 1000))
        executor = CodexExecutor(config)
        result = executor.exec_tests(code, test_code)

        if result.success:
            return True, result.stdout or ""
        elif result.timed_out:
            return False, "Execution timed out"
        else:
            return False, result.stderr or result.error or ""
    except Exception as e:
        return False, str(e)


def check_python_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code has valid syntax.

    Args:
        code: Python code to check

    Returns:
        Tuple of (valid, error message if invalid)
    """
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)


@dataclass
class HumanEvalTask(Task):
    """HumanEval coding task.

    Each task has:
    - A prompt with function signature and docstring
    - Test cases to verify the implementation
    - An entry point (function name)
    """

    task_id: str
    prompt: str  # Function signature + docstring
    test_code: str  # Test cases (assert statements)
    entry_point: str  # Function name
    canonical_solution: str = ""  # Reference solution (not shown to model)

    task_type: str = field(default="humaneval", init=False)

    def to_prompt(self) -> str:
        user = (
            "Complete the following Python function.\n\n"
            f"{self.prompt}\n\n"
            "Return ONLY the code in the final channel (either raw code or a ```python fenced block).\n"
            "Use the analysis channel for any reasoning.\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        final = parsed.final_content
        if not final.strip():
            return None
        return extract_code_block(final, "python") or final.strip()

    def verify(self, answer: str | None) -> bool:
        """Verify by running tests."""
        if answer is None:
            return False

        # Check syntax first
        syntax_ok, _ = check_python_syntax(answer)
        if not syntax_ok:
            return False

        # Run tests
        success, _ = run_python_tests(answer, self.test_code, timeout=10.0)
        return success

    def get_metadata(self) -> dict:
        return {
            "task_type": self.task_type,
            "task_id": self.task_id,
            "entry_point": self.entry_point,
        }


@dataclass
class MBPPTask(Task):
    """MBPP (Mostly Basic Python Problems) task.

    Similar to HumanEval but with natural language descriptions.
    """

    task_id: int
    description: str  # Natural language description
    test_code: str  # Test cases
    code: str = ""  # Reference solution

    task_type: str = field(default="mbpp", init=False)

    def to_prompt(self) -> str:
        user = (
            "Write a Python function to solve the following problem:\n\n"
            f"{self.description}\n\n"
            "Return ONLY the code in the final channel (either raw code or a ```python fenced block).\n"
            "Use the analysis channel for any reasoning.\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        final = parsed.final_content
        if not final.strip():
            return None
        return extract_code_block(final, "python") or final.strip()

    def verify(self, answer: str | None) -> bool:
        if answer is None:
            return False

        syntax_ok, _ = check_python_syntax(answer)
        if not syntax_ok:
            return False

        success, _ = run_python_tests(answer, self.test_code, timeout=10.0)
        return success


# =============================================================================
# Reward Signals for Code Tasks
# =============================================================================

@dataclass
class CodeRewardSignals:
    """Detailed reward signals for code tasks (GDPO multi-reward)."""

    syntax_valid: float = 0.0  # Code parses (syntax check)
    compiles: float = 0.0  # Code compiles (no import errors on load)
    runs: float = 0.0  # Code executes without runtime error
    tests_pass: float = 0.0  # All tests pass
    partial_tests: float = 0.0  # Fraction of tests that pass (0-1)


def compute_code_rewards(
    code: str | None,
    test_code: str,
    timeout: float = 10.0,
) -> CodeRewardSignals:
    """Compute detailed code reward signals.

    Args:
        code: Generated code
        test_code: Test code to run
        timeout: Execution timeout

    Returns:
        CodeRewardSignals with binary/fractional rewards
    """
    signals = CodeRewardSignals()

    if code is None:
        return signals

    # Syntax check
    syntax_ok, _ = check_python_syntax(code)
    signals.syntax_valid = 1.0 if syntax_ok else 0.0

    if not syntax_ok:
        return signals

    # Try to compile (check imports, etc.)
    try:
        compile(code, "<string>", "exec")
        signals.compiles = 1.0
    except Exception:
        return signals

    # Try to run (executes without error)
    # Run just the code without tests first using sandboxed executor
    try:
        config = CodexConfig(timeout_ms=int(timeout * 500))  # Half timeout for run-only
        executor = CodexExecutor(config)
        result = executor.exec_python(code)
        signals.runs = 1.0 if result.success else 0.0
    except Exception:
        signals.runs = 0.0

    if signals.runs == 0.0:
        return signals

    # Run with tests
    success, _ = run_python_tests(code, test_code, timeout=timeout)
    signals.tests_pass = 1.0 if success else 0.0
    signals.partial_tests = signals.tests_pass  # Could be extended for partial

    return signals


# =============================================================================
# Data Loading
# =============================================================================

def load_humaneval_tasks(
    max_examples: int = 164,  # HumanEval has 164 problems
    source: str = "openai/openai_humaneval",
) -> list[HumanEvalTask]:
    """Load HumanEval tasks from HuggingFace.

    Args:
        max_examples: Maximum number of examples
        source: HuggingFace dataset name

    Returns:
        List of HumanEvalTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    ds = load_dataset(source, split="test")

    tasks = []
    for ex in ds:
        if len(tasks) >= max_examples:
            break

        task_id = ex.get("task_id", "")
        prompt = ex.get("prompt", "")
        test = ex.get("test", "")
        entry_point = ex.get("entry_point", "")
        canonical = ex.get("canonical_solution", "")

        if not prompt or not test or not entry_point:
            continue

        tasks.append(HumanEvalTask(
            task_id=task_id,
            prompt=prompt,
            test_code=test,
            entry_point=entry_point,
            canonical_solution=canonical,
        ))

    return tasks


def load_mbpp_tasks(
    max_examples: int = 1000,
    split: str = "test",
    source: str = "google-research-datasets/mbpp",
) -> list[MBPPTask]:
    """Load MBPP tasks from HuggingFace.

    Args:
        max_examples: Maximum number of examples
        split: Dataset split
        source: HuggingFace dataset name

    Returns:
        List of MBPPTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    ds = load_dataset(source, split=split)

    tasks = []
    for ex in ds:
        if len(tasks) >= max_examples:
            break

        task_id = ex.get("task_id", 0)
        description = ex.get("text", "")
        test_list = ex.get("test_list", [])
        code = ex.get("code", "")

        if not description or not test_list:
            continue

        # Convert test list to test code
        test_code = "\n".join(test_list)

        tasks.append(MBPPTask(
            task_id=task_id,
            description=description,
            test_code=test_code,
            code=code,
        ))

    return tasks


def iter_humaneval(
    source: str = "openai/openai_humaneval",
) -> Iterator[HumanEvalTask]:
    """Iterator for HumanEval tasks.

    Args:
        source: HuggingFace dataset name

    Yields:
        HumanEvalTask instances
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    ds = load_dataset(source, split="test")

    for ex in ds:
        task_id = ex.get("task_id", "")
        prompt = ex.get("prompt", "")
        test = ex.get("test", "")
        entry_point = ex.get("entry_point", "")
        canonical = ex.get("canonical_solution", "")

        if not prompt or not test or not entry_point:
            continue

        yield HumanEvalTask(
            task_id=task_id,
            prompt=prompt,
            test_code=test,
            entry_point=entry_point,
            canonical_solution=canonical,
        )
