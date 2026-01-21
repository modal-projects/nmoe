"""Tests for nmoe.rl.tasks modules."""
import pytest
import torch


class TestSympyVerification:
    """Test sympy-based math verification."""

    def test_basic_equality(self):
        """Should match identical values."""
        from nmoe.rl.tasks.math import verify_sympy

        assert verify_sympy("4", "4")
        assert verify_sympy("42", "42")
        assert verify_sympy("-5", "-5")

    def test_fraction_equivalence(self):
        """Should match equivalent fractions."""
        from nmoe.rl.tasks.math import verify_sympy, are_equal_sympy

        assert verify_sympy("1/2", "0.5")
        assert are_equal_sympy("2/4", "1/2")
        assert are_equal_sympy("3/6", "0.5")

    def test_algebraic_equivalence(self):
        """Should match algebraically equivalent expressions."""
        from nmoe.rl.tasks.math import are_equal_sympy

        assert are_equal_sympy("x^2 + 2*x + 1", "(x+1)^2")
        assert are_equal_sympy("(a+b)^2", "a^2 + 2*a*b + b^2")

    def test_non_equivalent_rejected(self):
        """Should reject non-equivalent expressions."""
        from nmoe.rl.tasks.math import verify_sympy, are_equal_sympy

        assert not verify_sympy("4", "5")
        assert not are_equal_sympy("x", "x+1")
        assert not are_equal_sympy("2", "3")

    def test_handles_none(self):
        """Should handle None inputs gracefully."""
        from nmoe.rl.tasks.math import verify_sympy

        assert not verify_sympy(None, "4")
        assert not verify_sympy("4", None)
        assert not verify_sympy(None, None)


class TestPassAtK:
    """Test Pass@k estimation."""

    def test_all_pass(self):
        """All passing should give 1.0."""
        from nmoe.rl.tasks.code import pass_at_k

        assert pass_at_k(n=10, c=10, k=1) == 1.0
        assert pass_at_k(n=10, c=10, k=5) == 1.0

    def test_none_pass(self):
        """None passing should give 0.0."""
        from nmoe.rl.tasks.code import pass_at_k

        assert pass_at_k(n=10, c=0, k=1) == 0.0
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_half_pass_at_1(self):
        """Half passing should give 0.5 for pass@1."""
        from nmoe.rl.tasks.code import pass_at_k

        result = pass_at_k(n=10, c=5, k=1)
        assert abs(result - 0.5) < 0.01

    def test_compute_pass_at_k(self):
        """Test batch computation."""
        from nmoe.rl.tasks.code import compute_pass_at_k

        results = [True, True, True, False, False]  # 3/5 pass
        metrics = compute_pass_at_k(results, k_values=[1, 5])

        assert abs(metrics["pass@1"] - 0.6) < 0.01
        assert metrics["pass@5"] == 1.0  # c >= k

    def test_aggregate_pass_at_k(self):
        """Test aggregation across problems."""
        from nmoe.rl.tasks.code import aggregate_pass_at_k

        all_results = [
            [True, False, False],   # 1/3 pass
            [True, True, False],    # 2/3 pass
            [False, False, False],  # 0/3 pass
        ]
        agg = aggregate_pass_at_k(all_results, k_values=[1])

        # Mean of individual pass@1 values
        assert 0.0 < agg["pass@1"] < 1.0


class TestAgenticTasks:
    """Test agentic task classes."""

    def test_code_edit_task_creation(self):
        """CodeEditTask should initialize correctly."""
        from nmoe.rl.tasks.agentic import CodeEditTask

        task = CodeEditTask(
            task_id="test-1",
            issue_description="Fix the bug",
            repo_path="/tmp/repo",
            files_to_edit=["foo.py"],
        )

        assert task.task_id == "test-1"
        assert task.task_type == "code_edit"
        assert len(task.history) == 0

    def test_conversation_flow(self):
        """Should track conversation history."""
        from nmoe.rl.tasks.agentic import CodeEditTask
        from nmoe.rl.tools import ToolCall, ToolResult, ToolType

        task = CodeEditTask(
            task_id="test",
            issue_description="Fix bug",
            repo_path="/tmp",
        )

        task.add_assistant_message("Let me check.")
        task.add_tool_result(
            ToolCall(type=ToolType.BASH, call_id="1", command="cat foo.py"),
            ToolResult(call_id="1", success=True, output="code here")
        )

        assert len(task.history) == 2
        assert task.history[0]["role"] == "assistant"
        assert task.history[1]["role"] == "tool_result"

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        from nmoe.rl.tasks.agentic import CodeEditTask

        task = CodeEditTask(task_id="t", issue_description="x", repo_path="/")
        task.add_assistant_message("test")

        assert len(task.history) == 1
        task.reset()
        assert len(task.history) == 0

    def test_file_search_answer_extraction(self):
        """Should extract answer from tags."""
        from nmoe.rl.tasks.agentic import FileSearchTask

        task = FileSearchTask(
            question="Where is config?",
            search_path="/",
            gold_answer="config.py",
        )

        answer = task.extract_answer("Found it: <answer>config.py</answer>")
        assert answer == "config.py"

        # No tags
        answer = task.extract_answer("The file is config.py")
        assert answer is None

    def test_file_search_reward(self):
        """Should reward correct answer."""
        from nmoe.rl.tasks.agentic import FileSearchTask

        task = FileSearchTask(
            question="Where?",
            search_path="/",
            gold_answer="target.py",
        )

        # Correct answer
        task.history.append({"role": "assistant", "content": "<answer>target.py</answer>"})
        assert task.compute_reward() == 1.0

        # Wrong answer
        task.reset()
        task.history.append({"role": "assistant", "content": "<answer>wrong.py</answer>"})
        assert task.compute_reward() == 0.0

    def test_multi_step_math_extraction(self):
        """Should extract boxed answer."""
        from nmoe.rl.tasks.agentic import MultiStepMathTask

        task = MultiStepMathTask(problem="2+2=?", gold_answer="4")

        answer = task.extract_answer(r"Therefore \boxed{4}")
        assert answer == "4"

        # Multiple boxed - take last
        answer = task.extract_answer(r"First \boxed{2}, then \boxed{4}")
        assert answer == "4"


class TestTaskPool:
    def test_weighted_sampling_by_task_type(self):
        from nmoe.rl.tasks import Task, TaskPool

        class _T(Task):
            def __init__(self, t: str):
                super().__init__()
                self.task_type = t

            def to_prompt(self) -> str:
                return "x"

            def extract_answer(self, response: str) -> str | None:
                return response

            def verify(self, answer: str | None) -> bool:
                return True

        tasks = [_T("a"), _T("b")]
        pool = TaskPool(tasks, weights={"a": 1.0, "b": 0.0}, seed=0)
        sample = pool.sample(50, replace=True)
        assert all(t.task_type == "a" for t in sample)
