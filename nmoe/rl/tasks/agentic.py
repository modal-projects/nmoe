"""Agentic tasks for multi-turn RLVR training.

Supports SWEBench-style tasks with tool use, file operations,
and multi-step problem solving.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from nmoe.rl.tasks import Task
from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
from nmoe.rl.tools import ToolCall, ToolResult


@dataclass
class AgenticTask(Task):
    """Base class for multi-turn agentic tasks.

    Unlike single-turn tasks, agentic tasks:
    - Support multiple rounds of model output + tool execution
    - Track conversation history
    - May have intermediate rewards per turn
    - Have complex verification (e.g., code passes tests, file modified correctly)
    """

    task_id: str = ""
    max_turns: int = 10
    task_type: str = field(default="agentic", init=False)

    # Conversation state
    history: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def reset(self) -> None:
        """Reset task state for new episode."""
        self.history = []
        self.tool_calls = []
        self.tool_results = []

    @abstractmethod
    def get_initial_prompt(self) -> str:
        """Get the initial task prompt."""
        pass

    def get_current_prompt(self) -> str:
        """Get prompt including conversation history."""
        if not self.history:
            return self.get_initial_prompt()

        # Build prompt with history
        parts = [self.get_initial_prompt()]
        for entry in self.history:
            if entry["role"] == "assistant":
                parts.append(f"\nAssistant: {entry['content']}")
            elif entry["role"] == "tool_result":
                parts.append(f"\n[Tool Output]: {entry['content'][:500]}")

        return "\n".join(parts)

    def to_prompt(self) -> str:
        """Alias for get_current_prompt for Task interface."""
        return self.get_current_prompt()

    def add_assistant_message(self, content: str) -> None:
        """Add assistant response to history."""
        self.history.append({"role": "assistant", "content": content})

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        """Add tool call and result to history."""
        self.tool_calls.append(call)
        self.tool_results.append(result)
        self.history.append({
            "role": "tool_result",
            "content": result.output if result.success else f"Error: {result.error}",
            "call": call,
            "result": result,
        })

    def is_done(self) -> bool:
        """Check if task should terminate."""
        return len(self.history) >= self.max_turns * 2  # Each turn = response + result

    @abstractmethod
    def compute_reward(self) -> float:
        """Compute final reward for the trajectory."""
        pass

    def extract_answer(self, response: str) -> str | None:
        """Extract final answer from response (for compatibility)."""
        return response

    def verify(self, answer: str | None) -> bool:
        """Verify by computing reward > 0."""
        return self.compute_reward() > 0


@dataclass
class CodeEditTask(AgenticTask):
    """Task requiring code edits to fix bugs or add features.

    Similar to SWEBench: given a repo state and issue description,
    make code changes that pass tests.

    Fields:
        test_command: Primary test command shown to the agent.
        hidden_test_command: Eval-only test command (never shown to agent).
            If set, verification requires BOTH to pass.
        bootstrap: Workspace bootstrap configuration (deps/setup before tests).
    """

    issue_description: str = ""
    repo_path: str = ""
    test_command: str = "python -m pytest"
    hidden_test_command: str = ""  # Eval-only, never shown in prompts
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    files_to_edit: list[str] = field(default_factory=list)
    gold_patch: str = ""  # Reference solution (for analysis, not shown to model)

    task_type: str = field(default="code_edit", init=False)

    def get_initial_prompt(self) -> str:
        files_hint = ""
        if self.files_to_edit:
            files_hint = f"\n\nRelevant files: {', '.join(self.files_to_edit)}"

        return (
            f"You are a software engineer. Fix the following issue:\n\n"
            f"## Issue\n{self.issue_description}\n"
            f"{files_hint}\n\n"
            f"Use tools to read files, make edits, and run tests.\n"
            f"When done, say 'DONE' with a summary of changes."
        )

    def compute_reward(self) -> float:
        """Reward based on test results."""
        # Check if any tool result indicates test success
        for result in self.tool_results:
            if result.success and "passed" in result.output.lower():
                return 1.0
            if result.success and "failed" not in result.output.lower():
                # Heuristic: no failures mentioned
                return 0.5

        return 0.0

    def verify(self, answer: str | None) -> bool:
        """Verify by running the task's test command in the repo workspace.

        This is the correctness oracle for code self-play: the agent "wins" iff
        the tests pass in the current repo state. The textual assistant answer
        is not used for verification.

        If hidden_test_command is set, verification requires BOTH test_command
        AND hidden_test_command to pass (prevents overfitting to revealed tests).
        """
        _ = answer
        from pathlib import Path

        from nmoe.rl.tasks.code_workspace import is_writable_dir, verify_workspace

        repo = Path(self.repo_path)
        if not is_writable_dir(repo):
            return False
        return verify_workspace(
            workspace_path=repo,
            test_command=self.test_command,
            timeout_ms=120_000,
            bootstrap=self.bootstrap if (self.bootstrap.strategy != BootstrapStrategy.NONE or self.bootstrap.commands) else None,
            hidden_test_command=self.hidden_test_command or None,
        )

    def get_metadata(self) -> dict:
        return {
            "task_type": self.task_type,
            "task_id": self.task_id,
            "repo_path": self.repo_path,
            "num_files": len(self.files_to_edit),
        }


@dataclass
class FileSearchTask(AgenticTask):
    """Task requiring searching through files to find information."""

    question: str = ""
    search_path: str = ""
    gold_answer: str = ""
    gold_files: list[str] = field(default_factory=list)  # Files containing answer

    task_type: str = field(default="file_search", init=False)

    def get_initial_prompt(self) -> str:
        return (
            f"Search through the codebase to answer this question:\n\n"
            f"{self.question}\n\n"
            f"Search path: {self.search_path}\n\n"
            f"Use read and search tools to find the answer.\n"
            f"Respond with your answer in <answer>...</answer> tags."
        )

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from tags."""
        import re
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def compute_reward(self) -> float:
        """Reward based on answer correctness."""
        # Get last assistant response
        for entry in reversed(self.history):
            if entry["role"] == "assistant":
                answer = self.extract_answer(entry["content"])
                if answer and self.gold_answer.lower() in answer.lower():
                    return 1.0
                break
        return 0.0

    def verify(self, answer: str | None) -> bool:
        if answer is None:
            return False
        return self.gold_answer.lower() in answer.lower()


@dataclass
class MultiStepMathTask(AgenticTask):
    """Math task that may require calculator/code tools."""

    problem: str = ""
    gold_answer: str = ""
    allow_code: bool = True

    task_type: str = field(default="multi_step_math", init=False)

    def get_initial_prompt(self) -> str:
        tool_hint = ""
        if self.allow_code:
            tool_hint = "\n\nYou may use Python to compute intermediate results."

        return (
            f"Solve this problem step by step:\n\n"
            f"{self.problem}\n"
            f"{tool_hint}\n\n"
            f"Put your final answer in \\boxed{{}}."
        )

    def extract_answer(self, response: str) -> str | None:
        """Extract boxed answer."""
        import re
        # Find last \boxed{...}
        matches = list(re.finditer(r"\\boxed\{([^}]*)\}", response))
        if matches:
            return matches[-1].group(1).strip()
        return None

    def compute_reward(self) -> float:
        """Reward based on final answer."""
        for entry in reversed(self.history):
            if entry["role"] == "assistant":
                answer = self.extract_answer(entry["content"])
                if answer is not None:
                    # Normalize and compare
                    try:
                        from nmoe.rl.tasks.math import verify_sympy, normalize_number

                        # Try sympy first
                        if verify_sympy(self.gold_answer, answer):
                            return 1.0

                        # Fall back to numeric
                        norm_ans = normalize_number(answer)
                        norm_gold = normalize_number(self.gold_answer)
                        if norm_ans == norm_gold:
                            return 1.0
                    except Exception:
                        pass

                    # String comparison fallback
                    if answer.strip() == self.gold_answer.strip():
                        return 1.0
                break
        return 0.0

    def verify(self, answer: str | None) -> bool:
        if answer is None:
            return False
        try:
            from nmoe.rl.tasks.math import verify_sympy
            return verify_sympy(self.gold_answer, answer)
        except Exception:
            return answer.strip() == self.gold_answer.strip()


# =============================================================================
# Task Pool for Agentic Tasks
# =============================================================================

@dataclass
class AgenticTaskPool:
    """Pool of agentic tasks with sampling support."""

    tasks: list[AgenticTask] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def add(self, task: AgenticTask, weight: float = 1.0) -> None:
        """Add task to pool."""
        self.tasks.append(task)
        task_type = task.task_type
        self.weights[task_type] = self.weights.get(task_type, 0) + weight

    def sample(self, n: int = 1) -> list[AgenticTask]:
        """Sample tasks from pool."""
        import random

        if not self.tasks:
            return []

        # Weight by task type
        type_weights = []
        for task in self.tasks:
            type_weights.append(self.weights.get(task.task_type, 1.0))

        total = sum(type_weights)
        probs = [w / total for w in type_weights]

        indices = random.choices(range(len(self.tasks)), weights=probs, k=n)
        return [self.tasks[i] for i in indices]

    def __len__(self) -> int:
        return len(self.tasks)
