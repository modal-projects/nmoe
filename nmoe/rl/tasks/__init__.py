"""Task definitions for RLVR training.

Provides abstract Task interface and TaskPool for sampling tasks
during training. Tasks encapsulate:
- Prompt generation
- Answer extraction
- Verification (binary correctness)
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence


@dataclass
class Task(ABC):
    """Abstract base class for RLVR tasks.

    A task represents a single problem that the model should solve.
    Tasks are responsible for:
    1. Generating prompts
    2. Extracting answers from model output
    3. Verifying correctness (binary)
    """

    task_type: str = field(default="", init=False)

    @abstractmethod
    def to_prompt(self) -> str:
        """Generate the prompt for this task.

        Returns:
            Prompt string to send to the model
        """
        ...

    @abstractmethod
    def extract_answer(self, response: str) -> str | None:
        """Extract the answer from model response.

        Args:
            response: Model's generated response

        Returns:
            Extracted answer string, or None if extraction failed
        """
        ...

    @abstractmethod
    def verify(self, answer: str | None) -> bool:
        """Verify if the extracted answer is correct.

        Args:
            answer: Extracted answer (may be None)

        Returns:
            True if correct, False otherwise
        """
        ...

    @property
    def has_ground_truth(self) -> bool:
        """Whether this task has a verifiable ground truth."""
        return True

    def to_messages(self) -> list[dict[str, str]]:
        """Convert task to chat message format.

        Returns:
            List of message dicts with 'role' and 'content'
        """
        return [{"role": "user", "content": self.to_prompt()}]

    def get_metadata(self) -> dict[str, Any]:
        """Get task metadata for logging/analysis.

        Returns:
            Dict of metadata (task_type, difficulty, etc.)
        """
        return {"task_type": self.task_type}


class TaskPool:
    """Pool of tasks for sampling during training.

    Supports:
    - Sampling with/without replacement
    - Weighted sampling across task types
    - Shuffling and iteration
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        weights: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        """Initialize task pool.

        Args:
            tasks: Sequence of tasks
            weights: Optional weights by task_type (default: uniform)
            seed: Random seed for reproducibility
        """
        self.tasks = list(tasks)
        self.weights = weights or {}
        self.rng = random.Random(seed)

        # Index tasks by type for weighted sampling
        self._by_type: dict[str, list[Task]] = {}
        for task in self.tasks:
            if task.task_type not in self._by_type:
                self._by_type[task.task_type] = []
            self._by_type[task.task_type].append(task)

    def __len__(self) -> int:
        return len(self.tasks)

    def sample(self, n: int, replace: bool = True, *, seed: int | None = None) -> list[Task]:
        """Sample n tasks from the pool.

        Args:
            n: Number of tasks to sample
            replace: Sample with replacement (default: True)
            seed: Optional deterministic seed for this call (does not advance pool RNG)

        Returns:
            List of sampled tasks
        """
        if not self.tasks:
            return []

        rng = self.rng if seed is None else random.Random(int(seed))

        if not self.weights:
            # Uniform sampling
            if replace:
                return rng.choices(self.tasks, k=n)
            else:
                return rng.sample(self.tasks, min(n, len(self.tasks)))

        # Weighted sampling by task type
        types = list(self._by_type.keys())
        type_weights = [self.weights.get(t, 1.0) for t in types]

        sampled = []
        for _ in range(n):
            # Sample task type
            task_type = rng.choices(types, weights=type_weights, k=1)[0]
            # Sample task from that type
            task = rng.choice(self._by_type[task_type])
            sampled.append(task)

        return sampled

    def shuffle(self) -> None:
        """Shuffle tasks in-place."""
        self.rng.shuffle(self.tasks)

    def iter_epochs(self, batch_size: int, epochs: int = 1) -> Iterator[list[Task]]:
        """Iterate through tasks in epochs.

        Args:
            batch_size: Number of tasks per batch
            epochs: Number of epochs

        Yields:
            Batches of tasks
        """
        for _ in range(epochs):
            self.shuffle()
            for i in range(0, len(self.tasks), batch_size):
                yield self.tasks[i:i + batch_size]

    @classmethod
    def from_config(cls, config: dict) -> "TaskPool":
        """Create TaskPool from configuration dict.

        Config format:
            {
                "sources": ["gsm8k", "humaneval"],
                "weights": {"gsm8k": 1.0, "humaneval": 0.5},
                "max_examples": 10000,
                "seed": 42,
            }

        Args:
            config: Configuration dict

        Returns:
            Configured TaskPool
        """
        from nmoe.rl.tasks.math import load_gsm8k_tasks
        from nmoe.rl.tasks.code import load_humaneval_tasks

        sources = config.get("sources", ["gsm8k"])
        weights = config.get("weights", {})
        max_examples = config.get("max_examples", 10000)
        seed = config.get("seed", None)

        tasks = []
        for source in sources:
            if source == "gsm8k":
                tasks.extend(load_gsm8k_tasks(max_examples=max_examples))
            elif source == "humaneval":
                tasks.extend(load_humaneval_tasks(max_examples=max_examples))
            # Add more sources as needed

        return cls(tasks, weights=weights, seed=seed)

    @classmethod
    def from_hf_dataset(
        cls,
        *,
        dataset: str,
        split: str = "train",
        subset: str | None = None,
        data_files: str | list[str] | None = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        task_type: str = "math",
        problem_field: str = "problem",
        answer_field: str = "answer",
        gold_extractor: str = "raw",
        max_examples: int = 10_000,
        seed: int | None = None,
    ) -> "TaskPool":
        """Create a TaskPool from a HuggingFace dataset split.

        This is a small glue layer that maps dataset rows into existing Task
        types while keeping Harmony as the only output contract.
        """
        from nmoe.rl.tasks.hf_tasks import HFDatasetTaskSpec, load_tasks_from_hf

        spec = HFDatasetTaskSpec(
            dataset=dataset,
            split=split,
            subset=subset,
            data_files=data_files,
            streaming=streaming,
            trust_remote_code=trust_remote_code,
            task_type=task_type,
            problem_field=problem_field,
            answer_field=answer_field,
            gold_extractor=gold_extractor,
            max_examples=max_examples,
        )
        tasks = load_tasks_from_hf(spec)
        return cls(tasks, seed=seed)


# Re-export task types
from nmoe.rl.tasks.math import GSM8KTask, MATHTask, load_gsm8k_tasks
from nmoe.rl.tasks.code import HumanEvalTask, load_humaneval_tasks
from nmoe.rl.tasks.git_episodes import (
    GitCommitTaskPool,
    GitEpisode,
    MultiRepoTaskPool,
    RepoConfig,
    clone_repo,
    create_git_task_pool,
)
from nmoe.rl.tasks.agents import AgentSelfPlayTaskPool, MultiToolGCDTask
from nmoe.rl.tasks.proof import ProofMetaVerifierTask, ProofVerifierTask

__all__ = [
    "Task",
    "TaskPool",
    "GSM8KTask",
    "MATHTask",
    "HumanEvalTask",
    "load_gsm8k_tasks",
    "load_humaneval_tasks",
    "GitCommitTaskPool",
    "GitEpisode",
    "MultiRepoTaskPool",
    "RepoConfig",
    "clone_repo",
    "create_git_task_pool",
]
