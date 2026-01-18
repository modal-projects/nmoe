"""WSD-style curriculum for agentic RL training.

Implements Warmup-Sustain-Decay curriculum scheduling based on example counts,
with adaptive adjustment based on rolling metrics. Also includes thinking budget
control for efficient reasoning.

Reference:
- WSD learning rate schedules
- Curriculum learning for RL
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class StageType(str, Enum):
    """Standard WSD curriculum stages.

    Inherits from str for easy string comparison and JSON serialization.
    Custom stages can use plain strings instead.
    """
    WARMUP = "warmup"
    SUSTAIN = "sustain"
    DECAY = "decay"

    def __str__(self) -> str:
        return self.value


@dataclass
class CurriculumStage:
    """A stage in the WSD-style curriculum."""
    name: str | StageType            # Stage name (use StageType for standard, str for custom)
    examples_count: int              # Duration in examples (not steps)
    weights: dict[str, float]        # Reward weights for this stage
    weight_decay_rate: float = 0.0   # Per-example exponential decay within stage
    transitions_to: str | StageType | None = None  # Next stage name (None = terminal)

    def __post_init__(self):
        # Normalize to string for internal use (str(StageType.WARMUP) == "warmup")
        self.name = str(self.name)
        if self.transitions_to is not None:
            self.transitions_to = str(self.transitions_to)


class RollingMetrics:
    """Track rolling statistics for adaptive curriculum adjustment."""

    def __init__(self, window: int = 500):
        self.window = window
        self._data: dict[str, deque] = {}

    def add(self, metrics: dict[str, float]) -> None:
        """Add a batch of metrics."""
        for k, v in metrics.items():
            if k not in self._data:
                self._data[k] = deque(maxlen=self.window)
            self._data[k].append(v)

    def mean(self, key: str, default: float = 0.0) -> float:
        """Get rolling mean for a metric."""
        if key not in self._data or len(self._data[key]) == 0:
            return default
        return sum(self._data[key]) / len(self._data[key])

    def count(self, key: str) -> int:
        """Get count of samples for a metric."""
        return len(self._data.get(key, []))

    def clear(self) -> None:
        """Clear all metrics."""
        self._data.clear()


# =============================================================================
# Default Curriculum Stages
# =============================================================================

# Warmup: Learn structure and basic tool use
WARMUP_WEIGHTS = {
    # Structure: high weight (learn the format)
    "has_reasoning": 1.0,
    "has_final_response": 1.0,
    # Harmony channels
    "struct_has_start": 1.0,
    "struct_has_end": 1.0,
    "struct_proper_nesting": 1.0,
    "chan_has_analysis": 0.5,
    "chan_has_final": 0.5,
    # Tools: learn to use them
    "tool_syntax_valid": 1.0,
    "tool_executed": 0.5,
    "tool_output_used": 0.3,
    # Code tools: basics
    "python_compiles": 0.5,
    "bash_valid": 0.5,
    # Correctness: lower (still learning)
    "answer_correct": 0.3,
    # Efficiency: none yet
    "thinking_efficiency": 0.0,
    "tool_efficiency": 0.0,
}

# Sustain: Focus on tool effectiveness and correctness
SUSTAIN_WEIGHTS = {
    # Structure: decaying (should be learned)
    "has_reasoning": 0.3,
    "has_final_response": 0.3,
    "struct_has_start": 0.3,
    "struct_has_end": 0.3,
    "struct_proper_nesting": 0.3,
    "chan_has_analysis": 0.2,
    "chan_has_final": 0.2,
    # Tools: full focus on quality
    "tool_syntax_valid": 0.5,
    "tool_executed": 1.0,
    "tool_output_used": 1.5,  # Key: actually use outputs!
    # Code tools: full execution
    "python_compiles": 0.3,
    "python_runs": 1.0,
    "python_correct": 1.5,
    "bash_valid": 0.3,
    "bash_succeeds": 1.0,
    # Correctness: primary objective
    "answer_correct": 2.0,
    # Efficiency: starting
    "thinking_efficiency": 0.3,
    "tool_efficiency": 0.3,
}

# Decay: Correctness + efficiency, penalize structure violations
DECAY_WEIGHTS = {
    # Structure: only penalize failures (negative weights)
    "has_reasoning": -0.1,
    "has_final_response": -0.2,
    "struct_has_start": -0.1,
    "struct_has_end": -0.1,
    "struct_proper_nesting": -0.2,
    "chan_has_analysis": 0.0,
    "chan_has_final": 0.0,
    # Tools: effectiveness over form
    "tool_syntax_valid": 0.0,  # Should be learned
    "tool_executed": 0.5,
    "tool_output_used": 2.0,  # High: must use outputs
    # Code tools: correctness matters
    "python_compiles": 0.0,
    "python_runs": 0.3,
    "python_correct": 2.0,
    "bash_valid": 0.0,
    "bash_succeeds": 0.5,
    # Correctness: dominant
    "answer_correct": 3.0,
    # Efficiency: significant
    "thinking_efficiency": 1.0,
    "tool_efficiency": 1.0,
}


DEFAULT_STAGES = [
    CurriculumStage(
        name=StageType.WARMUP,
        examples_count=500,
        weights=WARMUP_WEIGHTS,
        weight_decay_rate=0.0,
        transitions_to=StageType.SUSTAIN,
    ),
    CurriculumStage(
        name=StageType.SUSTAIN,
        examples_count=5000,
        weights=SUSTAIN_WEIGHTS,
        weight_decay_rate=0.0001,  # Slow decay of structure weights
        transitions_to=StageType.DECAY,
    ),
    CurriculumStage(
        name=StageType.DECAY,
        examples_count=50000,
        weights=DECAY_WEIGHTS,
        weight_decay_rate=0.0,  # Stable in final stage
        transitions_to=None,  # Terminal
    ),
]


class WSDCurriculum:
    """Warmup-Sustain-Decay curriculum for reward weights.

    Similar to WSD learning rate schedules, but for reward weights.
    Transitions are based on example counts, with optional adaptive
    adjustment based on rolling success metrics.
    """

    def __init__(
        self,
        stages: list[CurriculumStage] | None = None,
        metrics_window: int = 500,
        adaptive: bool = True,
    ):
        """Initialize curriculum.

        Args:
            stages: List of curriculum stages (default: warmup/sustain/decay)
            metrics_window: Window size for rolling metrics
            adaptive: Enable adaptive weight adjustment
        """
        stages = stages or DEFAULT_STAGES
        self.stages = {s.name: s for s in stages}
        self.stage_order = [s.name for s in stages]
        self.current_stage = self.stage_order[0]
        self.examples_in_stage = 0
        self.total_examples = 0
        self.metrics = RollingMetrics(window=metrics_window)
        self.adaptive = adaptive

    def get_weights(self) -> dict[str, float]:
        """Get current reward weights with decay applied."""
        stage = self.stages[self.current_stage]
        weights = dict(stage.weights)

        # Apply within-stage exponential decay
        if stage.weight_decay_rate > 0:
            decay_factor = (1.0 - stage.weight_decay_rate) ** self.examples_in_stage
            for k in list(weights.keys()):
                if weights[k] > 0:  # Only decay positive weights
                    weights[k] *= decay_factor

        return weights

    def step(self, batch_metrics: dict[str, float] | None = None) -> str | None:
        """Update curriculum state after a batch.

        Args:
            batch_metrics: Metrics from the batch (for adaptive adjustment)

        Returns:
            New stage name if transition occurred, None otherwise
        """
        if batch_metrics:
            self.metrics.add(batch_metrics)

        self.examples_in_stage += 1
        self.total_examples += 1

        # Check for stage transition
        stage = self.stages[self.current_stage]
        if self.examples_in_stage >= stage.examples_count:
            if stage.transitions_to is not None:
                old_stage = self.current_stage
                self.current_stage = stage.transitions_to
                self.examples_in_stage = 0
                return self.current_stage

        return None

    def adapt_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Adaptive weight adjustment based on success rates.

        Only modifies weights if adaptive=True was set in constructor.

        Args:
            weights: Current weights from get_weights()

        Returns:
            Adjusted weights
        """
        if not self.adaptive:
            return weights

        weights = dict(weights)

        # Structure mastered (>95% success) -> accelerate decay
        struct_rate = self.metrics.mean("struct_success", default=0.0)
        if struct_rate > 0.95:
            for k in list(weights.keys()):
                if k.startswith("struct_") or k.startswith("has_"):
                    if weights[k] > 0:
                        weights[k] *= 0.9  # 10% faster decay

        # Accuracy struggling (<30%) -> boost
        accuracy_rate = self.metrics.mean("accuracy", default=0.5)
        if accuracy_rate < 0.30:
            weights["answer_correct"] = min(
                4.0, weights.get("answer_correct", 1.0) * 1.1
            )

        # Tool use struggling (<50% execution) -> boost tool rewards
        tool_rate = self.metrics.mean("tool_executed", default=0.5)
        if tool_rate < 0.50:
            weights["tool_syntax_valid"] = min(
                2.0, weights.get("tool_syntax_valid", 0.5) * 1.2
            )
            weights["tool_executed"] = min(
                2.0, weights.get("tool_executed", 0.5) * 1.2
            )

        return weights

    def get_stage_info(self) -> dict:
        """Get current curriculum state for logging."""
        stage = self.stages[self.current_stage]
        return {
            "stage": self.current_stage,
            "examples_in_stage": self.examples_in_stage,
            "total_examples": self.total_examples,
            "stage_progress": self.examples_in_stage / max(stage.examples_count, 1),
            "next_stage": stage.transitions_to,
        }


# =============================================================================
# Thinking Budget Control
# =============================================================================

@dataclass
class ThinkingBudgetConfig:
    """Configuration for thinking/reasoning token budget.

    Controls how much reasoning the model is allowed/encouraged to do.
    Works with the thinking_efficiency reward signal.
    """
    # Base budget (max reasoning tokens before penalty)
    max_reasoning_tokens: int = 2048

    # Penalty for exceeding budget
    penalty_per_token: float = 0.0005  # Soft penalty per token over budget

    # Bonus for efficiency (correct + under budget)
    efficiency_bonus: float = 0.1

    # Curriculum-based budget schedule: (example_threshold, new_budget)
    # Budget decreases as training progresses
    budget_schedule: list[tuple[int, int]] = field(default_factory=lambda: [
        (0, 4096),      # Warmup: generous
        (500, 2048),    # Sustain: tighter
        (5500, 1024),   # Decay: efficient
    ])


def get_thinking_budget(config: ThinkingBudgetConfig, total_examples: int) -> int:
    """Get current thinking budget based on curriculum progress.

    Args:
        config: Thinking budget configuration
        total_examples: Total examples seen so far

    Returns:
        Current max reasoning tokens
    """
    budget = config.max_reasoning_tokens
    for threshold, new_budget in config.budget_schedule:
        if total_examples >= threshold:
            budget = new_budget
    return budget


def thinking_budget_reward(
    reasoning_tokens: int,
    answer_correct: bool,
    config: ThinkingBudgetConfig,
    total_examples: int,
) -> float:
    """Compute efficiency reward based on thinking budget.

    Only rewards efficiency when the answer is correct - no point
    being fast if you're wrong.

    Args:
        reasoning_tokens: Number of tokens in reasoning/analysis
        answer_correct: Whether the final answer was correct
        config: Thinking budget configuration
        total_examples: Total examples seen (for curriculum)

    Returns:
        Efficiency reward (negative if over budget, positive bonus if under)
    """
    if not answer_correct:
        return 0.0  # No efficiency bonus for wrong answers

    budget = get_thinking_budget(config, total_examples)

    if reasoning_tokens > budget:
        # Over budget: soft penalty proportional to overage
        overage = reasoning_tokens - budget
        return -config.penalty_per_token * overage
    else:
        # Under budget: bonus scaled by how much under
        savings_ratio = 1.0 - (reasoning_tokens / max(budget, 1))
        return config.efficiency_bonus * savings_ratio


# =============================================================================
# Convenience Factory
# =============================================================================

def create_curriculum(
    mode: str = "default",
    **kwargs,
) -> WSDCurriculum:
    """Factory function to create curriculum configurations.

    Args:
        mode: Curriculum mode:
            - "default": Standard warmup/sustain/decay
            - "aggressive": Faster transitions, higher efficiency pressure
            - "conservative": Slower transitions, more structure emphasis
            - "accuracy_only": Skip format learning, focus on correctness
        **kwargs: Override arguments for WSDCurriculum

    Returns:
        Configured WSDCurriculum instance
    """
    if mode == "default":
        return WSDCurriculum(**kwargs)

    elif mode == "aggressive":
        stages = [
            CurriculumStage(
                name="warmup",
                examples_count=200,  # Faster
                weights=WARMUP_WEIGHTS,
                transitions_to="sustain",
            ),
            CurriculumStage(
                name="sustain",
                examples_count=2000,  # Faster
                weights=SUSTAIN_WEIGHTS,
                weight_decay_rate=0.0005,  # Faster decay
                transitions_to="decay",
            ),
            CurriculumStage(
                name="decay",
                examples_count=20000,
                weights={**DECAY_WEIGHTS, "thinking_efficiency": 2.0},  # More efficiency pressure
                transitions_to=None,
            ),
        ]
        return WSDCurriculum(stages=stages, **kwargs)

    elif mode == "conservative":
        stages = [
            CurriculumStage(
                name="warmup",
                examples_count=1000,  # Longer
                weights=WARMUP_WEIGHTS,
                transitions_to="sustain",
            ),
            CurriculumStage(
                name="sustain",
                examples_count=10000,  # Longer
                weights=SUSTAIN_WEIGHTS,
                weight_decay_rate=0.00005,  # Slower decay
                transitions_to="decay",
            ),
            CurriculumStage(
                name="decay",
                examples_count=100000,
                weights=DECAY_WEIGHTS,
                transitions_to=None,
            ),
        ]
        return WSDCurriculum(stages=stages, **kwargs)

    elif mode == "accuracy_only":
        # Skip format learning entirely - for models that already know format
        stages = [
            CurriculumStage(
                name="main",
                examples_count=100000,
                weights={
                    "answer_correct": 3.0,
                    "tool_executed": 1.0,
                    "tool_output_used": 1.5,
                    "thinking_efficiency": 0.5,
                },
                transitions_to=None,
            ),
        ]
        return WSDCurriculum(stages=stages, **kwargs)

    else:
        raise ValueError(f"Unknown curriculum mode: {mode}")
