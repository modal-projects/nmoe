"""Rule-based rewards with composable Rubric architecture.

Supports multiple reward signals (accuracy, format, tool use) that can be
composed with weights. Each reward function returns both a scalar value
and a category for debugging/analysis.

Reference: primeintellect/verifiers rubric pattern
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardResult:
    """Reward result with value and category for debugging."""
    value: float
    category: str  # e.g. "correct", "format_error", "timeout", "wrong_answer"
    details: dict = field(default_factory=dict)


# =============================================================================
# Tag Extraction
# =============================================================================

def extract_tag(text: str, tag: str) -> Optional[str]:
    """Extract content from <tag>...</tag>."""
    if tag not in ("think", "answer"):
        raise ValueError(f"unsupported tag: {tag}")
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if m is None:
        return None
    return m.group(1).strip()


# =============================================================================
# Format Reward
# =============================================================================

def format_reward(text: str) -> RewardResult:
    """Reward for structural compliance: <think>...</think><answer>...</answer>.

    Returns:
        RewardResult with value 1.0 if valid format, 0.0 otherwise.
        Category indicates the specific failure mode.
    """
    think = extract_tag(text, "think")
    answer = extract_tag(text, "answer")

    if think is None and answer is None:
        return RewardResult(0.0, "no_tags")
    if think is None:
        return RewardResult(0.0, "no_think_tag")
    if answer is None:
        return RewardResult(0.0, "no_answer_tag")

    # Ensure order: </think> precedes <answer>
    end_think = re.search(r"</think>", text, flags=re.IGNORECASE)
    start_answer = re.search(r"<answer>", text, flags=re.IGNORECASE)
    if end_think is None or start_answer is None:
        return RewardResult(0.0, "malformed_tags")
    if end_think.start() > start_answer.start():
        return RewardResult(0.0, "wrong_tag_order")
    if answer == "":
        return RewardResult(0.0, "empty_answer")

    return RewardResult(1.0, "valid_format")


# =============================================================================
# Math (GSM8K) Reward
# =============================================================================

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _normalize_math(s: str) -> str:
    """Normalize math answer for comparison."""
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.strip().strip(".")
    return s


def _extract_boxed_answer(answer_text: str) -> Optional[str]:
    """Extract content from \\boxed{...}."""
    m = _BOXED_RE.search(answer_text)
    if m is None:
        return None
    return m.group(1).strip()


def gsm8k_gold_from_answer_field(answer_field: str) -> Optional[str]:
    """Extract GSM8K gold from the dataset `answer` field (uses '####')."""
    if "####" not in answer_field:
        return None
    gold = answer_field.split("####", 1)[1].strip()
    m = _NUM_RE.search(gold)
    return m.group(0) if m else None


def gsm8k_accuracy_reward(text: str, *, gold: str) -> RewardResult:
    """Deterministic GSM8K reward using boxed final answers inside <answer>.

    Requires:
    - valid <think>/<answer> structure
    - a '\\boxed{...}' expression inside <answer>
    - numeric match to gold after normalization
    """
    fmt = format_reward(text)
    if fmt.value < 1.0:
        return RewardResult(0.0, f"format_{fmt.category}")

    answer = extract_tag(text, "answer") or ""
    boxed = _extract_boxed_answer(answer)
    if boxed is None:
        return RewardResult(0.0, "no_boxed")

    m = _NUM_RE.search(boxed)
    if m is None:
        return RewardResult(0.0, "no_number_in_boxed")

    pred = _normalize_math(m.group(0))
    gold_n = _normalize_math(gold)

    if pred == gold_n:
        return RewardResult(1.0, "correct", {"pred": pred, "gold": gold_n})
    else:
        return RewardResult(0.0, "wrong_answer", {"pred": pred, "gold": gold_n})


# =============================================================================
# Code Execution Reward
# =============================================================================

def _strip_md_fences(code: str) -> str:
    """Strip markdown code fences."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip() + "\n"
    return code + ("\n" if not code.endswith("\n") else "")


def python_tests_reward(
    text: str,
    *,
    tests: str | Sequence[str],
    timeout_s: float = 5.0,
) -> RewardResult:
    """Run python tests against code inside <answer>.

    Uses CodexExecutor for sandboxed execution (Landlock + Seccomp).
    """
    fmt = format_reward(text)
    if fmt.value < 1.0:
        return RewardResult(0.0, f"format_{fmt.category}")

    answer = extract_tag(text, "answer")
    if answer is None:
        return RewardResult(0.0, "no_answer_content")

    code = _strip_md_fences(answer)
    tests_src = "\n".join(tests) if isinstance(tests, (list, tuple)) else str(tests)

    try:
        config = CodexConfig(timeout_ms=int(timeout_s * 1000))
        executor = CodexExecutor(config)
        result = executor.exec_tests(code, tests_src)
    except TimeoutError:
        return RewardResult(0.0, "timeout")
    except Exception as e:
        return RewardResult(0.0, "execution_error", {"error": str(e)})

    if result.success:
        return RewardResult(1.0, "tests_passed")

    stderr = (result.stderr or "")[:500]
    return RewardResult(0.0, "tests_failed", {"stderr": stderr})


# =============================================================================
# Async Reward with Timeout
# =============================================================================

async def async_reward_with_timeout(
    func: Callable[..., RewardResult],
    *args,
    timeout_s: float = 5.0,
    **kwargs,
) -> RewardResult:
    """Run reward function with async timeout protection."""
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: func(*args, **kwargs)),
            timeout=timeout_s,
        )
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Reward function {func.__name__} timed out after {timeout_s}s")
        return RewardResult(0.0, "timeout")
    except Exception as e:
        logger.error(f"Reward function {func.__name__} failed: {e}")
        return RewardResult(0.0, "error", {"error": str(e)})


# =============================================================================
# Rubric: Composable Reward Functions
# =============================================================================

@dataclass
class RewardFunc:
    """A reward function with weight and name."""
    func: Callable[..., RewardResult]
    weight: float = 1.0
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.func.__name__


class Rubric:
    """Composable reward rubric with multiple weighted signals.

    Example:
        rubric = Rubric()
        rubric.add(format_reward, weight=0.5)
        rubric.add(gsm8k_accuracy_reward, weight=1.0)

        result = rubric.score(text, gold="42")
        # result.value = 0.5 * format + 1.0 * accuracy
    """

    def __init__(self, funcs: list[RewardFunc] | None = None):
        self.funcs = funcs or []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add(self, func: Callable[..., RewardResult], weight: float = 1.0, name: str | None = None):
        """Add a reward function with weight."""
        self.funcs.append(RewardFunc(func=func, weight=weight, name=name))

    def score(self, text: str, **kwargs) -> RewardResult:
        """Score text with all reward functions.

        Returns aggregated reward with per-function breakdown in details.
        Uses signature inspection to pass only required kwargs to each function.
        """
        import inspect

        total = 0.0
        categories = []
        metrics = {}

        for rf in self.funcs:
            try:
                # Inspect function signature to pass only valid kwargs
                sig = inspect.signature(rf.func)
                param_names = set(sig.parameters.keys())
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
                result = rf.func(text, **filtered_kwargs)
                weighted = result.value * rf.weight
                total += weighted
                categories.append(f"{rf.name}:{result.category}")
                metrics[rf.name] = result.value
                metrics[f"{rf.name}_weighted"] = weighted
            except Exception as e:
                self.logger.error(f"Reward function {rf.name} failed: {e}")
                metrics[rf.name] = 0.0
                metrics[f"{rf.name}_weighted"] = 0.0
                categories.append(f"{rf.name}:error")

        return RewardResult(
            value=total,
            category="|".join(categories),
            details={"metrics": metrics},
        )

    async def score_async(self, text: str, timeout_s: float = 5.0, **kwargs) -> RewardResult:
        """Score with async timeout protection."""
        return await async_reward_with_timeout(
            self.score, text, timeout_s=timeout_s, **kwargs
        )


# =============================================================================
# JudgeRubric: LLM-as-Judge Evaluation
# =============================================================================

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    """Rubric that uses LLM-as-judge for evaluation.

    Uses an external LLM to judge if responses are correct.
    Useful for subjective tasks where rule-based verification is insufficient.

    Reference: primeintellect/verifiers JudgeRubric

    Example:
        from openai import AsyncOpenAI

        judge = JudgeRubric(
            client=AsyncOpenAI(),
            model="gpt-4o-mini",
            prompt=CUSTOM_JUDGE_PROMPT,
        )
        result = await judge.score_async(
            response,
            question="What is 2+2?",
            answer="4"
        )
    """

    def __init__(
        self,
        client=None,  # AsyncOpenAI or compatible
        model: str = "gpt-4o-mini",
        prompt: str = DEFAULT_JUDGE_PROMPT,
        sampling_args: dict | None = None,
        cache_responses: bool = True,
        funcs: list[RewardFunc] | None = None,
    ):
        """Initialize JudgeRubric.

        Args:
            client: OpenAI-compatible async client (creates default if None)
            model: Model to use for judging
            prompt: Judge prompt template with {question}, {answer}, {response}
            sampling_args: Sampling parameters for judge model
            cache_responses: Cache judge responses to avoid redundant calls
            funcs: Additional reward functions to compose
        """
        super().__init__(funcs)
        self.client = client
        self.model = model
        self.prompt = prompt
        self.sampling_args = sampling_args or {}
        self.cache_responses = cache_responses
        self._cache: dict[str, str] = {}

    def _get_client(self):
        """Lazy-load client."""
        if self.client is None:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI()
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self.client

    def _format_judge_prompt(
        self,
        question: str,
        answer: str,
        response: str,
    ) -> str:
        """Format the judge prompt."""
        return self.prompt.format(
            question=question,
            answer=answer,
            response=response,
        )

    async def judge(
        self,
        question: str,
        answer: str,
        response: str,
    ) -> tuple[bool, str]:
        """Call LLM judge and parse response.

        Args:
            question: The original question/prompt
            answer: Ground truth answer
            response: Model response to judge

        Returns:
            Tuple of (is_correct, raw_judge_response)
        """
        judge_prompt = self._format_judge_prompt(question, answer, response)

        # Check cache
        if self.cache_responses and judge_prompt in self._cache:
            judge_response = self._cache[judge_prompt]
        else:
            # Normalize sampling args
            args = dict(self.sampling_args)
            if "max_tokens" in args:
                args["max_completion_tokens"] = args.pop("max_tokens")

            try:
                client = self._get_client()
                result = await client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    **args,
                )
                judge_response = str(result.choices[0].message.content).strip().lower()
            except Exception as e:
                self.logger.error(f"Judge call failed: {e}")
                return False, f"error: {e}"

            # Cache response
            if self.cache_responses:
                self._cache[judge_prompt] = judge_response

        # Parse yes/no
        is_correct = judge_response.startswith("yes")
        return is_correct, judge_response

    async def score_async(
        self,
        text: str,
        question: str = "",
        answer: str = "",
        timeout_s: float = 30.0,
        **kwargs,
    ) -> RewardResult:
        """Score using LLM judge.

        Args:
            text: Response text to judge
            question: Original question/prompt
            answer: Ground truth answer
            timeout_s: Timeout for judge call
            **kwargs: Additional kwargs for composed reward functions

        Returns:
            RewardResult with judge verdict
        """
        try:
            is_correct, judge_response = await asyncio.wait_for(
                self.judge(question, answer, text),
                timeout=timeout_s,
            )

            # Compute any additional rewards from composed functions
            metrics = {"judge_correct": 1.0 if is_correct else 0.0}
            total = 1.0 if is_correct else 0.0

            # Add composed reward functions
            for rf in self.funcs:
                try:
                    result = rf.func(text, **kwargs)
                    weighted = result.value * rf.weight
                    total += weighted
                    metrics[rf.name] = result.value
                except Exception as e:
                    self.logger.error(f"Reward {rf.name} failed: {e}")
                    metrics[rf.name] = 0.0

            return RewardResult(
                value=total,
                category="correct" if is_correct else "incorrect",
                details={"judge_response": judge_response, "metrics": metrics},
            )

        except asyncio.TimeoutError:
            return RewardResult(0.0, "timeout")
        except Exception as e:
            return RewardResult(0.0, "error", {"error": str(e)})

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()


# =============================================================================
# Pre-built Rubrics
# =============================================================================

def math_rubric(format_weight: float = 0.5, accuracy_weight: float = 1.0) -> Rubric:
    """Create a rubric for math tasks (GSM8K style)."""
    rubric = Rubric()
    rubric.add(format_reward, weight=format_weight, name="format")
    rubric.add(gsm8k_accuracy_reward, weight=accuracy_weight, name="accuracy")
    return rubric


def code_rubric(format_weight: float = 0.5, tests_weight: float = 1.0) -> Rubric:
    """Create a rubric for code tasks."""
    rubric = Rubric()
    rubric.add(format_reward, weight=format_weight, name="format")
    rubric.add(python_tests_reward, weight=tests_weight, name="tests")
    return rubric


# =============================================================================
# GDPO: Conditional Rewards (arxiv 2601.05242)
# =============================================================================

def condition_reward(
    secondary: float,
    primary: float,
    *,
    threshold: float = 1.0,
) -> float:
    """Condition secondary reward on primary reward achievement.

    GDPO insight: When rewards compete (e.g., length vs correctness),
    condition the easier reward on the harder one to prevent gaming.

    Example: Length reward only counts if correctness reward >= threshold.
    This prevents model from optimizing for short wrong answers.

    Args:
        secondary: The reward to condition (e.g., length/format)
        primary: The reward that must be achieved (e.g., correctness)
        threshold: Primary reward threshold for secondary to count

    Returns:
        secondary if primary >= threshold, else 0.0
    """
    if primary >= threshold:
        return secondary
    return 0.0


def conditioned_length_reward(
    text: str,
    *,
    accuracy_reward: float,
    max_length: int = 4096,
    accuracy_threshold: float = 1.0,
) -> RewardResult:
    """Length reward conditioned on accuracy (GDPO pattern).

    Only gives length credit if the answer is correct. This prevents
    the model from gaming length at the expense of correctness.

    Args:
        text: The generated text
        accuracy_reward: The accuracy reward value
        max_length: Maximum expected length for normalization
        accuracy_threshold: Accuracy threshold for length reward

    Returns:
        RewardResult with value 1.0 - (len/max_len) if accurate, else 0.0
    """
    if accuracy_reward < accuracy_threshold:
        return RewardResult(0.0, "accuracy_gate_failed")

    # Shorter is better: reward = 1 - (len / max_len)
    length = len(text)
    length_ratio = min(length / max_length, 1.0)
    reward = 1.0 - length_ratio

    return RewardResult(reward, "length_ok", {"length": length, "ratio": length_ratio})


# =============================================================================
# Convenience Functions (Backwards Compatibility)
# =============================================================================

def reward_math_gsm8k(text: str, *, gold: str) -> RewardResult:
    """Combined format + accuracy reward for GSM8K."""
    rubric = math_rubric()
    return rubric.score(text, gold=gold)


def reward_code_unittest(text: str, *, tests: str | Sequence[str]) -> RewardResult:
    """Combined format + tests reward for code."""
    rubric = code_rubric()
    return rubric.score(text, tests=tests)
