"""Harmony-first PRM verifier tasks.

We avoid custom markup (\boxed, <answer>, etc.) and rely on Harmony channels:
- user provides the task
- assistant responds with Harmony messages
- correctness is judged from the `final` channel only
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nmoe.rl.rewards_harmony import CHANNELS, harmony_message, parse_harmony_text
from nmoe.rl.tasks import Task


def _normalize_score_0_05_1(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    if t in {"0", "0.0"}:
        return "0"
    if t in {"0.5", ".5"}:
        return "0.5"
    if t in {"1", "1.0"}:
        return "1"
    return None


def _normalize_bit(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    if t in {"0"}:
        return "0"
    if t in {"1"}:
        return "1"
    return None


@dataclass
class HarmonyVerifierScoreTask(Task):
    """Whole-solution verifier scoring task (final in {0,0.5,1})."""

    problem: str
    solution: str
    gold_score: str  # "0" | "0.5" | "1"

    task_type: str = field(default="harmony_verifier_score", init=False)

    def to_prompt(self) -> str:
        user = (
            "You are a verifier. Evaluate the solution.\n\n"
            "Return ONLY the overall score in the final channel as one of: 0, 0.5, 1.\n"
            "Use the analysis channel for any reasoning.\n\n"
            f"Problem:\n{self.problem}\n\n"
            f"Solution:\n{self.solution}\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        return _normalize_score_0_05_1(parsed.final_content)

    def verify(self, answer: str | None) -> bool:
        return _normalize_score_0_05_1(answer) == _normalize_score_0_05_1(self.gold_score)

    def get_metadata(self) -> dict:
        return {"task_type": self.task_type, "gold_score": self.gold_score}


@dataclass
class HarmonyPRMStepLabelTask(Task):
    """Per-step PRM label task (final in {0,1})."""

    problem: str
    steps: list[str]
    step_idx: int  # 1-based
    gold_label: str  # "0" | "1"

    task_type: str = field(default="harmony_prm_step_label", init=False)

    def to_prompt(self) -> str:
        if self.step_idx <= 0:
            raise ValueError(f"step_idx must be >= 1 (got {self.step_idx})")
        shown = self.steps[: self.step_idx]
        sol = "\n".join(f"Step {i}: {s.strip()}" for i, s in enumerate(shown, start=1) if str(s).strip())
        user = (
            "You are a process reward model. Judge whether the last step is correct.\n\n"
            "Return ONLY a single token in the final channel:\n"
            "- 1 if the last step is correct\n"
            "- 0 if the last step is incorrect\n\n"
            f"Problem:\n{self.problem}\n\n"
            f"Solution prefix (through Step {self.step_idx}):\n{sol}\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        return _normalize_bit(parsed.final_content)

    def verify(self, answer: str | None) -> bool:
        return _normalize_bit(answer) == _normalize_bit(self.gold_label)

    def get_metadata(self) -> dict:
        return {"task_type": self.task_type, "gold_label": self.gold_label, "step_idx": self.step_idx}
