"""Math‑V2-style proof verification tasks (verifier + meta-verifier).

These tasks are **Harmony-only**:
- Prompts are a single Harmony `user` message.
- The model returns the score as the entire `final` channel content.

No \\boxed parsing, no custom tags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from nmoe.rl.rewards_harmony import CHANNELS, harmony_message, parse_harmony_text
from nmoe.rl.tasks import Task


def _normalize_score(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if s in {"0", "0.0"}:
        return "0"
    if s in {"0.5", ".5"}:
        return "0.5"
    if s in {"1", "1.0"}:
        return "1"
    return None


@dataclass
class ProofVerifierTask(Task):
    """Math-V2 verifier training task.

    The ground-truth signal is the annotated proof score in {0, 0.5, 1}.
    """

    problem: str
    proof: str
    gold_score: str  # "0" | "0.5" | "1"

    task_type: str = field(default="proof_verifier", init=False)

    def to_prompt(self) -> str:
        user = (
            "You are a proof verifier.\n\n"
            "Score the proof attempt.\n"
            "Return ONLY the overall score in the final channel as one of: 0, 0.5, 1.\n"
            "Use the analysis channel for any reasoning.\n\n"
            f"Problem:\n{self.problem}\n\n"
            f"Proof attempt:\n{self.proof}\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        return _normalize_score(parsed.final_content)

    def verify(self, answer: str | None) -> bool:
        pred = _normalize_score(answer)
        gold = _normalize_score(self.gold_score)
        if pred is None or gold is None:
            return False
        return pred == gold

    def get_metadata(self) -> dict:
        return {"task_type": self.task_type, "gold_score": self.gold_score}


@dataclass
class ProofMetaVerifierTask(Task):
    """Math-V2 meta-verifier training task (faithfulness of verifier analysis)."""

    problem: str
    proof: str
    verifier_response: str
    gold_meta_score: str  # "0" | "0.5" | "1"

    task_type: str = field(default="proof_meta_verifier", init=False)

    def to_prompt(self) -> str:
        user = (
            "You are a meta-verifier.\n\n"
            "Check whether the verifier's evaluation is correct and justified.\n"
            "Return ONLY the overall meta-score in the final channel as one of: 0, 0.5, 1.\n"
            "Use the analysis channel for any reasoning.\n\n"
            f"Problem:\n{self.problem}\n\n"
            f"Proof attempt:\n{self.proof}\n\n"
            f"Verifier response:\n{self.verifier_response}\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        return _normalize_score(parsed.final_content)

    def verify(self, answer: str | None) -> bool:
        pred = _normalize_score(answer)
        gold = _normalize_score(self.gold_meta_score)
        if pred is None or gold is None:
            return False
        return pred == gold

    def get_metadata(self) -> dict:
        return {"task_type": self.task_type, "gold_meta_score": self.gold_meta_score}


def iter_proof_verifier_jsonl(path: str | Path) -> Iterator[ProofVerifierTask]:
    """Load ProofVerifierTask from a JSONL file.

    JSONL schema (one per line):
      {"problem": "...", "proof": "...", "score": 0|0.5|1}
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem = obj.get("problem")
            proof = obj.get("proof")
            score = obj.get("score")
            if not isinstance(problem, str) or not isinstance(proof, str):
                continue
            score_s = _normalize_score(str(score))
            if score_s is None:
                continue
            yield ProofVerifierTask(problem=problem, proof=proof, gold_score=score_s)


def iter_proof_meta_verifier_jsonl(path: str | Path) -> Iterator[ProofMetaVerifierTask]:
    """Load ProofMetaVerifierTask from a JSONL file.

    JSONL schema (one per line):
      {"problem": "...", "proof": "...", "verifier_response": "...", "meta_score": 0|0.5|1}
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem = obj.get("problem")
            proof = obj.get("proof")
            verifier_response = obj.get("verifier_response")
            meta_score = obj.get("meta_score")
            if not all(isinstance(x, str) for x in [problem, proof, verifier_response]):
                continue
            score_s = _normalize_score(str(meta_score))
            if score_s is None:
                continue
            yield ProofMetaVerifierTask(
                problem=problem,
                proof=proof,
                verifier_response=verifier_response,
                gold_meta_score=score_s,
            )


@dataclass
class ProofTaskPool:
    """Pool of proof verification tasks with sampling support."""

    tasks: list[Task] = field(default_factory=list)
    task_type: str = "verifier"  # "verifier" or "meta_verifier"

    def sample(self, n: int = 1) -> list[Task]:
        """Sample n tasks from pool (with replacement)."""
        import random

        if not self.tasks:
            return []
        return random.choices(self.tasks, k=n)

    def __len__(self) -> int:
        return len(self.tasks)

    @classmethod
    def from_jsonl(cls, path: Path, task_type: str = "verifier") -> "ProofTaskPool":
        """Load pool from JSONL file.

        Args:
            path: Path to JSONL file
            task_type: "verifier" or "meta_verifier"

        Returns:
            ProofTaskPool instance
        """
        if task_type == "verifier":
            tasks = list(iter_proof_verifier_jsonl(path))
        elif task_type == "meta_verifier":
            tasks = list(iter_proof_meta_verifier_jsonl(path))
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return cls(tasks=tasks, task_type=task_type)
