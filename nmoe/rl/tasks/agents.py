"""Self-contained agentic self-play tasks (tools-first).

These tasks are designed to be:
- deterministic (hard verifiers)
- self-contained (no external datasets required)
- tool-friendly (answers are impractical to compute by hand)

They serve as a production smoke environment for:
Tasks → Rollout (multi-turn) → Tools → RewardSignals/GDPO → GRPO.
"""

from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from nmoe.rl.rewards_harmony import CHANNELS, harmony_message, parse_harmony_text
from nmoe.rl.tasks import Task


@dataclass
class MultiToolGCDTask(Task):
    """Compute gcd(a,b) from a file path.

    The file contains two large integers on separate lines. The intended
    solution path is to use tools (bash/read + python) rather than mental math.
    """

    task_id: str
    inputs_path: str
    gold_gcd: str

    task_type: str = field(default="agent_gcd", init=False)

    def to_prompt(self) -> str:
        user = (
            "Use tools to solve this task.\n\n"
            f"1) Read the file at: {self.inputs_path}\n"
            "2) The file contains two integers (a and b), one per line.\n"
            "3) Compute gcd(a, b).\n\n"
            "Return ONLY the gcd(a,b) in the final channel.\n"
            "Use the analysis channel for any reasoning.\n"
        )
        return harmony_message(role="user", channel=CHANNELS["commentary"], content=user)

    def extract_answer(self, response: str) -> str | None:
        parsed = parse_harmony_text(response)
        s = parsed.final_content.strip()
        return s if s else None

    def verify(self, answer: str | None) -> bool:
        if answer is None:
            return False
        return answer.strip() == self.gold_gcd

    def get_metadata(self) -> dict:
        return {"task_type": self.task_type, "task_id": self.task_id, "inputs_path": self.inputs_path}


class AgentSelfPlayTaskPool:
    """On-demand generator for agentic self-play tasks.

    This is intentionally a minimal interface: train_agentic only requires a
    `.sample(n)` method that returns Tasks.
    """

    def __init__(
        self,
        *,
        root_dir: str | Path,
        seed: int = 0,
        digits: int = 120,
    ):
        if digits < 32:
            raise ValueError(f"digits must be >= 32 (got {digits})")
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.digits = digits
        self._counter = 0

    def _rand_int(self) -> int:
        return self._rand_int_digits(self.digits)

    def _rand_int_digits(self, digits: int) -> int:
        if digits <= 0:
            raise ValueError(f"digits must be > 0 (got {digits})")
        lo = 10 ** (digits - 1)
        hi = (10 ** digits) - 1
        return self.rng.randint(lo, hi)

    def sample(self, n: int, replace: bool = True) -> list[Task]:
        _ = replace
        if n <= 0:
            return []

        tasks: list[Task] = []
        for _ in range(n):
            self._counter += 1
            task_id = f"agent_gcd_{self._counter:08d}"
            ws = self.root_dir / task_id
            if ws.exists():
                shutil.rmtree(ws)
            ws.mkdir(parents=True, exist_ok=True)

            # Guarantee a non-trivial gcd so the task is genuinely tools-first.
            g_digits = max(32, self.digits // 3)
            tail_digits = self.digits - g_digits
            if tail_digits < 16:
                raise ValueError(
                    f"digits too small for tools-first gcd task (digits={self.digits}, g_digits={g_digits})"
                )

            g = self._rand_int_digits(g_digits)
            x = self._rand_int_digits(tail_digits)
            y = self._rand_int_digits(tail_digits)

            # Make x,y coprime so gcd(a,b) == g exactly.
            d = math.gcd(x, y)
            x //= d
            y //= d

            a = g * x
            b = g * y
            gold = str(g)

            inputs = ws / "inputs.txt"
            inputs.write_text(f"{a}\n{b}\n", encoding="utf-8")

            tasks.append(MultiToolGCDTask(
                task_id=task_id,
                inputs_path=str(inputs),
                gold_gcd=gold,
            ))

        return tasks
