"""Step-level PRM dataset adapters (PRM800K / Math-Shepherd).

These datasets provide (prompt, step-completions, step-labels). We expose:
- a minimal HF loader (schema-tolerant)
- a small deterministic pool that can sample:
  - whole-solution verifier tasks (score in {0,0.5,1})
  - per-step label examples (0/1) for process supervision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from nmoe.rl.tasks.proof import ProofVerifierTask


@dataclass(frozen=True)
class StepwisePRMExample:
    prompt: str
    steps: list[str]
    step_labels: list[float]
    raw: dict[str, Any]


@dataclass(frozen=True)
class StepwisePRMStepLabel:
    prompt: str
    steps: list[str]
    step_idx: int  # 1-based
    gold_label: int  # 0 or 1


def _as_str_list(x: Any) -> list[str] | None:
    if not isinstance(x, list):
        return None
    if not all(isinstance(s, str) for s in x):
        return None
    return [s for s in x]


def _as_float_list(x: Any) -> list[float] | None:
    if not isinstance(x, list):
        return None
    out: list[float] = []
    for v in x:
        if isinstance(v, bool):
            out.append(1.0 if v else 0.0)
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            return None
    return out


def overall_score_0_05_1(step_labels: Sequence[float]) -> str:
    """Map step-level labels to a discrete {0, 0.5, 1} overall score."""
    if not step_labels:
        return "0"
    ok = [float(x) >= 0.5 for x in step_labels]
    if all(ok):
        return "1"
    if any(ok):
        return "0.5"
    return "0"


def iter_hf_stepwise_prm(
    dataset_name: str,
    *,
    split: str = "train",
    streaming: bool = False,
) -> Iterator[StepwisePRMExample]:
    """Iterate a Hugging Face step-level PRM dataset.

    Expected schema variants:
      - prompt key: "prompt" or "pompt"
      - steps key: "completions"
      - labels key: "labels"
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Hugging Face datasets is required to load PRM datasets") from e

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    for row in ds:
        ex = _row_to_stepwise_example(row)
        if ex is not None:
            yield ex


def iter_prm800k(*, split: str = "train", streaming: bool = False) -> Iterator[StepwisePRMExample]:
    return iter_hf_stepwise_prm("trl-lib/prm800k", split=split, streaming=streaming)


def iter_math_shepherd(*, split: str = "train", streaming: bool = False) -> Iterator[StepwisePRMExample]:
    return iter_hf_stepwise_prm("trl-lib/math_shepherd", split=split, streaming=streaming)


def to_verifier_task(ex: StepwisePRMExample) -> ProofVerifierTask:
    """Convert a stepwise PRM example into a whole-solution verifier task."""
    proof_lines: list[str] = []
    for i, s in enumerate(ex.steps, start=1):
        if not isinstance(s, str):
            continue
        ss = s.strip()
        if not ss:
            continue
        proof_lines.append(f"Step {i}: {ss}")
    proof = "\n".join(proof_lines)
    return ProofVerifierTask(problem=ex.prompt, proof=proof, gold_score=overall_score_0_05_1(ex.step_labels))


def _row_to_stepwise_example(row: Any) -> StepwisePRMExample | None:
    if not isinstance(row, dict):
        return None
    prompt = row.get("prompt")
    if not isinstance(prompt, str):
        prompt = row.get("pompt")
    steps = _as_str_list(row.get("completions"))
    labels = _as_float_list(row.get("labels"))
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if steps is None or labels is None:
        return None
    n = min(len(steps), len(labels))
    steps = [s for s in steps[:n]]
    labels = [float(x) for x in labels[:n]]
    return StepwisePRMExample(prompt=prompt, steps=steps, step_labels=labels, raw=row)


class PRMTaskPool:
    """Deterministic pool for PRM800K / Math-Shepherd.

    - `sample()` returns whole-solution ProofVerifierTask (score in {0,0.5,1})
    - `sample_step_labels()` returns per-step labels for process supervision
    """

    def __init__(
        self,
        tasks: Sequence[ProofVerifierTask],
        *,
        seed: int = 0,
        examples: Sequence[StepwisePRMExample] | None = None,
    ):
        self._tasks: list[ProofVerifierTask] = list(tasks)
        self._examples: list[StepwisePRMExample] | None = list(examples) if examples is not None else None
        self._seed = int(seed)
        self._sample_count = 0
        self._by_score: dict[str, list[int]] = {"0": [], "0.5": [], "1": []}
        for i, t in enumerate(self._tasks):
            if t.gold_score in self._by_score:
                self._by_score[t.gold_score].append(i)

    def __len__(self) -> int:
        return len(self._tasks)

    def score_distribution(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._by_score.items()}

    def sample(self, n: int = 1, *, stratified: bool = False) -> list[ProofVerifierTask]:
        import random

        if not self._tasks:
            return []
        rng = random.Random(self._seed + self._sample_count)
        self._sample_count += 1

        if not stratified:
            idxs = rng.choices(range(len(self._tasks)), k=int(n))
            return [self._tasks[i] for i in idxs]

        out: list[ProofVerifierTask] = []
        scores = ["0", "0.5", "1"]
        for i in range(int(n)):
            score = scores[i % 3]
            pool = self._by_score.get(score, [])
            if not pool:
                pool = list(range(len(self._tasks)))
            out.append(self._tasks[rng.choice(pool)])
        return out

    def sample_examples(self, n: int = 1, *, stratified: bool = False) -> list[StepwisePRMExample]:
        if self._examples is None:
            raise RuntimeError("PRMTaskPool was not constructed with examples (use from_hf())")
        # The examples list is aligned to tasks by construction in from_hf().
        idx_tasks = self.sample(int(n), stratified=stratified)
        # Map back by identity on problem+proof+gold_score; stable because tasks came from examples.
        # For minimalism: rebuild a dict once per call (n is small in sanity).
        key_to_ex: dict[tuple[str, str, str], StepwisePRMExample] = {}
        for ex in self._examples:
            t = to_verifier_task(ex)
            key_to_ex[(t.problem, t.proof, t.gold_score)] = ex
        out: list[StepwisePRMExample] = []
        for t in idx_tasks:
            ex = key_to_ex.get((t.problem, t.proof, t.gold_score))
            if ex is not None:
                out.append(ex)
        if len(out) != int(n):
            # Fall back to uniform sampling from examples (still deterministic).
            import random

            rng = random.Random(self._seed + self._sample_count + 9973)
            out = rng.choices(self._examples, k=int(n))
        return out

    def sample_step_labels(
        self,
        n: int,
        *,
        stop_at_first_incorrect: bool = True,
        max_steps_per_example: int | None = None,
    ) -> list[StepwisePRMStepLabel]:
        """Sample per-step labels.

        When stop_at_first_incorrect=True, only steps up to (and including) the
        first incorrect step are eligible, matching the PRM labeling contract.
        """
        import random

        if self._examples is None:
            raise RuntimeError("PRMTaskPool was not constructed with examples (use from_hf())")

        rng = random.Random(self._seed + self._sample_count + 1337)
        self._sample_count += 1
        out: list[StepwisePRMStepLabel] = []
        for _ in range(int(n)):
            ex = rng.choice(self._examples)
            steps = list(ex.steps)
            labels = list(ex.step_labels)
            if max_steps_per_example is not None:
                steps = steps[: int(max_steps_per_example)]
                labels = labels[: int(max_steps_per_example)]
            if not steps or not labels:
                continue
            n_steps = min(len(steps), len(labels))
            cut = n_steps
            if stop_at_first_incorrect:
                for i in range(n_steps):
                    if float(labels[i]) < 0.5:
                        cut = i + 1
                        break
            if cut <= 0:
                continue
            step_idx = rng.randrange(1, cut + 1)  # 1-based
            gold = 1 if float(labels[step_idx - 1]) >= 0.5 else 0
            out.append(StepwisePRMStepLabel(prompt=ex.prompt, steps=steps, step_idx=step_idx, gold_label=gold))
        return out

    @classmethod
    def from_hf(
        cls,
        source: str = "prm800k",
        *,
        max_examples: int = 10_000,
        seed: int = 0,
        split: str = "train",
        streaming: bool = False,
    ) -> "PRMTaskPool":
        import random

        examples: list[StepwisePRMExample] = []
        sources: list[tuple[str, Any]] = []
        if source in ("prm800k", "both"):
            sources.append(("prm800k", iter_prm800k))
        if source in ("math_shepherd", "both"):
            sources.append(("math_shepherd", iter_math_shepherd))
        if not sources:
            raise ValueError(f"Unknown source: {source!r} (expected prm800k, math_shepherd, or both)")

        per_source = max_examples if len(sources) == 1 else max_examples // len(sources)
        for _name, iter_fn in sources:
            count = 0
            for ex in iter_fn(split=split, streaming=streaming):
                examples.append(ex)
                count += 1
                if count >= int(per_source):
                    break

        rng = random.Random(int(seed))
        rng.shuffle(examples)
        examples = examples[: int(max_examples)]

        tasks = [to_verifier_task(ex) for ex in examples]
        return cls(tasks, seed=int(seed), examples=examples)

