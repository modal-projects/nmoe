"""HuggingFace dataset → Harmony RL tasks glue.

This intentionally keeps the surface small:
- Load a dataset split via `datasets.load_dataset(...)`
- Map fields into existing Task types (math / gsm8k today)
- Extract a verifiable `gold_answer` via a small built-in allowlist

No custom output formats: tasks always use Harmony via existing Task.to_prompt()
and parse model responses with `parse_harmony_text(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nmoe.rl.tasks import Task
from nmoe.rl.tasks.math import GSM8KTask, MATHTask, extract_boxed, extract_last_number


def _load_dataset(
    *,
    dataset: str,
    split: str,
    subset: str | None,
    streaming: bool,
    trust_remote_code: bool,
    data_files: str | list[str] | None,
):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("datasets package required: pip install datasets") from e

    kwargs: dict[str, Any] = {
        "path": dataset,
        "split": split,
        "streaming": bool(streaming),
        "trust_remote_code": bool(trust_remote_code),
    }
    if subset:
        kwargs["name"] = subset
    if data_files is not None:
        kwargs["data_files"] = data_files
    return load_dataset(**kwargs)


def _extract_gold_answer(answer_text: str, *, gold_extractor: str) -> str | None:
    if not isinstance(answer_text, str):
        return None

    if gold_extractor == "raw":
        s = answer_text.strip()
        return s if s else None

    if gold_extractor == "gsm8k_hash":
        # GSM8K answers are formatted as: "... #### <number>"
        if "####" not in answer_text:
            return None
        parts = answer_text.split("####")
        if len(parts) < 2:
            return None
        s = parts[-1].strip().replace(",", "")
        return s if s else None

    if gold_extractor == "boxed":
        return extract_boxed(answer_text)

    if gold_extractor == "last_number":
        return extract_last_number(answer_text)

    raise ValueError(f"unknown gold_extractor={gold_extractor!r}")


@dataclass(frozen=True)
class HFDatasetTaskSpec:
    dataset: str
    split: str = "train"
    subset: str | None = None
    data_files: str | list[str] | None = None
    streaming: bool = False
    trust_remote_code: bool = False

    task_type: str = "math"  # "math" | "gsm8k"
    problem_field: str = "problem"
    answer_field: str = "answer"
    gold_extractor: str = "raw"  # "raw" | "gsm8k_hash" | "boxed" | "last_number"

    max_examples: int = 10_000


def load_tasks_from_hf(spec: HFDatasetTaskSpec) -> list[Task]:
    if not isinstance(spec.dataset, str) or not spec.dataset:
        raise ValueError("dataset must be a non-empty string")
    if not isinstance(spec.split, str) or not spec.split:
        raise ValueError("split must be a non-empty string")
    if not isinstance(spec.problem_field, str) or not spec.problem_field:
        raise ValueError("problem_field must be a non-empty string")
    if not isinstance(spec.answer_field, str) or not spec.answer_field:
        raise ValueError("answer_field must be a non-empty string")
    if spec.max_examples <= 0:
        raise ValueError("max_examples must be > 0")

    ds = _load_dataset(
        dataset=spec.dataset,
        split=spec.split,
        subset=spec.subset,
        streaming=spec.streaming,
        trust_remote_code=spec.trust_remote_code,
        data_files=spec.data_files,
    )

    tasks: list[Task] = []
    for ex in ds:
        if len(tasks) >= int(spec.max_examples):
            break

        problem = ex.get(spec.problem_field, "")
        answer_text = ex.get(spec.answer_field, "")
        if not isinstance(problem, str) or not problem.strip():
            continue
        if not isinstance(answer_text, str) or not answer_text.strip():
            continue

        gold = _extract_gold_answer(answer_text, gold_extractor=spec.gold_extractor)
        if gold is None:
            continue

        if spec.task_type == "math":
            tasks.append(MATHTask(problem=problem, gold_answer=gold))
        elif spec.task_type == "gsm8k":
            tasks.append(GSM8KTask(question=problem, gold_answer=gold, full_solution=answer_text))
        else:
            raise ValueError(f"unknown task_type={spec.task_type!r}")

    return tasks

