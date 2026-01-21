"""Environment abstraction for RL post-training.

The goal is a single integration surface that binds together:
- a task source (TaskPool)
- an optional tool harness (ToolConfig / AsyncToolExecutor)
- an output format contract (e.g., Harmony)

This is deliberately small: trainers can depend on Environment without
knowing whether tasks come from static datasets, git-episodes, or self-play.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nmoe.rl.tasks import TaskPool
from nmoe.rl.tools import AsyncToolExecutor, ToolConfig


@dataclass(frozen=True)
class Environment:
    env_id: str
    task_pool: TaskPool
    format_type: str = "harmony"
    tool_config: ToolConfig | None = None

    def sample(self, n: int, *, seed: int | None = None) -> list:
        try:
            return self.task_pool.sample(int(n), seed=seed)
        except TypeError:
            return self.task_pool.sample(int(n))

    def build_tool_executor(self) -> AsyncToolExecutor | None:
        if self.tool_config is None:
            return None
        return AsyncToolExecutor(self.tool_config)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Environment":
        p = Path(path)
        obj = tomllib.loads(p.read_text(encoding="utf-8"))
        return environment_from_dict(obj)


def environment_from_dict(obj: dict[str, Any]) -> Environment:
    env_id = obj.get("env_id")
    if not isinstance(env_id, str) or not env_id:
        raise ValueError("env_id must be a non-empty string")

    format_type = obj.get("format_type", "harmony")
    if not isinstance(format_type, str) or not format_type:
        raise ValueError("format_type must be a non-empty string")

    pool_cfg = obj.get("task_pool")
    if not isinstance(pool_cfg, dict):
        raise ValueError("missing [task_pool] table")
    task_pool = _build_task_pool(pool_cfg)

    tool_cfg_obj = obj.get("tools", None)
    tool_cfg = None
    if tool_cfg_obj is not None:
        if not isinstance(tool_cfg_obj, dict):
            raise ValueError("[tools] must be a table")
        tool_cfg = _build_tool_config(tool_cfg_obj)

    return Environment(env_id=env_id, task_pool=task_pool, format_type=format_type, tool_config=tool_cfg)


def _build_task_pool(cfg: dict[str, Any]) -> TaskPool:
    pool_type = cfg.get("type", "selfplay")
    if not isinstance(pool_type, str) or not pool_type:
        raise ValueError("task_pool.type must be a non-empty string")

    if pool_type == "mixture":
        from nmoe.rl.tasks.mixture import MixtureSource, MixtureTaskPool

        sources = cfg.get("sources")
        if not isinstance(sources, list) or not all(isinstance(x, dict) for x in sources):
            raise ValueError("task_pool.sources must be a list of tables for type=mixture")

        seed = int(cfg.get("seed", 0))

        built: list[MixtureSource] = []
        for i, scfg_any in enumerate(sources):
            scfg = dict(scfg_any)
            stype = scfg.get("type", None)
            if stype is None:
                # Default: interpret as HF source when a dataset is provided.
                stype = "hf" if "dataset" in scfg else ""
            scfg["type"] = stype

            weight = scfg.pop("weight", 1.0)
            try:
                w = float(weight)
            except Exception as e:
                raise ValueError(f"task_pool.sources[{i}].weight must be a number") from e
            if w <= 0.0:
                raise ValueError(f"task_pool.sources[{i}].weight must be > 0")

            name = scfg.get("name", None)
            if not isinstance(name, str) or not name:
                name = f"source_{i}"

            pool = _build_task_pool(scfg)
            built.append(MixtureSource(name=name, weight=w, pool=pool))

        return MixtureTaskPool(built, seed=seed)  # type: ignore[return-value]

    if pool_type == "selfplay":
        from nmoe.rl.train_agentic import build_selfplay_pool

        task_type = cfg.get("task_type")
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_pool.task_type must be a non-empty string for type=selfplay")

        repo_paths = cfg.get("repo_paths", None)
        if repo_paths is not None and not (isinstance(repo_paths, list) and all(isinstance(x, str) for x in repo_paths)):
            raise ValueError("task_pool.repo_paths must be a list[str] if provided")

        proof_dataset = cfg.get("proof_dataset", None)
        if proof_dataset is not None and not isinstance(proof_dataset, str):
            raise ValueError("task_pool.proof_dataset must be a string if provided")

        workspaces_dir = str(cfg.get("workspaces_dir", "/tmp/nmoe_workspaces"))
        seed = int(cfg.get("seed", 0))
        return build_selfplay_pool(
            task_type=task_type,
            repo_paths=list(repo_paths) if repo_paths is not None else None,
            proof_dataset=proof_dataset,
            workspaces_dir=workspaces_dir,
            seed=seed,
        )

    if pool_type == "static":
        # Uses TaskPool.from_config schema (gsm8k/humaneval loaders).
        return TaskPool.from_config(cfg)

    if pool_type == "hf":
        dataset = cfg.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("task_pool.dataset must be a non-empty string for type=hf")

        split = cfg.get("split", "train")
        if not isinstance(split, str) or not split:
            raise ValueError("task_pool.split must be a non-empty string for type=hf")

        subset = cfg.get("subset", None)
        if subset is not None and not isinstance(subset, str):
            raise ValueError("task_pool.subset must be a string if provided")

        data_files = cfg.get("data_files", None)
        if data_files is not None and not (
            isinstance(data_files, str) or (isinstance(data_files, list) and all(isinstance(x, str) for x in data_files))
        ):
            raise ValueError("task_pool.data_files must be a string or list[str] if provided")

        task_type = cfg.get("task_type", "math")
        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_pool.task_type must be a non-empty string for type=hf")

        problem_field = cfg.get("problem_field", "problem")
        if not isinstance(problem_field, str) or not problem_field:
            raise ValueError("task_pool.problem_field must be a non-empty string for type=hf")

        answer_field = cfg.get("answer_field", "answer")
        if not isinstance(answer_field, str) or not answer_field:
            raise ValueError("task_pool.answer_field must be a non-empty string for type=hf")

        gold_extractor = cfg.get("gold_extractor", "raw")
        if not isinstance(gold_extractor, str) or not gold_extractor:
            raise ValueError("task_pool.gold_extractor must be a non-empty string for type=hf")

        max_examples = int(cfg.get("max_examples", 10_000))
        seed = cfg.get("seed", None)
        seed_i = int(seed) if seed is not None else None

        return TaskPool.from_hf_dataset(
            dataset=dataset,
            split=split,
            subset=subset,
            data_files=data_files,
            streaming=bool(cfg.get("streaming", False)),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            task_type=task_type,
            problem_field=problem_field,
            answer_field=answer_field,
            gold_extractor=gold_extractor,
            max_examples=max_examples,
            seed=seed_i,
        )

    raise ValueError(f"unknown task_pool.type={pool_type!r}")


def _build_tool_config(cfg: dict[str, Any]) -> ToolConfig:
    executor_type = cfg.get("executor_type", "codex_python")
    if not isinstance(executor_type, str) or not executor_type:
        raise ValueError("tools.executor_type must be a non-empty string")

    timeout_default_ms = int(cfg.get("timeout_default_ms", 30000))
    allow_network = bool(cfg.get("allow_network", False))

    cwd = cfg.get("cwd", "")
    if not isinstance(cwd, str):
        raise ValueError("tools.cwd must be a string")

    allowed_paths = cfg.get("allowed_paths", [])
    if not isinstance(allowed_paths, list) or not all(isinstance(x, str) for x in allowed_paths):
        raise ValueError("tools.allowed_paths must be a list[str]")

    return ToolConfig(
        executor_type=executor_type,
        timeout_default_ms=timeout_default_ms,
        allow_network=allow_network,
        cwd=cwd,
        allowed_paths=list(allowed_paths),
    )
