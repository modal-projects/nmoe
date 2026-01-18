"""Replay bundle persistence for post-hoc debugging.

Artifacts:
- trajectory_record.json: token-exact transcript + tool spans/outputs
- provenance.json: environment/workspace provenance (or {"present": false})
- failure_summary.json: compact failure/reward summary for aggregation

Design constraints:
- Deterministic sampling (no cross-rank nondeterminism).
- Fail-closed when enabled (write errors should stop training, not add noise).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nmoe.rl.rewards_gdpo import RewardSignals
from nmoe.rl.trajectory_record import TrajectoryRecord


def _safe_name(s: str) -> str:
    """Filesystem-safe identifier (stable, minimal)."""
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:128] if out else "unknown"


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, sort_keys=True, indent=2)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class ReplayBundleWriter:
    """Deterministic replay bundle writer."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        run_id: str,
        sample_every: int,
        seed: int = 0,
        rank: int = 0,
    ):
        if sample_every < 0:
            raise ValueError(f"sample_every must be >= 0 (got {sample_every})")
        self.base_dir = Path(base_dir)
        self.run_id = _safe_name(run_id)
        self.sample_every = int(sample_every)
        self.seed = int(seed)
        self.rank = int(rank)

    def should_write(self, *, step: int, task_id: str, sample_idx: int) -> bool:
        if self.sample_every == 0:
            return False
        if self.sample_every == 1:
            return True
        msg = f"{self.seed}:{self.run_id}:{self.rank}:{step}:{task_id}:{sample_idx}".encode()
        h = hashlib.sha256(msg).digest()
        v = int.from_bytes(h[:8], byteorder="big", signed=False)
        return (v % self.sample_every) == 0

    def write(
        self,
        *,
        step: int,
        task_id: str,
        sample_idx: int,
        record: TrajectoryRecord,
        rewards: RewardSignals | None = None,
        provenance: dict[str, Any] | None = None,
        provenance_path: Path | None = None,
    ) -> Path:
        """Write the replay bundle and return its directory path."""
        tid = _safe_name(task_id)
        bundle_dir = (
            self.base_dir
            / self.run_id
            / f"rank_{self.rank:03d}"
            / f"step_{int(step):08d}"
            / f"{tid}_s{int(sample_idx):02d}"
        )

        traj_path = bundle_dir / "trajectory_record.json"
        prov_path = bundle_dir / "provenance.json"
        fail_path = bundle_dir / "failure_summary.json"

        _atomic_write_json(traj_path, record.to_dict())

        prov_obj: dict[str, Any]
        if provenance is not None:
            prov_obj = dict(provenance)
            prov_obj.setdefault("present", True)
        elif provenance_path is not None and provenance_path.exists():
            prov_obj = _read_json(provenance_path)
            prov_obj.setdefault("present", True)
        else:
            prov_obj = {
                "present": False,
                "task_id": task_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        _atomic_write_json(prov_path, prov_obj)

        # Failure summary is intentionally compact and aggregation-friendly.
        counts: dict[str, int] = {}
        for ev in record.tool_events:
            cat = ev.result.failure_category or "unknown"
            counts[cat] = counts.get(cat, 0) + 1

        summary: dict[str, Any] = {
            "task_id": task_id,
            "step": int(step),
            "sample_idx": int(sample_idx),
            "rank": int(self.rank),
            "tool_failure_categories": counts,
        }
        if rewards is not None:
            summary["rewards"] = rewards.to_dict()
        _atomic_write_json(fail_path, summary)

        return bundle_dir

    def maybe_write(
        self,
        *,
        step: int,
        task_id: str,
        sample_idx: int,
        record: TrajectoryRecord,
        rewards: RewardSignals | None = None,
        provenance: dict[str, Any] | None = None,
        provenance_path: Path | None = None,
    ) -> Path | None:
        if not self.should_write(step=step, task_id=task_id, sample_idx=sample_idx):
            return None
        return self.write(
            step=step,
            task_id=task_id,
            sample_idx=sample_idx,
            record=record,
            rewards=rewards,
            provenance=provenance,
            provenance_path=provenance_path,
        )
