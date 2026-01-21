"""Replay bundle persistence tests (no GPU required)."""

from __future__ import annotations

import json
from pathlib import Path


def test_replay_bundle_writes_three_files(tmp_path: Path):
    from nmoe.rl.replay_bundle import ReplayBundleWriter
    from nmoe.rl.rewards_gdpo import RewardSignals
    from nmoe.rl.trajectory_record import TrajectoryRecord

    writer = ReplayBundleWriter(base_dir=tmp_path, run_id="run123", sample_every=1, seed=0, rank=3)
    assert writer.should_write(step=0, task_id="t", sample_idx=0) is True

    record = TrajectoryRecord(
        prompt_tokens=[1, 2],
        tokens=[1, 2, 3],
        tool_events=[],
    )
    rewards = RewardSignals(answer_correct=1.0, struct_proper_nesting=1.0)

    out_dir = writer.write(step=7, task_id="task/with:badchars", sample_idx=2, record=record, rewards=rewards)
    assert out_dir.exists()

    traj = out_dir / "trajectory_record.json"
    prov = out_dir / "provenance.json"
    fail = out_dir / "failure_summary.json"
    assert traj.exists() and prov.exists() and fail.exists()

    traj_obj = json.loads(traj.read_text(encoding="utf-8"))
    assert traj_obj["prompt_tokens"] == [1, 2]
    assert traj_obj["tokens"] == [1, 2, 3]

    prov_obj = json.loads(prov.read_text(encoding="utf-8"))
    assert prov_obj["present"] is False

    fail_obj = json.loads(fail.read_text(encoding="utf-8"))
    assert fail_obj["rank"] == 3
    assert fail_obj["step"] == 7
    assert fail_obj["tool_failure_categories"] == {}
    assert fail_obj["rewards"]["answer_correct"] == 1.0


def test_replay_bundle_uses_provenance_path_if_present(tmp_path: Path):
    from nmoe.rl.replay_bundle import ReplayBundleWriter
    from nmoe.rl.trajectory_record import TrajectoryRecord

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".nmoe_workspace.json").write_text(json.dumps({"base_sha": "a", "present": True}), encoding="utf-8")

    writer = ReplayBundleWriter(base_dir=tmp_path / "out", run_id="run", sample_every=1)
    out_dir = writer.write(
        step=0,
        task_id="t",
        sample_idx=0,
        record=TrajectoryRecord(prompt_tokens=[1], tokens=[1], tool_events=[]),
        provenance_path=ws / ".nmoe_workspace.json",
    )
    prov_obj = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert prov_obj["base_sha"] == "a"
    assert prov_obj["present"] is True


def test_replay_bundle_maybe_write_respects_sampling(tmp_path: Path):
    from nmoe.rl.replay_bundle import ReplayBundleWriter
    from nmoe.rl.trajectory_record import TrajectoryRecord

    writer = ReplayBundleWriter(base_dir=tmp_path, run_id="run", sample_every=2, seed=0, rank=0)
    record = TrajectoryRecord(prompt_tokens=[1], tokens=[1], tool_events=[])

    wrote = 0
    for i in range(20):
        out = writer.maybe_write(step=0, task_id="t", sample_idx=i, record=record)
        wrote += 1 if out is not None else 0

    # Deterministic but not necessarily exactly 10; should be non-trivial and stable.
    assert 0 < wrote < 20
