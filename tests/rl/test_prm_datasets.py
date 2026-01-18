from __future__ import annotations

import os

import pytest


def test_overall_score_mapping():
    from nmoe.rl.tasks.prm_datasets import StepwisePRMExample, to_verifier_task

    ex_all = StepwisePRMExample(prompt="p", steps=["a"], step_labels=[1.0], raw={})
    assert to_verifier_task(ex_all).gold_score == "1"

    ex_none = StepwisePRMExample(prompt="p", steps=["a"], step_labels=[0.0], raw={})
    assert to_verifier_task(ex_none).gold_score == "0"

    ex_mix = StepwisePRMExample(prompt="p", steps=["a", "b"], step_labels=[1.0, 0.0], raw={})
    assert to_verifier_task(ex_mix).gold_score == "0.5"


def test_iter_hf_stepwise_prm_schema_variants():
    from nmoe.rl.tasks.prm_datasets import iter_hf_stepwise_prm

    ds = [
        {"prompt": "p", "completions": ["a"], "labels": [True]},
        {"pompt": "p2", "completions": ["a", "b"], "labels": [1.0, 0.0]},
        {"prompt": "", "completions": ["a"], "labels": [1.0]},
        {"prompt": "p3", "completions": "nope", "labels": [1.0]},
    ]

    def _fake_load_dataset(_name, split, streaming):
        _ = (split, streaming)
        return list(ds)

    # Monkeypatch via assignment to keep this test dependency-free.
    # The adapter only needs "for row in ds:" behavior.
    orig = None
    try:
        from datasets import load_dataset as orig  # type: ignore
    except Exception:
        orig = None

    try:
        import datasets  # type: ignore

        datasets.load_dataset = _fake_load_dataset  # type: ignore[attr-defined]
        out = list(iter_hf_stepwise_prm("x", split="train", streaming=False))
    finally:
        if orig is not None:
            datasets.load_dataset = orig  # type: ignore[attr-defined]

    assert len(out) == 2
    assert out[0].prompt == "p"
    assert out[0].steps == ["a"]
    assert out[0].step_labels == [1.0]
    assert out[1].prompt == "p2"


def test_prm800k_math_shepherd_smoke_external():
    if os.environ.get("NMOE_RUN_HF_PRM_SMOKE") != "1":
        pytest.skip("set NMOE_RUN_HF_PRM_SMOKE=1 to run HuggingFace PRM dataset smoke test")

    from nmoe.rl.tasks.prm_datasets import iter_math_shepherd, iter_prm800k, to_verifier_task

    ex0 = next(iter_prm800k(split="train[:1]"))
    task0 = to_verifier_task(ex0)
    assert task0.problem.strip()
    assert task0.proof.strip()
    assert task0.gold_score in {"0", "0.5", "1"}

    ex1 = next(iter_math_shepherd(split="train[:1]"))
    task1 = to_verifier_task(ex1)
    assert task1.problem.strip()
    assert task1.proof.strip()
    assert task1.gold_score in {"0", "0.5", "1"}


# =============================================================================
# PRMTaskPool tests
# =============================================================================


def test_prm_task_pool_deterministic_sampling():
    """PRMTaskPool sampling is deterministic with fixed seed."""
    from nmoe.rl.tasks.prm_datasets import PRMTaskPool
    from nmoe.rl.tasks.proof import ProofVerifierTask

    # Create synthetic tasks
    tasks = [
        ProofVerifierTask(problem=f"P{i}", proof=f"proof{i}", gold_score=["0", "0.5", "1"][i % 3])
        for i in range(100)
    ]

    pool1 = PRMTaskPool(tasks, seed=42)
    pool2 = PRMTaskPool(tasks, seed=42)

    # Same seed → same samples
    batch1 = pool1.sample(10)
    batch2 = pool2.sample(10)
    assert [t.problem for t in batch1] == [t.problem for t in batch2]

    # Different seed → different samples
    pool3 = PRMTaskPool(tasks, seed=99)
    batch3 = pool3.sample(10)
    assert [t.problem for t in batch1] != [t.problem for t in batch3]


def test_prm_task_pool_stratified_sampling():
    """Stratified sampling balances across score classes."""
    from nmoe.rl.tasks.prm_datasets import PRMTaskPool
    from nmoe.rl.tasks.proof import ProofVerifierTask

    # Create imbalanced tasks: mostly score=1
    tasks = []
    for i in range(10):
        tasks.append(ProofVerifierTask(problem=f"bad{i}", proof="x", gold_score="0"))
    for i in range(10):
        tasks.append(ProofVerifierTask(problem=f"mid{i}", proof="x", gold_score="0.5"))
    for i in range(80):
        tasks.append(ProofVerifierTask(problem=f"good{i}", proof="x", gold_score="1"))

    pool = PRMTaskPool(tasks, seed=42)
    dist = pool.score_distribution()
    assert dist["0"] == 10
    assert dist["0.5"] == 10
    assert dist["1"] == 80

    # Stratified: should be more balanced
    batch = pool.sample(30, stratified=True)
    scores = [t.gold_score for t in batch]
    # Round-robin: 10 of each
    assert scores.count("0") == 10
    assert scores.count("0.5") == 10
    assert scores.count("1") == 10


def test_prm_task_pool_empty():
    """Empty pool returns empty samples."""
    from nmoe.rl.tasks.prm_datasets import PRMTaskPool

    pool = PRMTaskPool([], seed=0)
    assert len(pool) == 0
    assert pool.sample(10) == []


def test_prm_task_pool_from_hf_smoke_external():
    """PRMTaskPool.from_hf loads real data (opt-in)."""
    if os.environ.get("NMOE_RUN_HF_PRM_SMOKE") != "1":
        pytest.skip("set NMOE_RUN_HF_PRM_SMOKE=1 to run HuggingFace PRM dataset smoke test")

    from nmoe.rl.tasks.prm_datasets import PRMTaskPool

    pool = PRMTaskPool.from_hf(source="prm800k", max_examples=100, seed=42, split="train[:100]")
    assert len(pool) == 100

    batch = pool.sample(10)
    assert len(batch) == 10
    for t in batch:
        assert t.problem.strip()
        assert t.proof.strip()
        assert t.gold_score in {"0", "0.5", "1"}

    # Verify score distribution is populated
    dist = pool.score_distribution()
    assert sum(dist.values()) == 100


def test_prm_task_pool_sample_step_labels_deterministic():
    from nmoe.rl.tasks.prm_datasets import PRMTaskPool, StepwisePRMExample, to_verifier_task

    exs = [
        StepwisePRMExample(prompt="p0", steps=["a", "b", "c"], step_labels=[1.0, 1.0, 1.0], raw={}),
        StepwisePRMExample(prompt="p1", steps=["a", "b", "c"], step_labels=[1.0, 0.0, 1.0], raw={}),
    ]
    tasks = [to_verifier_task(e) for e in exs]
    p0 = PRMTaskPool(tasks, seed=123, examples=exs)
    p1 = PRMTaskPool(tasks, seed=123, examples=exs)

    s0 = p0.sample_step_labels(20, stop_at_first_incorrect=True)
    s1 = p1.sample_step_labels(20, stop_at_first_incorrect=True)
    assert s0 == s1
    assert all(1 <= x.step_idx <= len(x.steps) for x in s0)
    # For ex with first incorrect at step 2, stop_at_first_incorrect forbids idx>2.
    for x in s0:
        if x.prompt == "p1":
            assert x.step_idx <= 2
