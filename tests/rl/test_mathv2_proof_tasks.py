"""Math-V2 verifier primitives (Harmony-first)."""

from __future__ import annotations


def test_harmony_mathv2_verifier_task_extracts_final_score_only():
    from nmoe.rl.tasks.proof import ProofVerifierTask

    task = ProofVerifierTask(problem="P", proof="Y", gold_score="0.5")

    # Non-Harmony text -> no extracted score.
    assert task.extract_answer("0.5") is None

    # Harmony with final score.
    resp = (
        "<|start|>assistant<|channel|>analysis<|message|>x<|end|>"
        "<|start|>assistant<|channel|>final<|message|>0.5<|end|>"
    )
    assert task.extract_answer(resp) == "0.5"
    assert task.verify(task.extract_answer(resp)) is True


def test_harmony_mathv2_meta_verifier_task_extracts_final_score_only():
    from nmoe.rl.tasks.proof import ProofMetaVerifierTask

    task = ProofMetaVerifierTask(problem="P", proof="Y", verifier_response="V", gold_meta_score="1")
    resp = (
        "<|start|>assistant<|channel|>analysis<|message|>x<|end|>"
        "<|start|>assistant<|channel|>final<|message|>1.0<|end|>"
    )
    assert task.extract_answer(resp) == "1"
    assert task.verify(task.extract_answer(resp)) is True
