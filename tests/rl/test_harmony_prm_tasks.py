from __future__ import annotations


def test_harmony_verifier_score_task_extract_verify():
    from nmoe.rl.tasks.harmony_prm import HarmonyVerifierScoreTask

    t = HarmonyVerifierScoreTask(problem="p", solution="s", gold_score="0.5")
    resp = (
        "<|start|>assistant<|channel|>analysis<|message|>x<|end|>"
        "<|start|>assistant<|channel|>final<|message|>0.5<|end|>"
    )
    assert t.extract_answer(resp) == "0.5"
    assert t.verify("0.5") is True
    assert t.verify("1") is False


def test_harmony_prm_step_label_task_extract_verify():
    from nmoe.rl.tasks.harmony_prm import HarmonyPRMStepLabelTask

    t = HarmonyPRMStepLabelTask(problem="p", steps=["a"], step_idx=1, gold_label="1")
    resp = "<|start|>assistant<|channel|>final<|message|>1<|end|>"
    assert t.extract_answer(resp) == "1"
    assert t.verify("1") is True
    assert t.verify("0") is False

