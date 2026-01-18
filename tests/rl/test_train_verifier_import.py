from __future__ import annotations


def test_train_verifier_importable():
    import nmoe.rl.train_verifier as tv

    assert hasattr(tv, "train")
    assert hasattr(tv, "main")
    assert hasattr(tv, "_build_optimizers")
    assert hasattr(tv, "_require_harmony_tokens")
