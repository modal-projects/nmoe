from __future__ import annotations


def test_replay_sample_every_from_rate():
    from nmoe.rl.selfplay import _replay_sample_every_from_rate

    assert _replay_sample_every_from_rate(0.0) == 0
    assert _replay_sample_every_from_rate(-1.0) == 0
    assert _replay_sample_every_from_rate(1.0) == 1
    assert _replay_sample_every_from_rate(0.1) == 10
    assert _replay_sample_every_from_rate(0.01) == 100

