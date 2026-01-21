from __future__ import annotations


def test_rl_main_dispatch_help(capsys):
    import sys

    from nmoe.rl.__main__ import main

    argv0 = list(sys.argv)
    try:
        # Simulate `python -m nmoe.rl --help` (argv[1] is the flag).
        sys.argv = ["nmoe.rl", "--help"]
        try:
            main()
        except SystemExit as e:
            assert int(e.code) == 0
        out = capsys.readouterr().out
        assert "Commands:" in out
        assert "train" in out
        assert "verifier" in out
    finally:
        sys.argv = argv0
