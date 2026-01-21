"""nmoe.rl: opinionated entrypoints for post-training.

This is a thin dispatcher. The source of truth for each mode remains its TOML
config; CLI flags are only for small overrides.

Examples:
  python -m nmoe.rl train <config.toml>
  python -m nmoe.rl verifier <config.toml> --prm_source=prm800k --prm_split=train[:1024]
  torchrun --nproc_per_node=8 -m nmoe.rl tests-dist
"""

from __future__ import annotations

import sys


def _usage() -> None:
    print(
        "Usage: python -m nmoe.rl <command> ...\n\n"
        "Commands:\n"
        "  train        GRPO post-training (R1-Zero style)\n"
        "  verifier     GRPO verifier training (PRM datasets)\n"
        "  selfplay     Math-V2 self-play data generation\n"
        "  tests-dist   8-GPU distributed smoke tests\n"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _usage()
        raise SystemExit(0)

    cmd = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if cmd == "train":
        from nmoe.rl.train import main as _main
        _main()
        return
    if cmd == "verifier":
        from nmoe.rl.train_verifier import main as _main
        _main()
        return
    if cmd == "selfplay":
        from nmoe.rl.selfplay import main as _main
        _main()
        return
    if cmd in {"tests-dist", "tests_dist"}:
        from nmoe.rl.tests_dist import main as _main
        rc = _main()
        raise SystemExit(int(rc))

    _usage()
    raise SystemExit(f"Unknown command: {cmd!r}")


if __name__ == "__main__":
    main()
