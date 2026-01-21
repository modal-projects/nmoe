from __future__ import annotations

import sys
from pathlib import Path


def test_runtime_adds_third_party_to_sys_path(tmp_path: Path):
    from nmoe import runtime

    third_party = tmp_path / "third_party"
    third_party.mkdir(parents=True, exist_ok=True)

    orig = list(sys.path)
    try:
        runtime._maybe_add_repo_third_party_to_sys_path(repo_root=tmp_path)
        assert sys.path[0] == str(third_party)

        # Idempotent.
        runtime._maybe_add_repo_third_party_to_sys_path(repo_root=tmp_path)
        assert sys.path.count(str(third_party)) == 1
    finally:
        sys.path[:] = orig

