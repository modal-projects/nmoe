"""TaskPool that mixes multiple sub-pools by weight.

This is used for curriculum-style mixtures (difficulty/domain) while keeping
the trainer surface unchanged: `Environment.sample(n, seed=...)`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def _u64(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8, person=b"nmoe.rl.mix")
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little", signed=False)


def _sample_one(pool: Any, *, seed: int) -> list:
    try:
        return pool.sample(1, seed=seed)
    except TypeError:
        return pool.sample(1)


@dataclass(frozen=True)
class MixtureSource:
    name: str
    weight: float
    pool: Any


class MixtureTaskPool:
    """Mixes multiple pools by weight with deterministic per-call seeding."""

    def __init__(self, sources: list[MixtureSource], *, seed: int | None = None):
        if not sources:
            raise ValueError("MixtureTaskPool requires at least one source")
        for s in sources:
            if not isinstance(s.name, str) or not s.name:
                raise ValueError("MixtureTaskPool source.name must be a non-empty string")
            if not (isinstance(s.weight, int | float) and float(s.weight) > 0.0):
                raise ValueError("MixtureTaskPool source.weight must be > 0")
        self._sources = list(sources)
        self._seed = int(seed) if seed is not None else 0

        # Fail fast if any component pool is empty (common config error).
        for s in self._sources:
            try:
                if len(s.pool) == 0:
                    raise ValueError(f"MixtureTaskPool source {s.name!r} is empty")
            except TypeError:
                # Some pools may not implement __len__; allow, but sampling must work.
                pass

    def __len__(self) -> int:
        total = 0
        for s in self._sources:
            try:
                total += int(len(s.pool))
            except TypeError:
                pass
        return total

    def sample(self, n: int, replace: bool = True, *, seed: int | None = None) -> list:
        _ = replace  # sampling is with replacement across sources
        n_i = int(n)
        if n_i <= 0:
            return []

        call_seed = int(seed) if seed is not None else self._seed
        total_w = sum(float(s.weight) for s in self._sources)

        out: list = []
        for i in range(n_i):
            r = (_u64(call_seed, i, "pick") / 2**64) * total_w
            cum = 0.0
            chosen = self._sources[-1]
            for s in self._sources:
                cum += float(s.weight)
                if r < cum:
                    chosen = s
                    break

            t_seed = _u64(call_seed, i, chosen.name, "task")
            got = _sample_one(chosen.pool, seed=t_seed)
            if not got:
                raise RuntimeError(f"MixtureTaskPool source {chosen.name!r} returned no task")
            out.append(got[0])

        return out

