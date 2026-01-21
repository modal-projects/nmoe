from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from nmoe.perl.ldora import LDoRALinear, LDoRAInit


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise ValueError(msg)


@dataclass(frozen=True)
class AdaptedWeight:
  path: str
  shape: tuple[int, int]
  rank: int
  alpha: float


def _set_submodule(root: nn.Module, path: str, new: nn.Module) -> None:
  parts = path.split(".")
  parent = root
  for p in parts[:-1]:
    parent = getattr(parent, p)
    if not isinstance(parent, nn.Module):
      raise RuntimeError(f"bad module path: {path}")
  setattr(parent, parts[-1], new)


def apply_ldora(
  model: nn.Module,
  *,
  rank: int,
  alpha: Optional[float] = None,
  filter_fn: Optional[Callable[[str, nn.Linear], bool]] = None,
  freeze_base: bool = True,
) -> tuple[dict[str, LDoRALinear], list[AdaptedWeight]]:
  """Replace nn.Linear modules with LDoRALinear (Mode A).

  Returns:
    (adapters_by_path, manifest)
  """
  _require(rank > 0, f"rank must be > 0 (got {rank})")
  init = LDoRAInit(rank=int(rank), alpha=alpha)

  adapters: dict[str, LDoRALinear] = {}
  manifest: list[AdaptedWeight] = []

  for name, module in list(model.named_modules()):
    if not isinstance(module, nn.Linear):
      continue
    if filter_fn is not None and not filter_fn(name, module):
      continue
    ld = LDoRALinear.from_linear(module, init=init, freeze_base=freeze_base)
    _set_submodule(model, name, ld)
    adapters[name] = ld
    manifest.append(AdaptedWeight(path=name, shape=tuple(ld.weight.shape), rank=int(rank), alpha=float(ld.init.alpha or rank)))

  _require(adapters, "apply_ldora: matched 0 nn.Linear modules")
  return adapters, manifest
