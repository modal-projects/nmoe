from __future__ import annotations

from typing import Mapping

import torch

from nmoe.perl.ldora import LDoRALinear


def validate_optimizer_contract(
  param_groups_or_optimizer,  # torch optimizer or dense_groups list
  adapters: Mapping[str, LDoRALinear],
) -> None:
  """Validate training-harness policy for L/DoRA Mode A.

  This is intentionally strict and fail-fast:
  - Adapter params (A,B) MUST be present in optimizer param groups.
  - Adapter params MUST have weight_decay=0.0.
  - Base weights (W0) MUST NOT be present in optimizer groups (they are frozen).

  This module does not build optimizers; it only validates.
  """
  if not adapters:
    raise ValueError("adapters must be non-empty")

  if hasattr(param_groups_or_optimizer, "param_groups"):
    groups = getattr(param_groups_or_optimizer, "param_groups")
  else:
    groups = param_groups_or_optimizer

  if not isinstance(groups, (list, tuple)):
    raise ValueError("param_groups must be a list/tuple or an optimizer with .param_groups")

  wd_by_param: dict[int, float] = {}
  for gi, group in enumerate(groups):
    try:
      wd = float(group.get("weight_decay", 0.0))
    except Exception:
      raise ValueError(f"optimizer param group {gi} has invalid weight_decay")
    params = group.get("params", [])
    if not isinstance(params, (list, tuple)):
      raise ValueError(f"optimizer param group {gi} params must be a list/tuple")
    for p in params:
      pid = id(p)
      if pid in wd_by_param:
        raise ValueError("optimizer param appears in multiple param groups")
      wd_by_param[pid] = wd

  missing: list[str] = []
  bad_wd: list[tuple[str, float]] = []
  base_in_opt: list[str] = []

  for name, m in adapters.items():
    for pname, p in (("A", m.A), ("B", m.B)):
      wd = wd_by_param.get(id(p), None)
      if wd is None:
        missing.append(f"{name}.{pname}")
      elif wd != 0.0:
        bad_wd.append((f"{name}.{pname}", float(wd)))

    if id(m.weight) in wd_by_param:
      base_in_opt.append(f"{name}.weight")
    if m.bias is not None and id(m.bias) in wd_by_param:
      base_in_opt.append(f"{name}.bias")

  if missing or bad_wd or base_in_opt:
    lines: list[str] = ["PERL optimizer contract violated:"]
    if missing:
      lines.append(f"- missing adapter params: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")
    if bad_wd:
      ex = ", ".join([f"{n} (wd={wd})" for n, wd in bad_wd[:6]])
      lines.append(f"- adapter params must have weight_decay=0.0: {ex}{'...' if len(bad_wd) > 6 else ''}")
    if base_in_opt:
      lines.append(f"- base weights must not be optimized: {', '.join(base_in_opt[:8])}{'...' if len(base_in_opt) > 8 else ''}")
    raise ValueError("\n".join(lines))
