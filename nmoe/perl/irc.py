from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from nmoe.perl.ldora import LDoRALinear


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise ValueError(msg)


def _q(x: torch.Tensor, q: float) -> torch.Tensor:
  """Conservative quantile (kthvalue) for 1D tensors."""
  _require(x.ndim == 1, f"expected 1D tensor (got shape={tuple(x.shape)})")
  _require(0.0 <= q <= 1.0, f"q must be in [0,1] (got {q})")
  n = int(x.numel())
  if n == 0:
    raise ValueError("quantile on empty tensor")
  # Smallest k such that k/n >= q (i.e., 'higher' style).
  k0 = int(math.ceil(q * n)) - 1
  k0 = max(0, min(n - 1, k0))
  return x.kthvalue(k0 + 1).values


@dataclass(frozen=True)
class IrcThresholds:
  rho_warn: float = math.log(1.25)
  rho_abort: float = math.log(1.50)
  delta_frac_warn: float = 0.10
  delta_frac_abort: float = 0.25
  radial_frac_warn: float = 0.10
  radial_frac_abort: float = 0.25


@dataclass(frozen=True)
class IrcPerMatrix:
  rho_q99: float
  delta_frac_q99: float
  radial_frac_q99: float


@dataclass(frozen=True)
class IrcSummary:
  # Contract scalars: max_W q99_rows(...)
  rho: float
  delta_frac: float
  radial_frac: float

  # Per-matrix breakdown (diagnostic)
  per_matrix: Mapping[str, IrcPerMatrix]


@torch.no_grad()
def _rowwise_irc(module: LDoRALinear, *, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return (rho, delta_frac, radial_frac) per output row (float32)."""
  # Efficient per-row IRC without materializing V or ΔW.
  #
  # V = W0 + s * (B@A) where s = alpha/r.
  # Let Δ_i = s * B_i A (row i of B is length-r, A is [r, in]).
  #
  # We need:
  #   ||Δ_i||^2 = s^2 * B_i (A A^T) B_i^T
  #   <W0_i, Δ_i> = s * (W0_i A^T) · B_i^T
  #   ||V_i||^2 = ||W0_i||^2 + 2 <W0_i,Δ_i> + ||Δ_i||^2
  #
  # Mode A: g = g0 = ||W0|| rowwise and frozen.
  W0 = module.weight.detach()
  A = module.A.detach()
  B = module.B.detach()
  g0 = module.g0.detach().float()
  s = float(module.lora_scale())
  eps_f = float(eps)

  # ||Δ||^2 via quadratic form B (A A^T) B^T, rowwise.
  A_f = A.float()
  B_f = B.float()
  G = A_f @ A_f.T  # [r, r]
  BG = B_f @ G  # [out, r]
  delta_norm_sq = (BG * B_f).sum(dim=1).clamp_min(0.0) * (s * s)

  # <W0,Δ> via (W0 A^T) · B, rowwise.
  U = (W0 @ A.T).float()  # [out, r]
  cross = (U * B_f).sum(dim=1) * s

  # ||V||^2 = ||W0||^2 + 2 cross + ||Δ||^2.
  v_norm_sq = (g0 * g0) + (2.0 * cross) + delta_norm_sq
  v_norm = torch.sqrt(v_norm_sq.clamp_min(eps_f * eps_f))

  delta_norm = torch.sqrt(delta_norm_sq)
  denom = (g0 + eps_f)

  # rho := log(g/||V||) - log(g0/||W0||). With Mode A g=g0 and g0=||W0||,
  # baseline term is 0; rho reduces to log(||W0||/||V||).
  rho = torch.log((g0 + eps_f) / (v_norm + eps_f))
  delta_frac = delta_norm / denom
  radial_frac = (v_norm - g0).abs() / denom
  return rho, delta_frac, radial_frac


@torch.no_grad()
def compute_irc_summary(
  modules: Mapping[str, LDoRALinear],
  *,
  eps: float = 1e-12,
) -> IrcSummary:
  """Compute contract IRC scalars over a pinned module population.

  Contract aggregation:
    IRC_* := max_{W in population} q99_rows(metric(W))
  """
  _require(modules, "modules must be non-empty")
  per: dict[str, IrcPerMatrix] = {}
  rho_max = 0.0
  delta_max = 0.0
  radial_max = 0.0
  for name, m in modules.items():
    rho, delta_frac, radial_frac = _rowwise_irc(m, eps=eps)
    rho_q99 = float(_q(rho.abs(), 0.99).item())
    delta_q99 = float(_q(delta_frac, 0.99).item())
    radial_q99 = float(_q(radial_frac, 0.99).item())
    per[name] = IrcPerMatrix(rho_q99=rho_q99, delta_frac_q99=delta_q99, radial_frac_q99=radial_q99)
    rho_max = max(rho_max, rho_q99)
    delta_max = max(delta_max, delta_q99)
    radial_max = max(radial_max, radial_q99)
  return IrcSummary(rho=rho_max, delta_frac=delta_max, radial_frac=radial_max, per_matrix=per)
