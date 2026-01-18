from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def _require(cond: bool, msg: str) -> None:
  if not cond:
    raise ValueError(msg)


@dataclass(frozen=True)
class LDoRAInit:
  """L/DoRA initialization contract (v0).

  - Mode A: g is frozen at init (g0 = ||W0|| rowwise).
  - LoRA: B0 = 0 and A0 is rank-independent.
  """

  rank: int
  alpha: Optional[float] = None  # default: alpha=rank -> scale=1
  eps: float = 1e-12
  a_init_std_scale: float = 1.0  # A0 std = scale / sqrt(in_features)

  @property
  def scale(self) -> float:
    r = int(self.rank)
    _require(r > 0, f"rank must be > 0 (got {r})")
    alpha = float(self.alpha) if self.alpha is not None else float(r)
    return alpha / float(r)


class LDoRALinear(nn.Module):
  """Linear layer with L/DoRA (DoRA + LoRA), Mode A.

  Parameterization (per output row i):
    V = W0 + (alpha/r) * (B @ A)
    W = g0 * V / ||V||,  with g0 := ||W0|| and g0 frozen.

  Notes:
  - Bias is supported as a frozen base parameter (added after DoRA scaling).
  - This is a reference implementation: correct gradients, CPU-testable, and
    shaped to be swappable with future kernelized paths.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    init: LDoRAInit,
    bias: bool = False,
    weight_dtype: torch.dtype = torch.bfloat16,
    adapter_dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    _require(in_features > 0, f"in_features must be > 0 (got {in_features})")
    _require(out_features > 0, f"out_features must be > 0 (got {out_features})")
    _require(init.rank > 0, f"rank must be > 0 (got {init.rank})")
    _require(init.eps >= 0.0, f"eps must be >= 0 (got {init.eps})")
    _require(adapter_dtype == torch.bfloat16, "adapter_dtype must be BF16 (wgrad contract).")

    self.in_features = int(in_features)
    self.out_features = int(out_features)
    self.init = init

    # Base weight (W0). In PERL this will be frozen by the caller.
    self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=weight_dtype))

    # Optional bias (b0). If present, it is part of the frozen base module.
    if bias:
      self.bias = nn.Parameter(torch.empty(self.out_features, dtype=weight_dtype))
    else:
      self.register_parameter("bias", None)

    # LoRA params: ΔW = (alpha/r) * (B @ A)
    r = int(init.rank)
    self.A = nn.Parameter(torch.empty(r, self.in_features, dtype=adapter_dtype))
    self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=adapter_dtype))

    # DoRA magnitude (Mode A): g0 := ||W0||, frozen.
    self.register_buffer("g0", torch.empty(self.out_features, dtype=torch.float32), persistent=True)

    self.reset_parameters()

  @classmethod
  def from_linear(
    cls,
    linear: nn.Linear,
    *,
    init: LDoRAInit,
    freeze_base: bool = True,
  ) -> "LDoRALinear":
    _require(isinstance(linear, nn.Linear), f"expected nn.Linear (got {type(linear)})")
    m = cls(
      linear.in_features,
      linear.out_features,
      init=init,
      bias=(linear.bias is not None),
      weight_dtype=linear.weight.dtype,
      adapter_dtype=torch.bfloat16,
    )
    # Preserve device placement when patching an already-moved model.
    m = m.to(device=linear.weight.device)
    with torch.no_grad():
      m.weight.copy_(linear.weight)
      if linear.bias is not None and m.bias is not None:
        m.bias.copy_(linear.bias)
      m._reset_g0_from_weight()
    if freeze_base:
      m.weight.requires_grad_(False)
      if m.bias is not None:
        m.bias.requires_grad_(False)
    return m

  def reset_parameters(self) -> None:
    # Base weight: caller usually overwrites from a pretrained model.
    nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)
    if self.bias is not None:
      nn.init.zeros_(self.bias)
    self._reset_g0_from_weight()

    # LoRA init: B0=0, A0 rank-independent.
    a_std = float(self.init.a_init_std_scale) / math.sqrt(float(self.in_features))
    nn.init.normal_(self.A, mean=0.0, std=a_std)
    nn.init.zeros_(self.B)

  @torch.no_grad()
  def _reset_g0_from_weight(self) -> None:
    # g0 := ||W0|| rowwise, computed in FP32 for stability.
    w = self.weight.detach().float()
    self.g0.copy_(torch.linalg.vector_norm(w, ord=2, dim=1))

  def lora_scale(self) -> float:
    return float(self.init.scale)

  def _v_and_v_norm(
    self,
    *,
    eps: Optional[float] = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return V (float32) and ||V|| (float32) without materializing ΔW."""
    eps_f = float(self.init.eps if eps is None else eps)
    W0 = self.weight.float()
    A = self.A.float()
    B = self.B.float()
    s = float(self.lora_scale())

    # V = W0 + s * (B@A)
    V = torch.addmm(W0, B, A, beta=1.0, alpha=s)
    v_norm = torch.linalg.vector_norm(V, ord=2, dim=1).clamp_min(eps_f)
    return V, v_norm

  def effective_weight(self) -> torch.Tensor:
    """Materialize effective DoRA weight W in float32 (for tests/merge checks)."""
    V, v_norm = self._v_and_v_norm()
    scale = (self.g0 / v_norm).unsqueeze(1)
    return V * scale

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _require(x.size(-1) == self.in_features, "shape mismatch: last dim must be in_features")

    # Base path: x @ W0^T (optionally using fused bias epilogue).
    #
    # Note: for BF16, F.linear(..., bias=...) may fuse bias in the GEMM epilogue
    # and apply the final cast after adding bias. To preserve init equivalence
    # with nn.Linear(bias=True) while keeping bias outside DoRA scaling, we use:
    #   out_v = (linear(x; W0,b0) - b0) in float32, then apply DoRA, then +b0.
    if self.bias is None:
      out_v = F.linear(x, self.weight)
      out_dtype = out_v.dtype
    else:
      out_with_bias = F.linear(x, self.weight, self.bias)
      out_dtype = out_with_bias.dtype
      bias_f = self.bias.float()
      out_v = out_with_bias.float() - bias_f

    # LoRA path: x @ (s * (B@A))^T = s * ((x @ A^T) @ B^T)
    z = F.linear(x.to(dtype=self.A.dtype), self.A)  # [..., r]
    delta = F.linear(z, self.B)  # [..., out]
    if self.bias is None:
      out_v = out_v + (delta.to(dtype=out_v.dtype) * float(self.lora_scale()))
    else:
      out_v = out_v + (delta.float() * float(self.lora_scale()))

    # DoRA scaling: W = g0 * V / ||V||  ⇒  y = (x @ V^T) * (g0/||V||) rowwise
    _, v_norm = self._v_and_v_norm()
    if self.bias is None:
      row_scale = (self.g0 / v_norm).to(dtype=out_v.dtype)
      return out_v * row_scale

    row_scale_f = (self.g0 / v_norm)
    out = out_v * row_scale_f
    out = out + bias_f
    return out.to(dtype=out_dtype)
