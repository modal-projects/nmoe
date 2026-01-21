"""Nanochat-style deterministic manifolds for MoE scaling experiments.

Design:
  - One dial: depth (n_layers)
  - Everything else is a deterministic function of depth.
  - Horizon is token-indexed. Different contracts define the horizon in terms of:
      - total params:  T_total(d) = DN * P_total(d)   (atlas/capacity-fair)
      - active params: T_total(d) = DN * P_active(d)  (compute-fair for MoE)
      - tokens/expert: T_total(d) scales with E/K to hold tokens-per-expert fixed

This module is stdlib-only and does not instantiate the model. It provides:
  - a manifold spec for (d -> cfg overrides)
  - analytic param counts (total + active) for planning horizons/checkpoints
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def round_to_multiple(x: int, multiple: int) -> int:
  if multiple <= 0:
    raise ValueError(f"multiple must be > 0 (got {multiple})")
  x = int(x)
  m = int(multiple)
  return ((x + m - 1) // m) * m


def _pow2_int(k: int) -> int:
  if k < 0:
    raise ValueError(f"power must be >= 0 (got {k})")
  return 1 << int(k)


@dataclass(frozen=True)
class MoEManifoldSpec:
  # Dial
  n_layers: int

  # Deterministic functions of depth
  dim: int
  n_heads: int
  head_dim: int
  inter_dim: int
  moe_inter_dim: int
  n_routed_experts: int
  n_shared_experts: int
  n_activated_experts: int
  n_dense_layers: int

  # Planning
  params_total: int
  params_active: int


def build_moe_depth_manifold(
  *,
  n_layers: int,
  k_total: int = 8,
  n_shared_experts: int = 0,
  n_dense_layers: int = 0,
  dim_per_layer: int = 64,
  dim_round_to: int = 128,
  head_dim: int = 128,
  max_experts: int = 4096,
  vocab_size: int = 50304,
  n_routed_experts: int | None = None,
) -> MoEManifoldSpec:
  """Locked manifold spec (dial = n_layers).

  Defaults match the currently agreed "pure routed" manifold:
    E(d) = min(4096, 2^(d-4))
    dim(d) = round_to_multiple(64*d, 128)
    head_dim = 128  => n_heads = dim / 128
    inter_dim = 4*dim
    moe_inter_dim = inter_dim/8  (so K_total*moe_inter_dim = inter_dim)
    K_total = 8, shared=0, dense_prefix=0
  """
  d = int(n_layers)
  if d <= 0:
    raise ValueError(f"n_layers must be > 0 (got {d})")
  if k_total <= 0:
    raise ValueError(f"k_total must be > 0 (got {k_total})")
  s = int(n_shared_experts)
  if s < 0 or s > k_total:
    raise ValueError(f"n_shared_experts must be in [0,{k_total}] (got {s})")
  n_dense = int(n_dense_layers)
  if n_dense < 0 or n_dense > d:
    raise ValueError(f"n_dense_layers must be in [0,{d}] (got {n_dense})")

  dim = round_to_multiple(int(dim_per_layer) * d, int(dim_round_to))
  if dim % int(head_dim) != 0:
    raise ValueError(f"dim must be divisible by head_dim={head_dim} (got dim={dim})")
  n_heads = dim // int(head_dim)
  if n_heads <= 0:
    raise ValueError(f"n_heads must be > 0 (got {n_heads})")

  # E(d) default: min(max_experts, 2^(d-4)), unless explicitly fixed.
  if n_routed_experts is None:
    E = min(int(max_experts), _pow2_int(max(0, d - 4)))
  else:
    E = int(n_routed_experts)
    if E < 0:
      raise ValueError(f"n_routed_experts must be >= 0 (got {E})")
    E = min(int(max_experts), E)

  inter_dim = 4 * dim
  if inter_dim % k_total != 0:
    raise ValueError(f"inter_dim must be divisible by k_total={k_total} (got inter_dim={inter_dim})")
  moe_inter_dim = inter_dim // int(k_total)

  k_routed = int(k_total) - s

  params_total = estimate_total_params(
    vocab_size=int(vocab_size),
    dim=dim,
    n_layers=d,
    n_heads=n_heads,
    inter_dim=inter_dim,
    n_dense_layers=n_dense,
    n_routed_experts=E,
    moe_inter_dim=moe_inter_dim,
    n_shared_experts=s,
  )
  params_active = estimate_active_params(
    vocab_size=int(vocab_size),
    dim=dim,
    n_layers=d,
    n_heads=n_heads,
    inter_dim=inter_dim,
    n_dense_layers=n_dense,
    n_routed_experts=E,
    n_activated_experts=k_routed,
    moe_inter_dim=moe_inter_dim,
    n_shared_experts=s,
  )

  return MoEManifoldSpec(
    n_layers=d,
    dim=dim,
    n_heads=n_heads,
    head_dim=int(head_dim),
    inter_dim=inter_dim,
    moe_inter_dim=moe_inter_dim,
    n_routed_experts=E,
    n_shared_experts=s,
    n_activated_experts=k_routed,
    n_dense_layers=n_dense,
    params_total=int(params_total),
    params_active=int(params_active),
  )


def build_dense_depth_manifold(
  *,
  n_layers: int,
  dim_per_layer: int = 64,
  dim_round_to: int = 128,
  head_dim: int = 128,
  vocab_size: int = 50304,
) -> MoEManifoldSpec:
  """Dense-only manifold spec (all layers use dense FFN).

  Represented as MoEManifoldSpec with:
    - n_dense_layers = n_layers
    - n_routed_experts = 0, n_shared_experts = 0, n_activated_experts = 0
  """
  d = int(n_layers)
  if d <= 0:
    raise ValueError(f"n_layers must be > 0 (got {d})")

  dim = round_to_multiple(int(dim_per_layer) * d, int(dim_round_to))
  if dim % int(head_dim) != 0:
    raise ValueError(f"dim must be divisible by head_dim={head_dim} (got dim={dim})")
  n_heads = dim // int(head_dim)
  if n_heads <= 0:
    raise ValueError(f"n_heads must be > 0 (got {n_heads})")

  inter_dim = 4 * dim
  n_dense_layers = d

  params_total = estimate_total_params(
    vocab_size=int(vocab_size),
    dim=dim,
    n_layers=d,
    n_heads=n_heads,
    inter_dim=inter_dim,
    n_dense_layers=n_dense_layers,
    n_routed_experts=0,
    moe_inter_dim=0,
    n_shared_experts=0,
  )
  params_active = estimate_active_params(
    vocab_size=int(vocab_size),
    dim=dim,
    n_layers=d,
    n_heads=n_heads,
    inter_dim=inter_dim,
    n_dense_layers=n_dense_layers,
    n_routed_experts=0,
    n_activated_experts=0,
    moe_inter_dim=0,
    n_shared_experts=0,
  )

  return MoEManifoldSpec(
    n_layers=d,
    dim=dim,
    n_heads=n_heads,
    head_dim=int(head_dim),
    inter_dim=inter_dim,
    moe_inter_dim=0,
    n_routed_experts=0,
    n_shared_experts=0,
    n_activated_experts=0,
    n_dense_layers=n_dense_layers,
    params_total=int(params_total),
    params_active=int(params_active),
  )


def estimate_total_params(
  *,
  vocab_size: int,
  dim: int,
  n_layers: int,
  n_heads: int,
  inter_dim: int,
  n_dense_layers: int,
  n_routed_experts: int,
  moe_inter_dim: int,
  n_shared_experts: int,
) -> int:
  """Analytic total parameter count for current nmoe Transformer(+MoE) shape.

  Notes:
    - Matches nmoe/model.py semantics: untied embedding + lm_head.
    - Attention counted as 4 dense projections (wq,wk,wv,wo), each [dim,dim].
    - Dense MLP counted as SwiGLU: W1,W3:[dim,inter_dim], W2:[inter_dim,dim] => 3*dim*inter_dim.
    - MoE experts counted across *all* routed experts (total params, not active).
      W1,W3:[E,dim,moe_inter_dim], W2:[E,moe_inter_dim,dim] => 3*E*dim*moe_inter_dim.
    - Router gate is [dim,E] per MoE layer.
    - Shared experts are implemented as a dense MLP with inter_dim = S*moe_inter_dim.
  """
  V = int(vocab_size)
  D = int(dim)
  L = int(n_layers)
  H = int(n_heads)
  if D % H != 0:
    raise ValueError(f"dim must be divisible by n_heads (got dim={D}, n_heads={H})")

  # Embedding + output head (untied).
  p_embed = 2 * V * D

  # Per-layer attention weights: wq,wk,wv,wo each D*D.
  p_attn_per = 4 * D * D
  p_attn = L * p_attn_per

  # FFN weights.
  n_dense = int(n_dense_layers)
  n_moe = L - n_dense
  if n_moe < 0:
    raise ValueError(f"n_dense_layers must be <= n_layers (got {n_dense} > {L})")

  p_dense_ffn_per = 3 * D * int(inter_dim)
  p_dense_ffn = n_dense * p_dense_ffn_per

  E = int(n_routed_experts)
  p_experts_per_layer = 3 * E * D * int(moe_inter_dim)
  p_router_per_layer = D * E
  p_shared_per_layer = 0
  S = int(n_shared_experts)
  if S:
    p_shared_per_layer = 3 * D * (S * int(moe_inter_dim))
  p_moe_ffn = n_moe * (p_experts_per_layer + p_router_per_layer + p_shared_per_layer)

  return int(p_embed + p_attn + p_dense_ffn + p_moe_ffn)


def estimate_active_params(
  *,
  vocab_size: int,
  dim: int,
  n_layers: int,
  n_heads: int,
  inter_dim: int,
  n_dense_layers: int,
  n_routed_experts: int,
  n_activated_experts: int,
  moe_inter_dim: int,
  n_shared_experts: int,
) -> int:
  """Analytic *active* parameter count per token for current nmoe shape.

  Interpretation:
    - "Active params" are the parameters exercised per-token in a forward pass
      (a compute proxy). For MoE this counts only the K routed experts actually
      selected (plus shared experts, if any), not all E experts.

  Notes:
    - Matches estimate_total_params conventions (no biases/norm params).
    - Router gate weights [dim,E] are considered active in MoE layers.
    - Shared experts are counted as a dense MLP with inter_dim = S*moe_inter_dim.
  """
  V = int(vocab_size)
  D = int(dim)
  L = int(n_layers)
  H = int(n_heads)
  if D % H != 0:
    raise ValueError(f"dim must be divisible by n_heads (got dim={D}, n_heads={H})")

  # Embedding + output head (untied): active every token.
  p_embed = 2 * V * D

  # Attention weights: active every token.
  p_attn = L * (4 * D * D)

  # FFN: dense vs MoE layers.
  n_dense = int(n_dense_layers)
  n_moe = L - n_dense
  if n_moe < 0:
    raise ValueError(f"n_dense_layers must be <= n_layers (got {n_dense} > {L})")

  p_dense_ffn = n_dense * (3 * D * int(inter_dim))

  # MoE: active K experts, plus router gate and optional shared experts.
  E = int(n_routed_experts)
  K = int(n_activated_experts)
  S = int(n_shared_experts)
  if E < 0 or K < 0 or S < 0:
    raise ValueError("expert counts must be >= 0")
  if n_moe == 0:
    return int(p_embed + p_attn + p_dense_ffn)

  p_router_per_layer = D * E
  p_active_experts_per_layer = 3 * K * D * int(moe_inter_dim)
  p_shared_per_layer = 0
  if S:
    p_shared_per_layer = 3 * D * (S * int(moe_inter_dim))
  p_moe_active = n_moe * (p_router_per_layer + p_active_experts_per_layer + p_shared_per_layer)

  return int(p_embed + p_attn + p_dense_ffn + p_moe_active)


def tokens_total_from_params(params_total: int, *, coef_tokens_per_param: float) -> int:
  if coef_tokens_per_param <= 0:
    raise ValueError(f"coef_tokens_per_param must be > 0 (got {coef_tokens_per_param})")
  return int(math.ceil(float(coef_tokens_per_param) * float(params_total)))


def steps_for_token_budget(tokens: int, *, tokens_per_step: int) -> int:
  tps = int(tokens_per_step)
  if tps <= 0:
    raise ValueError(f"tokens_per_step must be > 0 (got {tps})")
  return int(math.ceil(int(tokens) / float(tps)))
