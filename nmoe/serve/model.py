# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-shaped inference model core (DSA + FlashMLA + DeepEP).

This module is intentionally minimal and *kernel-contract* focused:

- Attention: FlashMLA paged MLA on B200 (SM100), which requires:
  - FP8 KV cache (packed 656 bytes/token) and
  - token-level sparse attention via `indices` (DeepSeek Sparse Attention).
- Indices: learned lightning indexer (DSA) computed from per-layer projections:
  - q_idx(x), k_idx(x), w_idx(x)
  - streaming top-k selection (memory O(S * topk), compute O(S^2) for prefill)
- MoE: DeepEP normal dispatch/combine (BF16 expert compute for now).

This file does not implement checkpoint loading/mapping; that belongs in serve
engine/ckpt code. It also does not implement batching policies; it assumes the
caller buckets/pads inputs to satisfy kernel constraints.
"""

from __future__ import annotations

import os
from pathlib import Path

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nmoe.triton.dsa import compute_indexer_scores
from nmoe.triton.flashmla_kv import flashmla_pack_kv_fp8_ue8m0_scatter
from nmoe.serve.kernels.fp8_quant import quantize_fp8_ue8m0, weighted_scatter_add, silu_mul_fp8


def _require(cond: bool, msg: str) -> None:
  if not isinstance(cond, bool):
    raise TypeError(f"_require expects bool, got {type(cond)} (did you pass a Tensor?)")
  if not cond:
    raise RuntimeError(msg)

def _maybe_set_cutlass_path() -> None:
  """Best-effort CUTLASS_PATH discovery for DeepGEMM JIT (container-first, opt-in)."""
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """Dequantize FP8 weight using block-wise scales (from DeepSeek inference code)."""
  shape = weight.shape
  assert weight.dim() == 2
  weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size)
  weight = weight.transpose(1, 2).contiguous().view(-1, block_size * block_size)
  weight = (weight.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
  weight = weight.view(shape[0] // block_size, shape[1] // block_size, block_size, block_size)
  weight = weight.transpose(1, 2).contiguous().view(shape)
  return weight


def _sm100_only(device: torch.device) -> None:
  _require(torch.cuda.is_available(), "CUDA required for FlashMLA (B200 / SM100).")
  major, minor = torch.cuda.get_device_capability(device)
  _require(major == 10, f"FlashMLA MLA serve path targets SM100 only (B200). Got {major}.{minor}.")


@dataclass(frozen=True)
class ModelConfig:
  # Transformer
  vocab_size: int = 129280
  hidden_size: int = 7168
  intermediate_size: int = 18432
  num_layers: int = 61
  num_dense_layers: int = 3  # V3.2-Speciale has 3 dense layers (0, 1, 2)
  num_heads: int = 128

  # Attention type selection:
  # - "dsa": DeepSeek Sparse Attention (for Speciale) - uses FlashMLA with learned sparse indices
  # - "mla": Dense MLA (for DeepSeek-V3-0324, Kimi-K2) - uses CuTeDSL dense attention
  attention_type: Literal["dsa", "mla"] = "dsa"

  # MLA (latent attention)
  q_lora_rank: int = 1536
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128

  # DSA indexer (V3.2) - only used when attention_type="dsa"
  dsa_n_idx_heads: int = 64
  dsa_idx_dim: int = 128
  dsa_topk: int = 2048

  # MoE
  num_experts: int = 256
  num_shared_experts: int = 1
  num_experts_per_tok: int = 8
  num_expert_groups: int = 8
  num_limited_groups: int = 4
  route_scale: float = 2.5
  moe_intermediate_size: int = 2048

  # RoPE / YaRN scaling
  rope_theta: float = 10000.0
  rope_factor: float = 40.0
  max_seq_len: int = 163840
  original_seq_len: int = 4096
  beta_fast: float = 32.0
  beta_slow: float = 1.0
  mscale: float = 1.0


_world_size: int = 1  # EP size (expert parallelism)
_tp_size: int = 1     # TP size (tensor parallelism for attention)
_rank: int = 0


def init_distributed(rank: int, world_size: int, tp_size: int = None) -> None:
  """Initialize distributed settings.

  Args:
    rank: Global rank of this process
    world_size: Total number of processes (used for EP)
    tp_size: Tensor parallelism size for attention. Defaults to world_size.
             Set to 1 for TP=1, EP=N mode (no attention all_reduce).
  """
  global _world_size, _tp_size, _rank
  _world_size = int(world_size)
  _tp_size = int(tp_size) if tp_size is not None else _world_size
  _rank = int(rank)


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = float(eps)
    # PERF: Use bfloat16 to enable fused RMSNorm kernel (same dtype as input)
    self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)


class Linear(nn.Module):
  """BF16 linear layer (for embeddings, norms, etc.)."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    dtype: torch.dtype = torch.bfloat16,
  ) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
    self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight, self.bias)


class FP8Linear(nn.Module):
  """FP8 linear layer using DeepGEMM fp8_gemm_nt."""

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    block_size: int = 128,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.block_size = block_size

    # FP8 weight [out, in]
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
    # Block scales [out_blocks, in_blocks]
    out_blocks = (out_features + block_size - 1) // block_size
    in_blocks = (in_features + block_size - 1) // block_size
    self.weight_scale_inv = nn.Parameter(torch.ones(out_blocks, in_blocks, dtype=torch.float32))
    self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _maybe_set_cutlass_path()
    from deep_gemm import fp8_gemm_nt

    x_flat = x.view(-1, self.in_features)
    # Quantize input to FP8 without host sync (.item()).
    # DeepGEMM expects per-token-group (K//128) UE8M0 (power-of-2) scales.
    block = 128
    _require(self.in_features % block == 0, f"in_features ({self.in_features}) must be divisible by {block}.")
    x_view = x_flat.to(torch.bfloat16).view(x_flat.size(0), self.in_features // block, block)
    # Match DeepSeek / DeepGEMM activation quantization contract: clamp amax >= 1e-4.
    scales = x_view.float().abs().amax(dim=-1).clamp(min=1e-4) / 448.0  # [T, K//128]
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))  # UE8M0
    x_fp8 = (x_view.float() / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).view_as(x_flat)
    x_scale = scales.to(torch.float32)

    # Output tensor
    out = torch.empty(x_flat.size(0), self.out_features, dtype=torch.bfloat16, device=x.device)

    # FP8 GEMM: out = x @ W.T
    fp8_gemm_nt(
      (x_fp8, x_scale),
      (self.weight, self.weight_scale_inv),
      out,
      self.bias,
    )

    return out.view(*x.shape[:-1], self.out_features)


class ColumnParallelLinear(Linear):
  """Column-parallel linear using _tp_size for tensor parallelism."""

  def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
    if _tp_size > 1:
      _require(out_features % _tp_size == 0, f"out_features ({out_features}) must be divisible by tp_size ({_tp_size}).")
      super().__init__(in_features, out_features // _tp_size, **kwargs)
    else:
      super().__init__(in_features, out_features, **kwargs)


class VocabParallelLinear(Linear):
  """Vocab-parallel linear using _world_size for sharding.

  Used for lm_head to keep vocab sharded across GPUs even when _tp_size=1.
  This matches LMSYS's "DP attention" setup: attention TP=1, but lm_head
  still vocab-parallel for efficiency.
  """

  def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
    if _world_size > 1:
      _require(out_features % _world_size == 0, f"out_features ({out_features}) must be divisible by world_size ({_world_size}).")
      super().__init__(in_features, out_features // _world_size, **kwargs)
    else:
      super().__init__(in_features, out_features, **kwargs)


class RowParallelLinear(Linear):
  """Row-parallel linear using _tp_size for tensor parallelism."""

  def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
    if _tp_size > 1:
      _require(in_features % _tp_size == 0, f"in_features ({in_features}) must be divisible by tp_size ({_tp_size}).")
      super().__init__(in_features // _tp_size, out_features, **kwargs)
    else:
      super().__init__(in_features, out_features, **kwargs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, self.weight)
    if _tp_size > 1:
      dist.all_reduce(y)
    if self.bias is not None:
      y = y + self.bias
    return y


class FP8ColumnParallelLinear(FP8Linear):
  """FP8 column-parallel linear using _tp_size for tensor parallelism."""

  def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
    if _tp_size > 1:
      _require(out_features % _tp_size == 0, f"out_features ({out_features}) must be divisible by tp_size ({_tp_size}).")
      super().__init__(in_features, out_features // _tp_size, **kwargs)
    else:
      super().__init__(in_features, out_features, **kwargs)


class FP8RowParallelLinear(FP8Linear):
  """FP8 row-parallel linear using _tp_size for tensor parallelism."""

  def __init__(self, in_features: int, out_features: int, *, reduce_output: bool = True, **kwargs) -> None:
    if _tp_size > 1:
      _require(in_features % _tp_size == 0, f"in_features ({in_features}) must be divisible by tp_size ({_tp_size}).")
      super().__init__(in_features // _tp_size, out_features, **kwargs)
    else:
      super().__init__(in_features, out_features, **kwargs)
    self.reduce_output = reduce_output

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = super().forward(x)
    if _tp_size > 1 and self.reduce_output:
      dist.all_reduce(y)
    return y


def precompute_freqs_cis(cfg: ModelConfig, device: torch.device) -> torch.Tensor:
  """DeepSeek-V3 YaRN-style rotary precompute (must-match)."""
  dim = int(cfg.qk_rope_head_dim)
  seqlen = int(cfg.max_seq_len)
  base = float(cfg.rope_theta)
  factor = float(cfg.rope_factor)

  def find_correction_dim(num_rotations: float, dim_: int, base_: float, max_seq_len: int) -> float:
    return dim_ * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base_))

  def find_correction_range(
    low_rot: float, high_rot: float, dim_: int, base_: float, max_seq_len: int
  ) -> tuple[int, int]:
    low = math.floor(find_correction_dim(low_rot, dim_, base_, max_seq_len))
    high = math.ceil(find_correction_dim(high_rot, dim_, base_, max_seq_len))
    return max(low, 0), min(high, dim_ - 1)

  def linear_ramp_factor(min_: int, max_: int, dim_: int) -> torch.Tensor:
    if min_ == max_:
      max_ += 1
    t = (torch.arange(dim_, dtype=torch.float32, device=device) - float(min_)) / float(max_ - min_)
    return torch.clamp(t, 0.0, 1.0)

  freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
  if seqlen > int(cfg.original_seq_len):
    low, high = find_correction_range(float(cfg.beta_fast), float(cfg.beta_slow), dim, base, int(cfg.original_seq_len))
    smooth = 1.0 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1.0 - smooth) + freqs * smooth

  t = torch.arange(seqlen, device=device, dtype=torch.float32)
  freqs = torch.outer(t, freqs)
  return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
  """Apply rotary to a [..., rope_dim] tensor, where freqs_cis is per-position complex.

  Args:
    x: Input tensor with shape [..., rope_dim]
    freqs_cis: Complex frequency tensor for positions
    interleaved: If True (default), assumes pairs are interleaved (x0,x1,x2,x3,...).
                 If False, assumes pairs are split (x0,x2,x4,...,x1,x3,x5,...).
                 The DSA indexer uses interleaved=False.
  """
  dtype = x.dtype
  shape = x.shape
  if not interleaved:
    # Non-interleaved: reshape to separate even/odd, then transpose to pair them
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
  x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
  # freqs_cis can be [S, D/2] (shared) or [B, S, D/2] (batched)
  if freqs_cis.dim() == 2:
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))  # [1, S, 1, D/2]
  else:
    freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))  # [B, S, 1, D/2]
  y = torch.view_as_real(x * freqs_cis).flatten(3)
  if not interleaved:
    # Convert back to non-interleaved format
    y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
  return y.to(dtype)


# NOTE: DSA code below is duplicated in nmoe/serve/dsa.py
# Once dsa.py is verified working, remove from here.

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
  """Apply Hadamard transform to activations (required for DSA indexer).

  Reference: DeepSeek-V3.2 inference code uses fast_hadamard_transform.
  This is critical for the indexer to work correctly.
  """
  _require(x.dtype == torch.bfloat16, "rotate_activation expects BF16 input")
  from fast_hadamard_transform import hadamard_transform
  hidden_size = x.size(-1)
  return hadamard_transform(x, scale=hidden_size ** -0.5)


def _pack_flashmla_fp8_kv(latent: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
  """Pack (kv_latent, k_rope) into FlashMLA SM100 FP8 KV format.

  Args:
    latent: [T,512] BF16
    rope:   [T, 64] BF16

  Returns:
    packed: [T,656] uint8.
  """
  _require(latent.is_cuda and rope.is_cuda, "KV pack expects CUDA tensors.")
  _require(latent.dtype == torch.bfloat16 and rope.dtype == torch.bfloat16, "KV pack expects BF16 inputs.")
  _require(latent.ndim == 2 and latent.size(1) == 512, f"latent must be [T,512] (got {tuple(latent.shape)}).")
  _require(rope.ndim == 2 and rope.size(1) == 64, f"rope must be [T,64] (got {tuple(rope.shape)}).")
  _require(latent.is_contiguous() and rope.is_contiguous(), "KV pack expects contiguous inputs.")
  T = int(latent.size(0))
  packed = torch.empty((T, 656), device=latent.device, dtype=torch.uint8)
  loc = torch.arange(T, device=latent.device, dtype=torch.int64)
  flashmla_pack_kv_fp8_ue8m0_scatter(latent, rope, loc, packed)
  return packed


def _phys_token_ids(block_table_1d: torch.Tensor, context_len: int) -> torch.Tensor:
  """Build physical token ids [context_len] for one sequence: block_id*64 + offset."""
  _require(block_table_1d.is_cuda and block_table_1d.dtype == torch.int32 and block_table_1d.ndim == 1, "block_table must be CUDA int32 [num_blocks].")
  if context_len <= 0:
    return torch.empty((0,), device=block_table_1d.device, dtype=torch.int32)
  pos = torch.arange(context_len, device=block_table_1d.device, dtype=torch.int64)
  page = torch.div(pos, 64, rounding_mode="floor").to(torch.int64)
  off = (pos % 64).to(torch.int64)
  blk = block_table_1d.index_select(0, page).to(torch.int64)
  return (blk * 64 + off).to(torch.int32)



class DsaFlashMla(nn.Module):
  """One attention layer: DSA indexer + FlashMLA sparse attention."""

  def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
    super().__init__()
    self.layer_idx = int(layer_idx)
    self.hidden_size = int(cfg.hidden_size)
    self.num_heads = int(cfg.num_heads)
    self.num_local_heads = int(cfg.num_heads // _world_size)

    self.q_lora_rank = int(cfg.q_lora_rank)
    self.kv_lora_rank = int(cfg.kv_lora_rank)
    self.qk_nope_head_dim = int(cfg.qk_nope_head_dim)
    self.qk_rope_head_dim = int(cfg.qk_rope_head_dim)
    self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
    self.v_head_dim = int(cfg.v_head_dim)
    # IMPORTANT: Match DeepSeek reference scaling.
    # The MLA kernel consumes a 576-d "latent query" (kv_lora_rank + rope_dim),
    # but the reference model scales attention by 1/sqrt(qk_head_dim=192).
    # Using 1/sqrt(576) changes numerics enough to destroy generation while
    # staying finite. Keep scaling anchored to qk_head_dim.
    self.softmax_scale = (self.qk_head_dim**-0.5)
    if int(cfg.max_seq_len) > int(cfg.original_seq_len):
      mscale = 0.1 * float(cfg.mscale) * math.log(float(cfg.rope_factor)) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

    if self.q_lora_rank > 0:
      self.wq_a = FP8Linear(self.hidden_size, self.q_lora_rank)
      self.q_norm = RMSNorm(self.q_lora_rank)
      self.wq_b = FP8ColumnParallelLinear(self.q_lora_rank, self.num_heads * self.qk_head_dim)
    else:
      self.wq = FP8ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim)

    self.wkv_a = FP8Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
    self.kv_norm = RMSNorm(self.kv_lora_rank)
    self.wkv_b = FP8ColumnParallelLinear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim))
    self.wo = FP8RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size)
    # Cached dequantized wkv_b for absorbed attention (populated on first forward)
    self._wkv_b_dequant: Optional[torch.Tensor] = None

    # DSA indexer projections
    # NOTE: Q_idx shares wq_a with attention Q, then has its own wq_idx (wq_b in checkpoint)
    self.n_idx_heads = int(cfg.dsa_n_idx_heads)
    self.idx_dim = int(cfg.dsa_idx_dim)
    self.topk = int(cfg.dsa_topk)
    _require(self.topk > 0 and self.topk % 64 == 0, f"DSA topk must be >0 and divisible by 64 (got {self.topk}).")
    # wq_idx: from LoRA latent (q_lora_rank) to indexer output (n_idx_heads * idx_dim)
    self.wq_idx = FP8Linear(self.q_lora_rank, self.n_idx_heads * self.idx_dim)
    self.wk_idx = FP8Linear(self.hidden_size, self.idx_dim)
    self.k_norm = nn.LayerNorm(self.idx_dim, dtype=torch.bfloat16)  # LayerNorm for K indexer (has bias)
    # Reference stores weights_proj in BF16 but runs it in FP32 for numerical stability / convenience.
    self.w_idx = Linear(self.hidden_size, self.n_idx_heads, dtype=torch.float32)

  @torch.inference_mode()
  def forward(
    self,
    x: torch.Tensor,                # [B,S,H]
    freqs_cis: torch.Tensor,        # [B,S,rope_dim/2] complex
    *,
    kv_cache: torch.Tensor,         # [num_blocks,64,1,656] uint8
    idx_k_cache: torch.Tensor,      # [num_blocks,64,idx_dim] bf16
    block_table: torch.Tensor,      # [B,max_blocks] int32
    cache_seqlens: torch.Tensor,    # [B] int32
    cache_seqlens_cpu: Optional[list[int]] = None,
    out_loc: torch.Tensor,          # [B,S] int32 physical slot ids (block*64 + off)
    positions: torch.Tensor,        # [B,S] int64 absolute positions (0..cache_len-1)
  ) -> torch.Tensor:
    _sm100_only(x.device)
    _require(x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 3, "x must be CUDA BF16 [B,S,H].")
    B, S, _ = x.shape
    _require(block_table.is_cuda and block_table.dtype == torch.int32 and block_table.shape[0] == B, "block_table must be CUDA int32 [B,max_blocks].")
    _require(cache_seqlens.is_cuda and cache_seqlens.dtype == torch.int32 and cache_seqlens.numel() == B, "cache_seqlens must be CUDA int32 [B].")
    _require(out_loc.is_cuda and out_loc.dtype == torch.int32 and out_loc.shape == (B, S), "out_loc must be CUDA int32 [B,S].")
    # Contract: out_loc must be non-negative (no padding). Enforce without host sync.
    torch._assert_async((out_loc >= 0).all(), "out_loc must be non-negative (no padding in this layer contract).")
    _require(positions.is_cuda and positions.shape == (B, S), "positions must be CUDA [B,S].")
    _require(kv_cache.is_cuda and kv_cache.dtype == torch.uint8 and kv_cache.ndim == 4 and kv_cache.size(1) == 64 and kv_cache.size(2) == 1 and kv_cache.size(3) == 656, "kv_cache must be [num_blocks,64,1,656] uint8.")
    _require(idx_k_cache.is_cuda and idx_k_cache.dtype == torch.bfloat16 and idx_k_cache.ndim == 3 and idx_k_cache.size(1) == 64 and idx_k_cache.size(2) == self.idx_dim, "idx_k_cache must be [num_blocks,64,idx_dim] bf16.")

    # === MLA projections ===
    # Compute Q LoRA latent (shared with DSA indexer Q)
    if self.q_lora_rank > 0:
      q_latent = self.q_norm(self.wq_a(x))  # [B,S,q_lora_rank]
      q = self.wq_b(q_latent)
    else:
      q = self.wq(x)
      q_latent = None
    q = q.view(B, S, self.num_local_heads, self.qk_head_dim)
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs_cis)

    kv = self.wkv_a(x)
    kv_latent, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_latent = self.kv_norm(kv_latent)
    k_rope = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2).contiguous()  # [B,S,64]

    # === Store KV + indexer K into caches ===
    kv_latent2 = kv_latent.reshape(B * S, 512).contiguous()
    k_rope2 = k_rope.reshape(B * S, 64).contiguous()
    loc = out_loc.reshape(B * S).to(torch.int64)
    flashmla_pack_kv_fp8_ue8m0_scatter(
      kv_latent2,
      k_rope2,
      loc,
      kv_cache.view(-1, 656),
    )

    # === Compute sparse indices (DSA) using triton kernel ===
    # Q indexer uses shared q_latent from MLA LoRA
    if q_latent is not None:
      q_idx_all = self.wq_idx(q_latent).view(B, S, self.n_idx_heads, self.idx_dim)
    else:
      # Fallback if no LoRA (shouldn't happen for V3.2)
      q_idx_all = self.wq_idx(x).view(B, S, self.n_idx_heads, self.idx_dim)

    # DSA indexer RoPE: non-interleaved format (reference uses interleaved=False)
    # Split into rope and non-rope parts, apply RoPE to rope part only
    q_idx_pe, q_idx_nope = torch.split(q_idx_all, [self.qk_rope_head_dim, self.idx_dim - self.qk_rope_head_dim], dim=-1)
    q_idx_pe = apply_rotary_emb(q_idx_pe, freqs_cis, interleaved=False)
    q_idx_all = torch.cat([q_idx_pe, q_idx_nope], dim=-1)

    # K indexer: project, normalize, apply RoPE (non-interleaved)
    k_idx_raw = self.k_norm(self.wk_idx(x))  # [B, S, idx_dim]
    k_idx_pe, k_idx_nope = torch.split(k_idx_raw, [self.qk_rope_head_dim, self.idx_dim - self.qk_rope_head_dim], dim=-1)
    k_idx_pe = apply_rotary_emb(k_idx_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
    k_idx_processed = torch.cat([k_idx_pe, k_idx_nope], dim=-1)

    # Apply Hadamard transform (critical for DSA to work correctly)
    q_idx_all = rotate_activation(q_idx_all.to(torch.bfloat16))
    k_idx_processed = rotate_activation(k_idx_processed.to(torch.bfloat16))

    # Store processed K indexer in cache (after RoPE and Hadamard)
    k_idx_new = k_idx_processed.reshape(B * S, self.idx_dim).contiguous()
    idx_k_cache.view(-1, self.idx_dim).index_copy_(0, loc, k_idx_new)

    # Reference: weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
    # Reference: weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    # We use BF16 (no q_scale needed), but softmax_scale is critical for correct scores.
    idx_softmax_scale = self.idx_dim ** -0.5  # 128 ** -0.5 = 0.0884
    w_idx_all = self.w_idx(x.float()).view(B, S, self.n_idx_heads) * (self.n_idx_heads ** -0.5) * idx_softmax_scale

    # FlashMLA sparse path accepts invalid indices as -1 (or >= total_seq_len_kv).
    # Follow sglang/vLLM convention: fill unused entries with -1, never duplicate
    # a real token to pad (duplicates bias the softmax).
    indices = torch.full((B, S, self.topk), -1, device=x.device, dtype=torch.int32)
    _require(
      cache_seqlens_cpu is not None and len(cache_seqlens_cpu) == B,
      "cache_seqlens_cpu must be provided as a CPU list[int] of length B (no GPU scalar reads in hot path).",
    )
    for b in range(B):
      ctx_len = int(cache_seqlens_cpu[b])
      _require(ctx_len > 0, "cache_seqlens must be > 0.")

      # Gather idx_k for this batch's context from paged cache
      bt = block_table[b]
      phys_ids = _phys_token_ids(bt, ctx_len).to(torch.int64)
      k_ctx = idx_k_cache.view(-1, self.idx_dim).index_select(0, phys_ids)  # [ctx_len, D]

      # Prepare inputs for triton kernel: [1, S, H, D], [1, N, D], [1, S, H]
      q_idx = q_idx_all[b : b + 1]
      k_idx = k_ctx.unsqueeze(0)
      w_idx = w_idx_all[b : b + 1]

      # Compute scores using triton kernel (causal=False, we apply our own mask)
      scores = compute_indexer_scores(q_idx, k_idx, w_idx, causal=False)  # [1, S, ctx_len]
      scores = scores.squeeze(0)  # [S, ctx_len]

      # Apply position-based causal mask: key_pos <= query_pos
      q_pos = positions[b].to(torch.int64)  # [S]
      k_pos = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
      causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [S, ctx_len]
      scores = scores.masked_fill(causal_mask, float("-inf"))

      # Select top-k from the masked scores. For early query positions where the
      # causal mask excludes most keys, scores will contain -inf. We must not
      # pass those masked keys to FlashMLA; mark them as -1 (invalid) instead.
      k_sel = min(self.topk, ctx_len)
      vals, topk_logical = scores.topk(k_sel, dim=-1)  # [S, k_sel]

      # FlashMLA expects indices in *physical* KV coordinates (flattened):
      # global_block_id * 64 + offset. Our scores are computed against the
      # gathered context `k_ctx`, whose rows correspond 1:1 with `phys_ids`.
      topk_physical = phys_ids.index_select(0, topk_logical.to(torch.int64).reshape(-1)).view(S, k_sel)
      topk_physical_i32 = topk_physical.to(torch.int32)
      topk_physical_i32 = torch.where(torch.isfinite(vals), topk_physical_i32, torch.full_like(topk_physical_i32, -1))

      indices[b, :, :k_sel] = topk_physical_i32

    # === FlashMLA sparse attention ===
    # Absorb q_nope via wkv_b (first 128 rows) to produce 512-d latent query.
    # Dequantize wkv_b weight once and cache (FP8 -> BF16)
    if self._wkv_b_dequant is None:
      self._wkv_b_dequant = weight_dequant(self.wkv_b.weight, self.wkv_b.weight_scale_inv)
    wkv_b_w = self._wkv_b_dequant.view(self.num_local_heads, -1, self.kv_lora_rank)
    q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_w[:, : self.qk_nope_head_dim])
    q_for_attn = torch.cat([q_nope_abs, q_pe], dim=-1).contiguous()  # [B,S,H,576]

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache  # type: ignore
    # FlashMLA sparse decode kernel schedules work over `topk` indices, not the
    # true cache length. Follow vLLM/sglang convention: use cache_seqlens=topk
    # for metadata construction (and keep invalid indices as -1).
    cache_seqlens_meta = torch.full_like(cache_seqlens, self.topk)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens_meta,
      num_q_tokens_per_head_k=int(S * self.num_local_heads),
      num_heads_k=1,
      num_heads_q=self.num_local_heads,
      is_fp8_kvcache=True,
      topk=self.topk,
    )
    out_latent, _lse = flash_mla_with_kvcache(
      q_for_attn,
      kv_cache,
      block_table,
      cache_seqlens,
      head_dim_v=512,
      tile_scheduler_metadata=tile_scheduler_metadata,
      num_splits=num_splits,
      softmax_scale=float(self.softmax_scale),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )

    # Project latent -> v_head_dim and output projection.
    out = torch.einsum("bshc,hdc->bshd", out_latent, wkv_b_w[:, -self.v_head_dim :])
    return self.wo(out.flatten(2))


class MLP(nn.Module):
  """FP8 dense MLP (for layer 0 and shared_experts)."""

  def __init__(self, hidden: int, inter: int, *, reduce_output: bool = True) -> None:
    super().__init__()
    self.w1 = FP8ColumnParallelLinear(hidden, inter)
    self.w2 = FP8RowParallelLinear(inter, hidden, reduce_output=reduce_output)
    self.w3 = FP8ColumnParallelLinear(hidden, inter)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Reference computes gate*up in float32 for precision
    return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).to(x.dtype))


class MoEGate(nn.Module):
  def __init__(self, cfg: ModelConfig) -> None:
    super().__init__()
    self.num_experts = int(cfg.num_experts)
    self.topk = int(cfg.num_experts_per_tok)
    self.num_groups = int(cfg.num_expert_groups)
    self.topk_groups = int(cfg.num_limited_groups)
    self.route_scale = float(cfg.route_scale)

    # Gate must be numerically stable and deterministic across TP layouts.
    # Match DeepSeek reference: gate linear is computed in FP32.
    self.weight = nn.Parameter(torch.empty(self.num_experts, int(cfg.hidden_size), dtype=torch.float32))
    # DeepSeek-V3 gate bias is present for the 7168-hidden checkpoints; keep the contract explicit.
    self.bias = nn.Parameter(torch.zeros(self.num_experts, dtype=torch.float32)) if int(cfg.hidden_size) == 7168 else None

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(x.float(), self.weight).sigmoid()
    original_scores = scores
    scores_for_choice = scores if self.bias is None else (scores + self.bias)

    if self.num_groups > 1:
      scores_for_choice = scores_for_choice.view(-1, self.num_groups, self.num_experts // self.num_groups)
      # Match DeepSeek reference:
      # - bias absent: group score is max
      # - bias present: group score is top2 sum
      group_scores = (
        scores_for_choice.amax(dim=-1) if self.bias is None else scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
      )
      group_idx = group_scores.topk(self.topk_groups, dim=-1)[1]
      mask = torch.ones_like(group_scores, dtype=torch.bool).scatter_(1, group_idx, False)
      scores_for_choice = scores_for_choice.masked_fill(mask.unsqueeze(-1), float("-inf"))
      scores_for_choice = scores_for_choice.flatten(1)

    indices = scores_for_choice.topk(self.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * self.route_scale
    return weights.to(torch.float32), indices.to(torch.int64)


class MoE(nn.Module):
  """Model-parallel MoE with configurable dispatch strategy.

  Supports two modes:
    - single_node=True: local_mask + all_reduce (faster for single-node NVLink)
    - single_node=False: DeepEP dispatch/combine (required for multi-node RDMA)

  Expert matmuls use DeepGEMM contiguous grouped GEMM, which requires per-expert
  token counts to be padded to M alignment with padding rows marked by
  m_indices == -1.
  """

  def __init__(self, cfg: ModelConfig, buffer, single_node: bool = True) -> None:
    super().__init__()
    self.hidden = int(cfg.hidden_size)
    self.inter = int(cfg.moe_intermediate_size)
    self.num_experts = int(cfg.num_experts)
    self.num_local = self.num_experts // _world_size
    self.experts_start = _rank * self.num_local
    self.experts_end = self.experts_start + self.num_local
    self.gate = MoEGate(cfg)
    self.buffer = buffer
    self.single_node = single_node

    # FP8 expert weights: stacked [num_local, out, in] for grouped GEMM
    # w13 = concat(w1, w3) for gate-up projection: [num_local, 2*inter, hidden]
    # w2 = down projection: [num_local, hidden, inter]
    self.w13 = nn.Parameter(torch.empty(self.num_local, 2 * self.inter, self.hidden, dtype=torch.float8_e4m3fn))
    self.w2 = nn.Parameter(torch.empty(self.num_local, self.hidden, self.inter, dtype=torch.float8_e4m3fn))
    # Block scales (128x128): [num_local, out_blocks, in_blocks]
    self.w13_scale = nn.Parameter(torch.ones(self.num_local, (2 * self.inter) // 128, self.hidden // 128, dtype=torch.float32))
    self.w2_scale = nn.Parameter(torch.ones(self.num_local, self.hidden // 128, self.inter // 128, dtype=torch.float32))

    # Reference uses reduce_output=False for shared experts, with one all-reduce at end of MoE
    self.shared = MLP(self.hidden, int(cfg.num_shared_experts) * self.inter, reduce_output=False)

  def _quantize_act(self, x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-group quantize to FP8 with UE8M0 scales (SM100 requirement).

    Uses fused CUDA kernel for 3x speedup over naive PyTorch.
    """
    T, K = x.shape
    _require(K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size}).")
    _require(x.dtype == torch.bfloat16, f"Expected bfloat16, got {x.dtype}")
    return quantize_fp8_ue8m0(x)

  def forward(self, x: torch.Tensor, low_latency: bool = False) -> torch.Tensor:
    """
    MoE forward with DeepEP dispatch/combine.

    Args:
      x: [B, S, hidden] or [T, hidden] input tensor
      low_latency: Use low-latency dispatch for decode (S=1), normal dispatch for prefill
    """
    from deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    shape = x.shape
    x2 = x.view(-1, self.hidden)
    T = x2.size(0)
    weights, indices = self.gate(x2)  # [T, topk], [T, topk]

    # Degenerate EP case: single-rank runs experts locally (no communication).
    # This is not a fallback; it's the correct EP identity when world_size == 1.
    # PERF: Vectorized implementation - no .item() calls except one for allocation.
    if _world_size == 1:
      K = indices.size(1)
      flat_expert = indices.reshape(-1).to(torch.int64)  # [T*K]
      perm = flat_expert.sort().indices  # group by expert id

      token_ids = torch.arange(T, device=x2.device, dtype=torch.int64).repeat_interleave(K)  # [T*K]
      x_rep = x2.index_select(0, token_ids).index_select(0, perm)  # [T*K, hidden] grouped by expert
      m_indices_sorted = flat_expert.index_select(0, perm)  # [T*K] sorted by expert

      # DeepGEMM grouped GEMM requires 128-aligned token counts per expert.
      # PERF: Vectorized padding - compute all indices on GPU, single .item() for alloc.
      M_ALIGN = 128
      expert_counts = torch.bincount(m_indices_sorted.to(torch.int64), minlength=self.num_local)
      aligned_counts = ((expert_counts + M_ALIGN - 1) // M_ALIGN) * M_ALIGN

      # Compute destination offsets using cumsum
      dst_offsets = torch.zeros(self.num_local + 1, dtype=torch.int64, device=x2.device)
      dst_offsets[1:] = aligned_counts.cumsum(0)

      # Source offsets (where each expert's tokens start in sorted array)
      src_offsets = torch.zeros(self.num_local + 1, dtype=torch.int64, device=x2.device)
      src_offsets[1:] = expert_counts.cumsum(0)

      # ONE sync for allocation size
      total_padded = int(dst_offsets[-1].item())

      # Compute per-token destination index vectorized
      N = m_indices_sorted.shape[0]
      exp = m_indices_sorted  # [N]
      token_src_offset = torch.arange(N, device=x2.device, dtype=torch.int64) - src_offsets[exp]
      dst_indices = dst_offsets[exp] + token_src_offset  # [N]

      # Allocate padded tensors
      x_padded = torch.zeros(total_padded, self.hidden, device=x2.device, dtype=x_rep.dtype)
      m_indices = torch.full((total_padded,), -1, device=x2.device, dtype=torch.int32)
      real_mask = torch.zeros(total_padded, device=x2.device, dtype=torch.bool)

      # Scatter data to padded positions (vectorized - no Python for-loop)
      x_padded[dst_indices] = x_rep
      m_indices[dst_indices] = m_indices_sorted.to(torch.int32)
      real_mask[dst_indices] = True

      x_rep_q, x_rep_scale = self._quantize_act(x_padded.to(torch.bfloat16))
      gateup_out = torch.empty(total_padded, 2 * self.inter, device=x2.device, dtype=torch.bfloat16)
      m_grouped_fp8_gemm_nt_contiguous(
        (x_rep_q, x_rep_scale),
        (self.w13, self.w13_scale),
        gateup_out,
        m_indices,
      )
      gate, up = gateup_out.chunk(2, dim=-1)
      # Fused SiLU * up -> FP8 (no intermediate bf16)
      down_in_q, down_in_scale = silu_mul_fp8(gate, up)
      down_out = torch.empty(total_padded, self.hidden, device=x2.device, dtype=torch.bfloat16)
      m_grouped_fp8_gemm_nt_contiguous(
        (down_in_q, down_in_scale),
        (self.w2, self.w2_scale),
        down_out,
        m_indices,
      )

      # Extract real (non-padding) outputs and restore original order
      down_out_real = down_out[real_mask]  # [T*K, hidden]
      inv = torch.empty_like(perm)
      inv.scatter_(0, perm, torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype))
      out_rep = down_out_real.index_select(0, inv).view(T, K, self.hidden)
      out = (out_rep * weights.to(out_rep.dtype).unsqueeze(-1)).sum(dim=1)
      return (out + self.shared(x2)).view(shape)

    # ==========================================================================
    # Single-node TP path: local_mask + all_reduce (faster than DeepEP for NVLink)
    #
    # IMPORTANT: This requires that all ranks execute the same token batch
    # (i.e. TP-style execution). For TP=1 "each GPU does its own thing", this is
    # invalid and we must use DeepEP dispatch/combine instead.
    # ==========================================================================
    if self.single_node and _tp_size > 1:
      M_ALIGN = 128
      K = indices.size(1)

      # Filter to local experts only
      local_mask = (indices >= self.experts_start) & (indices < self.experts_end)  # [T, topk]
      sel = local_mask.nonzero(as_tuple=False)  # [N, 2] (token_id, k_idx)
      N = sel.shape[0]

      y = torch.zeros((T, self.hidden), device=x2.device, dtype=torch.bfloat16)

      if N > 0:
        token_ids = sel[:, 0].to(torch.int64)
        k_idx = sel[:, 1].to(torch.int64)
        local_expert = (indices[token_ids, k_idx] - self.experts_start).to(torch.int64)
        pair_w = weights[token_ids, k_idx].to(torch.float32)
        x_rep = x2.index_select(0, token_ids)

        # Sort by expert for grouped GEMM
        perm = local_expert.sort().indices
        local_expert = local_expert.index_select(0, perm)
        x_rep = x_rep.index_select(0, perm)
        token_ids = token_ids.index_select(0, perm)
        pair_w = pair_w.index_select(0, perm)

        # Pad to M_ALIGN per expert (vectorized)
        expert_counts = torch.bincount(local_expert, minlength=self.num_local)
        aligned_counts = ((expert_counts + M_ALIGN - 1) // M_ALIGN) * M_ALIGN
        dst_offsets = torch.zeros(self.num_local + 1, dtype=torch.int64, device=x2.device)
        dst_offsets[1:] = aligned_counts.cumsum(0)
        src_offsets = torch.zeros(self.num_local + 1, dtype=torch.int64, device=x2.device)
        src_offsets[1:] = expert_counts.cumsum(0)

        total_padded = int(dst_offsets[-1].item())
        exp = local_expert
        token_src_offset = torch.arange(N, device=x2.device, dtype=torch.int64) - src_offsets[exp]
        dst_indices = dst_offsets[exp] + token_src_offset

        x_padded = torch.zeros(total_padded, self.hidden, device=x2.device, dtype=torch.bfloat16)
        m_indices = torch.full((total_padded,), -1, device=x2.device, dtype=torch.int32)
        real_mask = torch.zeros(total_padded, device=x2.device, dtype=torch.bool)
        token_ids_padded = torch.full((total_padded,), -1, device=x2.device, dtype=torch.int64)
        pair_w_padded = torch.zeros((total_padded,), device=x2.device, dtype=torch.float32)

        x_padded[dst_indices] = x_rep.to(torch.bfloat16)
        m_indices[dst_indices] = local_expert.to(torch.int32)
        real_mask[dst_indices] = True
        token_ids_padded[dst_indices] = token_ids
        pair_w_padded[dst_indices] = pair_w

        # Gate-up GEMM
        x_q, x_scale = self._quantize_act(x_padded)
        gateup_out = torch.empty(total_padded, 2 * self.inter, device=x2.device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
          (x_q, x_scale), (self.w13, self.w13_scale), gateup_out, m_indices,
        )
        gate, up = gateup_out.chunk(2, dim=-1)
        # Fused SiLU * up -> FP8 (no intermediate bf16)
        down_in_q, down_in_scale = silu_mul_fp8(gate, up)
        down_out = torch.empty(total_padded, self.hidden, device=x2.device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
          (down_in_q, down_in_scale), (self.w2, self.w2_scale), down_out, m_indices,
        )

        # Accumulate weighted outputs (SM100-optimized fused kernel)
        token_ids_real = token_ids_padded[real_mask]
        pair_w_real = pair_w_padded[real_mask]  # Keep as float32 for kernel
        expert_out_real = down_out[real_mask].contiguous()
        weighted_scatter_add(expert_out_real, token_ids_real, pair_w_real, y)

      y = y + self.shared(x2)

      if _tp_size > 1:
        dist.all_reduce(y)
      return y.view(shape)

    # ==========================================================================
    # DeepEP dispatch/combine for multi-node expert parallelism
    # ==========================================================================
    # Prefill: normal (high-throughput) dispatch/combine
    # Decode: low-latency dispatch/combine
    M_ALIGN = 128  # DeepGEMM alignment requirement

    # Low-latency dispatch requires the DeepEP Buffer to be constructed with
    # low_latency_mode=True (NVSHMEM/QPs + RDMA buffer layout).
    # If the buffer isn't configured for it, fall back to normal dispatch for
    # correctness (single-node NVLink-only bringup).
    if low_latency and bool(getattr(self.buffer, "low_latency_mode", False)):
      # ========== LOW-LATENCY PATH (decode) ==========
      # Pre-allocated buffers, no CPU sync, CUDA-graph compatible
      num_max_tokens = max(T, 256)  # Ensure minimum buffer size

      (recv_x_fp8, recv_x_scales), recv_count, handle, _, _ = self.buffer.low_latency_dispatch(
        x2.to(torch.bfloat16),
        indices,
        num_max_dispatch_tokens_per_rank=num_max_tokens,
        num_experts=self.num_experts,
        use_fp8=True,
        round_scale=True,
        use_ue8m0=True,  # SM100 requirement
      )
      # recv_x_fp8: [num_local, max_tokens * world, hidden] fp8
      # recv_x_scales: [num_local, max_tokens * world, hidden // 512] packed ue8m0
      # recv_count: [num_local] actual token counts

      # Build m_indices for grouped GEMM (per-expert tokens are contiguous)
      # Each expert e has recv_count[e] valid tokens at the start of its slice
      max_recv = recv_x_fp8.shape[1]
      total_padded = self.num_local * max_recv

      # Flatten to [num_local * max_recv, hidden] for grouped GEMM
      recv_x_flat = recv_x_fp8.view(total_padded, self.hidden)
      recv_scales_flat = recv_x_scales.view(total_padded, -1)

      # m_indices: expert ID for each slot, -1 for padding
      m_indices = torch.arange(self.num_local, device=x2.device, dtype=torch.int32)
      m_indices = m_indices.unsqueeze(1).expand(-1, max_recv).reshape(-1)  # [num_local * max_recv]

      # Mask out padding slots (where slot index >= recv_count for that expert)
      slot_idx = torch.arange(max_recv, device=x2.device).unsqueeze(0).expand(self.num_local, -1)
      valid_mask = slot_idx < recv_count.unsqueeze(1)  # [num_local, max_recv]
      m_indices = torch.where(valid_mask.view(-1), m_indices, torch.full_like(m_indices, -1))

      # Gate-up GEMM
      gateup_out = torch.empty(total_padded, 2 * self.inter, device=x2.device, dtype=torch.bfloat16)
      m_grouped_fp8_gemm_nt_contiguous(
        (recv_x_flat, recv_scales_flat),
        (self.w13, self.w13_scale),
        gateup_out,
        m_indices,
      )
      gate, up = gateup_out.chunk(2, dim=-1)
      down_in = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

      # Down GEMM
      down_in_q, down_in_scale = self._quantize_act(down_in)
      down_out = torch.empty(total_padded, self.hidden, device=x2.device, dtype=torch.bfloat16)
      m_grouped_fp8_gemm_nt_contiguous(
        (down_in_q, down_in_scale),
        (self.w2, self.w2_scale),
        down_out,
        m_indices,
      )

      # Reshape back to [num_local, max_recv, hidden] for combine
      expert_out = down_out.view(self.num_local, max_recv, self.hidden)

      # Low-latency combine
      y, _, _ = self.buffer.low_latency_combine(
        expert_out,
        indices,
        weights,
        handle,
      )

    else:
      # ========== NORMAL PATH (prefill) ==========
      # High-throughput dispatch/combine with FP8 dispatch (50% less bandwidth)

      # Quantize BEFORE dispatch to reduce communication bandwidth
      x2_fp8, x2_scale = self._quantize_act(x2.to(torch.bfloat16))

      # First compute the dispatch layout
      num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = self.buffer.get_dispatch_layout(
        indices,  # [T, topk] expert indices
        self.num_experts,
      )

      # Dispatch FP8 tokens to expert ranks (half the bandwidth of BF16)
      (recv_x_fp8, recv_x_scale), recv_topk_idx, recv_topk_weights, counts_list, handle, _ = self.buffer.dispatch(
        (x2_fp8, x2_scale),  # FP8 dispatch
        num_tokens_per_rank=num_tokens_per_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        topk_idx=indices,
        topk_weights=weights,
        expert_alignment=M_ALIGN,
      )
      # recv_x_fp8: [num_recv, hidden] FP8 - received tokens sorted by expert
      # recv_x_scale: [num_recv, hidden//128] - scales
      # counts_list: list[int] - aligned token count per local expert

      num_recv = recv_x_fp8.shape[0]
      if num_recv > 0:
        # Build m_indices from counts_list
        # counts_list[e] = aligned count for expert e
        m_indices = torch.empty(num_recv, device=x2.device, dtype=torch.int32)
        offset = 0
        for e, cnt in enumerate(counts_list):
          m_indices[offset:offset + cnt] = e
          offset += cnt

        # Gate-up GEMM (already have FP8 from dispatch)
        gateup_out = torch.empty(num_recv, 2 * self.inter, device=x2.device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
          (recv_x_fp8, recv_x_scale),
          (self.w13, self.w13_scale),
          gateup_out,
          m_indices,
        )
        gate, up = gateup_out.chunk(2, dim=-1)
        # Fused SiLU * up -> FP8 (no intermediate bf16)
        down_in_q, down_in_scale = silu_mul_fp8(gate, up)
        expert_out = torch.empty(num_recv, self.hidden, device=x2.device, dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_contiguous(
          (down_in_q, down_in_scale),
          (self.w2, self.w2_scale),
          expert_out,
          m_indices,
        )
      else:
        expert_out = torch.empty(0, self.hidden, device=x2.device, dtype=torch.bfloat16)

      # Combine results back to original token order
      y, _, _ = self.buffer.combine(
        expert_out,
        handle,
        topk_weights=recv_topk_weights,
      )

    # Shared experts (no dispatch needed - same on all ranks)
    y = y + self.shared(x2)
    return y.view(shape)


class TransformerBlock(nn.Module):
  def __init__(self, cfg: ModelConfig, layer_idx: int, buffer, single_node: bool = True) -> None:
    super().__init__()
    self.attention_type = cfg.attention_type
    self.attn_norm = RMSNorm(int(cfg.hidden_size))

    if cfg.attention_type == "dsa":
      from nmoe.serve.dsa import DSA
      self.attn = DSA(cfg, layer_idx)
    else:
      from nmoe.serve.mla import MLA
      self.attn = MLA(cfg, layer_idx)

    self.ffn_norm = RMSNorm(int(cfg.hidden_size))

    if layer_idx < int(cfg.num_dense_layers):
      self.ffn = MLP(int(cfg.hidden_size), int(cfg.intermediate_size))
      self.is_moe = False
    else:
      self.ffn = MoE(cfg, buffer, single_node=single_node)
      self.is_moe = True

  def forward(
    self,
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    # DSA cache (used when attention_type="dsa")
    kv_cache: Optional[torch.Tensor] = None,       # [num_blocks,64,1,656] uint8
    idx_k_cache: Optional[torch.Tensor] = None,    # [num_blocks,64,idx_dim] bf16
    # MLA cache (used when attention_type="mla")
    kv_cache_latent: Optional[torch.Tensor] = None,  # [num_pages, page_size, 512] bf16
    kv_cache_rope: Optional[torch.Tensor] = None,    # [num_pages, page_size, 64] bf16
    # Common args
    block_table: torch.Tensor = None,
    cache_seqlens: torch.Tensor = None,
    cache_seqlens_cpu: Optional[list[int]] = None,
    out_loc: torch.Tensor = None,
    positions: Optional[torch.Tensor] = None,
    prefill_mode: Optional[str] = None,  # MLA: "dense" | "paged" | None (decode)
  ) -> torch.Tensor:
    if self.attention_type == "dsa":
      x = x + self.attn(
        self.attn_norm(x),
        freqs_cis,
        kv_cache=kv_cache,
        idx_k_cache=idx_k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=cache_seqlens_cpu,
        out_loc=out_loc,
        positions=positions,
      )
    else:
      x = x + self.attn(
        self.attn_norm(x),
        freqs_cis,
        kv_cache_latent=kv_cache_latent,
        kv_cache_rope=kv_cache_rope,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        out_loc=out_loc,
        positions=positions,
        prefill_mode=prefill_mode,
      )
    x_norm = self.ffn_norm(x)
    # MoE uses low_latency dispatch for decode (prefill_mode=None means decode)
    if self.is_moe:
      low_latency = (prefill_mode is None)
      x = x + self.ffn(x_norm, low_latency=low_latency)
    else:
      x = x + self.ffn(x_norm)
    return x


class DeepSeekV3(nn.Module):
  """DeepSeek-V3 model core for serving.

  Supports two attention modes:
  - DSA (attention_type="dsa"): DeepSeek Sparse Attention for Speciale
  - MLA (attention_type="mla"): Dense MLA for DeepSeek-V3-0324, Kimi-K2

  Args:
    single_node: If True, use local_mask + all_reduce (faster for single-node NVLink).
                 If False, use DeepEP dispatch/combine (required for multi-node RDMA).
  """

  def __init__(self, cfg: ModelConfig, buffer, single_node: bool = True) -> None:
    super().__init__()
    self.cfg = cfg
    self.attention_type = cfg.attention_type
    self.embed = nn.Embedding(int(cfg.vocab_size), int(cfg.hidden_size), dtype=torch.bfloat16)
    self.layers = nn.ModuleList([TransformerBlock(cfg, i, buffer, single_node=single_node) for i in range(int(cfg.num_layers))])
    self.norm = RMSNorm(int(cfg.hidden_size))
    # Reference uses float32 for lm_head
    # Use VocabParallelLinear to keep vocab sharded even when tp_size=1
    self.lm_head = VocabParallelLinear(int(cfg.hidden_size), int(cfg.vocab_size), dtype=torch.float32)
    self.register_buffer("freqs_cis", None, persistent=False)

  def _ensure_freqs(self, device: torch.device) -> None:
    if self.freqs_cis is None or self.freqs_cis.device != device:
      self.freqs_cis = precompute_freqs_cis(self.cfg, device)

  @torch.inference_mode()
  def forward(
    self,
    input_ids: torch.Tensor,        # [B,S] int64
    positions: torch.Tensor,        # [B,S] int64 absolute positions
    *,
    # DSA caches (attention_type="dsa")
    kv_caches: Optional[list[torch.Tensor]] = None,      # per layer: [num_blocks,64,1,656] uint8
    idx_k_caches: Optional[list[torch.Tensor]] = None,   # per layer: [num_blocks,64,idx_dim] bf16
    # MLA caches (attention_type="mla")
    kv_caches_latent: Optional[list[torch.Tensor]] = None,  # per layer: [num_pages, page_size, 512] bf16
    kv_caches_rope: Optional[list[torch.Tensor]] = None,    # per layer: [num_pages, page_size, 64] bf16
    # Common args
    block_table: torch.Tensor = None,          # [B,max_blocks] int32
    cache_seqlens: torch.Tensor = None,        # [B] int32
    cache_seqlens_cpu: Optional[list[int]] = None,
    out_loc: torch.Tensor = None,              # [B,S] int32 physical slot ids for these tokens
    prefill_mode: Optional[str] = None,        # MLA: "dense" | "paged" | None (decode)
  ) -> torch.Tensor:
    _require(input_ids.ndim == 2, "input_ids must be [B,S].")
    B, S = input_ids.shape
    _require(positions.shape == (B, S), "positions must be [B,S].")
    _require(out_loc.shape == (B, S), "out_loc must be [B,S].")

    if self.attention_type == "dsa":
      _require(kv_caches is not None and len(kv_caches) == int(self.cfg.num_layers),
               "kv_caches must provide one cache per layer for DSA.")
      _require(idx_k_caches is not None and len(idx_k_caches) == int(self.cfg.num_layers),
               "idx_k_caches must provide one indexer cache per layer for DSA.")
    else:
      _require(kv_caches_latent is not None and len(kv_caches_latent) == int(self.cfg.num_layers),
               "kv_caches_latent must provide one cache per layer for MLA.")
      _require(kv_caches_rope is not None and len(kv_caches_rope) == int(self.cfg.num_layers),
               "kv_caches_rope must provide one cache per layer for MLA.")

    self._ensure_freqs(input_ids.device)
    freqs = self.freqs_cis.index_select(0, positions.reshape(-1).to(torch.int64)).reshape(B, S, -1)

    x = self.embed(input_ids)
    for i, layer in enumerate(self.layers):
      if self.attention_type == "dsa":
        x = layer(
          x,
          freqs,
          kv_cache=kv_caches[i],
          idx_k_cache=idx_k_caches[i],
          block_table=block_table,
          cache_seqlens=cache_seqlens,
          cache_seqlens_cpu=cache_seqlens_cpu,
          out_loc=out_loc,
          positions=positions,
        )
      else:
        x = layer(
          x,
          freqs,
          kv_cache_latent=kv_caches_latent[i],
          kv_cache_rope=kv_caches_rope[i],
          block_table=block_table,
          cache_seqlens=cache_seqlens,
          out_loc=out_loc,
          positions=positions,
          prefill_mode=prefill_mode,
        )
    x = self.norm(x)
    # Reference uses float32 for logits computation
    logits = self.lm_head(x.float())
    # NOTE: logits are vocab-sharded across tensor-parallel ranks (ColumnParallelLinear).
    # The serve Engine is responsible for all-gathering shards when required by the
    # output mode / sampling policy.
    return logits
