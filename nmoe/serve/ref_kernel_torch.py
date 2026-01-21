# SPDX-License-Identifier: Apache-2.0
"""Naive torch implementation of DeepSeek's `inference/kernel.py`.

This exists *only* to make the checkpoint-shipped reference implementation
executable in environments where TileLang/TVM cannot JIT for SM100a (B200).

It is intentionally slow and not used in production code paths.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch


FP8_MAX = 448.0


def _pow2_ceil(x: torch.Tensor) -> torch.Tensor:
  # Match kernel.fast_round_scale semantics: pow2(log2ceil(x)).
  log2 = torch.log2(x)
  return torch.pow(2.0, torch.ceil(log2))


def act_quant(x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
  # Reference asserts contiguity; for a naive correctness backend we accept and
  # normalize to avoid false negatives.
  if not x.is_contiguous():
    x = x.contiguous()
  assert x.size(-1) % block_size == 0, "Last dimension must be divisible by block_size"

  N = x.size(-1)
  M = x.numel() // N
  groups = N // block_size

  x2 = x.view(M, groups, block_size).float()
  amax = x2.abs().amax(dim=-1).clamp(min=1e-4)  # [M, groups]
  scale = amax * (1.0 / FP8_MAX)
  if scale_fmt is not None:
    scale = _pow2_ceil(scale)

  y = (x2 / scale.unsqueeze(-1)).clamp(min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
  y = y.view_as(x.view(M, N)).view_as(x)

  s = scale.to(torch.float32).view(*x.shape[:-1], groups)
  return y, s


def _expand_a_scale(a_s: torch.Tensor, *, K: int, block_size: int = 128) -> torch.Tensor:
  # a_s: [M, K//block]
  M = a_s.size(0)
  groups = K // block_size
  return a_s.view(M, groups, 1).expand(M, groups, block_size).reshape(M, K)


def _expand_b_scale(b_s: torch.Tensor, *, N: int, K: int, block_size: int = 128) -> torch.Tensor:
  # b_s: [N//block, K//block]
  n_groups = N // block_size
  k_groups = K // block_size
  return b_s.view(n_groups, 1, k_groups, 1).expand(n_groups, block_size, k_groups, block_size).reshape(N, K)


@lru_cache(maxsize=256)
def _cache_key(dtype: torch.dtype, device: torch.device, shape: tuple[int, ...]) -> tuple:
  return (str(device), str(dtype), shape)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """Dequantize FP8 weights using block-wise scales.

  Naive torch implementation of DeepSeek's kernel.weight_dequant triton kernel.

  Args:
    x: FP8 weight tensor [M, N]
    s: Scale tensor [M // block_size, N // block_size]
    block_size: Block size for quantization (default 128)

  Returns:
    Dequantized weight tensor in default dtype [M, N]
  """
  if not x.is_contiguous():
    x = x.contiguous()
  if not s.is_contiguous():
    s = s.contiguous()
  assert x.dim() == 2 and s.dim() == 2, "weight_dequant expects 2D tensors"

  M, N = x.size()
  m_blocks = (M + block_size - 1) // block_size
  n_blocks = (N + block_size - 1) // block_size

  # Pad if necessary
  M_pad = m_blocks * block_size
  N_pad = n_blocks * block_size
  if M_pad != M or N_pad != N:
    x_pad = torch.zeros((M_pad, N_pad), device=x.device, dtype=x.dtype)
    x_pad[:M, :N].copy_(x)
    x = x_pad

  # Reshape to blocks: [m_blocks, block_size, n_blocks, block_size]
  x_blocked = x.view(m_blocks, block_size, n_blocks, block_size).float()
  # Scale is [m_blocks, n_blocks], expand to match
  s_expanded = s.view(m_blocks, 1, n_blocks, 1).float()
  # Dequantize
  y = x_blocked * s_expanded
  # Reshape back
  y = y.view(M_pad, N_pad)[:M, :N]
  return y.to(torch.get_default_dtype())


def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
  if not a.is_contiguous():
    a = a.contiguous()
  if not b.is_contiguous():
    b = b.contiguous()
  if not a_s.is_contiguous():
    a_s = a_s.contiguous()
  if not b_s.is_contiguous():
    b_s = b_s.contiguous()

  K = a.size(-1)
  M = a.numel() // K
  N = b.size(0)
  assert K % 128 == 0, "fp8_gemm assumes K divisible by 128"

  k_blocks = K // 128
  n_blocks = (N + 127) // 128
  assert tuple(a_s.view(M, -1).shape) == (M, k_blocks)
  assert tuple(b_s.shape) == (n_blocks, k_blocks)

  # Dequantize A: [M,K]
  a_fp8 = a.view(M, k_blocks, 128).float()
  a_scale = a_s.view(M, k_blocks, 1).float()
  a_deq = (a_fp8 * a_scale).reshape(M, K)

  # Dequantize B with padding on N to full blocks: [N,K]
  if N == n_blocks * 128:
    b_pad = b
  else:
    b_pad = torch.zeros((n_blocks * 128, K), device=b.device, dtype=b.dtype)
    b_pad[:N].copy_(b)
  b_fp8 = b_pad.view(n_blocks, 128, k_blocks, 128).float()
  b_scale = b_s.view(n_blocks, 1, k_blocks, 1).float()
  b_deq = (b_fp8 * b_scale).reshape(n_blocks * 128, K)[:N]

  out = a_deq @ b_deq.t()
  out = out.to(torch.get_default_dtype()).view(*a.size()[:-1], N)
  return out


def fp8_index(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor, k_s: torch.Tensor) -> torch.Tensor:
  # Reference kernel supports (b,m,h,d) @ (b,n,d) with per-token scales.
  if not q.is_contiguous():
    q = q.contiguous()
  if not k.is_contiguous():
    k = k.contiguous()
  if not q_s.is_contiguous():
    q_s = q_s.contiguous()
  if not k_s.is_contiguous():
    k_s = k_s.contiguous()

  # Accept trailing singleton dims produced by act_quant(..., groups=1).
  if q_s.ndim == 4 and q_s.size(-1) == 1:
    q_s = q_s.squeeze(-1)
  if k_s.ndim == 3 and k_s.size(-1) == 1:
    k_s = k_s.squeeze(-1)

  q_f = q.float()  # [B,M,H,D]
  k_f = k.float()  # [B,N,D]

  # logits[b,m,h,n] = dot(k[b,n], q[b,m,h])
  logits = torch.einsum("bnd,bmhd->bmhn", k_f, q_f)
  logits = torch.relu(logits) * q_s.float().unsqueeze(-1)
  logits_sum = logits.sum(dim=2)  # sum over heads -> [B,M,N]
  return logits_sum * k_s.float().unsqueeze(1)
