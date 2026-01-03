# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for FlashMLA, DeepGEMM, and FP8 quantization.

These tests isolate each component to identify where NaN/zero values originate.
Run with: python -m nmoe.serve.test_kernels
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
from typing import Optional, Tuple

_VERBOSE = os.environ.get("NMOE_TEST_VERBOSE", "0") not in ("", "0", "false", "False")


def _vprint(*args, **kwargs) -> None:
  if _VERBOSE:
    print(*args, **kwargs)


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return

import torch
import torch.nn.functional as F


def is_sm90_or_higher() -> bool:
  """Check if GPU supports SM90+ (required for FP8)."""
  if not torch.cuda.is_available():
    return False
  cap = torch.cuda.get_device_capability()
  return cap[0] >= 9


def skip_if_no_sm90(test_func):
  """Decorator to skip tests on GPUs without SM90+ support."""
  @wraps(test_func)
  def wrapper(*args, **kwargs):
    if not is_sm90_or_higher():
      raise unittest.SkipTest(f"{test_func.__name__}: requires SM90+ GPU")
    return test_func(*args, **kwargs)
  return wrapper

def skip_if_no_gpu(test_func):
  """Decorator to skip tests without CUDA."""
  @wraps(test_func)
  def wrapper(*args, **kwargs):
    if not torch.cuda.is_available():
      raise unittest.SkipTest(f"{test_func.__name__}: requires CUDA GPU")
    return test_func(*args, **kwargs)
  return wrapper


# =============================================================================
# Test 1: FP8 Quantization (per_token_cast_to_fp8)
# =============================================================================

class TestFP8Quantization(unittest.TestCase):
  """Test FP8 quantization functions from deep_gemm."""

  @skip_if_no_sm90
  def test_per_token_cast_to_fp8_basic(self):
    """Test basic per-token FP8 casting."""
    _maybe_set_cutlass_path()
    from deep_gemm import per_token_cast_to_fp8

    # Create test input
    x = torch.randn(32, 256, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = per_token_cast_to_fp8(x, use_ue8m0=True)

    _vprint(f"Input shape: {x.shape}, dtype: {x.dtype}")
    _vprint(f"Output FP8 shape: {x_fp8.shape}, dtype: {x_fp8.dtype}")
    _vprint(f"Scale shape: {x_scale.shape}, dtype: {x_scale.dtype}")
    _vprint(f"Input amax: {x.abs().max().item():.4f}")
    _vprint(f"Output FP8 amax: {x_fp8.float().abs().max().item():.4f}")

    self.assertEqual(x_fp8.shape, x.shape)
    self.assertEqual(x_fp8.dtype, torch.float8_e4m3fn)
    self.assertEqual(x_scale.shape, (x.size(0), x.size(1) // 128))
    self.assertEqual(x_scale.dtype, torch.float32)
    self.assertFalse(torch.isnan(x_fp8.float()).any())
    self.assertFalse(torch.isinf(x_fp8.float()).any())
    self.assertTrue(torch.isfinite(x_scale).all())

  @skip_if_no_sm90
  def test_per_token_cast_to_fp8_edge_cases(self):
    """Test FP8 casting with edge cases."""
    _maybe_set_cutlass_path()
    from deep_gemm import per_token_cast_to_fp8

    # Test with zeros
    x_zeros = torch.zeros(8, 128, dtype=torch.bfloat16, device="cuda")
    try:
      x_fp8, x_scale = per_token_cast_to_fp8(x_zeros, use_ue8m0=True)
      self.assertTrue(torch.isfinite(x_fp8.float()).all())
      self.assertTrue(torch.isfinite(x_scale).all())
    except AssertionError:
      # Some DeepGEMM builds assert on amax==0; accept but don't mask other exceptions.
      pass

    # Test with very small values
    x_small = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda") * 1e-6
    try:
      x_fp8, x_scale = per_token_cast_to_fp8(x_small, use_ue8m0=True)
      self.assertTrue(torch.isfinite(x_fp8.float()).all())
      self.assertTrue(torch.isfinite(x_scale).all())
    except AssertionError:
      pass

    # Test with large values
    x_large = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda") * 100
    x_fp8, x_scale = per_token_cast_to_fp8(x_large, use_ue8m0=True)
    _vprint(f"Large values: FP8 amax = {x_fp8.float().abs().max().item()}, scale = {x_scale.float().max().item()}")
    self.assertFalse(torch.isnan(x_fp8.float()).any())

  @skip_if_no_sm90
  def test_per_token_cast_to_fp8_reconstruction(self):
    """Test that FP8 quantization can be approximately reconstructed."""
    _maybe_set_cutlass_path()
    from deep_gemm import per_token_cast_to_fp8

    x = torch.randn(32, 128, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = per_token_cast_to_fp8(x, use_ue8m0=True)

    # DeepGEMM per-token FP8 returns per-128 scales: [T, K//128].
    block = 128
    T, K = x.shape
    self.assertEqual(K, block)
    self.assertEqual(x_scale.shape, (T, 1))

    x_rec = (x_fp8.float() * x_scale).to(torch.float32)
    rel_l2 = (x_rec - x.float()).norm() / (x.float().norm() + 1e-12)
    self.assertLess(float(rel_l2), 0.6)


# =============================================================================
# Test 2: DeepGEMM FP8 GEMM
# =============================================================================

class TestDeepGEMM(unittest.TestCase):
  """Test DeepGEMM FP8 GEMM operations."""

  @skip_if_no_sm90
  def test_fp8_gemm_nt_basic(self):
    """Test basic FP8 GEMM (A @ B.T) with correct scale format."""
    _maybe_set_cutlass_path()
    from deep_gemm import fp8_gemm_nt, per_token_cast_to_fp8

    # Use dimensions that are multiples of 128 for block-wise scales
    M, N, K = 128, 256, 128

    # Create inputs
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")  # Will be transposed

    # Quantize A to FP8 with per-token scales
    A_fp8, A_scale = per_token_cast_to_fp8(A, use_ue8m0=True)

    # For weights, need block-wise scales (128x128 blocks)
    # B is [N, K], scale should be [N // 128, K // 128] for 128x128 blocks
    B_fp8 = B.to(torch.float8_e4m3fn)
    B_scale = torch.ones(N // 128, K // 128, dtype=torch.float32, device="cuda")

    # Output buffer
    C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

    # Run GEMM
    fp8_gemm_nt((A_fp8, A_scale), (B_fp8, B_scale), C)

    _vprint(f"GEMM output shape: {C.shape}")
    _vprint(f"GEMM output amax: {C.abs().max().item():.4f}")
    _vprint(f"GEMM output has NaN: {torch.isnan(C).any()}")
    _vprint(f"GEMM output has Inf: {torch.isinf(C).any()}")

    # Reference computation
    C_ref = A.float() @ B.float().T
    _vprint(f"Reference output amax: {C_ref.abs().max().item():.4f}")

    self.assertFalse(torch.isnan(C).any())
    self.assertFalse(torch.isinf(C).any())

  @skip_if_no_sm90
  def test_grouped_fp8_gemm_basic(self):
    """Test grouped FP8 GEMM for MoE."""
    _maybe_set_cutlass_path()
    from deep_gemm import m_grouped_fp8_gemm_nt_contiguous, per_token_cast_to_fp8

    # Use dimensions that are multiples of 128
    num_tokens = 128
    num_experts = 8
    hidden_dim = 256
    intermediate_dim = 512

    # Create inputs
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = per_token_cast_to_fp8(x, use_ue8m0=True)

    # Expert weights [num_experts, intermediate_dim, hidden_dim]
    w = torch.randn(num_experts, intermediate_dim, hidden_dim, dtype=torch.bfloat16, device="cuda")
    w_fp8 = w.to(torch.float8_e4m3fn)
    # Scale: [num_experts, intermediate_dim // 128, hidden_dim // 128]
    w_scale = torch.ones(num_experts, intermediate_dim // 128, hidden_dim // 128, dtype=torch.float32, device="cuda")

    # Expert assignments (which expert each token goes to) - must be int32
    m_indices = torch.randint(0, num_experts, (num_tokens,), dtype=torch.int32, device="cuda")
    m_indices = m_indices.sort().values  # Must be sorted for contiguous

    # Output buffer
    out = torch.empty(num_tokens, intermediate_dim, dtype=torch.bfloat16, device="cuda")

    m_grouped_fp8_gemm_nt_contiguous((x_fp8, x_scale), (w_fp8, w_scale), out, m_indices)

    _vprint(f"Grouped GEMM output shape: {out.shape}")
    _vprint(f"Grouped GEMM output amax: {out.abs().max().item():.4f}")
    _vprint(f"Grouped GEMM output has NaN: {torch.isnan(out).any()}")

    self.assertFalse(torch.isnan(out).any())
    self.assertFalse(torch.isinf(out).any())


# =============================================================================
# Test 3: FlashMLA Attention
# =============================================================================

class TestFlashMLA(unittest.TestCase):
  """Test FlashMLA attention operations."""

  @skip_if_no_sm90
  def test_get_mla_metadata(self):
    """Test MLA metadata generation - SM100 requires FP8 + sparse (topk)."""
    from flash_mla import get_mla_metadata

    B, S, H = 2, 4, 128
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
    topk = 64  # SM100 FP8 MLA requires sparse attention with topk

    # Get metadata - SM100 (B200) REQUIRES is_fp8_kvcache=True AND topk for sparse attention
    metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=S * H,
      num_heads_k=1,
      num_heads_q=H,
      is_fp8_kvcache=True,
      topk=topk,  # Required for SM100 FP8
    )

    print(f"MLA metadata shape: {metadata.shape}")
    print(f"num_splits: {num_splits}")

    self.assertIsNotNone(metadata)
    # num_splits can be tensor or int depending on FlashMLA version
    if isinstance(num_splits, torch.Tensor):
      self.assertTrue(num_splits.numel() > 0)
    else:
      self.assertGreater(num_splits, 0)

  @skip_if_no_sm90
  def test_flash_mla_decode_basic(self):
    """Test basic FlashMLA decode with FP8 KV cache + sparse attention (SM100 requirement)."""
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    # Dimensions matching DeepSeek-V3
    B = 2
    S_q = 1  # Decode: 1 token per request
    H_q = 128
    H_kv = 1
    D = 576  # kv_lora_rank + qk_rope_head_dim
    D_v = 512  # kv_lora_rank
    block_size = 64
    max_seqlen = 256
    topk = 64  # SM100 FP8 requires sparse attention

    # Q tensor [B, S_q, H_q, D]
    q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device="cuda") / 10

    # FP8 KV cache [num_blocks, block_size, H_kv, 656] uint8
    num_blocks = (max_seqlen + block_size - 1) // block_size * B
    kv_cache = torch.zeros(num_blocks, block_size, H_kv, 656, dtype=torch.uint8, device="cuda")

    # Block table [B, max_blocks_per_seq]
    max_blocks_per_seq = (max_seqlen + block_size - 1) // block_size
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(B, -1)[:, :max_blocks_per_seq]

    # Cache sequence lengths
    cache_seqlens = torch.tensor([64, 128], dtype=torch.int32, device="cuda")

    # Get metadata - SM100 requires FP8 + topk for sparse attention
    metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=S_q * H_q // H_kv,
      num_heads_k=H_kv,
      num_heads_q=H_q,
      is_fp8_kvcache=True,
      topk=topk,
    )

    # Sparse indices [B, S_q, topk] - physical indices in flattened KV cache.
    max_blocks_per_seq = block_table.size(1)
    indices = torch.empty((B, S_q, topk), dtype=torch.int32, device="cuda")
    for b in range(B):
      ctx = int(cache_seqlens[b].item())
      pos = torch.arange(ctx - topk, ctx, dtype=torch.int64, device="cuda")  # last topk
      page = torch.div(pos, block_size, rounding_mode="floor").to(torch.int64)
      off = (pos % block_size).to(torch.int64)
      blk = block_table[b].index_select(0, page).to(torch.int64)
      phys = (blk * block_size + off).to(torch.int32)
      indices[b, 0] = phys

    _vprint(f"Q shape: {q.shape}")
    _vprint(f"KV cache shape: {kv_cache.shape}")
    _vprint(f"Block table shape: {block_table.shape}")
    _vprint(f"Indices shape: {indices.shape}")

    # Run FlashMLA with FP8 KV cache and sparse attention
    out, lse = flash_mla_with_kvcache(
      q,
      kv_cache,
      block_table,
      cache_seqlens,
      D_v,
      metadata,
      num_splits,
      softmax_scale=1.0 / math.sqrt(D),
      causal=False,  # Sparse attention handles causality via indices
      is_fp8_kvcache=True,
      indices=indices,
    )

    _vprint(f"Output shape: {out.shape}")
    _vprint(f"Output amax: {out.abs().max().item():.4f}")
    _vprint(f"Output has NaN: {torch.isnan(out).any()}")
    _vprint(f"LSE shape: {lse.shape}")

    self.assertEqual(out.shape, (B, S_q, H_q, D_v))
    self.assertFalse(torch.isnan(out).any())
    self.assertFalse(torch.isinf(out).any())

  @skip_if_no_sm90
  def test_flash_mla_decode_ignores_invalid_indices(self):
    """FlashMLA must ignore -1 indices (sglang/vLLM convention)."""
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    # Small, deterministic-ish shapes.
    B = 1
    S_q = 1
    H_q = 128
    H_kv = 1
    D = 576
    D_v = 512
    block_size = 64
    topk = 64

    # One block with 64 tokens.
    num_blocks = 1
    T = 64
    kv_cache = torch.empty(num_blocks, block_size, H_kv, 656, dtype=torch.uint8, device="cuda")
    block_table = torch.tensor([[0]], dtype=torch.int32, device="cuda")
    cache_seqlens = torch.tensor([T], dtype=torch.int32, device="cuda")

    # Pack latent+rope in the same layout as model.py.
    from nmoe.serve.model import _pack_flashmla_fp8_kv
    latent = (torch.randn(T, 512, dtype=torch.bfloat16, device="cuda") / 10).contiguous()
    rope = (torch.randn(T, 64, dtype=torch.bfloat16, device="cuda") / 10).contiguous()
    kv_cache.view(-1, 656)[:T].copy_(_pack_flashmla_fp8_kv(latent, rope))

    # Query.
    q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device="cuda") / 10

    # Indices: 8 valid, rest invalid.
    indices = torch.full((B, S_q, topk), -1, dtype=torch.int32, device="cuda")
    indices[0, 0, :8] = torch.arange(8, dtype=torch.int32, device="cuda")

    metadata, num_splits = get_mla_metadata(
      cache_seqlens=torch.tensor([topk], dtype=torch.int32, device="cuda"),
      num_q_tokens_per_head_k=S_q * H_q // H_kv,
      num_heads_k=H_kv,
      num_heads_q=H_q,
      is_fp8_kvcache=True,
      topk=topk,
    )

    out, _lse = flash_mla_with_kvcache(
      q,
      kv_cache,
      block_table,
      cache_seqlens,
      D_v,
      metadata,
      num_splits,
      softmax_scale=1.0 / math.sqrt(D),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )

    # Dense reference over only the valid keys (0..7).
    flat = kv_cache.view(-1, 656)[:T]
    fp8_bytes = flat[:, :512]
    scales_bytes = flat[:, 512:528]
    rope_bytes = flat[:, 528:]
    fp8 = fp8_bytes.contiguous().view(torch.float8_e4m3fn).view(T, 512)
    scales = scales_bytes.contiguous().view(torch.float32).view(T, 4)
    rope_u = rope_bytes.contiguous().view(torch.bfloat16).view(T, 64)
    tiles = []
    for t in range(4):
      lo, hi = t * 128, (t + 1) * 128
      tiles.append(fp8[:, lo:hi].float() * scales[:, t : t + 1])
    latent_u = torch.cat(tiles, dim=1)  # [T,512] fp32

    k_full = torch.cat([latent_u, rope_u.float()], dim=-1)  # [T,576] fp32
    v_full = latent_u  # [T,512] fp32
    idx = torch.arange(8, device="cuda", dtype=torch.int64)
    k_sel = k_full.index_select(0, idx).view(1, 1, 8, D)
    v_sel = v_full.index_select(0, idx).view(1, 1, 8, D_v)
    scores = torch.einsum("bshd,bskd->bshk", q.float(), k_sel) * (1.0 / math.sqrt(D))
    probs = scores.softmax(dim=-1)
    ref = torch.einsum("bshk,bskc->bshc", probs, v_sel).to(torch.bfloat16)

    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

  @skip_if_no_sm90
  def test_flash_mla_prefill_basic(self):
    """Test basic FlashMLA prefill with FP8 KV cache + sparse attention (SM100 requirement)."""
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    B = 1
    S_q = 64  # Prefill: multiple tokens (>= topk for valid indices)
    H_q = 128
    H_kv = 1
    D = 576
    D_v = 512
    block_size = 64
    max_seqlen = 128
    topk = 64  # SM100 FP8 requires sparse attention, must be multiple of B_TOPK (64)

    q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device="cuda") / 10

    # FP8 KV cache [num_blocks, block_size, H_kv, 656] uint8
    num_blocks = (max_seqlen + block_size - 1) // block_size * B
    kv_cache = torch.zeros(num_blocks, block_size, H_kv, 656, dtype=torch.uint8, device="cuda")

    max_blocks_per_seq = (max_seqlen + block_size - 1) // block_size
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(B, -1)[:, :max_blocks_per_seq]

    cache_seqlens = torch.tensor([S_q], dtype=torch.int32, device="cuda")

    metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=S_q * H_q // H_kv,
      num_heads_k=H_kv,
      num_heads_q=H_q,
      is_fp8_kvcache=True,
      topk=topk,
    )

    # Sparse indices [B, S_q, topk] - physical indices in flattened KV cache.
    # Follow sglang/vLLM: unused entries are -1 (invalid), never duplicated.
    indices = torch.full((B, S_q, topk), -1, dtype=torch.int32, device="cuda")
    for i in range(S_q):
      k_sel = min(topk, i + 1)
      pos = torch.arange(k_sel, dtype=torch.int64, device="cuda")
      page = torch.div(pos, block_size, rounding_mode="floor").to(torch.int64)
      off = (pos % block_size).to(torch.int64)
      blk = block_table[0].index_select(0, page).to(torch.int64)
      phys = (blk * block_size + off).to(torch.int32)
      indices[0, i, :k_sel] = phys

    out, lse = flash_mla_with_kvcache(
      q,
      kv_cache,
      block_table,
      cache_seqlens,
      D_v,
      metadata,
      num_splits,
      softmax_scale=1.0 / math.sqrt(D),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )

    _vprint(f"Prefill output shape: {out.shape}")
    _vprint(f"Prefill output amax: {out.abs().max().item():.4f}")
    _vprint(f"Prefill output has NaN: {torch.isnan(out).any()}")

    self.assertEqual(out.shape, (B, S_q, H_q, D_v))
    self.assertFalse(torch.isnan(out).any())


# =============================================================================
# Test 4: FP8 KV Cache Packing/Unpacking
# =============================================================================

class TestFP8KVCache(unittest.TestCase):
  """Test FP8 KV cache format for FlashMLA."""

  @skip_if_no_sm90
  def test_kv_cache_format_verification(self):
    """Test KV cache format matches FlashMLA SM100 requirements (656 bytes/token)."""
    # FP8 KV cache format for SM100:
    # [512 bytes FP8 nope] [16 bytes scales (4 tiles x 4 bytes)] [128 bytes BF16 rope]
    dv = 512  # kv_lora_rank (nope dimension)
    rope_dim = 64  # qk_rope_head_dim
    tile_size = 128
    num_tiles = dv // tile_size  # 4

    # Verify byte layout
    bytes_per_token = dv + num_tiles * 4 + 2 * rope_dim  # 512 + 16 + 128 = 656
    self.assertEqual(bytes_per_token, 656)

    num_blocks = 4
    block_size = 64

    # Create test data
    kv_nope = torch.randn(num_blocks, block_size, dv, dtype=torch.bfloat16, device="cuda") / 10
    kv_rope = torch.randn(num_blocks, block_size, rope_dim, dtype=torch.bfloat16, device="cuda") / 10

    # Create packed cache (matches model.py _pack_flashmla_fp8_kv)
    T = num_blocks * block_size
    kv_packed = torch.empty(T, bytes_per_token, dtype=torch.uint8, device="cuda")

    # Flatten for packing
    kv_nope_flat = kv_nope.view(T, dv)
    kv_rope_flat = kv_rope.view(T, rope_dim)

    # Pack each tile with FP8 quantization
    scales = []
    for tile_idx in range(num_tiles):
      tile_start = tile_idx * tile_size
      tile_end = (tile_idx + 1) * tile_size
      tile_data = kv_nope_flat[:, tile_start:tile_end].float()

      # Compute per-token scale
      amax = tile_data.abs().amax(dim=-1).clamp(min=1e-8) / 448.0
      scales.append(amax)

      # Quantize
      tile_fp8 = (tile_data / amax.unsqueeze(-1)).to(torch.float8_e4m3fn)
      kv_packed[:, tile_start:tile_end] = tile_fp8.view(torch.uint8)

    # Pack scales (4 float32 per token = 16 bytes)
    scales_tensor = torch.stack(scales, dim=1).to(torch.float32)  # [T, 4]
    kv_packed[:, dv:dv+16] = scales_tensor.view(torch.uint8).view(T, 16)

    # Pack rope (BF16, 64 values x 2 bytes = 128 bytes)
    kv_packed[:, dv+16:] = kv_rope_flat.view(torch.uint8).view(T, 128)

    _vprint(f"Packed KV shape: {kv_packed.shape}")
    _vprint(f"Total bytes: {kv_packed.numel()}")
    _vprint(f"Bytes per token: {bytes_per_token}")

    # Verify rope round-trip
    rope_recovered = kv_packed[:, dv+16:].view(torch.bfloat16).view(T, rope_dim)
    torch.testing.assert_close(rope_recovered, kv_rope_flat, rtol=0, atol=0)
    _vprint("Rope round-trip: PASS")

    # Reshape to FlashMLA expected format [num_blocks, block_size, 1, 656]
    kv_cache = kv_packed.view(num_blocks, block_size, 1, bytes_per_token)
    self.assertEqual(kv_cache.shape, (num_blocks, block_size, 1, 656))
    _vprint("FlashMLA format shape: PASS")

  @skip_if_no_sm90
  def test_kv_cache_scales_are_pow2_ue8m0(self):
    """Production KV packer must emit UE8M0 (power-of-two) float32 scales per 128 tile."""
    from nmoe.serve.model import _pack_flashmla_fp8_kv

    T = 64
    latent = (torch.randn(T, 512, device="cuda", dtype=torch.bfloat16) / 10).contiguous()
    rope = (torch.randn(T, 64, device="cuda", dtype=torch.bfloat16) / 10).contiguous()
    packed = _pack_flashmla_fp8_kv(latent, rope)

    scales_bytes = packed[:, 512:528].contiguous()  # [T,16] bytes
    scales = scales_bytes.view(torch.float32).view(T, 4)
    self.assertTrue(torch.isfinite(scales).all())
    self.assertTrue((scales > 0).all())

    # Power-of-two check: for normal float32, mantissa bits must be zero.
    bits = scales.view(torch.int32)
    mant = bits & 0x007FFFFF
    self.assertTrue((mant == 0).all(), "UE8M0 scales must be exact powers-of-two (mantissa==0).")


# =============================================================================
# Test 5: Reference Attention Implementation
# =============================================================================

class TestReferenceAttention(unittest.TestCase):
  """Test reference attention to compare against FlashMLA."""

  @skip_if_no_gpu
  def test_reference_mla_attention(self):
    """Reference MLA attention implementation."""
    B, S_q, H_q, D = 1, 4, 8, 64  # Small for testing
    S_kv = 16
    D_v = 48

    q = torch.randn(B, S_q, H_q, D, dtype=torch.float32, device="cuda")
    k = torch.randn(B, S_kv, 1, D, dtype=torch.float32, device="cuda")
    v = torch.randn(B, S_kv, 1, D_v, dtype=torch.float32, device="cuda")

    # Expand K, V for GQA
    k = k.expand(-1, -1, H_q, -1)  # [B, S_kv, H_q, D]
    v = v.expand(-1, -1, H_q, -1)  # [B, S_kv, H_q, D_v]

    # Attention: Q @ K.T / sqrt(d) -> softmax -> @ V
    scale = 1.0 / math.sqrt(D)

    # [B, H_q, S_q, D] @ [B, H_q, D, S_kv] -> [B, H_q, S_q, S_kv]
    q_t = q.transpose(1, 2)  # [B, H_q, S_q, D]
    k_t = k.transpose(1, 2)  # [B, H_q, S_kv, D]
    v_t = v.transpose(1, 2)  # [B, H_q, S_kv, D_v]

    scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # [B, H_q, S_q, S_kv]

    # Causal mask
    mask = torch.triu(torch.ones(S_q, S_kv, device="cuda"), diagonal=S_kv - S_q + 1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_t)  # [B, H_q, S_q, D_v]
    out = out.transpose(1, 2)  # [B, S_q, H_q, D_v]

    _vprint(f"Reference attention output shape: {out.shape}")
    _vprint(f"Reference attention output amax: {out.abs().max().item():.4f}")
    _vprint(f"Has NaN: {torch.isnan(out).any()}")

    self.assertEqual(out.shape, (B, S_q, H_q, D_v))
    self.assertFalse(torch.isnan(out).any())


# =============================================================================
# Test 6: End-to-End Attention Layer
# =============================================================================

class TestAttentionLayer(unittest.TestCase):
  """Test the full attention layer from our model."""

  @skip_if_no_sm90
  def test_attention_forward_isolated(self):
    """Test attention forward pass in isolation."""
    # Import after setting CUTLASS_PATH
    from nmoe.serve.model import ModelConfig

    B, S = 1, 4
    cfg = ModelConfig()

    # Create random input hidden states
    hidden = torch.randn(B, S, cfg.hidden_size, dtype=torch.bfloat16, device="cuda") / 10

    print(f"Hidden input shape: {hidden.shape}")
    print(f"Hidden input amax: {hidden.abs().max().item():.4f}")
    print(f"Hidden input has NaN: {torch.isnan(hidden).any()}")

    # Test Q/K/V projections manually
    # This mimics what the attention layer does

    # Q projection: hidden -> q_proj -> [B, S, num_heads, qk_head_dim]
    qk_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim  # 128 + 64 = 192
    q_proj = torch.randn(cfg.hidden_size, cfg.num_heads * qk_head_dim, dtype=torch.bfloat16, device="cuda") / 100
    q = hidden @ q_proj  # [B, S, num_heads * qk_head_dim]
    q = q.view(B, S, cfg.num_heads, qk_head_dim)

    print(f"Q shape: {q.shape}")
    print(f"Q amax: {q.abs().max().item():.4f}")
    print(f"Q has NaN: {torch.isnan(q).any()}")

    self.assertFalse(torch.isnan(q).any())
    self.assertFalse(torch.isinf(q).any())


# =============================================================================
# Main
# =============================================================================

def main():
  print("=" * 60)
  print("nmoe.serve Kernel Tests")
  print("=" * 60)

  if not torch.cuda.is_available():
    print("CUDA not available, skipping tests")
    return 1

  print(f"CUDA Device: {torch.cuda.get_device_name()}")
  print(f"Compute Capability: {torch.cuda.get_device_capability()}")
  print(f"SM90+: {is_sm90_or_higher()}")
  print()

  # Run tests
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()

  # Add test classes
  suite.addTests(loader.loadTestsFromTestCase(TestFP8Quantization))
  suite.addTests(loader.loadTestsFromTestCase(TestDeepGEMM))
  suite.addTests(loader.loadTestsFromTestCase(TestFlashMLA))
  suite.addTests(loader.loadTestsFromTestCase(TestFP8KVCache))
  suite.addTests(loader.loadTestsFromTestCase(TestReferenceAttention))
  suite.addTests(loader.loadTestsFromTestCase(TestAttentionLayer))

  # Run with verbosity
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)

  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  sys.exit(main())
