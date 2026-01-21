# SPDX-License-Identifier: Apache-2.0
"""Isolated attention layer test to pinpoint NaN/zeros issue.

Tests each step of DsaFlashMla with random weights (no checkpoint loading).
This isolates architecture/math issues from weight loading issues.
"""

import os
from pathlib import Path


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return

_maybe_set_cutlass_path()

import math
import torch
import torch.nn.functional as F
import unittest


def check_tensor(name: str, t: torch.Tensor) -> bool:
  """Check tensor for NaN/Inf/zeros, return True if OK."""
  has_nan = torch.isnan(t).any().item()
  has_inf = torch.isinf(t).any().item()
  amax = t.abs().max().item() if t.numel() > 0 else 0
  pct_zeros = (t == 0).sum().item() / t.numel() * 100 if t.numel() > 0 else 0

  status = "OK"
  if has_nan:
    status = "NaN!"
  elif has_inf:
    status = "Inf!"
  elif amax == 0:
    status = "ALL ZEROS!"
  elif pct_zeros > 90:
    status = f"{pct_zeros:.1f}% zeros"

  print(f"  {name}: shape={tuple(t.shape)}, amax={amax:.6f}, status={status}")
  return not (has_nan or has_inf or amax == 0)


class TestAttentionIsolated(unittest.TestCase):
  """Test attention components in isolation."""

  @classmethod
  def setUpClass(cls):
    if not torch.cuda.is_available():
      raise unittest.SkipTest("CUDA required")

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
      raise unittest.SkipTest("SM90+ required")

    cls.device = torch.device("cuda:0")
    torch.cuda.set_device(cls.device)

  def test_01_fp8_linear_basic(self):
    """Test FP8 linear layer produces valid output."""
    from deep_gemm import fp8_gemm_nt, per_token_cast_to_fp8

    M, K, N = 32, 512, 256
    x = torch.randn(M, K, dtype=torch.bfloat16, device=self.device) / 10
    w = (torch.randn(N, K, dtype=torch.bfloat16, device=self.device) / 10).to(torch.float8_e4m3fn)
    w_scale = torch.ones(N // 128, K // 128, dtype=torch.float32, device=self.device)

    x_fp8, x_scale = per_token_cast_to_fp8(x, use_ue8m0=True)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=self.device)
    fp8_gemm_nt((x_fp8, x_scale), (w, w_scale), out)

    self.assertTrue(check_tensor("fp8_linear_out", out))

  def test_02_rope_basic(self):
    """Test RoPE application."""
    B, S, H, D = 2, 16, 4, 64
    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=self.device)

    # Simple RoPE (no YaRN for this test)
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=self.device).float() / D))
    t = torch.arange(S, device=self.device).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs).unsqueeze(0).expand(B, -1, -1)

    # Apply
    x_complex = torch.view_as_complex(x.float().view(B, S, H, -1, 2))
    freqs_cis_exp = freqs_cis.view(B, S, 1, D // 2)
    y = torch.view_as_real(x_complex * freqs_cis_exp).flatten(-2)
    y = y.to(torch.bfloat16)

    self.assertTrue(check_tensor("rope_out", y))
    # RoPE should preserve magnitude roughly
    self.assertLess(abs(y.abs().mean().item() - x.abs().mean().item()), 0.5)

  def test_03_kv_pack_unpack(self):
    """Test FP8 KV packing/unpacking."""
    T = 64
    latent = torch.randn(T, 512, dtype=torch.bfloat16, device=self.device) / 10
    rope = torch.randn(T, 64, dtype=torch.bfloat16, device=self.device) / 10

    # Pack
    packed = torch.empty((T, 656), device=self.device, dtype=torch.uint8)
    latent_f = latent.float()
    scales = []
    q_bytes = []
    for tile in range(4):
      lo = tile * 128
      hi = lo + 128
      tile_f = latent_f[:, lo:hi]
      sf = tile_f.abs().amax(dim=-1).clamp(min=1e-8) / 448.0
      scales.append(sf)
      tile_q = (tile_f / sf.unsqueeze(-1)).to(torch.float8_e4m3fn)
      q_bytes.append(tile_q.view(torch.uint8))

    packed[:, :512] = torch.cat(q_bytes, dim=1)
    sf32 = torch.stack(scales, dim=1).to(torch.float32)
    packed[:, 512:528] = sf32.view(torch.uint8).reshape(T, 16)
    packed[:, 528:] = rope.view(torch.uint8).reshape(T, 128)

    self.assertTrue(check_tensor("packed_kv", packed.float()))

    # Unpack and verify (approximate reconstruction)
    unpacked_rope = packed[:, 528:].view(torch.bfloat16).reshape(T, 64)
    self.assertTrue(torch.allclose(unpacked_rope, rope, atol=1e-6))
    print(f"  rope reconstruction: exact match")

  def test_04_absorbed_query_einsum(self):
    """Test the absorbed query computation (key operation in MLA)."""
    B, S = 2, 16
    num_heads = 128
    qk_nope_dim = 128
    kv_lora_rank = 512

    # q_nope: [B, S, H, qk_nope_dim]
    q_nope = torch.randn(B, S, num_heads, qk_nope_dim, dtype=torch.bfloat16, device=self.device) / 10

    # W_UK: [H, qk_nope_dim, kv_lora_rank] - projects nope to latent space
    w_uk = torch.randn(num_heads, qk_nope_dim, kv_lora_rank, dtype=torch.bfloat16, device=self.device) / math.sqrt(qk_nope_dim)

    # einsum: contracts d (qk_nope_dim), preserves h (heads)
    q_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, w_uk)

    self.assertTrue(check_tensor("q_absorbed", q_absorbed))
    self.assertEqual(q_absorbed.shape, (B, S, num_heads, kv_lora_rank))

    # Verify against explicit loop
    q_absorbed_ref = torch.zeros_like(q_absorbed)
    for h in range(num_heads):
      q_absorbed_ref[:, :, h] = q_nope[:, :, h] @ w_uk[h]

    diff = (q_absorbed - q_absorbed_ref).abs().max().item()
    print(f"  einsum vs loop diff: {diff:.8f}")
    self.assertLess(diff, 1e-4)

  def test_05_output_projection_einsum(self):
    """Test the output projection (latent -> v space)."""
    B, S = 2, 16
    num_heads = 128
    v_head_dim = 128
    kv_lora_rank = 512

    # out_latent: [B, S, H, kv_lora_rank] from attention
    out_latent = torch.randn(B, S, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=self.device) / 10

    # W_UV: [H, v_head_dim, kv_lora_rank] - projects latent to V
    w_uv = torch.randn(num_heads, v_head_dim, kv_lora_rank, dtype=torch.bfloat16, device=self.device) / math.sqrt(kv_lora_rank)

    # einsum: contracts c (kv_lora_rank), preserves h
    out = torch.einsum("bshc,hdc->bshd", out_latent, w_uv)

    self.assertTrue(check_tensor("out_proj", out))
    self.assertEqual(out.shape, (B, S, num_heads, v_head_dim))

  def test_06_flashmla_sparse_minimal(self):
    """Test FlashMLA sparse attention with minimal setup."""
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    B = 1
    S_q = 1  # Single decode token
    H_q = 128
    H_kv = 1
    D = 576  # kv_lora_rank + qk_rope_dim = 512 + 64
    D_v = 512
    topk = 64
    block_size = 64

    # Single block with 64 tokens
    num_blocks = 1
    cache_seqlen = 64

    q = torch.randn(B, S_q, H_q, D, dtype=torch.bfloat16, device=self.device) / 10
    kv_cache = torch.zeros(num_blocks, block_size, H_kv, 656, dtype=torch.uint8, device=self.device)

    # Fill KV cache with valid FP8 data
    for i in range(cache_seqlen):
      latent = torch.randn(512, dtype=torch.bfloat16, device=self.device) / 10
      rope = torch.randn(64, dtype=torch.bfloat16, device=self.device) / 10

      # Pack into FP8 format
      packed = torch.empty(656, dtype=torch.uint8, device=self.device)
      latent_f = latent.float()
      for tile in range(4):
        lo, hi = tile * 128, (tile + 1) * 128
        tile_f = latent_f[lo:hi]
        sf = tile_f.abs().max().clamp(min=1e-8) / 448.0
        tile_q = (tile_f / sf).to(torch.float8_e4m3fn)
        packed[lo:hi] = tile_q.view(torch.uint8)
        # sf is a scalar, need to reshape for view
        sf_tensor = sf.to(torch.float32).reshape(1)
        sf_bytes = sf_tensor.view(torch.uint8)
        packed[512 + tile * 4:512 + (tile + 1) * 4] = sf_bytes
      packed[528:] = rope.view(torch.uint8)
      kv_cache[0, i, 0] = packed

    block_table = torch.tensor([[0]], dtype=torch.int32, device=self.device)
    cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device=self.device)

    # Get metadata for sparse FP8 MLA
    metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=S_q * H_q // H_kv,
      num_heads_k=H_kv,
      num_heads_q=H_q,
      is_fp8_kvcache=True,
      topk=topk,
    )

    # Sparse indices: attend to last 64 tokens
    indices = torch.arange(topk, dtype=torch.int32, device=self.device).unsqueeze(0).unsqueeze(0)

    out, lse = flash_mla_with_kvcache(
      q, kv_cache, block_table, cache_seqlens, D_v,
      metadata, num_splits,
      softmax_scale=1.0 / math.sqrt(D),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )

    self.assertTrue(check_tensor("flashmla_out", out))
    self.assertTrue(check_tensor("flashmla_lse", lse))
    self.assertEqual(out.shape, (B, S_q, H_q, D_v))

  def test_07_dsa_indexer_scores(self):
    """Test DSA indexer score computation."""
    from nmoe.triton.dsa import compute_indexer_scores

    B, S, N = 1, 4, 64  # S query positions, N key positions
    n_idx_heads = 64
    idx_dim = 128

    q_idx = torch.randn(B, S, n_idx_heads, idx_dim, dtype=torch.bfloat16, device=self.device) / 10
    k_idx = torch.randn(B, N, idx_dim, dtype=torch.bfloat16, device=self.device) / 10
    w_idx = torch.randn(B, S, n_idx_heads, dtype=torch.bfloat16, device=self.device)

    scores = compute_indexer_scores(q_idx, k_idx, w_idx, causal=False)

    self.assertTrue(check_tensor("dsa_scores", scores))
    self.assertEqual(scores.shape, (B, S, N))

  def test_08_full_attention_flow(self):
    """Test complete attention flow (without loading checkpoint)."""
    print("\n=== Full Attention Flow Test ===")

    B, S = 1, 4
    hidden_size = 7168
    num_heads = 128
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_dim = 128
    qk_rope_dim = 64
    v_head_dim = 128
    n_idx_heads = 64
    idx_dim = 128
    topk = 64

    x = torch.randn(B, S, hidden_size, dtype=torch.bfloat16, device=self.device) / 10

    # Step 1: Q LoRA
    print("\n[Step 1] Q LoRA projection")
    wq_a = torch.randn(q_lora_rank, hidden_size, dtype=torch.bfloat16, device=self.device) / math.sqrt(hidden_size)
    q_latent = F.linear(x, wq_a)  # [B, S, q_lora_rank]
    q_latent = F.rms_norm(q_latent, (q_lora_rank,))
    self.assertTrue(check_tensor("q_latent", q_latent))

    wq_b = torch.randn(num_heads * (qk_nope_dim + qk_rope_dim), q_lora_rank, dtype=torch.bfloat16, device=self.device) / math.sqrt(q_lora_rank)
    q = F.linear(q_latent, wq_b).view(B, S, num_heads, qk_nope_dim + qk_rope_dim)
    self.assertTrue(check_tensor("q", q))

    q_nope, q_pe = q.split([qk_nope_dim, qk_rope_dim], dim=-1)
    self.assertTrue(check_tensor("q_nope", q_nope))
    self.assertTrue(check_tensor("q_pe", q_pe))

    # Step 2: KV projection
    print("\n[Step 2] KV projection")
    wkv_a = torch.randn(kv_lora_rank + qk_rope_dim, hidden_size, dtype=torch.bfloat16, device=self.device) / math.sqrt(hidden_size)
    kv = F.linear(x, wkv_a)
    kv_latent, k_pe = kv.split([kv_lora_rank, qk_rope_dim], dim=-1)
    kv_latent = F.rms_norm(kv_latent, (kv_lora_rank,))
    self.assertTrue(check_tensor("kv_latent", kv_latent))
    self.assertTrue(check_tensor("k_pe", k_pe))

    # Step 3: RoPE
    print("\n[Step 3] RoPE application")
    positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
    freqs = 1.0 / (10000 ** (torch.arange(0, qk_rope_dim, 2, device=self.device).float() / qk_rope_dim))
    t = positions.float().view(-1)
    freqs = torch.outer(t, freqs).view(B, S, -1)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def apply_rope(x, freqs_cis):
      x_complex = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
      freqs = freqs_cis.view(B, S, 1, -1)
      return torch.view_as_real(x_complex * freqs).flatten(-2).to(x.dtype)

    q_pe_rope = apply_rope(q_pe, freqs_cis)
    k_rope = apply_rope(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
    self.assertTrue(check_tensor("q_pe_rope", q_pe_rope))
    self.assertTrue(check_tensor("k_rope", k_rope))

    # Step 4: Absorbed query
    print("\n[Step 4] Absorbed query computation")
    wkv_b = torch.randn(num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank, dtype=torch.bfloat16, device=self.device) / math.sqrt(kv_lora_rank)
    wkv_b_w = wkv_b.view(num_heads, qk_nope_dim + v_head_dim, kv_lora_rank)
    w_uk = wkv_b_w[:, :qk_nope_dim]  # [H, qk_nope_dim, kv_lora_rank]

    q_absorbed = torch.einsum("bshd,hdc->bshc", q_nope, w_uk)  # [B, S, H, kv_lora_rank]
    self.assertTrue(check_tensor("q_absorbed", q_absorbed))

    q_for_attn = torch.cat([q_absorbed, q_pe_rope], dim=-1)  # [B, S, H, 576]
    self.assertTrue(check_tensor("q_for_attn", q_for_attn))

    # Step 5: Simulate attention output (skip FlashMLA for this test)
    print("\n[Step 5] Simulated attention output")
    out_latent = torch.randn(B, S, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=self.device) / 10
    self.assertTrue(check_tensor("out_latent (simulated)", out_latent))

    # Step 6: Output projection
    print("\n[Step 6] Output projection")
    w_uv = wkv_b_w[:, qk_nope_dim:]  # [H, v_head_dim, kv_lora_rank]
    out = torch.einsum("bshc,hdc->bshd", out_latent, w_uv)  # [B, S, H, v_head_dim]
    self.assertTrue(check_tensor("out", out))

    out_flat = out.flatten(2)  # [B, S, H * v_head_dim]
    self.assertTrue(check_tensor("out_flat", out_flat))

    # Step 7: Final projection (wo)
    print("\n[Step 7] Final output projection")
    wo = torch.randn(hidden_size, num_heads * v_head_dim, dtype=torch.bfloat16, device=self.device) / math.sqrt(num_heads * v_head_dim)
    final_out = F.linear(out_flat, wo)
    self.assertTrue(check_tensor("final_out", final_out))

    print("\n=== All steps passed! ===")


if __name__ == "__main__":
  unittest.main(verbosity=2)
