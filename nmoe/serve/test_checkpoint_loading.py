# SPDX-License-Identifier: Apache-2.0
"""Test checkpoint loading and weight verification."""

import os
from pathlib import Path
import tempfile


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

import torch
import torch.distributed as dist
import unittest


def check_tensor(name: str, t: torch.Tensor) -> dict:
  """Check tensor stats."""
  stats = {
    "name": name,
    "shape": tuple(t.shape),
    "dtype": str(t.dtype),
    "has_nan": bool(torch.isnan(t).any()),
    "has_inf": bool(torch.isinf(t).any()),
    "amax": float(t.abs().max().item()) if t.numel() > 0 else 0,
    "mean": float(t.float().mean().item()) if t.numel() > 0 else 0,
    "std": float(t.float().std().item()) if t.numel() > 0 else 0,
    "pct_zeros": float((t == 0).sum().item() / t.numel() * 100) if t.numel() > 0 else 0,
  }

  status = "OK"
  if stats["has_nan"]:
    status = "NaN!"
  elif stats["has_inf"]:
    status = "Inf!"
  elif stats["amax"] == 0:
    status = "ALL ZEROS!"
  elif stats["pct_zeros"] > 90:
    status = f"{stats['pct_zeros']:.1f}% zeros"

  print(f"  {name}: shape={stats['shape']}, dtype={stats['dtype']}, "
        f"amax={stats['amax']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, status={status}")
  return stats


class TestCheckpointLoading(unittest.TestCase):
  """Test checkpoint loading produces valid weights."""

  @classmethod
  def setUpClass(cls):
    if not torch.cuda.is_available():
      raise unittest.SkipTest("CUDA required")

    cls.device = torch.device("cuda:0")
    torch.cuda.set_device(cls.device)

    # Initialize distributed
    if not dist.is_initialized():
      tmp = tempfile.NamedTemporaryFile(prefix="nmoe_pg_", suffix=".tmp", delete=False)
      tmp.close()
      dist.init_process_group(backend="nccl", init_method=f"file://{tmp.name}", world_size=1, rank=0)

    from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
    from nmoe.serve.ckpt import load_checkpoint
    from deep_ep import Buffer

    init_distributed(0, 1)

    # Load model with 4 layers for testing
    cls.cfg = ModelConfig(num_layers=4, num_dense_layers=3)
    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    cls.model = DeepSeekV3(cls.cfg, buffer).to(cls.device)

    # Load checkpoint
    ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3.2-Speciale")
    print(f"\nLoading checkpoint from {ckpt_path}...")
    cls.missing, cls.unexpected = load_checkpoint(cls.model, ckpt_path, rank=0, world_size=1, cfg=cls.cfg)
    print(f"Missing keys: {len(cls.missing)}")
    print(f"Unexpected keys: {len(cls.unexpected)}")

    if cls.missing:
      print(f"Sample missing: {list(cls.missing)[:5]}")
    if cls.unexpected:
      print(f"Sample unexpected: {list(cls.unexpected)[:5]}")

  def test_embedding_loaded(self):
    """Test embedding weights are loaded."""
    print("\n=== Embedding ===")
    embed = self.model.embed.weight
    stats = check_tensor("embed.weight", embed)
    self.assertFalse(stats["has_nan"])
    self.assertGreater(stats["amax"], 0)

  def test_attn_projections_loaded(self):
    """Test attention projection weights are loaded correctly."""
    print("\n=== Attention Projections (Layer 0) ===")
    attn = self.model.layers[0].attn

    # wq_a (FP8)
    stats = check_tensor("wq_a.weight", attn.wq_a.weight.float())
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("wq_a.weight_scale_inv", attn.wq_a.weight_scale_inv)
    self.assertFalse(stats["has_nan"])

    # wq_b (FP8)
    stats = check_tensor("wq_b.weight", attn.wq_b.weight.float())
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("wq_b.weight_scale_inv", attn.wq_b.weight_scale_inv)
    self.assertFalse(stats["has_nan"])

    # wkv_a (FP8)
    stats = check_tensor("wkv_a.weight", attn.wkv_a.weight.float())
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("wkv_a.weight_scale_inv", attn.wkv_a.weight_scale_inv)
    self.assertFalse(stats["has_nan"])

    # wkv_b (FP8) - this is the key one for absorbed attention
    stats = check_tensor("wkv_b.weight", attn.wkv_b.weight.float())
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("wkv_b.weight_scale_inv", attn.wkv_b.weight_scale_inv)
    self.assertFalse(stats["has_nan"])

    # wo (FP8)
    stats = check_tensor("wo.weight", attn.wo.weight.float())
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("wo.weight_scale_inv", attn.wo.weight_scale_inv)
    self.assertFalse(stats["has_nan"])

  def test_wkv_b_dequantization(self):
    """Test wkv_b dequantization produces valid weights."""
    print("\n=== wkv_b Dequantization (Layer 0) ===")
    attn = self.model.layers[0].attn

    # Get FP8 weight and scale
    w = attn.wkv_b.weight  # [out, in] FP8
    s = attn.wkv_b.weight_scale_inv  # [out//128, in//128]

    print(f"  wkv_b.weight shape: {w.shape}, dtype: {w.dtype}")
    print(f"  wkv_b.weight_scale_inv shape: {s.shape}, dtype: {s.dtype}")

    # Dequantize using the model's function
    from nmoe.serve.model import weight_dequant
    w_dequant = weight_dequant(w, s)

    stats = check_tensor("wkv_b_dequant", w_dequant)
    self.assertFalse(stats["has_nan"])
    self.assertFalse(stats["has_inf"])
    self.assertGreater(stats["amax"], 0)

    # Reshape for absorbed attention
    num_heads = self.cfg.num_heads
    qk_nope = self.cfg.qk_nope_head_dim
    v_dim = self.cfg.v_head_dim
    kv_lora = self.cfg.kv_lora_rank

    # Shape should be [num_heads * (qk_nope + v), kv_lora]
    expected_shape = (num_heads * (qk_nope + v_dim), kv_lora)
    self.assertEqual(w_dequant.shape, expected_shape, f"Expected {expected_shape}, got {w_dequant.shape}")

    # Reshape to [H, qk_nope+v, kv_lora]
    wkv_b_w = w_dequant.view(num_heads, qk_nope + v_dim, kv_lora)
    stats = check_tensor("wkv_b_w reshaped", wkv_b_w)

    # Split into UK and UV
    w_uk = wkv_b_w[:, :qk_nope]  # [H, qk_nope, kv_lora]
    w_uv = wkv_b_w[:, qk_nope:]  # [H, v, kv_lora]

    stats = check_tensor("w_uk (absorbed Q)", w_uk)
    self.assertFalse(stats["has_nan"])

    stats = check_tensor("w_uv (output proj)", w_uv)
    self.assertFalse(stats["has_nan"])

  def test_dsa_indexer_loaded(self):
    """Test DSA indexer weights are loaded."""
    print("\n=== DSA Indexer (Layer 0) ===")
    attn = self.model.layers[0].attn

    # wq_idx (FP8)
    stats = check_tensor("wq_idx.weight", attn.wq_idx.weight.float())
    self.assertFalse(stats["has_nan"])

    # wk_idx (FP8)
    stats = check_tensor("wk_idx.weight", attn.wk_idx.weight.float())
    self.assertFalse(stats["has_nan"])

    # k_norm (LayerNorm)
    stats = check_tensor("k_norm.weight", attn.k_norm.weight)
    self.assertFalse(stats["has_nan"])
    stats = check_tensor("k_norm.bias", attn.k_norm.bias)
    self.assertFalse(stats["has_nan"])

    # w_idx (BF16 linear)
    stats = check_tensor("w_idx.weight", attn.w_idx.weight)
    self.assertFalse(stats["has_nan"])

  def test_norms_loaded(self):
    """Test norm weights are loaded."""
    print("\n=== Norms (Layer 0) ===")
    layer = self.model.layers[0]

    stats = check_tensor("attn_norm.weight", layer.attn_norm.weight)
    self.assertFalse(stats["has_nan"])
    self.assertGreater(stats["amax"], 0)

    stats = check_tensor("ffn_norm.weight", layer.ffn_norm.weight)
    self.assertFalse(stats["has_nan"])

    # Q norm
    stats = check_tensor("q_norm.weight", layer.attn.q_norm.weight)
    self.assertFalse(stats["has_nan"])

    # KV norm
    stats = check_tensor("kv_norm.weight", layer.attn.kv_norm.weight)
    self.assertFalse(stats["has_nan"])

  def test_single_layer_forward(self):
    """Test single attention layer forward pass with loaded weights."""
    print("\n=== Single Layer Forward ===")

    B, S = 1, 4
    x = torch.randn(B, S, self.cfg.hidden_size, dtype=torch.bfloat16, device=self.device) / 10

    # Get freqs_cis
    from nmoe.serve.model import precompute_freqs_cis
    freqs_cis = precompute_freqs_cis(self.cfg, self.device)
    positions = torch.arange(S, device=self.device).unsqueeze(0)
    freqs = freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

    # Create minimal KV cache
    num_blocks = 1
    kv_cache = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=self.device)
    idx_k_cache = torch.zeros(num_blocks, 64, self.cfg.dsa_idx_dim, dtype=torch.bfloat16, device=self.device)
    block_table = torch.zeros(B, 1, dtype=torch.int32, device=self.device)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=self.device)
    out_loc = torch.arange(S, dtype=torch.int32, device=self.device).unsqueeze(0)

    print("\nInput:")
    check_tensor("x", x)

    # Run attention layer
    attn = self.model.layers[0].attn

    # Step by step to identify issue
    print("\n[Step 1] Input norm")
    x_norm = self.model.layers[0].attn_norm(x)
    check_tensor("x_norm", x_norm)

    print("\n[Step 2] Q LoRA")
    q_latent = attn.q_norm(attn.wq_a(x_norm))
    check_tensor("q_latent", q_latent)

    q = attn.wq_b(q_latent).view(B, S, attn.num_local_heads, attn.qk_head_dim)
    check_tensor("q", q)

    print("\n[Step 3] KV projection")
    kv = attn.wkv_a(x_norm)
    check_tensor("kv", kv)

    kv_latent, k_pe = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_latent = attn.kv_norm(kv_latent)
    check_tensor("kv_latent", kv_latent)
    check_tensor("k_pe", k_pe)

    print("\n[Step 4] Absorbed Q")
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    check_tensor("q_nope", q_nope)

  @classmethod
  def tearDownClass(cls):
    if dist.is_initialized():
      dist.destroy_process_group()

  def test_full_attention_forward(self):
    """Test full attention layer forward with all components."""
    print("\n=== Full Attention Forward ===")

    B, S = 1, 4
    x = torch.randn(B, S, self.cfg.hidden_size, dtype=torch.bfloat16, device=self.device) / 10

    # Get freqs_cis
    from nmoe.serve.model import precompute_freqs_cis, apply_rotary_emb, _pack_flashmla_fp8_kv
    freqs_cis = precompute_freqs_cis(self.cfg, self.device)
    positions = torch.arange(S, device=self.device).unsqueeze(0)
    freqs = freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

    attn = self.model.layers[0].attn

    # Create KV cache (pre-fill with some data)
    num_blocks = 1
    kv_cache = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=self.device)
    idx_k_cache = torch.zeros(num_blocks, 64, self.cfg.dsa_idx_dim, dtype=torch.bfloat16, device=self.device)
    block_table = torch.zeros(B, 1, dtype=torch.int32, device=self.device)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=self.device)
    out_loc = torch.arange(S, dtype=torch.int32, device=self.device).unsqueeze(0)

    print("\n[Step 1] Input norm")
    x_norm = self.model.layers[0].attn_norm(x)
    check_tensor("x_norm", x_norm)

    print("\n[Step 2] Q projection")
    q_latent = attn.q_norm(attn.wq_a(x_norm))
    q = attn.wq_b(q_latent).view(B, S, attn.num_local_heads, attn.qk_head_dim)
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    q_pe_rope = apply_rotary_emb(q_pe, freqs)
    check_tensor("q_nope", q_nope)
    check_tensor("q_pe_rope", q_pe_rope)

    print("\n[Step 3] KV projection")
    kv = attn.wkv_a(x_norm)
    kv_latent, k_pe = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_latent = attn.kv_norm(kv_latent)
    k_rope = apply_rotary_emb(k_pe.unsqueeze(2), freqs).squeeze(2).contiguous()
    check_tensor("kv_latent", kv_latent)
    check_tensor("k_rope", k_rope)

    print("\n[Step 4] Pack KV cache")
    kv_pack = _pack_flashmla_fp8_kv(
      kv_latent.reshape(B * S, 512).contiguous(),
      k_rope.reshape(B * S, 64).contiguous(),
    )
    check_tensor("kv_pack", kv_pack.float())

    # Store in cache
    loc = out_loc.reshape(B * S).to(torch.int64)
    kv_cache.view(-1, 656).index_copy_(0, loc, kv_pack)

    print("\n[Step 5] DSA indexer K")
    k_idx_new = attn.k_norm(attn.wk_idx(x_norm)).reshape(B * S, attn.idx_dim).contiguous()
    check_tensor("k_idx_new", k_idx_new)
    idx_k_cache.view(-1, attn.idx_dim).index_copy_(0, loc, k_idx_new)

    print("\n[Step 6] DSA indexer Q and scores")
    q_idx_all = attn.wq_idx(q_latent).view(B, S, attn.n_idx_heads, attn.idx_dim)
    # weights_proj (w_idx) runs in FP32 in the reference implementation.
    w_idx_all = attn.w_idx(x_norm.float()).view(B, S, attn.n_idx_heads)
    check_tensor("q_idx_all", q_idx_all)
    check_tensor("w_idx_all", w_idx_all)

    from nmoe.triton.dsa import compute_indexer_scores
    from nmoe.serve.model import _phys_token_ids

    indices = torch.empty((B, S, attn.topk), device=self.device, dtype=torch.int32)
    ctx_len = S  # For this test

    # Gather idx_k for context
    bt = block_table[0]
    phys_ids = _phys_token_ids(bt, ctx_len).to(torch.int64)
    k_ctx = idx_k_cache.view(-1, attn.idx_dim).index_select(0, phys_ids)
    check_tensor("k_ctx", k_ctx)

    # Compute scores
    q_idx = q_idx_all[0:1]
    k_idx = k_ctx.unsqueeze(0)
    w_idx = w_idx_all[0:1]
    scores = compute_indexer_scores(q_idx, k_idx, w_idx, causal=False).squeeze(0)
    check_tensor("scores", scores)

    # Apply causal mask
    q_pos = positions[0].to(torch.int64)
    k_pos = torch.arange(ctx_len, device=self.device, dtype=torch.int64)
    causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    check_tensor("scores_masked", scores)

    # Top-k selection
    k_sel = min(attn.topk, ctx_len)
    _, topk_logical = scores.topk(k_sel, dim=-1)
    check_tensor("topk_logical", topk_logical.float())

    # Pad to topk
    if k_sel < attn.topk:
      topk_logical = torch.nn.functional.pad(topk_logical, (0, attn.topk - k_sel), value=-1)
    indices[0] = topk_logical.to(torch.int32)
    print(f"  indices[0] sample: {indices[0, 0, :10].tolist()}")

    print("\n[Step 7] Absorbed Q")
    from nmoe.serve.model import weight_dequant
    if attn._wkv_b_dequant is None:
      attn._wkv_b_dequant = weight_dequant(attn.wkv_b.weight, attn.wkv_b.weight_scale_inv)
    wkv_b_w = attn._wkv_b_dequant.view(attn.num_local_heads, -1, attn.kv_lora_rank)
    q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_w[:, :attn.qk_nope_head_dim])
    q_for_attn = torch.cat([q_nope_abs, q_pe_rope], dim=-1).contiguous()
    check_tensor("q_for_attn", q_for_attn)

    print("\n[Step 8] FlashMLA")
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    tile_scheduler_metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=int(S * attn.num_local_heads),
      num_heads_k=1,
      num_heads_q=attn.num_local_heads,
      is_fp8_kvcache=True,
      topk=attn.topk,
    )
    print(f"  metadata shape: {tile_scheduler_metadata.shape}")
    print(f"  num_splits: {num_splits}")

    out_latent, lse = flash_mla_with_kvcache(
      q_for_attn,
      kv_cache,
      block_table,
      cache_seqlens,
      head_dim_v=512,
      tile_scheduler_metadata=tile_scheduler_metadata,
      num_splits=num_splits,
      softmax_scale=float(attn.softmax_scale),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )
    check_tensor("out_latent", out_latent)
    check_tensor("lse", lse)

    print("\n[Step 9] Output projection")
    out = torch.einsum("bshc,hdc->bshd", out_latent, wkv_b_w[:, -attn.v_head_dim:])
    check_tensor("out", out)

    # Debug the wo projection step by step
    print("\n[Step 9a] Debug wo projection")
    out_flat = out.flatten(2)  # [B, S, H*v]
    check_tensor("out_flat", out_flat)

    # Check wo weights
    check_tensor("wo.weight", attn.wo.weight.float())
    check_tensor("wo.weight_scale_inv", attn.wo.weight_scale_inv)

    # Manual FP8 forward
    from deep_gemm import fp8_gemm_nt, per_token_cast_to_fp8

    x_for_wo = out_flat.view(-1, attn.wo.in_features).to(torch.bfloat16)
    check_tensor("x_for_wo", x_for_wo)

    print(f"  x_for_wo shape: {x_for_wo.shape}, wo.in_features: {attn.wo.in_features}")
    print(f"  wo.weight shape: {attn.wo.weight.shape}")
    print(f"  wo.weight_scale_inv shape: {attn.wo.weight_scale_inv.shape}")

    # Try quantization
    try:
      x_fp8, x_scale = per_token_cast_to_fp8(x_for_wo, use_ue8m0=True)
      check_tensor("x_fp8", x_fp8.float())
      check_tensor("x_scale", x_scale.float())
    except Exception as e:
      print(f"  per_token_cast_to_fp8 FAILED: {e}")
      raise

    # Try GEMM
    wo_out = torch.empty(x_for_wo.size(0), attn.wo.out_features, dtype=torch.bfloat16, device=self.device)
    try:
      fp8_gemm_nt(
        (x_fp8, x_scale),
        (attn.wo.weight, attn.wo.weight_scale_inv),
        wo_out,
        attn.wo.bias,
      )
      check_tensor("wo_out (manual)", wo_out)
    except Exception as e:
      print(f"  fp8_gemm_nt FAILED: {e}")
      raise

    out_final = attn.wo(out.flatten(2))
    check_tensor("out_final", out_final)

    self.assertFalse(torch.isnan(out_final).any(), "out_final has NaN")
    self.assertGreater(out_final.abs().max().item(), 0, "out_final is all zeros")


if __name__ == "__main__":
  unittest.main(verbosity=2)
