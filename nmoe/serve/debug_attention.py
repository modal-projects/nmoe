# SPDX-License-Identifier: Apache-2.0
"""Debug script to isolate attention output issue.

This script traces through the attention forward pass step-by-step
to identify where NaN/Inf/zero values originate.

Run with: python -m nmoe.serve.debug_attention --model-path /data/models/DeepSeek-V3.2-Speciale
"""

from __future__ import annotations

import argparse
import os
import sys
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

import torch
import torch.distributed as dist


def check_tensor(name: str, t: torch.Tensor, verbose: bool = True) -> dict:
  """Check tensor for NaN/Inf/zeros and return stats."""
  stats = {
    "name": name,
    "shape": tuple(t.shape),
    "dtype": str(t.dtype),
    "device": str(t.device),
    "has_nan": bool(torch.isnan(t).any()),
    "has_inf": bool(torch.isinf(t).any()),
    "amax": float(t.abs().max().item()) if t.numel() > 0 else 0,
    "amin": float(t.abs().min().item()) if t.numel() > 0 else 0,
    "mean": float(t.float().mean().item()) if t.numel() > 0 else 0,
    "std": float(t.float().std().item()) if t.numel() > 0 else 0,
    "num_zeros": int((t == 0).sum().item()),
    "pct_zeros": float((t == 0).sum().item() / t.numel() * 100) if t.numel() > 0 else 0,
  }

  if verbose:
    status = "OK"
    if stats["has_nan"]:
      status = "NaN!"
    elif stats["has_inf"]:
      status = "Inf!"
    elif stats["amax"] == 0:
      status = "ALL ZEROS!"
    elif stats["pct_zeros"] > 50:
      status = f"{stats['pct_zeros']:.1f}% zeros"

    print(f"  {name}: shape={stats['shape']}, dtype={stats['dtype']}, "
          f"amax={stats['amax']:.6f}, mean={stats['mean']:.6f}, "
          f"std={stats['std']:.6f}, status={status}")

  return stats


def debug_attention_forward(model, layer_idx: int, hidden: torch.Tensor, positions: torch.Tensor,
                            kv_cache: torch.Tensor, idx_k_cache: torch.Tensor,
                            block_table: torch.Tensor, cache_seqlens: torch.Tensor,
                            out_loc: torch.Tensor) -> torch.Tensor:
  """Debug a single attention layer forward pass."""
  print(f"\n{'='*60}")
  print(f"DEBUG: Attention Layer {layer_idx}")
  print(f"{'='*60}")

  attn = model.layers[layer_idx].attn

  # Check input
  check_tensor("hidden_input", hidden)
  check_tensor("positions", positions)

  B, S, D = hidden.shape

  # Step 1: Input norm
  print("\n[Step 1] Input LayerNorm")
  h_normed = model.layers[layer_idx].attn_norm(hidden)
  check_tensor("h_normed", h_normed)

  # Step 2: Q/KV projections
  print("\n[Step 2] Q/KV Projections")

  # Check projection weights
  if hasattr(attn, 'q_proj_weight'):
    check_tensor("q_proj_weight", attn.q_proj_weight)
  if hasattr(attn, 'q_proj_weight_scale'):
    check_tensor("q_proj_weight_scale", attn.q_proj_weight_scale)
  if hasattr(attn, 'kv_proj_weight'):
    check_tensor("kv_proj_weight", attn.kv_proj_weight)

  # Do Q projection manually to trace
  from deep_gemm import fp8_gemm_nt, per_token_cast_to_fp8

  h_flat = h_normed.view(-1, D)  # [B*S, D]
  check_tensor("h_flat", h_flat)

  # Cast to FP8
  print("\n[Step 2a] Cast hidden to FP8")
  try:
    h_fp8, h_scale = per_token_cast_to_fp8(h_flat.to(torch.bfloat16), use_ue8m0=True)
    check_tensor("h_fp8", h_fp8.float())
    check_tensor("h_scale", h_scale.float())
  except AssertionError as e:
    print(f"  ERROR: per_token_cast_to_fp8 failed: {e}")
    print(f"  h_flat amax: {h_flat.abs().max().item()}")
    print(f"  h_flat has zeros rows: {(h_flat.abs().sum(dim=-1) == 0).sum().item()}")
    return None

  # Q projection
  print("\n[Step 2b] Q Projection GEMM")
  q_dim = attn.num_local_heads * attn.qk_nope_dim + attn.num_local_heads * attn.qk_rope_dim
  q_out = torch.empty(B * S, q_dim, dtype=torch.bfloat16, device=hidden.device)

  try:
    fp8_gemm_nt(
      (h_fp8, h_scale),
      (attn.q_proj_weight, attn.q_proj_weight_scale),
      q_out
    )
    check_tensor("q_out", q_out)
  except Exception as e:
    print(f"  ERROR: Q projection GEMM failed: {e}")
    return None

  # KV projection
  print("\n[Step 2c] KV Projection GEMM")
  kv_dim = attn.kv_lora_rank + attn.qk_rope_dim
  kv_out = torch.empty(B * S, kv_dim, dtype=torch.bfloat16, device=hidden.device)

  try:
    fp8_gemm_nt(
      (h_fp8, h_scale),
      (attn.kv_proj_weight, attn.kv_proj_weight_scale),
      kv_out
    )
    check_tensor("kv_out", kv_out)
  except Exception as e:
    print(f"  ERROR: KV projection GEMM failed: {e}")
    return None

  # Step 3: RoPE
  print("\n[Step 3] RoPE Application")
  # Extract rope components
  q_rope = q_out[:, attn.num_local_heads * attn.qk_nope_dim:].view(B, S, attn.num_local_heads, attn.qk_rope_dim)
  k_rope = kv_out[:, attn.kv_lora_rank:].view(B, S, 1, attn.qk_rope_dim)
  check_tensor("q_rope_before", q_rope)
  check_tensor("k_rope_before", k_rope)

  # Apply RoPE
  cos, sin = attn._get_rope(positions, attn.qk_rope_dim)
  check_tensor("cos", cos)
  check_tensor("sin", sin)

  q_rope_after = attn._apply_rope(q_rope, cos, sin)
  k_rope_after = attn._apply_rope(k_rope, cos, sin)
  check_tensor("q_rope_after", q_rope_after)
  check_tensor("k_rope_after", k_rope_after)

  # Step 4: DSA Indexer (if present)
  if hasattr(attn, 'k_norm'):
    print("\n[Step 4] DSA Indexer")
    kv_latent = kv_out[:, :attn.kv_lora_rank].view(B, S, attn.kv_lora_rank)
    check_tensor("kv_latent", kv_latent)

    # k_norm
    k_normed = attn.k_norm(kv_latent)
    check_tensor("k_normed", k_normed)

    # idx_k projection
    if hasattr(attn, 'idx_k_proj_weight'):
      check_tensor("idx_k_proj_weight", attn.idx_k_proj_weight)
      idx_k = torch.nn.functional.linear(k_normed, attn.idx_k_proj_weight.to(k_normed.dtype))
      check_tensor("idx_k", idx_k)

  # Step 5: Absorbed Q computation
  print("\n[Step 5] Absorbed Q (q_for_attn)")
  q_nope = q_out[:, :attn.num_local_heads * attn.qk_nope_dim].view(B, S, attn.num_local_heads, attn.qk_nope_dim)
  check_tensor("q_nope", q_nope)

  # wkv_b dequantization
  if hasattr(attn, '_wkv_b_dequant') and attn._wkv_b_dequant is not None:
    wkv_b = attn._wkv_b_dequant
  else:
    wkv_b = attn.wkv_b
  check_tensor("wkv_b", wkv_b)

  # q_nope @ wkv_b[:kv_lora_rank, :].T
  kv_lora_rank = attn.kv_lora_rank
  w_uk = wkv_b[:kv_lora_rank, :].T  # [num_heads * head_dim, kv_lora_rank]
  check_tensor("w_uk", w_uk)

  # Compute q_absorbed = q_nope @ W_UK (for absorbed attention)
  # This gives us the latent space query
  q_absorbed = torch.einsum('bshd,hdk->bshk', q_nope, w_uk.view(attn.num_local_heads, attn.qk_nope_dim, kv_lora_rank))
  check_tensor("q_absorbed", q_absorbed)

  # Concatenate with rope
  q_for_attn = torch.cat([q_absorbed, q_rope_after], dim=-1)  # [B, S, H, kv_lora_rank + qk_rope_dim]
  check_tensor("q_for_attn", q_for_attn)

  # Step 6: Pack KV cache
  print("\n[Step 6] KV Cache Packing")
  kv_latent = kv_out[:, :attn.kv_lora_rank].view(B, S, attn.kv_lora_rank)
  kv_for_cache = torch.cat([kv_latent, k_rope_after.squeeze(2)], dim=-1)  # [B, S, kv_lora_rank + qk_rope_dim]
  check_tensor("kv_for_cache", kv_for_cache)

  # Step 7: FlashMLA
  print("\n[Step 7] FlashMLA Attention")
  from flash_mla import get_mla_metadata, flash_mla_with_kvcache
  import math

  # Prepare Q for FlashMLA: [B, S, H, D] where D = kv_lora_rank + qk_rope_dim
  q_mla = q_for_attn
  check_tensor("q_mla", q_mla)

  # Get metadata
  metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    S * attn.num_local_heads,
    1,
    is_fp8_kvcache=False  # Using BF16 cache for debugging
  )
  print(f"  metadata shape: {metadata.shape}, num_splits: {num_splits}")

  # Run FlashMLA
  try:
    out, lse = flash_mla_with_kvcache(
      q_mla,
      kv_cache,
      block_table,
      cache_seqlens,
      attn.kv_lora_rank,  # head_dim_v
      metadata,
      num_splits,
      softmax_scale=1.0 / math.sqrt(attn.kv_lora_rank + attn.qk_rope_dim),
      causal=True,
    )
    check_tensor("attn_out", out)
    check_tensor("lse", lse)
  except Exception as e:
    print(f"  ERROR: FlashMLA failed: {e}")
    import traceback
    traceback.print_exc()
    return None

  # Step 8: Output projection
  print("\n[Step 8] Output Projection")
  # out shape: [B, S, H, kv_lora_rank]
  # Need to project back to hidden_dim

  # For absorbed attention, we use wkv_b[kv_lora_rank:, :] as output projection
  w_vo = wkv_b[kv_lora_rank:, :]  # [num_heads * head_dim, hidden_dim] or similar
  check_tensor("w_vo", w_vo)

  # Reshape and project
  out_flat = out.view(B * S, -1)  # [B*S, H * kv_lora_rank]
  check_tensor("out_flat", out_flat)

  print("\n" + "="*60)
  print("DEBUG COMPLETE")
  print("="*60)

  return out


def main():
  parser = argparse.ArgumentParser(description="Debug attention layer")
  parser.add_argument("--model-path", type=str, default="/data/models/DeepSeek-V3.2-Speciale")
  parser.add_argument("--num-layers", type=int, default=4, help="Number of layers to load")
  parser.add_argument("--layer-idx", type=int, default=0, help="Which layer to debug")
  args = parser.parse_args()

  print("="*60)
  print("nmoe.serve Attention Debug")
  print("="*60)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)
  torch.set_default_device(device)

  print(f"Model: {args.model_path}")
  print(f"Layers: {args.num_layers}")
  print(f"Debug layer: {args.layer_idx}")
  print(f"Device: {device}")
  print(f"CUDA: {torch.cuda.get_device_name()}")

  # Initialize distributed
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500", world_size=1, rank=0)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(0, 1)

  # Model config
  num_dense = min(3, args.num_layers)
  cfg = ModelConfig(num_layers=args.num_layers, num_dense_layers=num_dense)

  print(f"\nCreating model with {cfg.num_layers} layers...")

  # DeepEP buffer
  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  # Load checkpoint
  print(f"Loading checkpoint...")
  missing, unexpected = load_checkpoint(model, args.model_path, rank=0, world_size=1, cfg=cfg)
  print(f"Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

  # Create test input
  B, S = 1, 4
  input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)
  positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

  print(f"\nTest input: B={B}, S={S}")
  check_tensor("input_ids", input_ids)
  check_tensor("positions", positions)

  # Get embeddings
  print("\n[Embedding]")
  hidden = model.embed(input_ids)
  check_tensor("embedding_output", hidden)

  # Create dummy KV cache
  page_size = 64
  num_pages = 4
  kv_cache = torch.zeros(num_pages, page_size, 1, cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                         dtype=torch.bfloat16, device=device)
  idx_k_cache = torch.zeros(num_pages, page_size, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)

  # Block table and seqlens
  block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  # Debug the attention layer
  debug_attention_forward(
    model, args.layer_idx, hidden, positions,
    kv_cache, idx_k_cache, block_table, cache_seqlens, out_loc
  )


if __name__ == "__main__":
  main()
