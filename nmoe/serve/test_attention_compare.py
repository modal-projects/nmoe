# SPDX-License-Identifier: Apache-2.0
"""Compare attention computation with reference implementation."""

import os
from pathlib import Path

def _maybe_set_cutlass_path():
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


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, weight_dequant, apply_rotary_emb
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  # Use 1 layer only
  cfg = ModelConfig(num_layers=1, num_dense_layers=1)

  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("Attention Computation Comparison")
    print("=" * 70)

  # Test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  model._ensure_freqs(device)
  freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

  attn = model.layers[0].attn

  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

  if rank == 0:
    print(f"\nInput x_norm: shape={x_norm.shape}, mean={x_norm.float().mean():.6f}")

  # === Compute Q ===
  with torch.no_grad():
    q_latent = attn.q_norm(attn.wq_a(x_norm))
    q = attn.wq_b(q_latent).view(B, S, attn.num_local_heads, attn.qk_head_dim)
    q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs)  # interleaved=True

  if rank == 0:
    print(f"\nQ computation:")
    print(f"  q_nope: shape={q_nope.shape}, mean={q_nope.float().mean():.6f}, amax={q_nope.float().abs().max():.4f}")
    print(f"  q_pe: shape={q_pe.shape}, mean={q_pe.float().mean():.6f}, amax={q_pe.float().abs().max():.4f}")

  # === Compute KV ===
  with torch.no_grad():
    kv = attn.wkv_a(x_norm)
    kv_latent, k_pe = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_latent = attn.kv_norm(kv_latent)
    k_rope = apply_rotary_emb(k_pe.unsqueeze(2), freqs).squeeze(2)  # interleaved=True

  if rank == 0:
    print(f"\nKV computation:")
    print(f"  kv_latent: shape={kv_latent.shape}, mean={kv_latent.float().mean():.6f}, amax={kv_latent.float().abs().max():.4f}")
    print(f"  k_rope: shape={k_rope.shape}, mean={k_rope.float().mean():.6f}, amax={k_rope.float().abs().max():.4f}")

  # === Reference-style attention (no FlashMLA) ===
  # Following reference decode path exactly
  with torch.no_grad():
    # Dequant wkv_b
    wkv_b_dequant = weight_dequant(attn.wkv_b.weight, attn.wkv_b.weight_scale_inv)
    wkv_b_w = wkv_b_dequant.view(attn.num_local_heads, -1, attn.kv_lora_rank)

    # Absorb q_nope: q_nope_abs = q_nope @ wkv_b[:, :qk_nope_head_dim]
    q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_w[:, :attn.qk_nope_head_dim])

  if rank == 0:
    print(f"\nAbsorbed Q:")
    print(f"  q_nope_abs: shape={q_nope_abs.shape}, mean={q_nope_abs.float().mean():.6f}, amax={q_nope_abs.float().abs().max():.4f}")

  with torch.no_grad():
    # Reference-style score computation:
    # scores = (q_nope_abs @ kv_latent.T + q_pe @ k_rope.T) * softmax_scale
    scores_nope = torch.einsum("bshc,btc->bsht", q_nope_abs, kv_latent)
    scores_pe = torch.einsum("bshr,btr->bsht", q_pe, k_rope)
    scores_ref = (scores_nope + scores_pe) * attn.softmax_scale

  if rank == 0:
    print(f"\nReference-style scores:")
    print(f"  scores_nope: mean={scores_nope.float().mean():.6f}, amax={scores_nope.float().abs().max():.4f}")
    print(f"  scores_pe: mean={scores_pe.float().mean():.6f}, amax={scores_pe.float().abs().max():.4f}")
    print(f"  scores_ref: mean={scores_ref.float().mean():.6f}, amax={scores_ref.float().abs().max():.4f}")

  # Apply causal mask - scores_ref is [B, S, H, T]
  with torch.no_grad():
    causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
    # Expand to [1, S, 1, T]
    scores_ref_masked = scores_ref.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
    attn_weights_ref = torch.softmax(scores_ref_masked, dim=-1)

    # Reference output: attn @ kv_latent, then project via wkv_b
    out_latent_ref = torch.einsum("bsht,btc->bshc", attn_weights_ref, kv_latent)
    out_ref = torch.einsum("bshc,hdc->bshd", out_latent_ref, wkv_b_w[:, -attn.v_head_dim:])

  if rank == 0:
    print(f"\nReference output:")
    print(f"  out_latent_ref: shape={out_latent_ref.shape}, mean={out_latent_ref.float().mean():.6f}")
    print(f"  out_ref: shape={out_ref.shape}, mean={out_ref.float().mean():.6f}")

  # === Now test FlashMLA ===
  # Setup caches
  num_blocks = 1
  kv_cache = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
  idx_k_cache = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  # Run full attention forward
  with torch.no_grad():
    attn_out = attn(
      x_norm, freqs,
      kv_cache=kv_cache, idx_k_cache=idx_k_cache,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc, positions=positions,
    )

  if rank == 0:
    print(f"\nFlashMLA attention output:")
    print(f"  attn_out: shape={attn_out.shape}, mean={attn_out.float().mean():.6f}")

    # Compare reference vs FlashMLA
    # Note: Reference uses full causal attention, FlashMLA uses sparse (topk)
    # They won't match exactly, but should be similar
    out_ref_proj = attn.wo(out_ref.flatten(2))
    print(f"\nComparison (reference vs FlashMLA):")
    print(f"  Reference (wo projected): mean={out_ref_proj.float().mean():.6f}, amax={out_ref_proj.float().abs().max():.4f}")
    print(f"  FlashMLA: mean={attn_out.float().mean():.6f}, amax={attn_out.float().abs().max():.4f}")

    diff = (out_ref_proj.float() - attn_out.float()).abs()
    print(f"  Difference: max={diff.max():.6f}, mean={diff.mean():.6f}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
