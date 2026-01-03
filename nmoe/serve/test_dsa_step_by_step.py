# SPDX-License-Identifier: Apache-2.0
"""Step-by-step verification of DSA indexer against reference logic.

This test isolates each component of the DSA computation to find bugs.
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

import torch
import torch.nn.functional as F
import torch.distributed as dist


def reference_indexer_score(q, k, w, softmax_scale):
  """Reference implementation of DSA indexer score.

  Args:
    q: [B, S, H, D] - queries after RoPE and Hadamard
    k: [B, N, D] - keys after RoPE and Hadamard
    w: [B, S, H] - weights (already scaled by n_heads**-0.5 * softmax_scale)

  Formula: I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . k_s)
  """
  B, S, H, D = q.shape
  N = k.shape[1]

  # Compute dot products: [B, S, H, N]
  qk = torch.einsum('bshd,bnd->bshn', q.float(), k.float())

  # Apply ReLU
  qk_relu = F.relu(qk)

  # Weighted sum over heads: [B, S, N]
  scores = torch.einsum('bsh,bshn->bsn', w.float(), qk_relu)

  return scores


def test_dsa_indexer_output():
  """Test that our DSA indexer produces reasonable indices."""

  # Initialize distributed (single GPU)
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, precompute_freqs_cis
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(0, 1)

  # Use minimal model (1 layer) for faster testing
  cfg = ModelConfig(num_layers=1, num_dense_layers=1)

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  # Load checkpoint
  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=0, world_size=1, cfg=cfg)

  print("=" * 60)
  print("DSA Indexer Step-by-Step Test")
  print("=" * 60)

  # Create simple test input
  B, S = 1, 8
  input_ids = torch.tensor([[1, 100, 200, 300, 400, 500, 600, 700]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  num_blocks = 2
  kv_caches = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  # Get the attention layer
  attn = model.layers[0].attn
  freqs_cis = precompute_freqs_cis(cfg, device)

  # Run embed and get hidden states
  x = model.embed(input_ids)  # [B, S, H]

  print(f"\n1. Input hidden states: shape={x.shape}, mean={x.mean():.4f}")

  # === Replicate DSA indexer computations ===

  # Get Q LoRA latent
  q_latent = attn.q_norm(attn.wq_a(x))  # [B, S, q_lora_rank]
  print(f"\n2. Q LoRA latent: shape={q_latent.shape}, mean={q_latent.mean():.4f}")

  # Indexer Q projection
  q_idx_all = attn.wq_idx(q_latent).view(B, S, attn.n_idx_heads, attn.idx_dim)
  print(f"\n3. Q indexer (before RoPE): shape={q_idx_all.shape}, mean={q_idx_all.mean():.4f}")

  # Apply RoPE (non-interleaved)
  from nmoe.serve.model import apply_rotary_emb, rotate_activation

  fc = freqs_cis[positions.flatten()].view(B, S, -1)
  q_idx_pe, q_idx_nope = torch.split(q_idx_all, [cfg.qk_rope_head_dim, attn.idx_dim - cfg.qk_rope_head_dim], dim=-1)
  q_idx_pe_rotated = apply_rotary_emb(q_idx_pe, fc, interleaved=False)
  q_idx_with_rope = torch.cat([q_idx_pe_rotated, q_idx_nope], dim=-1)
  print(f"\n4. Q indexer (after RoPE): mean={q_idx_with_rope.mean():.4f}")

  # Apply Hadamard
  q_idx_final = rotate_activation(q_idx_with_rope.to(torch.bfloat16))
  print(f"\n5. Q indexer (after Hadamard): mean={q_idx_final.mean():.4f}, std={q_idx_final.std():.4f}")

  # K indexer
  k_idx_raw = attn.k_norm(attn.wk_idx(x))
  k_idx_pe, k_idx_nope = torch.split(k_idx_raw, [cfg.qk_rope_head_dim, attn.idx_dim - cfg.qk_rope_head_dim], dim=-1)
  k_idx_pe_rotated = apply_rotary_emb(k_idx_pe.unsqueeze(2), fc, interleaved=False).squeeze(2)
  k_idx_with_rope = torch.cat([k_idx_pe_rotated, k_idx_nope], dim=-1)
  k_idx_final = rotate_activation(k_idx_with_rope.to(torch.bfloat16))
  print(f"\n6. K indexer (after Hadamard): mean={k_idx_final.mean():.4f}, std={k_idx_final.std():.4f}")

  # Weights
  idx_softmax_scale = attn.idx_dim ** -0.5
  w_idx = attn.w_idx(x.float()).view(B, S, attn.n_idx_heads) * (attn.n_idx_heads ** -0.5) * idx_softmax_scale
  print(f"\n7. Weights: mean={w_idx.mean():.4f}, std={w_idx.std():.4f}")

  # Compute scores using reference implementation
  k_ctx = k_idx_final.squeeze(0)  # [S, D] for this simple case
  scores_ref = reference_indexer_score(q_idx_final, k_ctx.unsqueeze(0), w_idx, idx_softmax_scale)
  print(f"\n8. Reference scores: shape={scores_ref.shape}")
  print(f"   scores[0,0,:] = {scores_ref[0, 0, :].tolist()}")  # Query 0 scores for all keys
  print(f"   scores[0,-1,:] = {scores_ref[0, -1, :].tolist()}")  # Last query scores

  # Compute scores using our triton kernel
  from nmoe.triton.dsa import compute_indexer_scores
  scores_triton = compute_indexer_scores(q_idx_final, k_ctx.unsqueeze(0), w_idx, causal=False)
  print(f"\n9. Triton scores: shape={scores_triton.shape}")
  print(f"   scores[0,0,:] = {scores_triton[0, 0, :].tolist()}")
  print(f"   scores[0,-1,:] = {scores_triton[0, -1, :].tolist()}")

  # Compare
  diff = (scores_ref - scores_triton).abs()
  print(f"\n10. Score difference: max={diff.max():.6f}, mean={diff.mean():.6f}")

  # Apply causal mask and get top-k
  for q_pos in range(S):
    mask = torch.arange(S, device=device) > q_pos
    masked_scores = scores_triton[0, q_pos, :].clone()
    masked_scores[mask] = float('-inf')
    topk_vals, topk_idx = masked_scores.topk(min(4, q_pos + 1))
    print(f"\n11. Position {q_pos}: top-k indices = {topk_idx.tolist()}, values = {[f'{v:.2f}' for v in topk_vals.tolist()]}")

  # Now run full forward and check indices
  print("\n" + "=" * 60)
  print("Running full forward pass...")
  print("=" * 60)

  with torch.no_grad():
    # We need to hook into the model to see the indices
    # For now, just verify the model runs without NaN
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  print(f"\nLogits: shape={logits.shape}")
  print(f"Has NaN: {torch.isnan(logits).any().item()}")
  print(f"Logits max: {logits.abs().max().item():.4f}")

  # Check argmax
  argmax = logits[0, -1, :].argmax().item()
  top5 = logits[0, -1, :].topk(5)
  print(f"\nLast position argmax: {argmax}")
  print(f"Top 5: indices={top5.indices.tolist()}, values={[f'{v:.2f}' for v in top5.values.tolist()]}")


if __name__ == "__main__":
  test_dsa_indexer_output()
