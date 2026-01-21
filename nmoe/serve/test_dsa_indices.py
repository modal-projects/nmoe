# SPDX-License-Identifier: Apache-2.0
"""Debug DSA indices during prefill vs decode."""

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
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Use just 1 layer to focus on DSA
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

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("DSA Index Debug Test")
    print("=" * 70)

  # Simple test: 5 tokens
  S = 5
  input_ids = torch.tensor([[1, 100, 1000, 10000, 50000]], device=device)

  # Setup caches
  num_blocks = 4
  kv_caches = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

  # === PREFILL ===
  if rank == 0:
    print(f"\n=== PREFILL ({S} tokens) ===")
  positions = torch.arange(S, device=device).unsqueeze(0)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

  # Patch the model to capture DSA indices
  layer = model.layers[0].attn
  _orig_forward = layer.forward

  captured_indices = []

  def capturing_forward(*args, **kwargs):
    result = _orig_forward(*args, **kwargs)
    return result

  # Manually trace through to capture indices
  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

    model._ensure_freqs(device)
    freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(1, S, -1)

    # Get q_latent and q_idx
    from nmoe.serve.model import apply_rotary_emb, rotate_activation
    q_latent = layer.q_norm(layer.wq_a(x_norm))
    q_idx_all = layer.wq_idx(q_latent).view(1, S, layer.n_idx_heads, layer.idx_dim)
    q_idx_pe, q_idx_nope = torch.split(q_idx_all, [layer.qk_rope_head_dim, layer.idx_dim - layer.qk_rope_head_dim], dim=-1)
    q_idx_pe = apply_rotary_emb(q_idx_pe, freqs, interleaved=False)
    q_idx_all = torch.cat([q_idx_pe, q_idx_nope], dim=-1)

    # K indexer
    k_idx_raw = layer.k_norm(layer.wk_idx(x_norm))
    k_idx_pe, k_idx_nope = torch.split(k_idx_raw, [layer.qk_rope_head_dim, layer.idx_dim - layer.qk_rope_head_dim], dim=-1)
    k_idx_pe = apply_rotary_emb(k_idx_pe.unsqueeze(2), freqs, interleaved=False).squeeze(2)
    k_idx_processed = torch.cat([k_idx_pe, k_idx_nope], dim=-1)

    # Hadamard
    q_idx_all = rotate_activation(q_idx_all.to(torch.bfloat16))
    k_idx_processed = rotate_activation(k_idx_processed.to(torch.bfloat16))

    # Store K indexer in cache
    k_idx_new = k_idx_processed.reshape(S, layer.idx_dim).contiguous()
    loc = out_loc.reshape(S).to(torch.int64)
    idx_k_caches[0].view(-1, layer.idx_dim).index_copy_(0, loc, k_idx_new)

    # Compute indexer scores manually
    from nmoe.triton.dsa import compute_indexer_scores
    idx_softmax_scale = layer.idx_dim ** -0.5
    w_idx_all = layer.w_idx(x_norm.float()).view(1, S, layer.n_idx_heads) * (layer.n_idx_heads ** -0.5) * idx_softmax_scale

    ctx_len = S
    phys_ids = torch.arange(ctx_len, device=device, dtype=torch.int64)
    k_ctx = idx_k_caches[0].view(-1, layer.idx_dim).index_select(0, phys_ids)

    q_idx = q_idx_all
    k_idx = k_ctx.unsqueeze(0)
    w_idx = w_idx_all

    scores = compute_indexer_scores(q_idx, k_idx, w_idx, causal=False)
    scores = scores.squeeze(0)  # [S, ctx_len]

    # Apply causal mask
    q_pos = positions.squeeze(0).to(torch.int64)
    k_pos = torch.arange(ctx_len, device=device, dtype=torch.int64)
    causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    scores_masked = scores.masked_fill(causal_mask, float("-inf"))

    # Select top-k
    k_sel = min(layer.topk, ctx_len)
    vals, topk_idx = scores_masked.topk(k_sel, dim=-1)

    if rank == 0:
      print(f"\nPrefill DSA scores shape: {scores.shape}")
      print(f"Prefill causal_mask sum: {causal_mask.sum()} (should mask future tokens)")
      print(f"Prefill scores (before mask):")
      print(f"  Position 0: {scores[0, :].tolist()}")
      print(f"  Position 4: {scores[4, :].tolist()}")
      print(f"Prefill top-k indices:")
      for s in range(S):
        valid_topk = topk_idx[s][torch.isfinite(vals[s])]
        print(f"  Position {s}: top-{len(valid_topk)} = {valid_topk[:5].tolist()}...")

  # === DECODE ===
  if rank == 0:
    print(f"\n=== DECODE (1 token at position {S}) ===")

  # Simulate processing next token at position S
  decode_input = torch.tensor([[99999]], device=device)  # Some token
  decode_pos = torch.tensor([[S]], dtype=torch.int64, device=device)
  decode_out_loc = torch.tensor([[S]], dtype=torch.int32, device=device)
  decode_cache_seqlens = torch.tensor([S + 1], dtype=torch.int32, device=device)

  with torch.no_grad():
    x_dec = model.embed(decode_input)
    x_dec_norm = model.layers[0].attn_norm(x_dec)

    freqs_dec = model.freqs_cis.index_select(0, decode_pos.view(-1).to(torch.int64)).view(1, 1, -1)

    q_latent_dec = layer.q_norm(layer.wq_a(x_dec_norm))
    q_idx_dec = layer.wq_idx(q_latent_dec).view(1, 1, layer.n_idx_heads, layer.idx_dim)
    q_idx_pe_dec, q_idx_nope_dec = torch.split(q_idx_dec, [layer.qk_rope_head_dim, layer.idx_dim - layer.qk_rope_head_dim], dim=-1)
    q_idx_pe_dec = apply_rotary_emb(q_idx_pe_dec, freqs_dec, interleaved=False)
    q_idx_dec = torch.cat([q_idx_pe_dec, q_idx_nope_dec], dim=-1)
    q_idx_dec = rotate_activation(q_idx_dec.to(torch.bfloat16))

    # K indexer for new token
    k_idx_raw_dec = layer.k_norm(layer.wk_idx(x_dec_norm))
    k_idx_pe_dec, k_idx_nope_dec = torch.split(k_idx_raw_dec, [layer.qk_rope_head_dim, layer.idx_dim - layer.qk_rope_head_dim], dim=-1)
    k_idx_pe_dec = apply_rotary_emb(k_idx_pe_dec.unsqueeze(2), freqs_dec, interleaved=False).squeeze(2)
    k_idx_dec = torch.cat([k_idx_pe_dec, k_idx_nope_dec], dim=-1)
    k_idx_dec = rotate_activation(k_idx_dec.to(torch.bfloat16))

    # Store new K indexer
    k_idx_new_dec = k_idx_dec.reshape(1, layer.idx_dim).contiguous()
    loc_dec = decode_out_loc.reshape(1).to(torch.int64)
    idx_k_caches[0].view(-1, layer.idx_dim).index_copy_(0, loc_dec, k_idx_new_dec)

    # Gather all ctx_len=S+1 idx_k from cache
    ctx_len_dec = S + 1
    phys_ids_dec = torch.arange(ctx_len_dec, device=device, dtype=torch.int64)
    k_ctx_dec = idx_k_caches[0].view(-1, layer.idx_dim).index_select(0, phys_ids_dec)

    # Compute indexer scores
    w_idx_dec = layer.w_idx(x_dec_norm.float()).view(1, 1, layer.n_idx_heads) * (layer.n_idx_heads ** -0.5) * idx_softmax_scale

    scores_dec = compute_indexer_scores(q_idx_dec, k_ctx_dec.unsqueeze(0), w_idx_dec, causal=False)
    scores_dec = scores_dec.squeeze(0)  # [1, ctx_len]

    # Apply causal mask
    q_pos_dec = decode_pos.squeeze(0).to(torch.int64)  # [1] = [S]
    k_pos_dec = torch.arange(ctx_len_dec, device=device, dtype=torch.int64)  # [0..S]
    causal_mask_dec = k_pos_dec.unsqueeze(0) > q_pos_dec.unsqueeze(1)  # Position S can attend to all 0..S
    scores_dec_masked = scores_dec.masked_fill(causal_mask_dec, float("-inf"))

    k_sel_dec = min(layer.topk, ctx_len_dec)
    vals_dec, topk_idx_dec = scores_dec_masked.topk(k_sel_dec, dim=-1)

    if rank == 0:
      print(f"\nDecode DSA scores shape: {scores_dec.shape}")
      print(f"Decode causal_mask sum: {causal_mask_dec.sum()} (should be 0 - position S can see all)")
      print(f"Decode scores: {scores_dec[0, :].tolist()}")
      print(f"Decode top-k indices: {topk_idx_dec[0, :10].tolist()}...")
      print(f"Decode top-k scores: {vals_dec[0, :10].tolist()}...")

      # Check if scores look reasonable
      print(f"\nAnalysis:")
      print(f"  Q_idx norm (decode): {q_idx_dec.float().norm():.4f}")
      print(f"  K_ctx norm (decode): {k_ctx_dec.float().norm():.4f}")
      print(f"  w_idx (decode): {w_idx_dec.squeeze().tolist()[:5]}...")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
