# SPDX-License-Identifier: Apache-2.0
"""Module-by-module comparison to find divergence from reference."""

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
import torch.nn.functional as F
import torch.distributed as dist


def reference_rope(x, freqs_cis, interleaved=True):
  """Reference RoPE implementation."""
  # x: [..., head_dim] where head_dim is the rope dimension
  # freqs_cis: [..., head_dim//2] complex
  xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
  freqs_cis = freqs_cis.view(*([1] * (x.ndim - 2)), freqs_cis.shape[-2], freqs_cis.shape[-1])

  if interleaved:
    # Interleaved: pairs of (real, imag) adjacent
    x_complex = torch.view_as_complex(xshaped)
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
  else:
    # Non-interleaved: first half real, second half imag
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = freqs_cis.real
    sin = freqs_cis.imag
    x_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

  return x_out.type_as(x)


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, precompute_freqs_cis
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  # Use 1 layer for detailed debugging
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
    print("Module-by-Module Comparison")
    print("=" * 70)

  # Simple test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  # Ensure freqs are computed
  model._ensure_freqs(device)
  freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

  def print_stats(name, tensor, check_nan=True):
    if rank == 0:
      has_nan = torch.isnan(tensor).any().item()
      has_inf = torch.isinf(tensor).any().item()
      mean = tensor.float().mean().item()
      std = tensor.float().std().item()
      amax = tensor.float().abs().max().item()
      status = "NaN!" if has_nan else ("Inf!" if has_inf else "OK")
      print(f"  {name}: mean={mean:.6f}, std={std:.6f}, amax={amax:.4f} [{status}]")

  # ===== EMBEDDING =====
  if rank == 0:
    print("\n--- Embedding ---")

  with torch.no_grad():
    x = model.embed(input_ids)
  print_stats("embed_out", x)

  # ===== LAYER 0: ATTENTION NORM =====
  if rank == 0:
    print("\n--- Layer 0: Attention ---")

  attn = model.layers[0].attn

  with torch.no_grad():
    x_norm = model.layers[0].attn_norm(x)
  print_stats("attn_norm", x_norm)

  # ===== Q LoRA =====
  with torch.no_grad():
    q_latent = attn.q_norm(attn.wq_a(x_norm))
  print_stats("q_latent (after q_norm)", q_latent)

  # ===== Q projection =====
  with torch.no_grad():
    q_full = attn.wq_b(q_latent)
    q_full = q_full.view(B, S, attn.num_local_heads, attn.qk_head_dim)
  print_stats("q_full", q_full)

  # ===== Q RoPE =====
  with torch.no_grad():
    q_pe = q_full[..., :cfg.qk_rope_head_dim]
    q_nope = q_full[..., cfg.qk_rope_head_dim:]
  print_stats("q_pe (before RoPE)", q_pe)
  print_stats("q_nope", q_nope)

  # Check RoPE application
  with torch.no_grad():
    from nmoe.serve.model import apply_rotary_emb
    q_pe_rotated = apply_rotary_emb(q_pe, freqs, interleaved=False)
  print_stats("q_pe (after RoPE, non-interleaved)", q_pe_rotated)

  # Compare with interleaved RoPE
  with torch.no_grad():
    q_pe_rotated_interleaved = apply_rotary_emb(q_pe, freqs, interleaved=True)
  print_stats("q_pe (after RoPE, interleaved)", q_pe_rotated_interleaved)

  # ===== KV LoRA =====
  with torch.no_grad():
    kv_a_out = attn.wkv_a(x_norm)  # [B, S, kv_lora_rank + qk_rope_head_dim]
    kv_latent = kv_a_out[..., :attn.kv_lora_rank]
    k_pe_raw = kv_a_out[..., attn.kv_lora_rank:]
  print_stats("kv_latent", kv_latent)
  print_stats("k_pe_raw", k_pe_raw)

  with torch.no_grad():
    kv_latent_normed = attn.kv_norm(kv_latent)
  print_stats("kv_latent_normed", kv_latent_normed)

  # ===== K RoPE =====
  with torch.no_grad():
    k_pe_rotated = apply_rotary_emb(k_pe_raw.unsqueeze(2), freqs, interleaved=False).squeeze(2)
  print_stats("k_pe (after RoPE)", k_pe_rotated)

  # ===== DSA Indexer =====
  if rank == 0:
    print("\n--- DSA Indexer ---")

  with torch.no_grad():
    # Q indexer
    q_idx_all = attn.wq_idx(q_latent).view(B, S, attn.n_idx_heads, attn.idx_dim)
  print_stats("q_idx (before RoPE)", q_idx_all)

  with torch.no_grad():
    q_idx_pe = q_idx_all[..., :cfg.qk_rope_head_dim]
    q_idx_nope = q_idx_all[..., cfg.qk_rope_head_dim:]
    q_idx_pe_rot = apply_rotary_emb(q_idx_pe, freqs, interleaved=False)
    q_idx_with_rope = torch.cat([q_idx_pe_rot, q_idx_nope], dim=-1)
  print_stats("q_idx (after RoPE)", q_idx_with_rope)

  # Hadamard
  with torch.no_grad():
    from nmoe.serve.model import rotate_activation
    q_idx_final = rotate_activation(q_idx_with_rope.to(torch.bfloat16))
  print_stats("q_idx (after Hadamard)", q_idx_final)

  # K indexer
  with torch.no_grad():
    k_idx_raw = attn.k_norm(attn.wk_idx(x_norm))
  print_stats("k_idx_raw", k_idx_raw)

  with torch.no_grad():
    k_idx_pe = k_idx_raw[..., :cfg.qk_rope_head_dim]
    k_idx_nope = k_idx_raw[..., cfg.qk_rope_head_dim:]
    k_idx_pe_rot = apply_rotary_emb(k_idx_pe.unsqueeze(2), freqs, interleaved=False).squeeze(2)
    k_idx_with_rope = torch.cat([k_idx_pe_rot, k_idx_nope], dim=-1)
    k_idx_final = rotate_activation(k_idx_with_rope.to(torch.bfloat16))
  print_stats("k_idx (after Hadamard)", k_idx_final)

  # Weights
  with torch.no_grad():
    idx_softmax_scale = attn.idx_dim ** -0.5
    w_idx = attn.w_idx(x_norm.float()).view(B, S, attn.n_idx_heads) * (attn.n_idx_heads ** -0.5) * idx_softmax_scale
  print_stats("w_idx", w_idx)

  # ===== Compute indexer scores =====
  if rank == 0:
    print("\n--- Indexer Scores ---")

  with torch.no_grad():
    # Reference score: I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h} . k_s)
    # q_idx_final: [B, S, H, D], k_idx_final: [B, S, D]
    qk = torch.einsum('bthd,bsd->bths', q_idx_final.float(), k_idx_final.float())
    qk_relu = F.relu(qk)
    scores = torch.einsum('bth,bths->bts', w_idx, qk_relu)
  print_stats("indexer_scores (ref impl)", scores)

  # Check top-k indices
  if rank == 0:
    for t in range(S):
      # Causal mask
      mask = torch.arange(S, device=device) > t
      masked_scores = scores[0, t, :].clone()
      masked_scores[mask] = float('-inf')
      topk_vals, topk_idx = masked_scores.topk(min(4, t + 1))
      print(f"  Token {t}: topk_idx={topk_idx.tolist()}, topk_vals={[f'{v:.2f}' for v in topk_vals.tolist()]}")

  # ===== FFN =====
  if rank == 0:
    print("\n--- Layer 0: FFN (Dense MLP) ---")

  with torch.no_grad():
    # Run full attention to get attn output
    num_blocks = 1
    kv_cache = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    idx_k_cache = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
    out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

    attn_out = attn(
      x_norm, freqs,
      kv_cache=kv_cache, idx_k_cache=idx_k_cache,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc, positions=positions,
    )
  print_stats("attn_out", attn_out)

  with torch.no_grad():
    x_after_attn = x + attn_out
    x_ffn_norm = model.layers[0].ffn_norm(x_after_attn)
  print_stats("x_after_attn", x_after_attn)
  print_stats("x_ffn_norm", x_ffn_norm)

  # FFN forward
  ffn = model.layers[0].ffn
  with torch.no_grad():
    w1_out = ffn.w1(x_ffn_norm)
    w3_out = ffn.w3(x_ffn_norm)
  print_stats("ffn.w1 output", w1_out)
  print_stats("ffn.w3 output", w3_out)

  with torch.no_grad():
    gate_up = F.silu(w1_out.float()) * w3_out.float()
  print_stats("gate * up (float32)", gate_up)

  with torch.no_grad():
    ffn_out = ffn.w2(gate_up.to(x_ffn_norm.dtype))
  print_stats("ffn_out (w2)", ffn_out)

  # ===== Final output =====
  if rank == 0:
    print("\n--- Final Output ---")

  with torch.no_grad():
    x_final = x_after_attn + ffn_out
    x_normed = model.norm(x_final)
    logits = model.lm_head(x_normed.float())

    if world_size > 1:
      all_logits = [torch.empty_like(logits) for _ in range(world_size)]
      dist.all_gather(all_logits, logits)
      logits = torch.cat(all_logits, dim=-1)

  print_stats("x_final", x_final)
  print_stats("x_normed", x_normed)
  print_stats("logits", logits)

  if rank == 0:
    argmax = logits[0, -1, :].argmax().item()
    top5 = logits[0, -1, :].topk(5)
    print(f"\n  Argmax: {argmax}")
    print(f"  Top 5: {top5.indices.tolist()}")
    print(f"  Top 5 logits: {[f'{v:.2f}' for v in top5.values.tolist()]}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
