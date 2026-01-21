# SPDX-License-Identifier: Apache-2.0
"""Module-by-module comparison: our impl vs PyTorch reference."""

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


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import (
    ModelConfig, DeepSeekV3, init_distributed, weight_dequant,
    apply_rotary_emb, rotate_activation
  )
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  # Use 1 layer for detailed analysis
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
    print("Module-by-Module Comparison: Our Impl vs PyTorch Reference")
    print("=" * 70)

  # Test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  model._ensure_freqs(device)
  freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

  attn = model.layers[0].attn
  ffn = model.layers[0].ffn

  def pstat(name, t):
    if rank == 0:
      print(f"  {name}: mean={t.float().mean():.6f}, std={t.float().std():.6f}, amax={t.float().abs().max():.4f}")

  # ========== EMBEDDING ==========
  if rank == 0:
    print("\n" + "="*50)
    print("EMBEDDING")
    print("="*50)

  with torch.no_grad():
    x = model.embed(input_ids)
  pstat("embed_out", x)

  # ========== ATTN NORM ==========
  if rank == 0:
    print("\n" + "="*50)
    print("ATTENTION NORM")
    print("="*50)

  with torch.no_grad():
    x_norm = model.layers[0].attn_norm(x)
  pstat("x_norm", x_norm)

  # ========== Q PROJECTION ==========
  if rank == 0:
    print("\n" + "="*50)
    print("Q PROJECTION (LoRA)")
    print("="*50)

  with torch.no_grad():
    # Our impl uses FP8 for wq_a, wq_b
    q_latent = attn.q_norm(attn.wq_a(x_norm))
    q_full = attn.wq_b(q_latent)
    q_full = q_full.view(B, S, attn.num_local_heads, attn.qk_head_dim)
    q_nope, q_pe = q_full.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

  pstat("q_latent", q_latent)
  pstat("q_full", q_full)
  pstat("q_nope (before RoPE)", q_nope)
  pstat("q_pe (before RoPE)", q_pe)

  # Reference Q using dequantized weights
  with torch.no_grad():
    wq_a_dequant = weight_dequant(attn.wq_a.weight, attn.wq_a.weight_scale_inv)
    wq_b_dequant = weight_dequant(attn.wq_b.weight, attn.wq_b.weight_scale_inv)

    q_latent_ref = F.rms_norm(F.linear(x_norm, wq_a_dequant), (attn.q_lora_rank,), attn.q_norm.weight, attn.q_norm.eps)
    q_full_ref = F.linear(q_latent_ref, wq_b_dequant)
    q_full_ref = q_full_ref.view(B, S, attn.num_local_heads, attn.qk_head_dim)

  if rank == 0:
    diff = (q_full.float() - q_full_ref.float()).abs()
    print(f"\n  Q diff (FP8 vs BF16 ref): max={diff.max():.6f}, mean={diff.mean():.6f}")

  # Apply RoPE
  with torch.no_grad():
    q_pe_rotated = apply_rotary_emb(q_pe, freqs)  # interleaved=True
  pstat("q_pe (after RoPE)", q_pe_rotated)

  # ========== KV PROJECTION ==========
  if rank == 0:
    print("\n" + "="*50)
    print("KV PROJECTION")
    print("="*50)

  with torch.no_grad():
    kv = attn.wkv_a(x_norm)
    kv_latent, k_pe = kv.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_latent = attn.kv_norm(kv_latent)
    k_rope = apply_rotary_emb(k_pe.unsqueeze(2), freqs).squeeze(2)

  pstat("kv (raw)", kv)
  pstat("kv_latent (normed)", kv_latent)
  pstat("k_pe (before RoPE)", k_pe)
  pstat("k_rope (after RoPE)", k_rope)

  # Reference KV - wkv_a has non-128-aligned dims (576), needs special handling
  def weight_dequant_padded(weight, scale, block_size=128):
    """Dequantize FP8 weight that may have non-128-aligned dimensions."""
    out_feat, in_feat = weight.shape
    out_tiles, in_tiles = scale.shape
    # Pad weight to match scale tile count
    out_padded = out_tiles * block_size
    in_padded = in_tiles * block_size
    w_pad = torch.zeros(out_padded, in_padded, dtype=weight.dtype, device=weight.device)
    w_pad[:out_feat, :in_feat] = weight
    # Dequant
    w_pad = w_pad.view(out_tiles, block_size, in_tiles, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(-1, block_size * block_size)
    w_pad = (w_pad.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
    w_pad = w_pad.view(out_tiles, in_tiles, block_size, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(out_padded, in_padded)
    # Unpad
    return w_pad[:out_feat, :in_feat]

  with torch.no_grad():
    wkv_a_dequant = weight_dequant_padded(attn.wkv_a.weight, attn.wkv_a.weight_scale_inv)
    kv_ref = F.linear(x_norm, wkv_a_dequant)
    kv_latent_ref, k_pe_ref = kv_ref.split([attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    kv_latent_ref = F.rms_norm(kv_latent_ref, (attn.kv_lora_rank,), attn.kv_norm.weight, attn.kv_norm.eps)

  if rank == 0:
    diff = (kv_latent.float() - kv_latent_ref.float()).abs()
    print(f"\n  KV latent diff (FP8 vs BF16 ref): max={diff.max():.6f}, mean={diff.mean():.6f}")

  # ========== ATTENTION SCORES (Reference Style) ==========
  if rank == 0:
    print("\n" + "="*50)
    print("ATTENTION SCORES (Reference Decode Style)")
    print("="*50)

  with torch.no_grad():
    # Dequant wkv_b
    wkv_b_dequant = weight_dequant(attn.wkv_b.weight, attn.wkv_b.weight_scale_inv)
    wkv_b_w = wkv_b_dequant.view(attn.num_local_heads, -1, attn.kv_lora_rank)

    # Absorb q_nope: [B,S,H,128] @ [H,128,512] -> [B,S,H,512]
    q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_w[:, :attn.qk_nope_head_dim])

    # Reference score: (q_nope_abs @ kv_latent) + (q_pe @ k_rope)
    # q_nope_abs: [B,S,H,512], kv_latent: [B,T,512] -> [B,S,H,T]
    # q_pe: [B,S,H,64], k_rope: [B,T,64] -> [B,S,H,T]
    scores_nope = torch.einsum("bshc,btc->bsht", q_nope_abs, kv_latent)
    scores_pe = torch.einsum("bshr,btr->bsht", q_pe_rotated, k_rope)
    scores_total = (scores_nope + scores_pe) * attn.softmax_scale

  pstat("q_nope_abs", q_nope_abs)
  pstat("scores_nope (q_abs @ kv)", scores_nope)
  pstat("scores_pe (q_pe @ k_rope)", scores_pe)
  pstat("scores_total (scaled)", scores_total)

  if rank == 0:
    # Ratio analysis
    nope_contrib = scores_nope.float().abs().mean()
    pe_contrib = scores_pe.float().abs().mean()
    print(f"\n  Score contribution ratio: nope={nope_contrib:.4f}, pe={pe_contrib:.4f}, ratio={pe_contrib/nope_contrib:.2f}x")

    # Expected: both should contribute meaningfully, pe shouldn't dominate by 1000x
    if pe_contrib / nope_contrib > 100:
      print(f"  ⚠️  WARNING: q_pe @ k_rope dominates! This may indicate RoPE or scale issue.")

  # ========== ATTENTION OUTPUT (Reference) ==========
  if rank == 0:
    print("\n" + "="*50)
    print("ATTENTION OUTPUT (Reference Decode Style)")
    print("="*50)

  with torch.no_grad():
    # Apply causal mask
    causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
    scores_masked = scores_total.masked_fill(causal_mask.unsqueeze(0).unsqueeze(2), float("-inf"))
    attn_weights = torch.softmax(scores_masked, dim=-1)

    # Output: attn @ kv_latent -> project via wkv_b
    out_latent = torch.einsum("bsht,btc->bshc", attn_weights, kv_latent)
    out_v = torch.einsum("bshc,hdc->bshd", out_latent, wkv_b_w[:, -attn.v_head_dim:])

    # Output projection (wo)
    wo_dequant = weight_dequant(attn.wo.weight, attn.wo.weight_scale_inv)
    attn_out_ref = F.linear(out_v.flatten(2), wo_dequant)

  pstat("attn_weights", attn_weights)
  pstat("out_latent", out_latent)
  pstat("out_v", out_v)
  pstat("attn_out_ref", attn_out_ref)

  # ========== FFN (Dense MLP) ==========
  if rank == 0:
    print("\n" + "="*50)
    print("FFN (Dense MLP)")
    print("="*50)

  with torch.no_grad():
    x_after_attn = x + attn_out_ref
    x_ffn_norm = model.layers[0].ffn_norm(x_after_attn)

    # Our FFN uses FP8
    w1_out = ffn.w1(x_ffn_norm)
    w3_out = ffn.w3(x_ffn_norm)
    gate_up = F.silu(w1_out.float()) * w3_out.float()
    ffn_out = ffn.w2(gate_up.to(x_ffn_norm.dtype))

  pstat("x_ffn_norm", x_ffn_norm)
  pstat("w1_out (gate)", w1_out)
  pstat("w3_out (up)", w3_out)
  pstat("gate_up", gate_up)
  pstat("ffn_out", ffn_out)

  # Reference FFN with dequantized weights
  with torch.no_grad():
    w1_dequant = weight_dequant(ffn.w1.weight, ffn.w1.weight_scale_inv)
    w2_dequant = weight_dequant(ffn.w2.weight, ffn.w2.weight_scale_inv)
    w3_dequant = weight_dequant(ffn.w3.weight, ffn.w3.weight_scale_inv)

    w1_out_ref = F.linear(x_ffn_norm, w1_dequant)
    w3_out_ref = F.linear(x_ffn_norm, w3_dequant)
    gate_up_ref = F.silu(w1_out_ref.float()) * w3_out_ref.float()
    ffn_out_ref = F.linear(gate_up_ref.to(torch.bfloat16), w2_dequant)

  if rank == 0:
    diff_w1 = (w1_out.float() - w1_out_ref.float()).abs()
    diff_ffn = (ffn_out.float() - ffn_out_ref.float()).abs()
    print(f"\n  w1 diff (FP8 vs BF16 ref): max={diff_w1.max():.6f}, mean={diff_w1.mean():.6f}")
    print(f"  ffn diff (FP8 vs BF16 ref): max={diff_ffn.max():.6f}, mean={diff_ffn.mean():.6f}")

  # ========== FINAL OUTPUT ==========
  if rank == 0:
    print("\n" + "="*50)
    print("FINAL OUTPUT")
    print("="*50)

  with torch.no_grad():
    x_final = x_after_attn + ffn_out_ref
    x_normed = model.norm(x_final)
    logits = model.lm_head(x_normed.float())

    if world_size > 1:
      all_logits = [torch.empty_like(logits) for _ in range(world_size)]
      dist.all_gather(all_logits, logits)
      logits = torch.cat(all_logits, dim=-1)

  pstat("x_final", x_final)
  pstat("logits", logits)

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
