# SPDX-License-Identifier: Apache-2.0
"""Check kv_latent vs k_pe magnitudes and compare with BF16 reference."""

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

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=1, num_dense_layers=1)
  model = DeepSeekV3(cfg, buffer=None).to(device)
  model.eval()

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("KV Magnitude Analysis: kv_latent vs k_pe")
    print("=" * 70)

  # Test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)

  attn = model.layers[0].attn

  def pstat(name, t):
    if rank == 0:
      print(f"  {name}: mean={t.float().mean():.6f}, std={t.float().std():.6f}, amax={t.float().abs().max():.4f}")

  # Get the input to attention
  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

  pstat("x_norm (input to wkv_a)", x_norm)

  # Our FP8 path
  if rank == 0:
    print("\n=== FP8 Forward Path ===")
  with torch.no_grad():
    kv_fp8 = attn.wkv_a(x_norm)
    kv_latent_part = kv_fp8[:, :, :attn.kv_lora_rank]
    k_pe_part = kv_fp8[:, :, attn.kv_lora_rank:]

  pstat("kv_fp8 (full)", kv_fp8)
  pstat("kv_latent_part (first 512)", kv_latent_part)
  pstat("k_pe_part (last 64)", k_pe_part)

  # Dequantize wkv_a and compute reference
  if rank == 0:
    print("\n=== BF16 Reference Path (dequantized) ===")

  def weight_dequant_padded(weight, scale, block_size=128):
    out_feat, in_feat = weight.shape
    out_tiles, in_tiles = scale.shape
    out_padded = out_tiles * block_size
    in_padded = in_tiles * block_size
    w_pad = torch.zeros(out_padded, in_padded, dtype=weight.dtype, device=weight.device)
    w_pad[:out_feat, :in_feat] = weight
    w_pad = w_pad.view(out_tiles, block_size, in_tiles, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(-1, block_size * block_size)
    w_pad = (w_pad.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
    w_pad = w_pad.view(out_tiles, in_tiles, block_size, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(out_padded, in_padded)
    return w_pad[:out_feat, :in_feat]

  with torch.no_grad():
    wkv_a_dequant = weight_dequant_padded(attn.wkv_a.weight, attn.wkv_a.weight_scale_inv)
    kv_ref = F.linear(x_norm, wkv_a_dequant)
    kv_latent_ref = kv_ref[:, :, :attn.kv_lora_rank]
    k_pe_ref = kv_ref[:, :, attn.kv_lora_rank:]

  pstat("kv_ref (full)", kv_ref)
  pstat("kv_latent_ref (first 512)", kv_latent_ref)
  pstat("k_pe_ref (last 64)", k_pe_ref)

  # Check weight statistics per output channel group
  if rank == 0:
    print("\n=== Weight Statistics (dequantized) ===")
    w = wkv_a_dequant.float()
    w_kv = w[:attn.kv_lora_rank, :]  # First 512 rows
    w_pe = w[attn.kv_lora_rank:, :]  # Last 64 rows
    print(f"  wkv_a_kv (rows 0-511): mean={w_kv.mean():.6f}, std={w_kv.std():.6f}, amax={w_kv.abs().max():.4f}")
    print(f"  wkv_a_pe (rows 512-575): mean={w_pe.mean():.6f}, std={w_pe.std():.6f}, amax={w_pe.abs().max():.4f}")

    # Ratio analysis
    print("\n=== Ratio Analysis ===")
    ratio = k_pe_part.float().std() / kv_latent_part.float().std()
    print(f"  k_pe std / kv_latent std = {ratio:.2f}x")

    if ratio > 10:
      print(f"  ⚠️  WARNING: k_pe is {ratio:.1f}x larger than kv_latent!")
      print(f"     This will cause RoPE component to dominate attention scores.")
      print(f"     This might be by design (check reference), or a weight loading bug.")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
