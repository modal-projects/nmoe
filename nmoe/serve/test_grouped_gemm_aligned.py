# SPDX-License-Identifier: Apache-2.0
"""Test grouped GEMM with proper 128-alignment."""

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


def weight_dequant_3d(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """Dequantize stacked FP8 expert weights [num_experts, out, in]."""
  num_experts, out_feat, in_feat = weight.shape
  result = torch.empty(num_experts, out_feat, in_feat, dtype=torch.bfloat16, device=weight.device)

  for e in range(num_experts):
    w = weight[e]
    s = scale[e]
    out_tiles = out_feat // block_size
    in_tiles = in_feat // block_size
    w = w.view(out_tiles, block_size, in_tiles, block_size)
    w = w.transpose(1, 2).contiguous().view(-1, block_size * block_size)
    w = (w.float() * s.view(-1, 1).float()).to(torch.bfloat16)
    w = w.view(out_tiles, in_tiles, block_size, block_size)
    w = w.transpose(1, 2).contiguous().view(out_feat, in_feat)
    result[e] = w

  return result


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer
  from deep_gemm import m_grouped_fp8_gemm_nt_contiguous
  from deep_gemm.utils.layout import get_mk_alignment_for_contiguous_layout

  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=4, num_dense_layers=3)

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

  M_ALIGN = get_mk_alignment_for_contiguous_layout()
  if rank == 0:
    print("=" * 70)
    print(f"Grouped GEMM Alignment Test (M_ALIGN={M_ALIGN})")
    print("=" * 70)

  moe = model.layers[3].ffn
  num_local_experts = moe.num_local

  # Dequantize expert weights for reference
  w13_dequant = weight_dequant_3d(moe.w13, moe.w13_scale)

  def quantize_act(x, block_size=128):
    T, D = x.shape
    x2 = x.view(T, D // block_size, block_size)
    amax = x2.abs().amax(dim=-1).clamp(min=1e-12)
    log2 = torch.log2(amax / 448.0)
    scale = torch.pow(2.0, torch.ceil(log2)).to(torch.float32)
    x_q = (x2 / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return x_q.view(T, D), scale

  # Test: 1 real token per expert, with proper 128-alignment padding
  num_experts_used = 4  # Use 4 experts
  real_tokens_per_expert = 1

  if rank == 0:
    print(f"\n=== Test: {real_tokens_per_expert} real token per expert, {num_experts_used} experts ===")
    print(f"With 128-alignment padding")

  # Create input with padding
  total_padded = num_experts_used * M_ALIGN  # 4 * 128 = 512 tokens total
  torch.manual_seed(42)
  x = torch.randn(total_padded, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  # Build m_indices: real tokens get expert id, padding gets -1
  m_indices = torch.full((total_padded,), -1, device=device, dtype=torch.int32)
  for e in range(num_experts_used):
    start = e * M_ALIGN
    m_indices[start:start + real_tokens_per_expert] = e

  if rank == 0:
    print(f"m_indices shape: {m_indices.shape}")
    print(f"m_indices (first 10 of each expert block):")
    for e in range(num_experts_used):
      start = e * M_ALIGN
      print(f"  Expert {e}: {m_indices[start:start+10].tolist()}")

  # Quantize and run DeepGEMM
  x_q, x_scale = quantize_act(x)
  gateup_deepgemm = torch.empty(total_padded, 2 * cfg.moe_intermediate_size, device=device, dtype=torch.bfloat16)
  m_grouped_fp8_gemm_nt_contiguous(
    (x_q, x_scale),
    (moe.w13, moe.w13_scale),
    gateup_deepgemm,
    m_indices,
  )

  # Zero out padding rows (as DeepGEMM test does)
  gateup_deepgemm = torch.where(
    (m_indices == -1).unsqueeze(1),
    torch.zeros_like(gateup_deepgemm),
    gateup_deepgemm
  )

  # Reference: per-expert matmul for real tokens only
  gateup_ref = torch.zeros_like(gateup_deepgemm)
  for e in range(num_experts_used):
    start = e * M_ALIGN
    end = start + real_tokens_per_expert
    x_e = x[start:end]
    gateup_ref[start:end] = F.linear(x_e, w13_dequant[e])

  # Compare only real tokens
  if rank == 0:
    print(f"\n=== Per-Expert Comparison (real tokens only) ===")
    for e in range(num_experts_used):
      start = e * M_ALIGN
      end = start + real_tokens_per_expert
      diff = (gateup_deepgemm[start:end].float() - gateup_ref[start:end].float()).abs()
      cos_sim = F.cosine_similarity(
        gateup_deepgemm[start:end].flatten().float().unsqueeze(0),
        gateup_ref[start:end].flatten().float().unsqueeze(0)
      ).item()
      print(f"Expert {e}: diff_max={diff.max():.6f}, diff_mean={diff.mean():.6f}, cos={cos_sim:.6f}")

    # Overall
    all_real_mask = (m_indices != -1)
    all_diff = (gateup_deepgemm[all_real_mask].float() - gateup_ref[all_real_mask].float()).abs()
    all_cos = F.cosine_similarity(
      gateup_deepgemm[all_real_mask].flatten().float().unsqueeze(0),
      gateup_ref[all_real_mask].flatten().float().unsqueeze(0)
    ).item()
    print(f"\nOverall: diff_max={all_diff.max():.6f}, diff_mean={all_diff.mean():.6f}, cos={all_cos:.6f}")

    if all_cos > 0.99:
      print("✓ With proper 128-alignment, grouped GEMM works correctly!")
    else:
      print("✗ Still failing even with alignment")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
