# SPDX-License-Identifier: Apache-2.0
"""Test grouped GEMM with small batch sizes (decode scenario)."""

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

  if rank == 0:
    print("=" * 70)
    print("Grouped GEMM Small Batch Test (Decode Scenario)")
    print("=" * 70)

  moe = model.layers[3].ffn
  num_local_experts = moe.num_local

  # Dequantize expert weights for reference
  w13_dequant = weight_dequant_3d(moe.w13, moe.w13_scale)
  w2_dequant = weight_dequant_3d(moe.w2, moe.w2_scale)

  if rank == 0:
    print(f"\nExpert weights: w13={moe.w13.shape}, w2={moe.w2.shape}")
    print(f"Num local experts: {num_local_experts}")

  # Test different batch sizes (simulating prefill vs decode)
  batch_sizes = [1, 2, 4, 8, 16, 64]  # 1 = decode, larger = prefill

  for batch_size in batch_sizes:
    if rank == 0:
      print(f"\n{'='*50}")
      print(f"Batch size: {batch_size} tokens")
      print(f"{'='*50}")

    # Create test input: tokens assigned to different experts
    torch.manual_seed(42 + batch_size)
    x = torch.randn(batch_size, cfg.hidden_size, device=device, dtype=torch.bfloat16)

    # Assign tokens to experts (round-robin for simplicity)
    expert_ids = torch.arange(batch_size, device=device) % num_local_experts
    m_indices = expert_ids.to(torch.int32)

    # Sort by expert (required for contiguous GEMM)
    perm = expert_ids.argsort()
    x_sorted = x[perm]
    m_indices_sorted = m_indices[perm]

    # Quantize input for FP8 GEMM
    def quantize_act(x, block_size=128):
      T, D = x.shape
      x2 = x.view(T, D // block_size, block_size)
      amax = x2.abs().amax(dim=-1).clamp(min=1e-12)
      log2 = torch.log2(amax / 448.0)
      scale = torch.pow(2.0, torch.ceil(log2)).to(torch.float32)
      x_q = (x2 / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
      return x_q.view(T, D), scale

    x_q, x_scale = quantize_act(x_sorted.to(torch.bfloat16))

    # === DeepGEMM grouped GEMM ===
    gateup_deepgemm = torch.empty(batch_size, 2 * cfg.moe_intermediate_size, device=device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_contiguous(
      (x_q, x_scale),
      (moe.w13, moe.w13_scale),
      gateup_deepgemm,
      m_indices_sorted,
    )

    # === Torch reference (per-expert matmul) ===
    gateup_ref = torch.zeros_like(gateup_deepgemm)
    for e in range(num_local_experts):
      mask = (m_indices_sorted == e)
      if mask.any():
        x_e = x_sorted[mask]
        # Dequant input and compute
        x_e_dequant = x_e  # Already BF16
        out_e = F.linear(x_e_dequant, w13_dequant[e])
        gateup_ref[mask] = out_e

    # Compare
    diff = (gateup_deepgemm.float() - gateup_ref.float()).abs()
    cos_sim = F.cosine_similarity(
      gateup_deepgemm.flatten().float().unsqueeze(0),
      gateup_ref.flatten().float().unsqueeze(0)
    ).item()

    if rank == 0:
      print(f"DeepGEMM output: mean={gateup_deepgemm.float().mean():.6f}, std={gateup_deepgemm.float().std():.6f}")
      print(f"Reference output: mean={gateup_ref.float().mean():.6f}, std={gateup_ref.float().std():.6f}")
      print(f"Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
      print(f"Cosine similarity: {cos_sim:.6f}")

      if cos_sim < 0.99:
        print(f"⚠️  WARNING: Low cosine similarity for batch_size={batch_size}!")

      # Check per-expert
      print(f"\nPer-expert analysis:")
      for e in range(min(4, num_local_experts)):
        mask = (m_indices_sorted == e)
        if mask.any():
          e_diff = diff[mask]
          e_cos = F.cosine_similarity(
            gateup_deepgemm[mask].flatten().float().unsqueeze(0),
            gateup_ref[mask].flatten().float().unsqueeze(0)
          ).item() if mask.sum() > 0 else 0
          print(f"  Expert {e}: {mask.sum().item()} tokens, diff_max={e_diff.max():.6f}, cos={e_cos:.6f}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
