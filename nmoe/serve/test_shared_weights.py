# SPDX-License-Identifier: Apache-2.0
"""Check shared experts weight loading between single and multi GPU."""

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

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
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

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  moe = model.layers[3].ffn
  shared = moe.shared

  if rank == 0:
    print("=" * 60)
    print(f"Shared Experts Weight Check (world_size={world_size})")
    print("=" * 60)

  # Print weight shapes and scale shapes
  w1_shape = shared.w1.weight.shape
  w1_scale_shape = shared.w1.weight_scale_inv.shape
  w2_shape = shared.w2.weight.shape
  w2_scale_shape = shared.w2.weight_scale_inv.shape
  w3_shape = shared.w3.weight.shape
  w3_scale_shape = shared.w3.weight_scale_inv.shape

  if rank == 0:
    print(f"\nw1 (gate): weight={w1_shape}, scale={w1_scale_shape}")
    print(f"w2 (down): weight={w2_shape}, scale={w2_scale_shape}")
    print(f"w3 (up):   weight={w3_shape}, scale={w3_scale_shape}")

    # Expected shapes for world_size=8:
    # w1: [2048//8, 7168] = [256, 7168], scale: [2, 56]
    # w2: [7168, 2048//8] = [7168, 256], scale: [56, 2]
    # w3: [2048//8, 7168] = [256, 7168], scale: [2, 56]
    print(f"\nExpected for world_size={world_size}:")
    print(f"w1 (gate): weight=[{2048//world_size}, 7168], scale=[{(2048//world_size)//128}, 56]")
    print(f"w2 (down): weight=[7168, {2048//world_size}], scale=[56, {(2048//world_size)//128}]")
    print(f"w3 (up):   weight=[{2048//world_size}, 7168], scale=[{(2048//world_size)//128}, 56]")

  # Gather all ranks' scales
  w1_scale_mean = shared.w1.weight_scale_inv.float().mean()
  all_w1_scale_means = [torch.empty_like(w1_scale_mean) for _ in range(world_size)]
  dist.all_gather(all_w1_scale_means, w1_scale_mean.contiguous())

  if rank == 0:
    print(f"\nw1 scale mean across ranks: {[f'{m:.6f}' for m in all_w1_scale_means]}")

  # Compare first block of scales
  w1_scale_first = shared.w1.weight_scale_inv[0, :5].float()
  all_w1_scale_first = [torch.empty_like(w1_scale_first) for _ in range(world_size)]
  dist.all_gather(all_w1_scale_first, w1_scale_first.contiguous())

  if rank == 0:
    print(f"\nw1 scale[0, :5] for each rank:")
    for r, s in enumerate(all_w1_scale_first):
      print(f"  Rank {r}: {s.tolist()}")

  # Check if scales are from different parts of the full scale tensor
  # For correct sharding, rank 0 should have rows 0-1, rank 1 should have rows 2-3, etc.

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
