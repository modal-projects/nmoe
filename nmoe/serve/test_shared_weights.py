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

  # Shared experts are converted to BF16 fused weights at load time.
  w13_shape = tuple(shared.w13_bf16.shape)  # [2*inter, hidden]
  w2_shape = tuple(shared.w2_bf16.shape)    # [hidden, inter]
  if rank == 0:
    print(f"\nshared.w13_bf16: {w13_shape} dtype={shared.w13_bf16.dtype}")
    print(f"shared.w2_bf16:  {w2_shape} dtype={shared.w2_bf16.dtype}")

  # Sanity: replicated weights should match across ranks (compare means).
  w13_mean = shared.w13_bf16.float().mean()
  all_w13_mean = [torch.empty_like(w13_mean) for _ in range(world_size)]
  dist.all_gather(all_w13_mean, w13_mean.contiguous())
  if rank == 0:
    print(f"\nshared.w13_bf16 mean across ranks: {[f'{m:.6f}' for m in all_w13_mean]}")

  # Check if scales are from different parts of the full scale tensor
  # For correct sharding, rank 0 should have rows 0-1, rank 1 should have rows 2-3, etc.

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
