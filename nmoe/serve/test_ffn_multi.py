# SPDX-License-Identifier: Apache-2.0
"""Check FFN w1 output across multiple GPUs."""

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

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, weight_dequant
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

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

  ffn = model.layers[0].ffn

  if rank == 0:
    print("=== Multi-GPU FFN Analysis ===")
    print(f"w1 weight shape: {ffn.w1.weight.shape}")
    w1_dequant = weight_dequant(ffn.w1.weight, ffn.w1.weight_scale_inv)
    print(f"Rank 0 w1 mean: {w1_dequant.float().mean():.6f}")

  # Test input
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)

  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

  # Check input is identical across ranks
  x_mean = x_norm.float().mean()
  all_means = [torch.empty_like(x_mean) for _ in range(world_size)]
  dist.all_gather(all_means, x_mean.contiguous())

  if rank == 0:
    print(f"\nInput x_norm mean per rank: {[f'{m:.6f}' for m in all_means]}")
    print(f"All same: {all(abs(float(m) - float(all_means[0])) < 1e-4 for m in all_means)}")

  # Check w1 output
  with torch.no_grad():
    w1_out = ffn.w1(x_norm)

  w1_mean = w1_out.float().mean()
  w1_min = w1_out.float().min()
  w1_max = w1_out.float().max()

  all_w1_means = [torch.empty_like(w1_mean) for _ in range(world_size)]
  all_w1_mins = [torch.empty_like(w1_min) for _ in range(world_size)]
  all_w1_maxs = [torch.empty_like(w1_max) for _ in range(world_size)]

  dist.all_gather(all_w1_means, w1_mean.contiguous())
  dist.all_gather(all_w1_mins, w1_min.contiguous())
  dist.all_gather(all_w1_maxs, w1_max.contiguous())

  if rank == 0:
    print(f"\nw1 output per rank:")
    for r in range(world_size):
      print(f"  Rank {r}: mean={float(all_w1_means[r]):.4f}, min={float(all_w1_mins[r]):.4f}, max={float(all_w1_maxs[r]):.4f}")

    # Global stats (what we'd see if we concatenated all shards)
    global_mean = sum(float(m) for m in all_w1_means) / world_size
    global_min = min(float(m) for m in all_w1_mins)
    global_max = max(float(m) for m in all_w1_maxs)
    print(f"\nGlobal (avg of means): {global_mean:.4f}")
    print(f"Global min: {global_min:.4f}, max: {global_max:.4f}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
