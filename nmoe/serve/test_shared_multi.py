# SPDX-License-Identifier: Apache-2.0
"""Test shared experts output in multi-GPU mode."""

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

  torch.manual_seed(12345)
  x_input = torch.randn(1, 5, cfg.hidden_size, device=device, dtype=torch.bfloat16)
  dist.broadcast(x_input, src=0)

  moe = model.layers[3].ffn
  x = x_input.view(-1, cfg.hidden_size)

  if rank == 0:
    print(f"Multi-GPU Shared Experts Test (world_size={world_size}):")

  with torch.no_grad():
    shared_out = moe.shared(x)

  if rank == 0:
    print(f"Shared experts mean: {shared_out.mean():.6f}")
    print(f"Shared experts first 10: {shared_out[0, :10].tolist()}")

    # Compare with reference
    ref = torch.load("/tmp/shared_ref.pt")
    ref_shared = ref["shared_output"]
    diff = (shared_out.cpu().float() - ref_shared.float()).abs()
    print(f"\n--- Comparison with single-GPU ---")
    print(f"Shared experts max_diff: {diff.max():.6f}")
    print(f"Shared experts mean_diff: {diff.mean():.6f}")

    if diff.max() > 0.01:
      print("\nSHARED EXPERTS MISMATCH!")
      print(f"Multi-GPU: {shared_out[0, :10].tolist()}")
      print(f"Single-GPU: {ref_shared[0, :10].tolist()}")
    else:
      print("\nShared experts: MATCH")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
