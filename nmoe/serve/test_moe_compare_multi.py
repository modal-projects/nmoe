# SPDX-License-Identifier: Apache-2.0
"""Compare multi-GPU MoE with saved single-GPU reference."""

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

  # Same random input as single-GPU test
  torch.manual_seed(12345)
  x_input = torch.randn(1, 5, cfg.hidden_size, device=device, dtype=torch.bfloat16)
  dist.broadcast(x_input, src=0)

  moe = model.layers[3].ffn
  x = x_input.view(-1, cfg.hidden_size)

  if rank == 0:
    print(f"Multi-GPU MoE (world_size={world_size}):")
    print(f"Input: shape={x.shape}, mean={x.mean():.6f}")

  with torch.no_grad():
    weights, indices = moe.gate(x)
    out = moe(x.unsqueeze(0)).squeeze(0)

  if rank == 0:
    print(f"Gate indices (token 0): {indices[0].tolist()}")
    print(f"Gate weights (token 0): {weights[0].tolist()}")
    print(f"Output mean: {out.mean():.6f}")
    print(f"Output token 0 first 10: {out[0, :10].tolist()}")

    # Load and compare reference
    ref = torch.load("/tmp/moe_ref.pt")
    print("\n--- Comparison with single-GPU ---")

    # Indices
    if torch.equal(indices.cpu(), ref['indices']):
      print("Indices: MATCH")
    else:
      print("Indices: MISMATCH!")

    # Weights
    w_diff = (weights.cpu() - ref['weights']).abs()
    print(f"Weights: max_diff={w_diff.max():.6f}")

    # Output
    o_diff = (out.cpu().float() - ref['output'].float()).abs()
    print(f"Output: max_diff={o_diff.max():.4f}, mean_diff={o_diff.mean():.4f}")

    if o_diff.max() > 0.1:
      print("\nOUTPUT MISMATCH!")
      print(f"Multi-GPU token 0 first 10: {out[0, :10].tolist()}")
      print(f"Single-GPU token 0 first 10: {ref['output'][0, :10].tolist()}")

      # Check shared experts
      with torch.no_grad():
        shared_out = moe.shared(x)
      print(f"\nMulti-GPU shared experts mean: {shared_out.mean():.6f}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
