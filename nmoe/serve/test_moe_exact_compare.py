# SPDX-License-Identifier: Apache-2.0
"""Exact comparison of MoE output between single-GPU and multi-GPU.

Run this with world_size=1 first to save reference output,
then with world_size=8 to compare.
"""

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

  if world_size > 1:
    hidden_bytes = cfg.hidden_size * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
      dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
      combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
  else:
    num_nvl_bytes = 0

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  # Use fixed random input based on seed for reproducibility
  torch.manual_seed(12345)
  x_input = torch.randn(1, 5, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  # Broadcast from rank 0 to ensure identical input in multi-GPU mode
  dist.broadcast(x_input, src=0)

  moe = model.layers[3].ffn
  x = x_input.view(-1, cfg.hidden_size)  # [5, 7168]

  if rank == 0:
    print("=" * 60)
    print(f"MoE Exact Comparison (world_size={world_size})")
    print("=" * 60)
    print(f"Input: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")

  # Get gate output first (should be identical)
  with torch.no_grad():
    weights, indices = moe.gate(x)

  if rank == 0:
    print(f"\nGate output:")
    print(f"  Indices (token 0): {indices[0].tolist()}")
    print(f"  Weights (token 0): {[f'{w:.4f}' for w in weights[0].tolist()]}")

  # Run full MoE
  with torch.no_grad():
    out = moe(x.unsqueeze(0)).squeeze(0)

  if rank == 0:
    print(f"\nMoE output:")
    print(f"  Shape: {out.shape}")
    print(f"  Mean: {out.mean():.6f}")
    print(f"  Std: {out.std():.6f}")
    print(f"  Token 0, first 10 values: {[f'{v:.4f}' for v in out[0, :10].tolist()]}")
    print(f"  Token 0, last 10 values: {[f'{v:.4f}' for v in out[0, -10:].tolist()]}")

    # Save reference for comparison (when running with world_size=1)
    ref_path = "/tmp/moe_reference.pt"
    if world_size == 1:
      torch.save({
        'indices': indices.cpu(),
        'weights': weights.cpu(),
        'output': out.cpu(),
      }, ref_path)
      print(f"\nSaved reference to {ref_path}")
    else:
      # Load reference and compare
      if os.path.exists(ref_path):
        ref = torch.load(ref_path)
        print(f"\n--- Comparison with single-GPU reference ---")

        # Compare indices
        if torch.equal(indices.cpu(), ref['indices']):
          print("Indices: MATCH")
        else:
          print("Indices: MISMATCH!")
          print(f"  Multi-GPU: {indices[0].tolist()}")
          print(f"  Single-GPU: {ref['indices'][0].tolist()}")

        # Compare weights
        w_diff = (weights.cpu() - ref['weights']).abs()
        print(f"Weights: max_diff={w_diff.max():.6f}, mean_diff={w_diff.mean():.6f}")

        # Compare output
        o_diff = (out.cpu() - ref['output']).abs()
        print(f"Output: max_diff={o_diff.max():.6f}, mean_diff={o_diff.mean():.6f}")

        if o_diff.max() > 0.01:
          print("\n  Output MISMATCH - finding where they differ...")
          # Find which tokens have largest differences
          token_diffs = o_diff.sum(dim=-1)
          worst_token = token_diffs.argmax().item()
          print(f"  Worst token: {worst_token}, diff={token_diffs[worst_token]:.4f}")
          print(f"  Multi-GPU: {out[worst_token, :10].tolist()}")
          print(f"  Single-GPU: {ref['output'][worst_token, :10].tolist()}")
      else:
        print(f"\nNo reference found at {ref_path}")
        print("Run with world_size=1 first to create reference.")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
