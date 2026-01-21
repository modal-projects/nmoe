# SPDX-License-Identifier: Apache-2.0
"""Test that dist.all_reduce works correctly."""

import torch
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  # Test 1: Simple all-reduce
  x = torch.tensor([float(rank)], device=device)
  dist.all_reduce(x)
  expected = sum(range(world_size))  # 0 + 1 + 2 + ... + 7 = 28

  if rank == 0:
    print(f"Test 1 - Simple all_reduce:")
    print(f"  Expected: {expected}")
    print(f"  Got: {x.item()}")
    print(f"  PASS: {abs(x.item() - expected) < 0.01}")

  # Gather results from all ranks
  results = [torch.empty_like(x) for _ in range(world_size)]
  dist.all_gather(results, x)

  if rank == 0:
    print(f"\n  Results across ranks: {[r.item() for r in results]}")
    print(f"  All same: {all(abs(r.item() - expected) < 0.01 for r in results)}")

  # Test 2: Larger tensor
  y = torch.full((100, 100), float(rank), device=device)
  dist.all_reduce(y)

  y_results = [torch.empty_like(y) for _ in range(world_size)]
  dist.all_gather(y_results, y)

  if rank == 0:
    print(f"\nTest 2 - Larger tensor all_reduce:")
    print(f"  Expected mean: {expected}")
    means = [r.mean().item() for r in y_results]
    print(f"  Means across ranks: {[f'{m:.2f}' for m in means]}")
    print(f"  All same: {all(abs(m - expected) < 0.01 for m in means)}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
