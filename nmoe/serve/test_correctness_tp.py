# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel correctness tests (math, not finiteness).

These tests validate that our TP linear sharding + collectives produce the same
result as the equivalent full (unsharded) matmul.

Run with:
  torchrun --nproc_per_node=8 -m nmoe.serve.test_correctness_tp
"""

from __future__ import annotations

import unittest

import torch
import torch.distributed as dist

from nmoe.serve.test_utils import init_torchrun_nccl


def _init_dist() -> tuple[int, int, torch.device]:
  rank, world_size, _local_rank, device = init_torchrun_nccl()
  return rank, world_size, device


class TestTPCorrectness(unittest.TestCase):
  def test_column_parallel_gather_matches_full(self) -> None:
    rank, world_size, device = _init_dist()

    B, hidden = 2, 256
    vocab = 512
    self.assertEqual(vocab % world_size, 0)

    # Full inputs/weights live on rank 0, then shard.
    if rank == 0:
      x_full = torch.randn(B, hidden, device=device, dtype=torch.float32)
      w_full = torch.randn(vocab, hidden, device=device, dtype=torch.float32) / 10
    else:
      x_full = torch.empty(B, hidden, device=device, dtype=torch.float32)
      w_full = torch.empty(vocab, hidden, device=device, dtype=torch.float32)

    dist.broadcast(x_full, src=0)
    dist.broadcast(w_full, src=0)

    shard = vocab // world_size
    w_shard = w_full.narrow(0, rank * shard, shard).contiguous()

    # Local logits shard, then gather like DeepSeekV3.forward.
    local = x_full @ w_shard.t()  # [B, shard]
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local.contiguous())
    logits_tp = torch.cat(gathered, dim=-1)  # [B, vocab]

    if rank == 0:
      logits_full = x_full @ w_full.t()
      # Exact equality is not guaranteed across devices/backends. Keep this
      # strict enough to catch sharding/collective mistakes, but not flaky.
      torch.testing.assert_close(logits_tp, logits_full, rtol=1e-6, atol=1e-6)

  def test_row_parallel_allreduce_matches_full(self) -> None:
    rank, world_size, device = _init_dist()

    B, hidden = 2, 256
    out_features = 128
    self.assertEqual(hidden % world_size, 0)

    # Full inputs/weights live on rank 0, then shard on input dim.
    if rank == 0:
      x_full = torch.randn(B, hidden, device=device, dtype=torch.float32)
      w_full = torch.randn(out_features, hidden, device=device, dtype=torch.float32) / 10
    else:
      x_full = torch.empty(B, hidden, device=device, dtype=torch.float32)
      w_full = torch.empty(out_features, hidden, device=device, dtype=torch.float32)

    dist.broadcast(x_full, src=0)
    dist.broadcast(w_full, src=0)

    shard = hidden // world_size
    x_shard = x_full.narrow(1, rank * shard, shard).contiguous()
    w_shard = w_full.narrow(1, rank * shard, shard).contiguous()

    local = x_shard @ w_shard.t()  # [B, out]
    dist.all_reduce(local)

    if rank == 0:
      logits_full = x_full @ w_full.t()
      # all_reduce summation order can introduce tiny fp32 rounding deltas.
      torch.testing.assert_close(local, logits_full, rtol=1e-6, atol=1e-6)


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTPCorrectness)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
