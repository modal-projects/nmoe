# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the DSA indexer Triton kernel vs torch reference.

This catches "looks fine" failures where routing scores are wrong but finite.
"""

from __future__ import annotations

import unittest

import torch

from nmoe.triton.dsa import compute_indexer_scores, lightning_indexer_ref


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 10


class TestIndexerKernelCorrectness(unittest.TestCase):
  def test_scores_match_reference(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires SM100 (B200)")

    torch.cuda.set_device(0)
    device = torch.device("cuda")

    B, M, N = 2, 32, 64
    H, D = 8, 128

    q = (torch.randn(B, M, H, D, device=device, dtype=torch.bfloat16) / 10).contiguous()
    k = (torch.randn(B, N, D, device=device, dtype=torch.bfloat16) / 10).contiguous()
    w = (torch.randn(B, M, H, device=device, dtype=torch.bfloat16) / 10).contiguous()

    s_triton = compute_indexer_scores(q, k, w, causal=False)
    s_ref, _idx = lightning_indexer_ref(q, k, w, top_k=min(16, N), causal=False)

    # `lightning_indexer_ref` returns top-k values, not full scores. Recompute full
    # reference scores (same formula) for a proper compare.
    full_ref = torch.einsum("bmhd,bnd->bmhn", q.float(), k.float())
    full_ref = torch.relu(full_ref)
    full_ref = (w.float().unsqueeze(-1) * full_ref).sum(dim=2)

    torch.testing.assert_close(s_triton, full_ref, rtol=2e-3, atol=2e-3)


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIndexerKernelCorrectness)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())

