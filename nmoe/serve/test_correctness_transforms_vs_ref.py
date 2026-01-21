# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for RoPE + Hadamard transforms vs DeepSeek reference.

Why this matters:
- These transforms feed the DSA indexer path. Subtle mismatches can keep all
  tensors finite while completely destroying routing/topk selection → garbage
  generation.

This test imports the DeepSeek checkpoint's reference `inference/model.py` and
compares only the *pure math* helpers (no TileLang kernels are executed).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 10


class TestTransformsVsReference(unittest.TestCase):
  def test_apply_rotary_emb_matches_reference(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires CUDA SM100 for representative dtype/device")

    model_path = os.environ.get("NMOE_MODEL_PATH", "").strip()
    if not model_path:
      raise unittest.SkipTest("NMOE_MODEL_PATH is required to import reference helpers.")

    ref_dir = Path(os.environ.get("NMOE_REFERENCE_DIR", f"{model_path}/inference"))
    if not ref_dir.is_dir():
      raise unittest.SkipTest(f"reference dir not found: {ref_dir}")

    sys.path.insert(0, str(ref_dir))
    try:
      import model as ref  # type: ignore
    finally:
      # Keep on path until after import; remove at end of test.
      pass

    from nmoe.serve.model import apply_rotary_emb as ours_apply

    torch.cuda.set_device(0)
    B, S, H, D = 2, 8, 4, 64  # D must be even
    x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16) / 10
    theta = torch.randn(S, D // 2, device="cuda", dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(theta), theta).to(torch.complex64)

    y_ref = ref.apply_rotary_emb(x, freqs_cis)  # default interleaved=True in ref for attention
    y_ours = ours_apply(x, freqs_cis, interleaved=True)

    torch.testing.assert_close(y_ours, y_ref, rtol=0, atol=0)

    sys.path.remove(str(ref_dir))

  def test_rotate_activation_matches_reference(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires CUDA SM100 for representative dtype/device")

    model_path = os.environ.get("NMOE_MODEL_PATH", "").strip()
    if not model_path:
      raise unittest.SkipTest("NMOE_MODEL_PATH is required to import reference helpers.")

    ref_dir = Path(os.environ.get("NMOE_REFERENCE_DIR", f"{model_path}/inference"))
    if not ref_dir.is_dir():
      raise unittest.SkipTest(f"reference dir not found: {ref_dir}")

    sys.path.insert(0, str(ref_dir))
    try:
      import model as ref  # type: ignore
    finally:
      pass

    from nmoe.serve.model import rotate_activation as ours_rotate

    torch.cuda.set_device(0)
    x = torch.randn(4, 8, 128, 192, device="cuda", dtype=torch.bfloat16) / 10

    y_ref = ref.rotate_activation(x)
    y_ours = ours_rotate(x)
    torch.testing.assert_close(y_ours, y_ref, rtol=0, atol=0)

    sys.path.remove(str(ref_dir))


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTransformsVsReference)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
