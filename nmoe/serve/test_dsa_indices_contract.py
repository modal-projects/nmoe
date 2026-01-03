# SPDX-License-Identifier: Apache-2.0
"""Contract tests for DSA sparse indices passed into FlashMLA.

This catches a subtle but severe correctness bug:
- When context_len < topk, indices must be padded by repeating valid indices.
- Passing -1 (or other invalid indices) can silently corrupt attention output
  without producing NaNs, leading to garbage generation.

Run with:
  python -m nmoe.serve.test_dsa_indices_contract
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from nmoe.serve.model import DsaFlashMla, ModelConfig, init_distributed


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  # In the debug container this exists; in other contexts tests may be skipped.
  cand = "/workspace/nmoe/third_party/DeepGEMM/third-party/cutlass"
  if os.path.isdir(cand):
    os.environ["CUTLASS_PATH"] = cand


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _minor = torch.cuda.get_device_capability()
  return major == 10


def _init_module_params(m: torch.nn.Module) -> None:
  # Avoid uninitialized fp8 weights/scales producing NaNs during the test.
  with torch.no_grad():
    for p in m.parameters(recurse=True):
      if p.dtype == torch.float8_e4m3fn:
        p.copy_((torch.randn_like(p.float()) / 10).to(torch.float8_e4m3fn))
      elif p.dtype in (torch.float16, torch.bfloat16, torch.float32):
        if p.ndim >= 2:
          torch.nn.init.normal_(p, mean=0.0, std=0.02)
        else:
          p.zero_()
      else:
        # Leave other dtypes unchanged.
        pass
    # Ensure all DeepGEMM scale tensors are UE8M0-compatible.
    for name, p in m.named_parameters(recurse=True):
      if name.endswith("weight_scale_inv") and p.dtype == torch.float32:
        p.fill_(1.0)


class TestDsaIndicesContract(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    _maybe_set_cutlass_path()
    if not _is_sm100():
      raise unittest.SkipTest("requires SM100 (B200) for FlashMLA path")
    init_distributed(0, 1)
    torch.cuda.set_device(0)

  def test_indices_no_negative_and_causal_when_ctx_lt_topk(self) -> None:
    # Use a short context (S=8) with topk=2048; the implementation must
    # pad by repeating valid indices (not -1).
    cfg = ModelConfig(num_layers=1, num_dense_layers=1, dsa_topk=2048)
    attn = DsaFlashMla(cfg, layer_idx=0).cuda().eval()
    _init_module_params(attn)

    B, S = 1, 8
    x = torch.randn(B, S, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) / 10
    positions = torch.arange(S, device="cuda", dtype=torch.int64).unsqueeze(0)
    out_loc = torch.arange(S, device="cuda", dtype=torch.int32).unsqueeze(0)

    num_blocks = 1
    kv_cache = torch.zeros(num_blocks, 64, 1, 656, device="cuda", dtype=torch.uint8)
    idx_k_cache = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros((B, 1), device="cuda", dtype=torch.int32)
    cache_seqlens = torch.tensor([S], device="cuda", dtype=torch.int32)
    cache_seqlens_cpu = [S]

    # Fake FlashMLA to validate indices without depending on attention numerics.
    def _fake_get_mla_metadata(**kwargs):
      # Return minimal metadata with correct dtype/device.
      md = torch.empty((1, 1), device="cuda", dtype=torch.int32)
      return md, torch.tensor([0], device="cuda", dtype=torch.int32)

    def _fake_flash_mla_with_kvcache(
      q,
      kv_cache_,
      block_table_,
      cache_seqlens_,
      head_dim_v,
      tile_scheduler_metadata,
      num_splits,
      *,
      softmax_scale,
      causal,
      is_fp8_kvcache,
      indices,
    ):
      self.assertEqual(indices.dtype, torch.int32)
      self.assertTrue(indices.is_cuda)
      self.assertEqual(indices.shape, (B, S, cfg.dsa_topk))

      # Must not contain invalid indices.
      self.assertTrue((indices >= 0).all(), "DSA indices must not contain -1 padding.")

      # With identity block_table (all zeros), physical indices are just token positions.
      # Enforce causality: each query position i may attend only to <= i.
      idx64 = indices.to(torch.int64)
      for i in range(S):
        self.assertTrue(bool((idx64[0, i] <= i).all()), f"non-causal indices at query pos {i}")

      # Return zeros with correct shapes.
      out = torch.zeros((B, S, cfg.num_heads, head_dim_v), device="cuda", dtype=q.dtype)
      lse = torch.zeros((B, cfg.num_heads, S), device="cuda", dtype=torch.float32)
      return out, lse

    import flash_mla  # type: ignore
    with patch.object(flash_mla, "get_mla_metadata", new=_fake_get_mla_metadata):
      with patch.object(flash_mla, "flash_mla_with_kvcache", new=_fake_flash_mla_with_kvcache):
        # Build freqs matching model contract.
        from nmoe.serve.model import precompute_freqs_cis
        freqs_all = precompute_freqs_cis(cfg, device=torch.device("cuda"))
        freqs = freqs_all.index_select(0, positions.view(-1)).view(B, S, -1)
        _ = attn(
          x,
          freqs,
          kv_cache=kv_cache,
          idx_k_cache=idx_k_cache,
          block_table=block_table,
          cache_seqlens=cache_seqlens,
          cache_seqlens_cpu=cache_seqlens_cpu,
          out_loc=out_loc,
          positions=positions,
        )


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDsaIndicesContract)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())

