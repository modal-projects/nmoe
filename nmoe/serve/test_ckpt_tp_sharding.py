# SPDX-License-Identifier: Apache-2.0
"""Regression tests for checkpoint tensor-parallel sharding.

These tests validate that `load_checkpoint()` shards and replicates tensors in
the same way as `nmoe/serve/model.py` expects for inference TP:
  - Column-parallel weights (e.g., `lm_head.weight`) shard on dim 0 (vocab).
  - Row-parallel weights (e.g., `attn.wo.weight`) shard on dim 1 (hidden-in).
  - Replicated weights (e.g., `embed.weight`, `norm.weight`) are identical on
    every rank.

Run with:
  python -m nmoe.serve.test_ckpt_tp_sharding
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from nmoe.serve.ckpt import load_checkpoint
from nmoe.serve.model import ModelConfig


def _is_power_of_two(x: torch.Tensor) -> bool:
  x = x.detach().abs().clamp(min=1e-12)
  log2 = torch.log2(x)
  return bool(torch.allclose(log2, log2.round(), atol=1e-5))


class _DummyTpModel(torch.nn.Module):
  def __init__(self, *, hidden: int, vocab: int, world_size: int) -> None:
    super().__init__()
    assert vocab % world_size == 0
    assert hidden % world_size == 0
    assert hidden % 128 == 0
    assert (hidden // world_size) % 128 == 0

    self.embed = torch.nn.Module()
    self.embed.weight = torch.nn.Parameter(torch.empty((vocab, hidden), dtype=torch.bfloat16))

    self.norm = torch.nn.Module()
    self.norm.weight = torch.nn.Parameter(torch.empty((hidden,), dtype=torch.float32))

    self.lm_head = torch.nn.Module()
    self.lm_head.weight = torch.nn.Parameter(torch.empty((vocab // world_size, hidden), dtype=torch.float32))

    self.layers = torch.nn.ModuleList([torch.nn.Module()])
    self.layers[0].attn = torch.nn.Module()
    self.layers[0].attn.wo = torch.nn.Module()
    self.layers[0].attn.wo.weight = torch.nn.Parameter(
      torch.empty((hidden, hidden // world_size), dtype=torch.float32)
    )
    self.layers[0].attn.wo.weight_scale_inv = torch.nn.Parameter(
      torch.empty((hidden // 128, (hidden // world_size) // 128), dtype=torch.float32)
    )


class TestCheckpointTPSharding(unittest.TestCase):
  def test_lm_head_and_wo_sharding(self) -> None:
    hidden = 256
    vocab = 16
    world_size = 2

    # Build deterministic, non-symmetric tensors so sharding mistakes are obvious.
    embed_full = torch.arange(vocab * hidden, dtype=torch.float32).view(vocab, hidden).to(torch.bfloat16)
    norm_full = torch.arange(hidden, dtype=torch.float32) / 100.0
    lm_head_full = torch.arange(vocab * hidden, dtype=torch.float32).view(vocab, hidden) / 10.0
    wo_full = torch.arange(hidden * hidden, dtype=torch.float32).view(hidden, hidden) / 1000.0

    # Non-UE8M0 scales to ensure loader converts (and still shards correctly).
    wo_scale_full = torch.full((hidden // 128, hidden // 128), 0.3, dtype=torch.float32)
    self.assertFalse(_is_power_of_two(wo_scale_full))

    with tempfile.TemporaryDirectory() as d:
      ckpt_dir = Path(d)
      shard = {
        "model.embed_tokens.weight": embed_full,
        "model.norm.weight": norm_full,
        "model.lm_head.weight": lm_head_full,
        "model.layers.0.self_attn.o_proj.weight": wo_full,
        "model.layers.0.self_attn.o_proj.weight_scale_inv": wo_scale_full,
      }
      save_file(shard, str(ckpt_dir / "model-00001-of-00001.safetensors"))

      cfg = ModelConfig(num_layers=1, num_dense_layers=1, hidden_size=hidden, vocab_size=vocab, num_experts=2)

      for rank in range(world_size):
        model = _DummyTpModel(hidden=hidden, vocab=vocab, world_size=world_size)
        missing, unexpected = load_checkpoint(
          model,
          str(ckpt_dir),
          rank=rank,
          world_size=world_size,
          cfg=cfg,
          strict=False,
        )
        self.assertEqual(len(unexpected), 0)

        # Replicated tensors.
        torch.testing.assert_close(model.embed.weight.detach(), embed_full, rtol=0, atol=0)
        torch.testing.assert_close(model.norm.weight.detach(), norm_full, rtol=0, atol=0)

        # Column-parallel vocab sharding (dim 0).
        vocab_shard = vocab // world_size
        exp_lm = lm_head_full.narrow(0, rank * vocab_shard, vocab_shard).contiguous()
        torch.testing.assert_close(model.lm_head.weight.detach(), exp_lm, rtol=0, atol=0)

        # Row-parallel input sharding (dim 1).
        hidden_shard = hidden // world_size
        exp_wo = wo_full.narrow(1, rank * hidden_shard, hidden_shard).contiguous()
        torch.testing.assert_close(model.layers[0].attn.wo.weight.detach(), exp_wo, rtol=0, atol=0)

        exp_scale = wo_scale_full.narrow(1, rank * (hidden_shard // 128), hidden_shard // 128).contiguous()
        got_scale = model.layers[0].attn.wo.weight_scale_inv.detach()
        self.assertTrue(_is_power_of_two(got_scale))
        self.assertTrue(torch.all(got_scale >= exp_scale))


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCheckpointTPSharding)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())

