# SPDX-License-Identifier: Apache-2.0
"""Regression tests for checkpoint scale formats required on SM100.

These tests are designed to catch the exact class of failure that caused NaNs
in `attn.wo` on B200: FP8 GEMM scale tensors not being in UE8M0 (power-of-2)
format when using DeepGEMM.

Run with:
  python -m nmoe.serve.test_ckpt_scale_fmt
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from nmoe.serve.ckpt import load_checkpoint


def _is_power_of_two(x: torch.Tensor) -> bool:
  # UE8M0 is power-of-2 scaling. Treat 0 as invalid; clamp away for log2.
  x = x.detach().abs().clamp(min=1e-12)
  log2 = torch.log2(x)
  return bool(torch.allclose(log2, log2.round(), atol=1e-5))


class _DummyModel(torch.nn.Module):
  """Minimal module exposing a wo weight_scale_inv param at the expected name."""

  def __init__(self) -> None:
    super().__init__()
    self.layers = torch.nn.ModuleList([torch.nn.Module()])
    self.layers[0].attn = torch.nn.Module()
    self.layers[0].attn.wo = torch.nn.Module()
    # Match the FP8Linear contract: [out_blocks, in_blocks] float32.
    self.layers[0].attn.wo.weight_scale_inv = torch.nn.Parameter(
      torch.empty((56, 16), dtype=torch.float32)
    )


class _DummyMoeModel(torch.nn.Module):
  """Minimal module exposing MoE expert scale params expected by the loader."""

  def __init__(self, *, num_local_experts: int, hidden: int, inter: int) -> None:
    super().__init__()
    self.layers = torch.nn.ModuleList([torch.nn.Module()])
    self.layers[0].ffn = torch.nn.Module()
    # Match names used by ckpt.py stacking: layers.{i}.ffn.w13, w2, w13_scale, w2_scale
    self.layers[0].ffn.w13 = torch.nn.Parameter(
      torch.empty((num_local_experts, 2 * inter, hidden), dtype=torch.bfloat16)
    )
    self.layers[0].ffn.w2 = torch.nn.Parameter(
      torch.empty((num_local_experts, hidden, inter), dtype=torch.bfloat16)
    )
    self.layers[0].ffn.w13_scale = torch.nn.Parameter(
      torch.empty((num_local_experts, (2 * inter) // 128, hidden // 128), dtype=torch.float32)
    )
    self.layers[0].ffn.w2_scale = torch.nn.Parameter(
      torch.empty((num_local_experts, hidden // 128, inter // 128), dtype=torch.float32)
    )


class TestCheckpointScaleFormat(unittest.TestCase):
  def test_weight_scale_inv_is_converted_to_ue8m0(self):
    # Create a fake checkpoint shard containing a non-power-of-2 scale.
    # HF name must map to layers.0.attn.wo.weight_scale_inv.
    # Match the dummy parameter shape so load_state_dict can succeed.
    non_po2 = torch.full((56, 16), 0.3, dtype=torch.float32)
    non_po2[0, 0] = 1.1
    non_po2[0, 1] = 3.3
    self.assertFalse(_is_power_of_two(non_po2))

    with tempfile.TemporaryDirectory() as d:
      ckpt_dir = Path(d)
      shard = {
        "model.layers.0.self_attn.o_proj.weight_scale_inv": non_po2,
      }
      save_file(shard, str(ckpt_dir / "model-00001-of-00001.safetensors"))

      model = _DummyModel()
      missing, unexpected = load_checkpoint(model, str(ckpt_dir), strict=False)
      self.assertEqual(len(unexpected), 0)

      got = model.layers[0].attn.wo.weight_scale_inv.detach()
      self.assertTrue(_is_power_of_two(got), f"Expected UE8M0 pow2 scales, got:\n{got}")
      # Ensure we never downscale (round up for safety).
      self.assertTrue(torch.all(got >= non_po2))

  def test_expert_scales_are_converted_to_ue8m0(self):
    # Exercise the expert stacking path and verify w13_scale/w2_scale are UE8M0.
    from nmoe.serve.model import ModelConfig

    hidden = 256
    inter = 128
    num_experts = 2
    cfg = ModelConfig(
      num_layers=1,
      num_dense_layers=0,  # layer 0 is MoE for this dummy
      hidden_size=hidden,
      moe_intermediate_size=inter,
      num_experts=num_experts,
    )

    # Non power-of-2 scales for expert weights.
    w1_scale = torch.full((1, hidden // 128), 0.3, dtype=torch.float32)
    w2_scale = torch.full((hidden // 128, 1), 0.7, dtype=torch.float32)
    w3_scale = torch.full((1, hidden // 128), 1.1, dtype=torch.float32)
    self.assertFalse(_is_power_of_two(w1_scale))
    self.assertFalse(_is_power_of_two(w2_scale))
    self.assertFalse(_is_power_of_two(w3_scale))

    # Minimal expert weights (dtype doesn't matter for this regression).
    w1 = torch.randn((inter, hidden), dtype=torch.bfloat16) / 10
    w2 = torch.randn((hidden, inter), dtype=torch.bfloat16) / 10
    w3 = torch.randn((inter, hidden), dtype=torch.bfloat16) / 10

    with tempfile.TemporaryDirectory() as d:
      ckpt_dir = Path(d)
      shard = {
        # Expert 0
        "model.layers.0.ffn.experts.0.gate_proj.weight": w1.clone(),
        "model.layers.0.ffn.experts.0.gate_proj.weight_scale_inv": w1_scale.clone(),
        "model.layers.0.ffn.experts.0.down_proj.weight": w2.clone(),
        "model.layers.0.ffn.experts.0.down_proj.weight_scale_inv": w2_scale.clone(),
        "model.layers.0.ffn.experts.0.up_proj.weight": w3.clone(),
        "model.layers.0.ffn.experts.0.up_proj.weight_scale_inv": w3_scale.clone(),
        # Expert 1
        "model.layers.0.ffn.experts.1.gate_proj.weight": w1.clone(),
        "model.layers.0.ffn.experts.1.gate_proj.weight_scale_inv": w1_scale.clone(),
        "model.layers.0.ffn.experts.1.down_proj.weight": w2.clone(),
        "model.layers.0.ffn.experts.1.down_proj.weight_scale_inv": w2_scale.clone(),
        "model.layers.0.ffn.experts.1.up_proj.weight": w3.clone(),
        "model.layers.0.ffn.experts.1.up_proj.weight_scale_inv": w3_scale.clone(),
      }
      save_file(shard, str(ckpt_dir / "model-00001-of-00001.safetensors"))

      model = _DummyMoeModel(num_local_experts=num_experts, hidden=hidden, inter=inter)
      _missing, unexpected = load_checkpoint(model, str(ckpt_dir), strict=False, cfg=cfg)
      self.assertEqual(len(unexpected), 0)

      got_w13 = model.layers[0].ffn.w13_scale.detach()
      got_w2 = model.layers[0].ffn.w2_scale.detach()
      self.assertTrue(_is_power_of_two(got_w13), f"Expected UE8M0 pow2 w13_scale, got:\n{got_w13}")
      self.assertTrue(_is_power_of_two(got_w2), f"Expected UE8M0 pow2 w2_scale, got:\n{got_w2}")


def main() -> int:
  # Ensure we run under an environment with torch installed.
  if not hasattr(torch, "cuda"):
    raise RuntimeError("torch is required to run these tests")
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCheckpointScaleFormat)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
