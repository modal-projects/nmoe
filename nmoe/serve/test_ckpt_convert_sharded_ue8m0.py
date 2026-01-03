# SPDX-License-Identifier: Apache-2.0
"""Regression: HF->mp sharder must emit UE8M0 (pow2) scales."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file, safe_open

from nmoe.serve.ckpt import convert_hf_to_sharded
from nmoe.serve.model import ModelConfig


def _is_power_of_two(x: torch.Tensor) -> bool:
  x = x.detach().abs().clamp(min=1e-12)
  log2 = torch.log2(x)
  return bool(torch.allclose(log2, log2.round(), atol=1e-5))


class TestConvertHfToShardedUE8M0(unittest.TestCase):
  def test_convert_emits_pow2_scales(self):
    cfg = ModelConfig(
      num_layers=1,
      num_dense_layers=0,
      hidden_size=64,
      intermediate_size=64,
      moe_intermediate_size=64,
      num_experts=2,
    )

    # Minimal synthetic HF shard: one dense FP8 weight+scale and a single MoE layer with 2 experts.
    w = (torch.randn(64, 64, dtype=torch.float32) / 10).to(torch.float8_e4m3fn)
    non_po2 = torch.tensor([[0.3]], dtype=torch.float32)
    self.assertFalse(_is_power_of_two(non_po2))

    hf = {
      # Dense fp8 weight (maps to layers.0.attn.wo.weight + weight_scale_inv).
      "model.layers.0.self_attn.o_proj.weight": w.clone(),
      "model.layers.0.self_attn.o_proj.weight_scale_inv": non_po2.clone(),
    }

    for e in range(cfg.num_experts):
      hf[f"model.layers.0.ffn.experts.{e}.gate_proj.weight"] = w.clone()
      hf[f"model.layers.0.ffn.experts.{e}.gate_proj.weight_scale_inv"] = non_po2.clone()
      hf[f"model.layers.0.ffn.experts.{e}.up_proj.weight"] = w.clone()
      hf[f"model.layers.0.ffn.experts.{e}.up_proj.weight_scale_inv"] = non_po2.clone()
      hf[f"model.layers.0.ffn.experts.{e}.down_proj.weight"] = w.t().contiguous().clone()
      hf[f"model.layers.0.ffn.experts.{e}.down_proj.weight_scale_inv"] = non_po2.clone()

    with tempfile.TemporaryDirectory() as d:
      root = Path(d)
      hf_dir = root / "hf"
      out_dir = root / "out"
      hf_dir.mkdir()
      out_dir.mkdir()

      save_file(hf, str(hf_dir / "model-00001-of-00001.safetensors"))

      convert_hf_to_sharded(str(hf_dir), str(out_dir), world_size=1, cfg=cfg)
      out_file = out_dir / "model0-mp1.safetensors"
      self.assertTrue(out_file.exists(), f"Expected output shard {out_file}")

      with safe_open(str(out_file), framework="pt", device="cpu") as f:
        dense_scale = f.get_tensor("layers.0.attn.wo.weight_scale_inv")
        self.assertTrue(_is_power_of_two(dense_scale), f"dense scale not pow2:\n{dense_scale}")

        # Expert scales are stacked into w13_scale/w2_scale (must be pow2).
        w13_scale = f.get_tensor("layers.0.ffn.w13_scale")
        w2_scale = f.get_tensor("layers.0.ffn.w2_scale")
        self.assertTrue(_is_power_of_two(w13_scale), f"expert w13_scale not pow2:\n{w13_scale}")
        self.assertTrue(_is_power_of_two(w2_scale), f"expert w2_scale not pow2:\n{w2_scale}")


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestConvertHfToShardedUE8M0)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
