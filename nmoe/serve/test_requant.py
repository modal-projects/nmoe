# SPDX-License-Identifier: Apache-2.0
"""Test that FP8 requantization preserves dequantized weights."""

import torch
from safetensors.torch import safe_open


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """Dequantize FP8 weight using block-wise scales."""
  shape = weight.shape
  weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size)
  weight = weight.transpose(1, 2).contiguous().view(-1, block_size * block_size)
  weight = (weight.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
  weight = weight.view(shape[0] // block_size, shape[1] // block_size, block_size, block_size)
  weight = weight.transpose(1, 2).contiguous().view(shape)
  return weight


def test_requant():
  """Test requantization on o_proj which has non-pow2 scales."""
  from nmoe.serve.ckpt import _is_ue8m0, _requantize_fp8_for_ue8m0

  # Load original o_proj weight and scale from checkpoint
  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"

  weight_orig = None
  scale_orig = None

  from glob import glob
  import os
  files = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))

  for fpath in files:
    with safe_open(fpath, framework="pt", device="cpu") as f:
      for name in f.keys():
        if "layers.0.self_attn.o_proj.weight" in name and "scale" not in name:
          weight_orig = f.get_tensor(name)
        if "layers.0.self_attn.o_proj.weight_scale_inv" in name:
          scale_orig = f.get_tensor(name)
    if weight_orig is not None and scale_orig is not None:
      break

  assert weight_orig is not None, "Could not find o_proj weight"
  assert scale_orig is not None, "Could not find o_proj scale"

  print(f"Original weight: {weight_orig.shape}, dtype={weight_orig.dtype}")
  print(f"Original scale: {scale_orig.shape}, range=[{scale_orig.min():.6e}, {scale_orig.max():.6e}]")
  print(f"Original scale is_pow2: {_is_ue8m0(scale_orig)}")

  # Dequantize original
  w_dequant_orig = weight_dequant(weight_orig, scale_orig)
  print(f"\nOriginal dequantized: mean={w_dequant_orig.float().mean():.6f}, amax={w_dequant_orig.float().abs().max():.4f}")

  # Requantize for UE8M0
  weight_new, scale_new = _requantize_fp8_for_ue8m0(weight_orig, scale_orig)

  print(f"\nNew scale: range=[{scale_new.min():.6e}, {scale_new.max():.6e}]")
  print(f"New scale is_pow2: {_is_ue8m0(scale_new)}")

  # Dequantize requantized
  w_dequant_new = weight_dequant(weight_new, scale_new)
  print(f"Requantized dequantized: mean={w_dequant_new.float().mean():.6f}, amax={w_dequant_new.float().abs().max():.4f}")

  # Compare
  diff = (w_dequant_orig.float() - w_dequant_new.float()).abs()
  print(f"\n=== Comparison ===")
  print(f"Max diff: {diff.max():.6f}")
  print(f"Mean diff: {diff.mean():.6f}")
  print(f"Relative max diff: {(diff / w_dequant_orig.float().abs().clamp(min=1e-6)).max():.6f}")

  # The max diff should be small (FP8 quantization noise)
  assert diff.max() < 0.1, f"Max diff too large: {diff.max()}"
  assert diff.mean() < 0.01, f"Mean diff too large: {diff.mean()}"

  print("\n✓ Requantization test PASSED - dequantized weights preserved!")


if __name__ == "__main__":
  test_requant()
