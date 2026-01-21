# SPDX-License-Identifier: Apache-2.0
"""Debug FFN layer issue - w1 output is extremely negative."""

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
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import safe_open
from glob import glob


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, weight_dequant
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=1, num_dense_layers=1)

  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("FFN Debug - Investigating negative w1 output")
    print("=" * 70)

  # Get FFN layer
  ffn = model.layers[0].ffn

  # Check w1 weight stats
  w1_weight = ffn.w1.weight  # FP8
  w1_scale = ffn.w1.weight_scale_inv  # float32

  if rank == 0:
    print(f"\n--- FFN W1 Weight Analysis ---")
    print(f"w1.weight shape: {w1_weight.shape}, dtype: {w1_weight.dtype}")
    print(f"w1.scale shape: {w1_scale.shape}, dtype: {w1_scale.dtype}")

    # Dequantize to bf16 for analysis
    w1_dequant = weight_dequant(w1_weight, w1_scale)
    print(f"\nDequantized w1:")
    print(f"  mean: {w1_dequant.float().mean():.6f}")
    print(f"  std:  {w1_dequant.float().std():.6f}")
    print(f"  min:  {w1_dequant.float().min():.6f}")
    print(f"  max:  {w1_dequant.float().max():.6f}")

    # Check scale values
    print(f"\nw1 scales:")
    print(f"  mean: {w1_scale.mean():.6f}")
    print(f"  min:  {w1_scale.min():.6f}")
    print(f"  max:  {w1_scale.max():.6f}")

  # Load reference weights directly from checkpoint
  if rank == 0:
    print(f"\n--- Reference Weights from Checkpoint ---")
    ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
    files = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))

    for fpath in files:
      with safe_open(fpath, framework="pt", device="cpu") as f:
        for name in f.keys():
          if "layers.0.mlp.gate_proj.weight" in name and "scale" not in name:
            ref_w1 = f.get_tensor(name)
            print(f"\nReference gate_proj (w1) from {fpath}:")
            print(f"  shape: {ref_w1.shape}, dtype: {ref_w1.dtype}")
            if ref_w1.dtype == torch.float8_e4m3fn:
              print(f"  FP8 values - need scale to interpret")
            else:
              print(f"  mean: {ref_w1.float().mean():.6f}")
              print(f"  std:  {ref_w1.float().std():.6f}")

          if "layers.0.mlp.gate_proj.weight_scale_inv" in name:
            ref_w1_scale = f.get_tensor(name)
            print(f"\nReference gate_proj scale:")
            print(f"  shape: {ref_w1_scale.shape}")
            print(f"  mean: {ref_w1_scale.mean():.6f}")
            print(f"  min:  {ref_w1_scale.min():.6f}")
            print(f"  max:  {ref_w1_scale.max():.6f}")

  # Test with known input
  if rank == 0:
    print(f"\n--- Test FFN with known input ---")

  # Create test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)

  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

  if rank == 0:
    print(f"\nInput to FFN (x_norm):")
    print(f"  shape: {x_norm.shape}")
    print(f"  mean: {x_norm.float().mean():.6f}")
    print(f"  std:  {x_norm.float().std():.6f}")
    print(f"  amax: {x_norm.float().abs().max():.6f}")

  # Test w1 forward
  with torch.no_grad():
    w1_out = ffn.w1(x_norm)

  if rank == 0:
    print(f"\nFFN w1 output:")
    print(f"  mean: {w1_out.float().mean():.6f}")
    print(f"  std:  {w1_out.float().std():.6f}")
    print(f"  min:  {w1_out.float().min():.6f}")
    print(f"  max:  {w1_out.float().max():.6f}")

  # Compare with reference computation (bf16 matmul instead of FP8)
  if rank == 0:
    print(f"\n--- Reference BF16 computation ---")

  with torch.no_grad():
    # Dequantize weight
    w1_dequant = weight_dequant(ffn.w1.weight, ffn.w1.weight_scale_inv)

    # BF16 matmul
    w1_out_ref = F.linear(x_norm.to(torch.bfloat16), w1_dequant)

  if rank == 0:
    print(f"\nBF16 w1 output (using dequantized weights):")
    print(f"  mean: {w1_out_ref.float().mean():.6f}")
    print(f"  std:  {w1_out_ref.float().std():.6f}")
    print(f"  min:  {w1_out_ref.float().min():.6f}")
    print(f"  max:  {w1_out_ref.float().max():.6f}")

    # Compare
    diff = (w1_out.float() - w1_out_ref.float()).abs()
    print(f"\nDifference (FP8 vs BF16):")
    print(f"  max_diff: {diff.max():.6f}")
    print(f"  mean_diff: {diff.mean():.6f}")

  # Check if the negative mean is in the weights
  if rank == 0:
    print(f"\n--- Check weight bias ---")

    # Row-wise mean of dequantized weights
    w1_dequant = weight_dequant(ffn.w1.weight, ffn.w1.weight_scale_inv)
    row_means = w1_dequant.float().mean(dim=1)
    print(f"Row-wise mean of w1:")
    print(f"  min:  {row_means.min():.6f}")
    print(f"  max:  {row_means.max():.6f}")
    print(f"  mean: {row_means.mean():.6f}")

    # The output is y = x @ W.T
    # If W has row-wise mean bias, it would show up when input has mean != 0
    print(f"\nInput mean: {x_norm.float().mean():.6f}")
    print(f"Expected output bias from weight means: {(x_norm.float().mean() * row_means.sum() / x_norm.shape[-1]):.6f}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
