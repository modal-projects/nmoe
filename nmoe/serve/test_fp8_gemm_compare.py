# SPDX-License-Identifier: Apache-2.0
"""Compare DeepGEMM FP8 outputs against torch dequant+matmul reference."""

import os
from pathlib import Path


def _maybe_set_cutlass_path() -> None:
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


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
  """Dequantize FP8 weight using block-wise scales."""
  shape = weight.shape
  out_tiles = shape[0] // block_size
  in_tiles = shape[1] // block_size

  # Handle non-aligned dimensions
  if shape[0] % block_size != 0 or shape[1] % block_size != 0:
    out_tiles_padded, in_tiles_padded = scale.shape
    out_padded = out_tiles_padded * block_size
    in_padded = in_tiles_padded * block_size
    w_pad = torch.zeros(out_padded, in_padded, dtype=weight.dtype, device=weight.device)
    w_pad[:shape[0], :shape[1]] = weight
    w_pad = w_pad.view(out_tiles_padded, block_size, in_tiles_padded, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(-1, block_size * block_size)
    w_pad = (w_pad.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
    w_pad = w_pad.view(out_tiles_padded, in_tiles_padded, block_size, block_size)
    w_pad = w_pad.transpose(1, 2).contiguous().view(out_padded, in_padded)
    return w_pad[:shape[0], :shape[1]]

  weight = weight.view(out_tiles, block_size, in_tiles, block_size)
  weight = weight.transpose(1, 2).contiguous().view(-1, block_size * block_size)
  weight = (weight.float() * scale.view(-1, 1).float()).to(torch.bfloat16)
  weight = weight.view(out_tiles, in_tiles, block_size, block_size)
  weight = weight.transpose(1, 2).contiguous().view(shape)
  return weight


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Load model
  cfg = ModelConfig(num_layers=4, num_dense_layers=3)

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

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("FP8 GEMM Comparison: DeepGEMM vs Torch Reference")
    print("=" * 70)

  # Test with layer 3's MoE shared experts (simpler than routed)
  shared_mlp = model.layers[3].ffn.shared

  # Test input
  torch.manual_seed(42)
  x = torch.randn(16, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  if rank == 0:
    print(f"\n=== Shared Experts MLP Test ===")
    print(f"Input shape: {x.shape}")
    print(f"w1 shape: {shared_mlp.w1.weight.shape}, dtype: {shared_mlp.w1.weight.dtype}")

  with torch.no_grad():
    # Our FP8 implementation
    w1_out_fp8 = shared_mlp.w1(x)
    w3_out_fp8 = shared_mlp.w3(x)

    # Torch reference (dequant + matmul)
    w1_dequant = weight_dequant(shared_mlp.w1.weight, shared_mlp.w1.weight_scale_inv)
    w3_dequant = weight_dequant(shared_mlp.w3.weight, shared_mlp.w3.weight_scale_inv)
    w1_out_ref = F.linear(x, w1_dequant)
    w3_out_ref = F.linear(x, w3_dequant)

  if rank == 0:
    # Compare w1 outputs
    w1_diff = (w1_out_fp8.float() - w1_out_ref.float()).abs()
    print(f"\n=== w1 (gate) Projection ===")
    print(f"FP8 output: mean={w1_out_fp8.float().mean():.6f}, std={w1_out_fp8.float().std():.6f}")
    print(f"Ref output: mean={w1_out_ref.float().mean():.6f}, std={w1_out_ref.float().std():.6f}")
    print(f"Diff: max={w1_diff.max():.6f}, mean={w1_diff.mean():.6f}")
    print(f"Relative max diff: {(w1_diff / w1_out_ref.float().abs().clamp(min=1e-6)).max():.6f}")
    cos_sim_w1 = F.cosine_similarity(w1_out_fp8.flatten().float().unsqueeze(0),
                                      w1_out_ref.flatten().float().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim_w1:.6f}")

    # Compare w3 outputs
    w3_diff = (w3_out_fp8.float() - w3_out_ref.float()).abs()
    print(f"\n=== w3 (up) Projection ===")
    print(f"FP8 output: mean={w3_out_fp8.float().mean():.6f}, std={w3_out_fp8.float().std():.6f}")
    print(f"Ref output: mean={w3_out_ref.float().mean():.6f}, std={w3_out_ref.float().std():.6f}")
    print(f"Diff: max={w3_diff.max():.6f}, mean={w3_diff.mean():.6f}")
    cos_sim_w3 = F.cosine_similarity(w3_out_fp8.flatten().float().unsqueeze(0),
                                      w3_out_ref.flatten().float().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim_w3:.6f}")

    # Full forward comparison
    print(f"\n=== Full Shared MLP Forward ===")
    with torch.no_grad():
      # Our implementation
      mlp_out_fp8 = shared_mlp(x)

      # Reference (using dequantized weights)
      w2_dequant = weight_dequant(shared_mlp.w2.weight, shared_mlp.w2.weight_scale_inv)
      gate_up_ref = F.silu(w1_out_ref.float()) * w3_out_ref.float()
      mlp_out_ref = F.linear(gate_up_ref.to(torch.bfloat16), w2_dequant)

    mlp_diff = (mlp_out_fp8.float() - mlp_out_ref.float()).abs()
    print(f"FP8 output: mean={mlp_out_fp8.float().mean():.6f}, std={mlp_out_fp8.float().std():.6f}")
    print(f"Ref output: mean={mlp_out_ref.float().mean():.6f}, std={mlp_out_ref.float().std():.6f}")
    print(f"Diff: max={mlp_diff.max():.6f}, mean={mlp_diff.mean():.6f}")
    cos_sim_mlp = F.cosine_similarity(mlp_out_fp8.flatten().float().unsqueeze(0),
                                       mlp_out_ref.flatten().float().unsqueeze(0)).item()
    print(f"Cosine similarity: {cos_sim_mlp:.6f}")

    if cos_sim_mlp < 0.999:
      print(f"\n⚠️  WARNING: MLP output differs significantly from reference!")
      print(f"This FP8 GEMM error will compound across layers.")

    # Check if diff could flip routing decisions
    print(f"\n=== Impact on Routing ===")
    # The router sees hidden states, small differences can flip sigmoid scores
    # near decision boundaries
    print(f"Hidden state diff max: {mlp_diff.max():.6f}")
    print(f"This can cause routing flips when sigmoid scores are near 0.5")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
