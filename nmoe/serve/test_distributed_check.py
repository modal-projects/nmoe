# SPDX-License-Identifier: Apache-2.0
"""Test distributed model output consistency across ranks."""

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
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  if rank == 0:
    print("=" * 60)
    print("Distributed Consistency Check")
    print(f"World size: {world_size}")
    print("=" * 60)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Use fewer layers to save memory
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

  # Test input - same on all ranks
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  num_blocks = 1
  kv_caches = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  with torch.no_grad():
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  # Check 1: All ranks should have same logits shape
  local_shape = torch.tensor(logits.shape, device=device)
  shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
  dist.all_gather(shapes, local_shape)

  if rank == 0:
    print(f"\n1. Logits shapes across ranks:")
    for r, s in enumerate(shapes):
      print(f"   Rank {r}: {s.tolist()}")

  # Check 2: All ranks should have same argmax
  local_argmax = logits[0, -1, :].argmax().to(torch.int64)
  argmaxes = [torch.empty_like(local_argmax) for _ in range(world_size)]
  dist.all_gather(argmaxes, local_argmax.contiguous())

  if rank == 0:
    print(f"\n2. Argmax across ranks:")
    for r, a in enumerate(argmaxes):
      print(f"   Rank {r}: {int(a)}")
    match = all(int(a) == int(argmaxes[0]) for a in argmaxes)
    print(f"   All match: {match}")

  # Check 3: Compare actual logits values
  # Gather logit samples from each rank
  sample_logits = logits[0, -1, :100].float()  # First 100 logits
  all_samples = [torch.empty_like(sample_logits) for _ in range(world_size)]
  dist.all_gather(all_samples, sample_logits.contiguous())

  if rank == 0:
    print(f"\n3. Logits sample consistency:")
    base = all_samples[0]
    for r in range(1, world_size):
      diff = (all_samples[r] - base).abs()
      print(f"   Rank 0 vs Rank {r}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")

  # Check 4: Verify hidden state consistency at different layers
  # Re-run forward with hooks to capture intermediate states
  if rank == 0:
    print(f"\n4. Checking embedding output...")

  # Just check embedding
  with torch.no_grad():
    embed_out = model.embed(input_ids)

  embed_mean = embed_out.float().mean()
  embed_means = [torch.empty_like(embed_mean) for _ in range(world_size)]
  dist.all_gather(embed_means, embed_mean.contiguous())

  if rank == 0:
    print(f"   Embedding means: {[f'{m:.6f}' for m in embed_means]}")
    match = all(abs(float(m) - float(embed_means[0])) < 1e-4 for m in embed_means)
    print(f"   All match: {match}")

  # Check 5: Layer norm outputs
  if rank == 0:
    print(f"\n5. Checking layer 0 attn_norm output...")

  with torch.no_grad():
    x = model.embed(input_ids)
    x_norm = model.layers[0].attn_norm(x)

  norm_mean = x_norm.float().mean()
  norm_means = [torch.empty_like(norm_mean) for _ in range(world_size)]
  dist.all_gather(norm_means, norm_mean.contiguous())

  if rank == 0:
    print(f"   Norm means: {[f'{m:.6f}' for m in norm_means]}")
    match = all(abs(float(m) - float(norm_means[0])) < 1e-4 for m in norm_means)
    print(f"   All match: {match}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
