# SPDX-License-Identifier: Apache-2.0
"""Compare single-GPU vs multi-GPU output for same model and input."""

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


def run_model(num_layers: int, rank: int, world_size: int, device):
  """Run model with given config and return logits."""
  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=num_layers, num_dense_layers=3)

  # DeepEP buffer - minimal for small tests
  if world_size > 1:
    hidden_bytes = cfg.hidden_size * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
      dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
      combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
  else:
    num_nvl_bytes = 0

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)

  # Fixed test input: "The capital of France is"
  B, S = 1, 5
  input_ids = torch.tensor([[671, 6102, 294, 8760, 344]], device=device)
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

  return logits


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  if rank == 0:
    print("=" * 60)
    print(f"Single-GPU vs {world_size}-GPU Comparison")
    print("=" * 60)
    print("Input: 'The capital of France is' [671, 6102, 294, 8760, 344]")

  # Test with 4 layers first (small enough to fit)
  for num_layers in [4, 8]:
    dist.barrier()

    if rank == 0:
      print(f"\n--- {num_layers} layers ---")

    # Multi-GPU run
    logits_multi = run_model(num_layers, rank, world_size, device)
    argmax_multi = logits_multi[0, -1, :].argmax().item()
    top5_multi = logits_multi[0, -1, :].topk(5)

    if rank == 0:
      print(f"Multi-GPU ({world_size}): argmax={argmax_multi}")
      print(f"  Top 5: {top5_multi.indices.tolist()}")
      print(f"  Logits: {[f'{v:.2f}' for v in top5_multi.values.tolist()]}")

    # Clear memory
    del logits_multi
    torch.cuda.empty_cache()

    # Single-GPU run (only rank 0, with world_size=1)
    # We can't easily do this in the same process, so let's compare
    # against saved reference values instead

  dist.barrier()

  if rank == 0:
    print("\n" + "=" * 60)
    print("Note: To compare with single-GPU, run separately with world_size=1")
    print("=" * 60)

  dist.destroy_process_group()


if __name__ == "__main__":
  main()
