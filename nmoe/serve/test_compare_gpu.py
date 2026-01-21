# SPDX-License-Identifier: Apache-2.0
"""Compare single-GPU vs multi-GPU outputs for 4-layer model."""

import os
import sys
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
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

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

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  # Same input as single-GPU test: "The capital of France is"
  input_ids = torch.tensor([[671, 6102, 294, 8760, 344]], device=device)
  S = 5
  positions = torch.arange(S, device=device).unsqueeze(0)
  kv_caches = [torch.zeros(1, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
  idx_k_caches = [torch.zeros(1, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
  block_table = torch.arange(1, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  with torch.no_grad():
    logits = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                   block_table=block_table, cache_seqlens=cache_seqlens, cache_seqlens_cpu=[S], out_loc=out_loc)

  if rank == 0:
    print(f"Multi-GPU 4-layer (world_size={world_size}):")
    print(f"  Argmax: {logits[0, -1, :].argmax().item()}")
    top5 = logits[0, -1, :].topk(5)
    print(f"  Top 5 indices: {top5.indices.tolist()}")
    print(f"  Top 5 logits: {[f'{v:.2f}' for v in top5.values.tolist()]}")
    print(f"  Token 11111 (Paris) logit: {logits[0, -1, 11111].item():.2f}")

    # Expected from single-GPU test
    print("\nExpected from single-GPU (world_size=1):")
    print("  Argmax: 44117")
    print("  Top 5 indices: [44117, 64415, 74993, 53494, 15215]")
    print("  Top 5 logits: ['10.79', '10.57', '10.45', '10.13', '10.00']")
    print("  Token 11111 (Paris) logit: -0.37")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
