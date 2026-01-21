# SPDX-License-Identifier: Apache-2.0
"""Compare KV cache contents between prefill and decode."""

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

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Use just 1 layer
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

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("KV Cache Comparison: Prefill vs Decode")
    print("=" * 70)

  # Same input for both tests
  input_ids = torch.tensor([[1, 100, 1000, 10000, 50000]], device=device)
  S = 5
  next_token = 223

  num_blocks = 4
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

  # === TEST 1: Prefill 6 tokens ===
  if rank == 0:
    print(f"\n=== TEST 1: Prefill {S+1} tokens ===")

  kv_cache_1 = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
  idx_k_cache_1 = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)

  input_full = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
  positions_1 = torch.arange(S + 1, device=device).unsqueeze(0)
  out_loc_1 = torch.arange(S + 1, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens_1 = torch.tensor([S + 1], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits_1 = model(
      input_full, positions_1,
      kv_caches=[kv_cache_1], idx_k_caches=[idx_k_cache_1],
      block_table=block_table, cache_seqlens=cache_seqlens_1,
      cache_seqlens_cpu=[S + 1], out_loc=out_loc_1,
    )

  # Save KV cache snapshot
  kv_cache_1_snapshot = kv_cache_1.clone()
  idx_k_cache_1_snapshot = idx_k_cache_1.clone()

  if rank == 0:
    print(f"KV cache for positions 0-5: {kv_cache_1[0, :S+1, 0, :8].tolist()}")
    print(f"idx_k cache for positions 0-5: {idx_k_cache_1[0, :S+1, :4].tolist()}")
    print(f"Logits[-1] argmax: {logits_1[0, -1, :].argmax().item()}")

  # === TEST 2: Prefill 5, then decode 1 ===
  if rank == 0:
    print(f"\n=== TEST 2: Prefill {S}, then decode 1 ===")

  kv_cache_2 = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
  idx_k_cache_2 = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)

  # Prefill
  positions_2a = torch.arange(S, device=device).unsqueeze(0)
  out_loc_2a = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens_2a = torch.tensor([S], dtype=torch.int32, device=device)

  with torch.no_grad():
    _ = model(
      input_ids, positions_2a,
      kv_caches=[kv_cache_2], idx_k_caches=[idx_k_cache_2],
      block_table=block_table, cache_seqlens=cache_seqlens_2a,
      cache_seqlens_cpu=[S], out_loc=out_loc_2a,
    )

  if rank == 0:
    print(f"After prefill - KV cache for positions 0-4: {kv_cache_2[0, :S, 0, :8].tolist()}")
    print(f"After prefill - idx_k cache for positions 0-4: {idx_k_cache_2[0, :S, :4].tolist()}")

  # Decode
  inp_decode = torch.tensor([[next_token]], device=device)
  pos_decode = torch.tensor([[S]], dtype=torch.int64, device=device)
  out_loc_decode = torch.tensor([[S]], dtype=torch.int32, device=device)
  cache_seqlens_decode = torch.tensor([S + 1], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits_2 = model(
      inp_decode, pos_decode,
      kv_caches=[kv_cache_2], idx_k_caches=[idx_k_cache_2],
      block_table=block_table, cache_seqlens=cache_seqlens_decode,
      cache_seqlens_cpu=[S + 1], out_loc=out_loc_decode,
    )

  if rank == 0:
    print(f"After decode - KV cache for positions 0-5: {kv_cache_2[0, :S+1, 0, :8].tolist()}")
    print(f"After decode - idx_k cache for positions 0-5: {idx_k_cache_2[0, :S+1, :4].tolist()}")
    print(f"Decode logits[0] argmax: {logits_2[0, 0, :].argmax().item()}")

  # === Compare caches ===
  if rank == 0:
    print(f"\n=== Cache Comparison ===")

    # Compare KV cache
    kv_diff = (kv_cache_1_snapshot[:, :S+1].float() - kv_cache_2[:, :S+1].float()).abs()
    print(f"KV cache diff max: {kv_diff.max().item()}")
    if kv_diff.max() > 0:
      print(f"  Position with max diff: {kv_diff.argmax()}")
      # Find which position differs
      for pos in range(S + 1):
        pos_diff = (kv_cache_1_snapshot[0, pos].float() - kv_cache_2[0, pos].float()).abs().max()
        if pos_diff > 0:
          print(f"  Position {pos} diff: {pos_diff.item()}")

    # Compare idx_k cache
    idx_diff = (idx_k_cache_1_snapshot[:, :S+1].float() - idx_k_cache_2[:, :S+1].float()).abs()
    print(f"idx_k cache diff max: {idx_diff.max().item()}")

    # Check if the issue is in the cache values
    if kv_diff.max() == 0 and idx_diff.max() == 0:
      print("\n✓ Caches are identical! Issue must be elsewhere (DSA or FlashMLA kernel).")
    else:
      print("\n⚠️ Caches differ! This is the root cause.")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
