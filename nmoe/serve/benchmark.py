# SPDX-License-Identifier: Apache-2.0
"""Benchmark prefill and decode throughput."""

import os
import time
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

  if rank == 0:
    print("=" * 70)
    print("nmoe.serve Benchmark")
    print(f"GPUs: {world_size}")
    print("=" * 70)

  cfg = ModelConfig(num_layers=61, num_dense_layers=3)

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
    print("\nModel loaded. Running benchmarks...\n")

  # Warmup
  warmup_tokens = 32
  input_ids = torch.randint(0, 10000, (1, warmup_tokens), device=device)
  positions = torch.arange(warmup_tokens, device=device).unsqueeze(0)
  num_blocks = 4
  kv_caches = [torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
  idx_k_caches = [torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([warmup_tokens], dtype=torch.int32, device=device)
  out_loc = torch.arange(warmup_tokens, dtype=torch.int32, device=device).unsqueeze(0)

  with torch.inference_mode():
    for _ in range(3):
      _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=[warmup_tokens], out_loc=out_loc)
  torch.cuda.synchronize()

  # ============ PREFILL BENCHMARK ============
  if rank == 0:
    print("=" * 50)
    print("PREFILL BENCHMARK")
    print("=" * 50)

  prefill_lengths = [128, 256, 512, 1024, 2048]
  num_iterations = 10

  for seq_len in prefill_lengths:
    num_blocks = (seq_len + 63) // 64 + 1
    input_ids = torch.randint(0, 10000, (1, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    kv_caches = [torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
    idx_k_caches = [torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    out_loc = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)

    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    with torch.inference_mode():
      for _ in range(num_iterations):
        _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                  block_table=block_table, cache_seqlens=cache_seqlens,
                  cache_seqlens_cpu=[seq_len], out_loc=out_loc)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_sec = (seq_len * num_iterations) / elapsed
    if rank == 0:
      print(f"  seq_len={seq_len:4d}: {tokens_per_sec:,.0f} tok/s  ({elapsed/num_iterations*1000:.1f} ms/iter)")

  # ============ DECODE BENCHMARK ============
  if rank == 0:
    print("\n" + "=" * 50)
    print("DECODE BENCHMARK")
    print("=" * 50)

  # First prefill some context
  context_len = 512
  num_blocks = 16
  kv_caches = [torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
  idx_k_caches = [torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

  # Prefill context
  input_ids = torch.randint(0, 10000, (1, context_len), device=device)
  positions = torch.arange(context_len, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([context_len], dtype=torch.int32, device=device)
  out_loc = torch.arange(context_len, dtype=torch.int32, device=device).unsqueeze(0)

  with torch.inference_mode():
    _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
              block_table=block_table, cache_seqlens=cache_seqlens,
              cache_seqlens_cpu=[context_len], out_loc=out_loc)
  torch.cuda.synchronize()

  # Decode benchmark
  num_decode_steps = 100
  decode_input = torch.randint(0, 10000, (1, 1), device=device)

  torch.cuda.synchronize()
  dist.barrier()

  start = time.perf_counter()
  with torch.inference_mode():
    for step in range(num_decode_steps):
      cur_pos = context_len + step
      positions = torch.tensor([[cur_pos]], dtype=torch.int64, device=device)
      out_loc = torch.tensor([[cur_pos]], dtype=torch.int32, device=device)
      cache_seqlens = torch.tensor([cur_pos + 1], dtype=torch.int32, device=device)

      _ = model(decode_input, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=[cur_pos + 1], out_loc=out_loc)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - start

  decode_tokens_per_sec = num_decode_steps / elapsed
  if rank == 0:
    print(f"  context={context_len}, steps={num_decode_steps}: {decode_tokens_per_sec:,.0f} tok/s  ({elapsed/num_decode_steps*1000:.1f} ms/step)")

  # ============ SUMMARY ============
  if rank == 0:
    print("\n" + "=" * 50)
    print("SUMMARY vs LMSYS targets (per node)")
    print("=" * 50)
    print(f"  Prefill target: 50-57K tok/s")
    print(f"  Decode target:  22K tok/s")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
