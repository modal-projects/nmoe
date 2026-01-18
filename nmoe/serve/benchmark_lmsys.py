# SPDX-License-Identifier: Apache-2.0
"""Benchmark targeting LMSYS performance numbers.

LMSYS targets (per 8x H100 node):
- Prefill: 52.3k tok/s (16,384 tokens/device, 4096 input length)
- Decode: 22.3k tok/s (256 sequences, 2000 input length)

Key optimizations needed:
1. Batched inference (not BS=1)
2. CUDA graphs for decode
3. DeepEP low-latency dispatch

Supports both attention types:
- DSA (DeepSeek Sparse Attention): for V3.2-Speciale
- MLA (Multi-Head Latent Attention): for V3-0324, Kimi-K2
"""

import argparse
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


def benchmark_prefill(model, cfg, device, rank, world_size, attention_type: str):
    """Benchmark prefill with batching matching LMSYS conditions."""
    if rank == 0:
        print("\n" + "=" * 60)
        print(f"PREFILL BENCHMARK ({attention_type.upper()})")
        print("=" * 60)

    # LMSYS: 16,384 tokens per device with 4,096 input length = BS=4
    configs = [
        (1, 512),    # baseline
        (4, 512),    # small batch
        (4, 2048),   # medium - closer to LMSYS
        (4, 4096),   # LMSYS config
        (8, 2048),   # larger batch
        (16, 1024),  # high batch, shorter seq
    ]

    num_iterations = 5
    page_size = 64

    for batch_size, seq_len in configs:
        total_tokens = batch_size * seq_len
        num_blocks_per_seq = (seq_len + page_size - 1) // page_size + 1
        total_blocks = batch_size * num_blocks_per_seq

        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Per-layer KV caches (format depends on attention type)
        if attention_type == "dsa":
            kv_caches = [
                torch.zeros(total_blocks, page_size, 1, 656, dtype=torch.uint8, device=device)
                for _ in range(cfg.num_layers)
            ]
            idx_k_caches = [
                torch.zeros(total_blocks, page_size, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
                for _ in range(cfg.num_layers)
            ]
            kv_caches_latent = None
            kv_caches_rope = None
        else:  # MLA
            kv_caches = None
            idx_k_caches = None
            kv_caches_latent = [
                torch.zeros(total_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
                for _ in range(cfg.num_layers)
            ]
            kv_caches_rope = [
                torch.zeros(total_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
                for _ in range(cfg.num_layers)
            ]

        # Block table: [B, max_blocks_per_seq]
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            block_table[b] = torch.arange(b * num_blocks_per_seq, (b + 1) * num_blocks_per_seq)

        cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        cache_seqlens_cpu = [seq_len] * batch_size

        # out_loc: [B, S] - where each token writes to cache
        out_loc = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=device)
        for b in range(batch_size):
            base = b * num_blocks_per_seq * page_size
            out_loc[b] = torch.arange(base, base + seq_len)

        # Build forward kwargs
        if attention_type == "dsa":
            fwd_kwargs = dict(
                kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                block_table=block_table, cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc,
            )
        else:
            fwd_kwargs = dict(
                kv_caches_latent=kv_caches_latent, kv_caches_rope=kv_caches_rope,
                block_table=block_table, cache_seqlens=cache_seqlens,
                out_loc=out_loc, prefill_mode="dense",
            )

        # Warmup
        with torch.inference_mode():
            _ = model(input_ids, positions, **fwd_kwargs)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(num_iterations):
                _ = model(input_ids, positions, **fwd_kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tokens_per_sec = (total_tokens * num_iterations) / elapsed
        ms_per_iter = (elapsed / num_iterations) * 1000

        if rank == 0:
            print(f"  BS={batch_size:2d} seq={seq_len:4d} ({total_tokens:6,d} tok): "
                  f"{tokens_per_sec:>8,.0f} tok/s  ({ms_per_iter:>6.1f} ms)")

    if rank == 0:
        print(f"\n  Target: 52,300 tok/s (LMSYS)")


def benchmark_decode(model, cfg, device, rank, world_size):
    """Benchmark decode with batching matching LMSYS conditions."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("DECODE BENCHMARK (batched)")
        print("=" * 60)

    # LMSYS: 256 sequences with 2000 input length
    # We'll test various batch sizes
    configs = [
        (1, 512),     # baseline
        (8, 512),     # small batch
        (32, 512),    # medium batch
        (64, 1024),   # larger
        (128, 1024),  # even larger
        (256, 2000),  # LMSYS config
    ]

    num_decode_steps = 20  # Steps per measurement

    for batch_size, context_len in configs:
        num_blocks_per_seq = (context_len + 100 + 63) // 64  # Room for decode
        total_blocks = batch_size * num_blocks_per_seq

        # Allocate caches
        kv_caches = [
            torch.zeros(total_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
            for _ in range(cfg.num_layers)
        ]
        idx_k_caches = [
            torch.zeros(total_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
            for _ in range(cfg.num_layers)
        ]

        # Block table
        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            block_table[b] = torch.arange(b * num_blocks_per_seq, (b + 1) * num_blocks_per_seq)

        # Prefill context first
        input_ids = torch.randint(0, 10000, (batch_size, context_len), device=device)
        positions = torch.arange(context_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cache_seqlens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
        cache_seqlens_cpu = [context_len] * batch_size

        out_loc = torch.zeros(batch_size, context_len, dtype=torch.int32, device=device)
        for b in range(batch_size):
            base = b * num_blocks_per_seq * 64
            out_loc[b] = torch.arange(base, base + context_len)

        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                      block_table=block_table, cache_seqlens=cache_seqlens,
                      cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc)
        torch.cuda.synchronize()

        # Decode benchmark - single token per sequence
        decode_input = torch.randint(0, 10000, (batch_size, 1), device=device)

        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        start = time.perf_counter()
        with torch.inference_mode():
            for step in range(num_decode_steps):
                cur_pos = context_len + step
                positions = torch.full((batch_size, 1), cur_pos, dtype=torch.int64, device=device)
                cache_seqlens = torch.full((batch_size,), cur_pos + 1, dtype=torch.int32, device=device)
                cache_seqlens_cpu = [cur_pos + 1] * batch_size

                # out_loc for decode token
                out_loc = torch.zeros(batch_size, 1, dtype=torch.int32, device=device)
                for b in range(batch_size):
                    page_idx = cur_pos // 64
                    slot_idx = cur_pos % 64
                    out_loc[b, 0] = block_table[b, page_idx] * 64 + slot_idx

                _ = model(decode_input, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                          block_table=block_table, cache_seqlens=cache_seqlens,
                          cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Total tokens = batch_size * num_decode_steps
        total_tokens = batch_size * num_decode_steps
        tokens_per_sec = total_tokens / elapsed
        ms_per_step = (elapsed / num_decode_steps) * 1000

        if rank == 0:
            print(f"  BS={batch_size:3d} ctx={context_len:4d}: "
                  f"{tokens_per_sec:>8,.0f} tok/s  ({ms_per_step:>6.1f} ms/step)")

    if rank == 0:
        print(f"\n  Target: 22,300 tok/s (LMSYS)")
        print(f"\n  Note: CUDA graphs not implemented yet (critical for decode)")


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
        print("=" * 60)
        print("nmoe.serve LMSYS Benchmark")
        print(f"GPUs: {world_size}")
        print("=" * 60)
        print("\nTargets (per 8x H100 node):")
        print("  Prefill: 52,300 tok/s")
        print("  Decode:  22,300 tok/s")

    cfg = ModelConfig(num_layers=61, num_dense_layers=3)

    # DeepEP buffer
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
    if rank == 0:
        print(f"\nLoading checkpoint from {ckpt_path}...")
    load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
    dist.barrier()

    if rank == 0:
        print("Model loaded.\n")

    # Run benchmarks
    benchmark_prefill(model, cfg, device, rank, world_size, cfg.attention_type)
    # benchmark_decode uses DSA-specific caches - skip for MLA for now

    # Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("OPTIMIZATION ROADMAP")
        print("=" * 60)
        print("1. [CRITICAL] CUDA Graphs for decode (10-50x speedup)")
        print("2. [HIGH] DeepEP low-latency dispatch for decode")
        print("3. [HIGH] Two-batch overlap (TBO) for prefill")
        print("4. [MEDIUM] Tune EP degree (EP32 prefill, EP72 decode)")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
