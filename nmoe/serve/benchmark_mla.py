# SPDX-License-Identifier: Apache-2.0
"""Benchmark MLA (V3-0324) targeting LMSYS numbers.

Target: 52.3k tok/s prefill per 8x H100 node
"""
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


def benchmark_mla_prefill(model, cfg, device, rank, world_size):
    """Benchmark MLA prefill."""
    if rank == 0:
        print("\n" + "=" * 60)
        print("MLA PREFILL BENCHMARK")
        print("=" * 60)

    configs = [
        (1, 512),
        (4, 512),
        (4, 2048),
        (4, 4096),   # LMSYS config: 16k tokens total
        (8, 2048),
        (16, 1024),
    ]

    num_iterations = 5
    page_size = 64

    for batch_size, seq_len in configs:
        total_tokens = batch_size * seq_len
        num_blocks_per_seq = (seq_len + page_size - 1) // page_size + 1
        total_blocks = batch_size * num_blocks_per_seq

        input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # MLA caches
        kv_caches_latent = [
            torch.zeros(total_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(cfg.num_layers)
        ]
        kv_caches_rope = [
            torch.zeros(total_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(cfg.num_layers)
        ]

        block_table = torch.zeros(batch_size, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch_size):
            block_table[b] = torch.arange(b * num_blocks_per_seq, (b + 1) * num_blocks_per_seq)

        cache_seqlens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        out_loc = torch.zeros(batch_size, seq_len, dtype=torch.int32, device=device)
        for b in range(batch_size):
            base = b * num_blocks_per_seq * page_size
            out_loc[b] = torch.arange(base, base + seq_len)

        fwd_kwargs = dict(
            kv_caches_latent=kv_caches_latent,
            kv_caches_rope=kv_caches_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            out_loc=out_loc,
            prefill_mode="dense",
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


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
    from nmoe.serve.ckpt import load_sharded_checkpoint
    from deep_ep import Buffer

    init_distributed(rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("MLA Benchmark (DeepSeek-V3-0324)")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # MLA config
    cfg = ModelConfig(
        num_layers=61,
        num_dense_layers=3,
        attention_type="mla",
    )

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

    ckpt_path = "/data/models/DeepSeek-V3-0324-mp8"
    if rank == 0:
        print(f"\nLoading checkpoint from {ckpt_path}...")
    load_sharded_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)
    dist.barrier()

    if rank == 0:
        print("Model loaded.\n")

    benchmark_mla_prefill(model, cfg, device, rank, world_size)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. CUDA graphs for decode")
        print("  2. Two-batch overlap (TBO)")
        print("  3. Tune EP degree")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
