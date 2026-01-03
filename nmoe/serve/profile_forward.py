# SPDX-License-Identifier: Apache-2.0
"""Profile forward pass to identify bottlenecks."""
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


def profile_components():
    """Profile individual components of forward pass."""
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
        print("Forward Pass Profiling")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # Full model config
    cfg = ModelConfig(num_layers=61, num_dense_layers=3)

    # Buffer for DeepEP
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

    # Load checkpoint
    ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
    if rank == 0:
        print(f"\nLoading checkpoint from {ckpt_path}...")
    load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
    dist.barrier()

    if rank == 0:
        print("Model loaded.\n")
        # Check weight dtypes
        for name, param in model.named_parameters():
            if "norm" in name.lower():
                print(f"  {name}: {param.dtype}")
                break

    # Simple test case
    B, S = 4, 512
    page_size = 64
    num_blocks = (S + page_size - 1) // page_size + 1
    total_blocks = B * num_blocks

    input_ids = torch.randint(0, 10000, (B, S), device=device)
    positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # DSA caches
    kv_caches = [
        torch.zeros(total_blocks, page_size, 1, 656, dtype=torch.uint8, device=device)
        for _ in range(cfg.num_layers)
    ]
    idx_k_caches = [
        torch.zeros(total_blocks, page_size, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
        for _ in range(cfg.num_layers)
    ]

    block_table = torch.zeros(B, num_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.arange(b * num_blocks, (b + 1) * num_blocks)
    cache_seqlens = torch.full((B,), S, dtype=torch.int32, device=device)
    cache_seqlens_cpu = [S] * B
    out_loc = torch.zeros(B, S, dtype=torch.int32, device=device)
    for b in range(B):
        base = b * num_blocks * page_size
        out_loc[b] = torch.arange(base, base + S)

    # Warmup
    if rank == 0:
        print("Warmup...")
    with torch.inference_mode():
        _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                  block_table=block_table, cache_seqlens=cache_seqlens,
                  cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc)
    torch.cuda.synchronize()

    # Profile with CUDA events
    if rank == 0:
        print("\nProfiling with CUDA events...\n")

    num_iters = 3
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    dist.barrier()

    start_event.record()
    with torch.inference_mode():
        for _ in range(num_iters):
            _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                      block_table=block_table, cache_seqlens=cache_seqlens,
                      cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    tokens = B * S
    tok_per_sec = tokens / (elapsed_ms / 1000)

    if rank == 0:
        print(f"=== Results ===")
        print(f"Config: B={B}, S={S}, total_tokens={tokens}")
        print(f"Forward time: {elapsed_ms:.1f} ms")
        print(f"Throughput: {tok_per_sec:,.0f} tok/s")
        print(f"Target: 52,300 tok/s")
        print(f"Gap: {52300 / tok_per_sec:.1f}x slower")

    # Profile with torch profiler
    if rank == 0:
        print("\n\nRunning torch profiler...")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            _ = model(input_ids, positions, kv_caches=kv_caches, idx_k_caches=idx_k_caches,
                      block_table=block_table, cache_seqlens=cache_seqlens,
                      cache_seqlens_cpu=cache_seqlens_cpu, out_loc=out_loc)
        torch.cuda.synchronize()

    if rank == 0:
        print("\nTop 20 CUDA operations by time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    profile_components()
