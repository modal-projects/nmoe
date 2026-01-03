# SPDX-License-Identifier: Apache-2.0
"""Benchmark CuTeDSL MLA kernel vs FlashMLA.

Compares dense attention performance between:
- CuTeDSL BlackwellMultiHeadLatentAttentionForward (SM100/Blackwell native)
- FlashMLA with FP8 KV cache (production kernel)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


@dataclass
class BenchmarkConfig:
    """Configuration for MLA benchmark."""
    batch_size: int
    num_heads: int
    seq_len: int
    latent_dim: int = 512
    rope_dim: int = 64
    page_size: int = 64
    warmup_iters: int = 10
    bench_iters: int = 100


def benchmark_cutedsl_mla(config: BenchmarkConfig, device: torch.device, use_zero_copy: bool = True) -> dict:
    """Benchmark CuTeDSL MLA kernel."""
    from nmoe.serve.mla import _CompiledMlaKernel

    B = config.batch_size
    H = config.num_heads
    L = config.latent_dim
    R = config.rope_dim
    page_size = config.page_size
    num_pages = (config.seq_len + page_size - 1) // page_size
    seq_len = num_pages * page_size

    # Helper to create tensors with stride[leading_dim]=1
    def make_tensor(shape: tuple, dtype: torch.dtype, leading_dim: int) -> torch.Tensor:
        perm = list(range(len(shape)))
        perm.remove(leading_dim)
        perm.append(leading_dim)
        inv_perm = [perm.index(i) for i in range(len(shape))]
        reordered = tuple(shape[p] for p in perm)
        t = torch.randn(reordered, dtype=dtype, device=device) * 0.1
        return t.permute(*inv_perm)

    # Create kernel (compiles during init)
    compile_start = time.perf_counter()
    kernel = _CompiledMlaKernel(
        num_heads=H,
        max_batch=B,
        max_seq_len=seq_len,
        page_size=page_size,
        device=device,
    )
    compile_time = time.perf_counter() - compile_start

    softmax_scale = 1.0 / math.sqrt(L + R)

    if use_zero_copy:
        # Zero-copy path: pre-populate kernel's backing tensors once
        q_latent = make_tensor((H, L, B), torch.float16, 1)
        q_rope = make_tensor((H, R, B), torch.float16, 1)
        c_latent = make_tensor((num_pages, page_size, L), torch.float16, 1)
        c_rope = make_tensor((num_pages, page_size, R), torch.float16, 1)
        page_table = torch.empty(B, num_pages, dtype=torch.int32, device=device)
        for i in range(num_pages):
            page_table[:, i] = i
        page_table = page_table.permute(1, 0)
        cache_seqs = torch.full((B,), seq_len, dtype=torch.int32, device=device)

        # Pre-populate backing tensors
        kernel.q_latent_buffer[:, :, :B].copy_(q_latent)
        kernel.q_rope_buffer[:, :, :B].copy_(q_rope)
        kernel.c_latent_buffer[:num_pages].copy_(c_latent)
        kernel.c_rope_buffer[:num_pages].copy_(c_rope)
        kernel.page_table_buffer[:num_pages, :B].copy_(page_table)
        kernel.cache_seqs_buffer[:B].copy_(cache_seqs)

        # Warmup
        for _ in range(config.warmup_iters):
            out, lse = kernel.run_inplace(B, num_pages, softmax_scale)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(config.bench_iters):
            out, lse = kernel.run_inplace(B, num_pages, softmax_scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    else:
        # Standard path with copies
        q_latent = make_tensor((H, L, B), torch.float16, 1)
        q_rope = make_tensor((H, R, B), torch.float16, 1)
        c_latent = make_tensor((num_pages, page_size, L), torch.float16, 1)
        c_rope = make_tensor((num_pages, page_size, R), torch.float16, 1)
        page_table = torch.empty(B, num_pages, dtype=torch.int32, device=device)
        for i in range(num_pages):
            page_table[:, i] = i
        page_table = page_table.permute(1, 0)
        cache_seqs = torch.full((B,), seq_len, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(config.warmup_iters):
            out, lse = kernel(q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, softmax_scale)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(config.bench_iters):
            out, lse = kernel(q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, softmax_scale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / config.bench_iters) * 1000

    # Memory stats
    torch.cuda.reset_peak_memory_stats()
    out, lse = kernel.run_inplace(B, num_pages, softmax_scale)
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "kernel": "CuTeDSL" + (" (zero-copy)" if use_zero_copy else ""),
        "compile_time_s": compile_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_iter_per_s": config.bench_iters / elapsed,
        "peak_mem_mb": peak_mem_mb,
        "output_shape": tuple(out.shape),
    }


def benchmark_flashmla(config: BenchmarkConfig, device: torch.device) -> dict:
    """Benchmark FlashMLA kernel (dense mode via full indices)."""
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    B = config.batch_size
    H_q = config.num_heads
    H_kv = 1  # MQA mode
    L = config.latent_dim
    R = config.rope_dim
    D_qk = L + R  # 576
    D_v = L  # 512
    page_size = config.page_size
    num_pages = (config.seq_len + page_size - 1) // page_size
    seq_len = num_pages * page_size
    topk = seq_len  # Dense attention: attend to all tokens

    # Query: [B, S_q, H_q, D_qk]
    S_q = 1  # Single decode token
    q = torch.randn(B, S_q, H_q, D_qk, dtype=torch.bfloat16, device=device) / 10

    # KV cache: [num_blocks, block_size, H_kv, 656] uint8 (FP8 packed)
    kv_cache = torch.zeros(num_pages, page_size, H_kv, 656, dtype=torch.uint8, device=device)

    # Fill KV cache with valid FP8 data
    for page in range(num_pages):
        for tok in range(page_size):
            latent = torch.randn(L, dtype=torch.bfloat16, device=device) / 10
            rope = torch.randn(R, dtype=torch.bfloat16, device=device) / 10

            packed = torch.empty(656, dtype=torch.uint8, device=device)
            latent_f = latent.float()
            for tile in range(4):
                lo, hi = tile * 128, (tile + 1) * 128
                tile_f = latent_f[lo:hi]
                sf = tile_f.abs().max().clamp(min=1e-8) / 448.0
                tile_q = (tile_f / sf).to(torch.float8_e4m3fn)
                packed[lo:hi] = tile_q.view(torch.uint8)
                sf_tensor = sf.to(torch.float32).reshape(1)
                sf_bytes = sf_tensor.view(torch.uint8)
                packed[512 + tile * 4:512 + (tile + 1) * 4] = sf_bytes
            packed[528:] = rope.view(torch.uint8)
            kv_cache[page, tok, 0] = packed

    # Block table and cache lengths
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1).contiguous()
    cache_seqlens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Get metadata for dense mode
    compile_start = time.perf_counter()
    metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=S_q * H_q // H_kv,
        num_heads_k=H_kv,
        num_heads_q=H_q,
        is_fp8_kvcache=True,
        topk=topk,
    )
    compile_time = time.perf_counter() - compile_start

    # Dense indices: attend to all tokens
    indices = torch.arange(seq_len, dtype=torch.int32, device=device).view(1, 1, seq_len).expand(B, S_q, seq_len).contiguous()

    softmax_scale = 1.0 / math.sqrt(D_qk)

    # Warmup
    for _ in range(config.warmup_iters):
        out, lse = flash_mla_with_kvcache(
            q, kv_cache, block_table, cache_seqlens, D_v,
            metadata, num_splits,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(config.bench_iters):
        out, lse = flash_mla_with_kvcache(
            q, kv_cache, block_table, cache_seqlens, D_v,
            metadata, num_splits,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / config.bench_iters) * 1000

    # Memory stats
    torch.cuda.reset_peak_memory_stats()
    out, lse = flash_mla_with_kvcache(
        q, kv_cache, block_table, cache_seqlens, D_v,
        metadata, num_splits,
        softmax_scale=softmax_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "kernel": "FlashMLA",
        "compile_time_s": compile_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_iter_per_s": config.bench_iters / elapsed,
        "peak_mem_mb": peak_mem_mb,
        "output_shape": tuple(out.shape),
    }


def print_results(results: list[dict], config: BenchmarkConfig) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"MLA Kernel Benchmark Results")
    print(f"Config: B={config.batch_size}, H={config.num_heads}, seq_len={config.seq_len}")
    print(f"        latent={config.latent_dim}, rope={config.rope_dim}, page_size={config.page_size}")
    print(f"        warmup={config.warmup_iters}, iters={config.bench_iters}")
    print("=" * 70)

    print(f"\n{'Kernel':<12} {'Compile(s)':<12} {'Latency(ms)':<14} {'Throughput':<14} {'Peak Mem(MB)':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['kernel']:<12} {r['compile_time_s']:<12.3f} {r['avg_latency_ms']:<14.3f} {r['throughput_iter_per_s']:<14.1f} {r['peak_mem_mb']:<12.1f}")

    # Speedup comparison
    if len(results) == 2:
        cutedsl = next((r for r in results if r['kernel'] == 'CuTeDSL'), None)
        flashmla = next((r for r in results if r['kernel'] == 'FlashMLA'), None)
        if cutedsl and flashmla:
            speedup = flashmla['avg_latency_ms'] / cutedsl['avg_latency_ms']
            print(f"\nCuTeDSL speedup vs FlashMLA: {speedup:.2f}x")

    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MLA kernels")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--heads", type=int, default=128, help="Number of heads")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--cutedsl-only", action="store_true", help="Only benchmark CuTeDSL")
    parser.add_argument("--flashmla-only", action="store_true", help="Only benchmark FlashMLA")
    args = parser.parse_args()

    if not _is_sm100():
        print("ERROR: Requires SM100 (B200/Blackwell)")
        return 1

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = BenchmarkConfig(
        batch_size=args.batch,
        num_heads=args.heads,
        seq_len=args.seq_len,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    results = []

    if not args.flashmla_only:
        print(f"Benchmarking CuTeDSL MLA kernel...")
        try:
            results.append(benchmark_cutedsl_mla(config, device))
        except Exception as e:
            print(f"CuTeDSL benchmark failed: {e}")

    if not args.cutedsl_only:
        print(f"Benchmarking FlashMLA kernel...")
        try:
            results.append(benchmark_flashmla(config, device))
        except Exception as e:
            print(f"FlashMLA benchmark failed: {e}")

    if results:
        print_results(results, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
