# SPDX-License-Identifier: Apache-2.0
"""Profile with CUDA events for accurate GPU timing."""

from __future__ import annotations

import math
import time

import torch


def profile_with_cuda_events():
    """Use CUDA events for accurate GPU-side timing."""
    from nmoe.serve.mla import _CompiledMlaKernel

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Config
    B = 2
    H = 128
    L, R = 512, 64
    page_size = 64
    num_pages = 4
    seq_len = page_size * num_pages
    softmax_scale = 1.0 / math.sqrt(L + R)

    # Create kernel
    print("Creating kernel...")
    kernel = _CompiledMlaKernel(
        num_heads=H, max_batch=B, max_seq_len=seq_len,
        page_size=page_size, device=device,
    )

    # Create CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    print("Warming up...")
    for _ in range(50):
        kernel.run_inplace(B, num_pages, softmax_scale)
    torch.cuda.synchronize()

    N = 100
    print(f"\nProfiling {N} iterations with CUDA events...\n")

    # Profile CuTeDSL with CUDA events
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(N):
        kernel._compiled(
            kernel._q_latent_ct, kernel._q_rope_ct,
            kernel._c_latent_ct, kernel._c_rope_ct,
            kernel._page_table_ct,
            kernel._o_ct, kernel._lse_ct,
            kernel._workspace_ct,
            kernel.split_kv,
            kernel._cache_seqs_ct,
            None,
            softmax_scale,
            1.0,
            kernel._stream,
        )
    end_event.record()
    torch.cuda.synchronize()
    cute_gpu_time = start_event.elapsed_time(end_event) / N
    print(f"CuTeDSL GPU time (CUDA events):  {cute_gpu_time:.4f} ms")

    # Profile CuTeDSL with wall clock for comparison
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        kernel._compiled(
            kernel._q_latent_ct, kernel._q_rope_ct,
            kernel._c_latent_ct, kernel._c_rope_ct,
            kernel._page_table_ct,
            kernel._o_ct, kernel._lse_ct,
            kernel._workspace_ct,
            kernel.split_kv,
            kernel._cache_seqs_ct,
            None,
            softmax_scale,
            1.0,
            kernel._stream,
        )
    torch.cuda.synchronize()
    cute_wall_time = (time.perf_counter() - t0) / N * 1000
    print(f"CuTeDSL wall time:               {cute_wall_time:.4f} ms")

    # Profile FlashMLA
    print("\n--- FlashMLA ---")
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    H_kv = 1
    D_qk = L + R
    D_v = L
    topk = seq_len
    S_q = 1

    q = torch.randn(B, S_q, H, D_qk, dtype=torch.bfloat16, device=device) / 10
    kv_cache = torch.zeros(num_pages, page_size, H_kv, 656, dtype=torch.uint8, device=device)
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1).contiguous()
    cache_seqlens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    indices = torch.arange(seq_len, dtype=torch.int32, device=device).view(1, 1, seq_len).expand(B, S_q, seq_len).contiguous()

    metadata, num_splits = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=S_q * H // H_kv,
        num_heads_k=H_kv,
        num_heads_q=H,
        is_fp8_kvcache=True,
        topk=topk,
    )
    scale = 1.0 / math.sqrt(D_qk)

    # Warmup FlashMLA
    for _ in range(50):
        flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, D_v,
                               metadata, num_splits, softmax_scale=scale,
                               causal=False, is_fp8_kvcache=True, indices=indices)
    torch.cuda.synchronize()

    # FlashMLA with CUDA events
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(N):
        flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, D_v,
                               metadata, num_splits, softmax_scale=scale,
                               causal=False, is_fp8_kvcache=True, indices=indices)
    end_event.record()
    torch.cuda.synchronize()
    flash_gpu_time = start_event.elapsed_time(end_event) / N
    print(f"FlashMLA GPU time (CUDA events): {flash_gpu_time:.4f} ms")

    # FlashMLA wall time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, D_v,
                               metadata, num_splits, softmax_scale=scale,
                               causal=False, is_fp8_kvcache=True, indices=indices)
    torch.cuda.synchronize()
    flash_wall_time = (time.perf_counter() - t0) / N * 1000
    print(f"FlashMLA wall time:              {flash_wall_time:.4f} ms")

    # Analysis
    print(f"\n{'='*55}")
    print("Analysis:")
    print(f"  CuTeDSL GPU:   {cute_gpu_time:.4f} ms")
    print(f"  CuTeDSL wall:  {cute_wall_time:.4f} ms")
    print(f"  CuTeDSL CPU overhead: {cute_wall_time - cute_gpu_time:.4f} ms")
    print(f"\n  FlashMLA GPU:  {flash_gpu_time:.4f} ms")
    print(f"  FlashMLA wall: {flash_wall_time:.4f} ms")
    print(f"  FlashMLA CPU overhead: {flash_wall_time - flash_gpu_time:.4f} ms")
    print(f"\n  GPU kernel gap: {cute_gpu_time - flash_gpu_time:.4f} ms")
    print(f"  CPU overhead gap: {(cute_wall_time - cute_gpu_time) - (flash_wall_time - flash_gpu_time):.4f} ms")


if __name__ == "__main__":
    profile_with_cuda_events()
