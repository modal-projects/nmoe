# SPDX-License-Identifier: Apache-2.0
"""Detailed profiling of CuTeDSL MLA kernel overhead."""

from __future__ import annotations

import math
import time

import torch


def profile_detailed():
    """Profile every micro-operation in the kernel call."""
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

    # Warmup
    print("Warming up...")
    for _ in range(20):
        kernel.run_inplace(B, num_pages, softmax_scale)
    torch.cuda.synchronize()

    N = 1000
    print(f"\nProfiling {N} iterations each...\n")

    # 1. Full run_inplace
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out, lse = kernel.run_inplace(B, num_pages, softmax_scale)
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - t0) / N * 1000
    print(f"run_inplace (full):           {full_time:.4f} ms")

    # 2. Just the compiled kernel call (no slicing)
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
    kernel_only = (time.perf_counter() - t0) / N * 1000
    print(f"_compiled() call only:        {kernel_only:.4f} ms")

    # 3. Output slicing overhead
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        o_view = kernel._o_data[:, :, :B]
        lse_view = kernel._lse_data[:, :B]
    torch.cuda.synchronize()
    slice_time = (time.perf_counter() - t0) / N * 1000
    print(f"Output slicing only:          {slice_time:.4f} ms")

    # 4. Empty Python function call overhead
    def empty_func():
        pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        empty_func()
    torch.cuda.synchronize()
    pyfunc_time = (time.perf_counter() - t0) / N * 1000
    print(f"Empty Python func call:       {pyfunc_time:.4f} ms")

    # 5. Attribute access overhead
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        _ = kernel._compiled
        _ = kernel._q_latent_ct
        _ = kernel._q_rope_ct
        _ = kernel._c_latent_ct
        _ = kernel._c_rope_ct
        _ = kernel._page_table_ct
        _ = kernel._o_ct
        _ = kernel._lse_ct
        _ = kernel._workspace_ct
        _ = kernel.split_kv
        _ = kernel._cache_seqs_ct
        _ = kernel._stream
    torch.cuda.synchronize()
    attr_time = (time.perf_counter() - t0) / N * 1000
    print(f"Attribute access (12x):       {attr_time:.4f} ms")

    # 6. Just cuda synchronize overhead
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        pass
    torch.cuda.synchronize()
    loop_time = (time.perf_counter() - t0) / N * 1000
    print(f"Empty loop iteration:         {loop_time:.4f} ms")

    # 7. Compare with FlashMLA
    print("\n--- FlashMLA comparison ---")
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    # Setup FlashMLA
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
    for _ in range(20):
        flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, D_v,
                               metadata, num_splits, softmax_scale=scale,
                               causal=False, is_fp8_kvcache=True, indices=indices)
    torch.cuda.synchronize()

    # Benchmark FlashMLA
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, D_v,
                               metadata, num_splits, softmax_scale=scale,
                               causal=False, is_fp8_kvcache=True, indices=indices)
    torch.cuda.synchronize()
    flash_time = (time.perf_counter() - t0) / N * 1000
    print(f"FlashMLA full call:           {flash_time:.4f} ms")

    # Summary
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"  CuTeDSL run_inplace:     {full_time:.4f} ms")
    print(f"  CuTeDSL kernel only:     {kernel_only:.4f} ms")
    print(f"  FlashMLA:                {flash_time:.4f} ms")
    print(f"\n  CuTeDSL overhead:        {full_time - kernel_only:.4f} ms")
    print(f"  CuTeDSL vs FlashMLA gap: {full_time - flash_time:.4f} ms")
    print(f"  Kernel-only gap:         {kernel_only - flash_time:.4f} ms")


if __name__ == "__main__":
    profile_detailed()
