# SPDX-License-Identifier: Apache-2.0
"""Profile overhead in CuTeDSL MLA kernel wrapper."""

from __future__ import annotations

import math
import time

import torch
from cuda import cuda


def profile_overhead():
    """Profile each component of the MLA kernel call."""
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

    # Helper to create tensors with stride[leading_dim]=1
    def make_tensor(shape: tuple, dtype: torch.dtype, leading_dim: int) -> torch.Tensor:
        perm = list(range(len(shape)))
        perm.remove(leading_dim)
        perm.append(leading_dim)
        inv_perm = [perm.index(i) for i in range(len(shape))]
        reordered = tuple(shape[p] for p in perm)
        t = torch.randn(reordered, dtype=dtype, device=device) * 0.1
        return t.permute(*inv_perm)

    # Create kernel
    print("Creating kernel (includes compile)...")
    kernel = _CompiledMlaKernel(
        num_heads=H, max_batch=B, max_seq_len=seq_len,
        page_size=page_size, device=device,
    )

    # Create inputs
    q_latent = make_tensor((H, L, B), torch.float16, 1)
    q_rope = make_tensor((H, R, B), torch.float16, 1)
    c_latent = make_tensor((num_pages, page_size, L), torch.float16, 1)
    c_rope = make_tensor((num_pages, page_size, R), torch.float16, 1)
    page_table = torch.empty(B, num_pages, dtype=torch.int32, device=device)
    for i in range(num_pages):
        page_table[:, i] = i
    page_table = page_table.permute(1, 0)
    cache_seqs = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    softmax_scale = 1.0 / math.sqrt(L + R)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        kernel(q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, softmax_scale)
    torch.cuda.synchronize()

    N = 100
    print(f"\nProfiling {N} iterations each...\n")

    # Profile full call (with clone)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out, lse = kernel(q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, softmax_scale, copy_out=True)
    torch.cuda.synchronize()
    full_clone_time = (time.perf_counter() - t0) / N * 1000
    print(f"__call__ (copy_out=True):  {full_clone_time:.3f} ms")

    # Profile full call (no clone)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out, lse = kernel(q_latent, q_rope, c_latent, c_rope, page_table, cache_seqs, softmax_scale, copy_out=False)
    torch.cuda.synchronize()
    full_view_time = (time.perf_counter() - t0) / N * 1000
    print(f"__call__ (copy_out=False): {full_view_time:.3f} ms")

    # Profile run_inplace (zero-copy)
    # Pre-populate backing tensors
    kernel.q_latent_buffer[:, :, :B].copy_(q_latent)
    kernel.q_rope_buffer[:, :, :B].copy_(q_rope)
    kernel.c_latent_buffer[:num_pages].copy_(c_latent)
    kernel.c_rope_buffer[:num_pages].copy_(c_rope)
    kernel.page_table_buffer[:num_pages, :B].copy_(page_table)
    kernel.cache_seqs_buffer[:B].copy_(cache_seqs)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        out, lse = kernel.run_inplace(B, num_pages, softmax_scale)
    torch.cuda.synchronize()
    inplace_time = (time.perf_counter() - t0) / N * 1000
    print(f"run_inplace (zero-copy):   {inplace_time:.3f} ms")

    # Profile copy_ operations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        kernel._q_latent_data[:, :, :B].copy_(q_latent)
        kernel._q_rope_data[:, :, :B].copy_(q_rope)
        kernel._c_latent_data[:num_pages].copy_(c_latent)
        kernel._c_rope_data[:num_pages].copy_(c_rope)
        kernel._page_table_data[:num_pages, :B].copy_(page_table)
        kernel._cache_seqs_data[:B].copy_(cache_seqs)
    torch.cuda.synchronize()
    copy_time = (time.perf_counter() - t0) / N * 1000
    print(f"\nInput copy_ (6x):          {copy_time:.3f} ms")

    # Profile clone operations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        o_clone = kernel._o_data[:, :, :B].clone()
        lse_clone = kernel._lse_data[:, :B].clone()
    torch.cuda.synchronize()
    clone_time = (time.perf_counter() - t0) / N * 1000
    print(f"Output clone (2x):         {clone_time:.3f} ms")

    # Calculate improvements
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Original (__call__ copy_out=True): {full_clone_time:.3f} ms")
    print(f"  No-clone (__call__ copy_out=False): {full_view_time:.3f} ms  ({(full_clone_time-full_view_time)/full_clone_time*100:.1f}% faster)")
    print(f"  Zero-copy (run_inplace):            {inplace_time:.3f} ms  ({(full_clone_time-inplace_time)/full_clone_time*100:.1f}% faster)")
    print(f"\n  Kernel-only (theoretical min):      {inplace_time:.3f} ms")
    print(f"  Overhead eliminated:                {full_clone_time - inplace_time:.3f} ms")


if __name__ == "__main__":
    profile_overhead()
