# SPDX-License-Identifier: Apache-2.0
"""Test non-persistent kernel mode."""

from __future__ import annotations

import math
import time

import torch


def main():
    """Compare persistent vs non-persistent kernel modes."""
    # We need to modify the kernel class to test this, so let's do it inline
    import sys
    sys.path.insert(0, "/workspace/nmoe/nmoe/serve")  # For the MLA file

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as ctorch
    import cuda.bindings.driver as cuda_driver

    # Import from our local mla.py which includes BlackwellMultiHeadLatentAttentionForward
    from nmoe.serve.mla import BlackwellMultiHeadLatentAttentionForward

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    B, H, L, R = 2, 128, 512, 64
    page_size, num_pages = 64, 4
    seq_len = page_size * num_pages
    softmax_scale = 1.0 / math.sqrt(L + R)

    def make_tensor(shape, dtype, leading_dim):
        perm = list(range(len(shape)))
        perm.remove(leading_dim)
        perm.append(leading_dim)
        inv_perm = [perm.index(i) for i in range(len(shape))]
        reordered = tuple(shape[p] for p in perm)
        return torch.empty(reordered, dtype=dtype, device=device).permute(*inv_perm)

    def make_cute(backing, cutlass_dtype, leading_dim, skip_compact=False):
        ct, _ = ctorch.cute_tensor_like(backing, cutlass_dtype, is_dynamic_layout=True, assumed_align=16)
        if not skip_compact:
            strides = backing.stride()
            stride_order = tuple(sorted(range(len(strides)), key=lambda i: strides[i], reverse=True))
            ct.mark_compact_shape_dynamic(mode=leading_dim, stride_order=stride_order, divisibility=(128 // cutlass_dtype.width))
        return ct

    # Tensors
    q_latent = make_tensor((H, L, B), torch.float16, 1)
    q_rope = make_tensor((H, R, B), torch.float16, 1)
    c_latent = make_tensor((num_pages, page_size, L), torch.float16, 1)
    c_rope = make_tensor((num_pages, page_size, R), torch.float16, 1)
    page_table = make_tensor((num_pages, B), torch.int32, 0)
    o = make_tensor((H, L, B), torch.float16, 1)
    lse = make_tensor((H, B), torch.float32, 0)
    cache_seqs = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    q_latent_ct = make_cute(q_latent, cutlass.Float16, 1)
    q_rope_ct = make_cute(q_rope, cutlass.Float16, 1)
    c_latent_ct = make_cute(c_latent, cutlass.Float16, 1)
    c_rope_ct = make_cute(c_rope, cutlass.Float16, 1)
    page_table_ct = make_cute(page_table, cutlass.Int32, 0, skip_compact=True)
    o_ct = make_cute(o, cutlass.Float16, 1)
    lse_ct = make_cute(lse, cutlass.Float32, 0, skip_compact=True)
    cache_seqs_ct = from_dlpack(cache_seqs, assumed_align=16, use_32bit_stride=True).mark_layout_dynamic()

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    # Get hardware info (from our existing mla.py)
    from nmoe.serve.mla import _CompiledMlaKernel
    temp_kernel = _CompiledMlaKernel(H, B, seq_len, page_size, device)
    max_active_clusters = temp_kernel.max_active_clusters
    del temp_kernel
    torch.cuda.empty_cache()

    results = []

    for is_persistent in [True, False]:
        print(f"\nTesting is_persistent={is_persistent}...")

        mma_qk_tiler_mn = (128, 128)
        mma_pv_tiler_mn = (128, 256)
        split_kv = BlackwellMultiHeadLatentAttentionForward.get_split_kv(
            B, seq_len, mma_qk_tiler_mn, max_active_clusters * 2
        )
        ws_size = BlackwellMultiHeadLatentAttentionForward.get_workspace_size(H, L, B, split_kv, cutlass.Float32)
        workspace = torch.empty(ws_size, dtype=torch.int8, device=device) if ws_size > 0 else None
        workspace_ct = from_dlpack(workspace, assumed_align=16, use_32bit_stride=True) if workspace is not None else None

        kernel = BlackwellMultiHeadLatentAttentionForward(
            cutlass.Float32, cutlass.Float32,
            mma_qk_tiler_mn, mma_pv_tiler_mn,
            max_active_clusters,
            is_persistent=is_persistent,
            is_cpasync=True,
            use_page_table=True, is_var_seq=True, is_var_split_kv=False,
        )

        compiled = cute.compile(
            kernel, q_latent_ct, q_rope_ct, c_latent_ct, c_rope_ct, page_table_ct,
            o_ct, lse_ct, workspace_ct, split_kv, cache_seqs_ct,
            None, softmax_scale, 1.0, stream,
        )

        # Warmup
        for _ in range(20):
            compiled(q_latent_ct, q_rope_ct, c_latent_ct, c_rope_ct, page_table_ct,
                     o_ct, lse_ct, workspace_ct, split_kv, cache_seqs_ct,
                     None, softmax_scale, 1.0, stream)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        N = 100

        start.record()
        for _ in range(N):
            compiled(q_latent_ct, q_rope_ct, c_latent_ct, c_rope_ct, page_table_ct,
                     o_ct, lse_ct, workspace_ct, split_kv, cache_seqs_ct,
                     None, softmax_scale, 1.0, stream)
        end.record()
        torch.cuda.synchronize()

        latency = start.elapsed_time(end) / N
        results.append((is_persistent, latency))
        print(f"  Latency: {latency:.4f} ms")

        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("Summary:")
    for is_persistent, latency in results:
        print(f"  is_persistent={is_persistent}: {latency:.4f} ms")


if __name__ == "__main__":
    main()
