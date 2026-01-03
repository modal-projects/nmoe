# SPDX-License-Identifier: Apache-2.0
"""Test persistent vs non-persistent at different scales."""

from __future__ import annotations

import math
import torch


def test_config(is_persistent: bool, B: int, seq_len: int, device: torch.device) -> float:
    """Test a specific configuration."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as ctorch
    import cuda.bindings.driver as cuda_driver
    from nmoe.serve.mla import BlackwellMultiHeadLatentAttentionForward

    H, L, R = 128, 512, 64
    page_size = 64
    num_pages = (seq_len + page_size - 1) // page_size
    seq_len = num_pages * page_size
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

    # Get max_active_clusters
    from nmoe.serve.mla import _CompiledMlaKernel
    temp = _CompiledMlaKernel(H, B, seq_len, page_size, device)
    max_active_clusters = temp.max_active_clusters
    del temp

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
    for _ in range(10):
        compiled(q_latent_ct, q_rope_ct, c_latent_ct, c_rope_ct, page_table_ct,
                 o_ct, lse_ct, workspace_ct, split_kv, cache_seqs_ct,
                 None, softmax_scale, 1.0, stream)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N = 50

    start.record()
    for _ in range(N):
        compiled(q_latent_ct, q_rope_ct, c_latent_ct, c_rope_ct, page_table_ct,
                 o_ct, lse_ct, workspace_ct, split_kv, cache_seqs_ct,
                 None, softmax_scale, 1.0, stream)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / N


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    configs = [
        (2, 256),      # Small
        (8, 4096),     # Medium
        (16, 8192),    # Large
        (32, 16384),   # Very large
    ]

    print("Persistent vs Non-Persistent Scaling\n")
    print(f"{'B':>4} {'Seq':>8} | {'Persistent':>12} {'Non-Pers':>12} | {'Diff':>8} {'Winner':>12}")
    print("-" * 70)

    for B, seq_len in configs:
        torch.cuda.empty_cache()
        try:
            pers = test_config(True, B, seq_len, device)
        except Exception as e:
            pers = float('inf')
            print(f"  Persistent failed at B={B}, seq={seq_len}: {e}")

        torch.cuda.empty_cache()
        try:
            non_pers = test_config(False, B, seq_len, device)
        except Exception as e:
            non_pers = float('inf')
            print(f"  Non-persistent failed at B={B}, seq={seq_len}: {e}")

        if pers != float('inf') and non_pers != float('inf'):
            diff = pers - non_pers
            winner = "Non-Pers" if diff > 0 else "Persistent"
            print(f"{B:>4} {seq_len:>8} | {pers:>12.4f} {non_pers:>12.4f} | {diff:>+8.4f} {winner:>12}")


if __name__ == "__main__":
    main()
