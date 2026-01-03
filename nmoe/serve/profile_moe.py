# SPDX-License-Identifier: Apache-2.0
"""Profile MoE dispatch/combine to identify bottlenecks."""
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
import torch.nn.functional as F


def profile_moe():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, MoEGate, init_distributed
    from deep_ep import Buffer
    from deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    init_distributed(rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("MoE Profile (DeepEP dispatch/combine)")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    cfg = ModelConfig()
    hidden = cfg.hidden_size  # 7168
    inter = cfg.moe_intermediate_size  # 2048
    num_experts = cfg.num_experts  # 256
    num_local = num_experts // world_size
    M_ALIGN = 128

    # Create buffer
    hidden_bytes = hidden * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )
    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)

    # Create gate and expert weights
    gate = MoEGate(cfg).to(device)
    w13 = torch.randn(num_local, 2 * inter, hidden, dtype=torch.float8_e4m3fn, device=device)
    w13_scale = torch.ones(num_local, (2 * inter) // 128, hidden // 128, dtype=torch.float32, device=device)
    w2 = torch.randn(num_local, hidden, inter, dtype=torch.float8_e4m3fn, device=device)
    w2_scale = torch.ones(num_local, hidden // 128, inter // 128, dtype=torch.float32, device=device)

    def quantize_act(x):
        T, K = x.shape
        x_view = x.view(T, K // 128, 128)
        scales = x_view.float().abs().amax(dim=-1).clamp(min=1e-4) / 448.0
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
        x_q = (x_view.float() / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return x_q.view(T, K), scales

    # Test configs
    configs = [
        (4, 512),    # 2048 tokens
        (4, 2048),   # 8192 tokens
        (4, 4096),   # 16384 tokens (LMSYS config)
    ]

    num_warmup = 2
    num_iters = 5

    for batch_size, seq_len in configs:
        total_tokens = batch_size * seq_len

        x = torch.randn(total_tokens, hidden, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(num_warmup):
            weights, indices = gate(x)
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                indices, num_experts
            )
            recv_x, recv_topk_idx, recv_topk_weights, counts_list, handle, _ = buffer.dispatch(
                x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert, topk_idx=indices, topk_weights=weights,
                expert_alignment=M_ALIGN,
            )
            if recv_x.shape[0] > 0:
                recv_x_q, recv_x_scale = quantize_act(recv_x)
                gateup_out = torch.empty(recv_x.shape[0], 2 * inter, device=device, dtype=torch.bfloat16)
                m_grouped_fp8_gemm_nt_contiguous((recv_x_q, recv_x_scale), (w13, w13_scale), gateup_out,
                    torch.zeros(recv_x.shape[0], device=device, dtype=torch.int32))
            y, _, _ = buffer.combine(recv_x if recv_x.shape[0] == 0 else gateup_out[:, :hidden], handle, topk_weights=recv_topk_weights)
        torch.cuda.synchronize()

        # Timed breakdown
        timings = {"gate": 0, "layout": 0, "dispatch": 0, "quantize": 0, "gemm1": 0, "gemm2": 0, "combine": 0}

        for _ in range(num_iters):
            # Gate
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            weights, indices = gate(x)
            torch.cuda.synchronize()
            timings["gate"] += time.perf_counter() - t0

            # Layout
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                indices, num_experts
            )
            torch.cuda.synchronize()
            timings["layout"] += time.perf_counter() - t0

            # Dispatch
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            recv_x, recv_topk_idx, recv_topk_weights, counts_list, handle, _ = buffer.dispatch(
                x, num_tokens_per_rank=num_tokens_per_rank, is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert, topk_idx=indices, topk_weights=weights,
                expert_alignment=M_ALIGN,
            )
            torch.cuda.synchronize()
            timings["dispatch"] += time.perf_counter() - t0

            num_recv = recv_x.shape[0]
            if num_recv > 0:
                # Build m_indices
                m_indices = torch.empty(num_recv, device=device, dtype=torch.int32)
                offset = 0
                for e, cnt in enumerate(counts_list):
                    m_indices[offset:offset + cnt] = e
                    offset += cnt

                # Quantize
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                recv_x_q, recv_x_scale = quantize_act(recv_x)
                torch.cuda.synchronize()
                timings["quantize"] += time.perf_counter() - t0

                # GEMM 1 (gate-up)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gateup_out = torch.empty(num_recv, 2 * inter, device=device, dtype=torch.bfloat16)
                m_grouped_fp8_gemm_nt_contiguous((recv_x_q, recv_x_scale), (w13, w13_scale), gateup_out, m_indices)
                torch.cuda.synchronize()
                timings["gemm1"] += time.perf_counter() - t0

                gate_out, up = gateup_out.chunk(2, dim=-1)
                down_in = (F.silu(gate_out.float()) * up.float()).to(torch.bfloat16)
                down_in_q, down_in_scale = quantize_act(down_in)

                # GEMM 2 (down)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                expert_out = torch.empty(num_recv, hidden, device=device, dtype=torch.bfloat16)
                m_grouped_fp8_gemm_nt_contiguous((down_in_q, down_in_scale), (w2, w2_scale), expert_out, m_indices)
                torch.cuda.synchronize()
                timings["gemm2"] += time.perf_counter() - t0
            else:
                expert_out = recv_x

            # Combine
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y, _, _ = buffer.combine(expert_out, handle, topk_weights=recv_topk_weights)
            torch.cuda.synchronize()
            timings["combine"] += time.perf_counter() - t0

        if rank == 0:
            total_ms = sum(timings.values()) / num_iters * 1000
            print(f"\nConfig: {total_tokens} tokens (B={batch_size}, S={seq_len})")
            print(f"  Total: {total_ms:.1f} ms")
            for name, t in timings.items():
                ms = t / num_iters * 1000
                pct = ms / total_ms * 100 if total_ms > 0 else 0
                print(f"    {name:10s}: {ms:6.2f} ms ({pct:5.1f}%)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    profile_moe()
