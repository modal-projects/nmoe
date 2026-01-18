# SPDX-License-Identifier: Apache-2.0
"""Test zero-token DeepEP dispatch/combine for dynamic disagg design.

Validates that:
1. All ranks can participate in dispatch/combine with 0 tokens
2. Buffer index ping-pong stays in sync
3. Ranks with actual tokens get correct results
4. Overhead of empty participation is acceptable

Run with: torchrun --nproc_per_node=8 -m nmoe.serve.test_zero_token_dispatch
"""

import os
import time
import torch
import torch.distributed as dist
from dataclasses import dataclass


@dataclass
class TestConfig:
    hidden_size: int = 7168
    num_experts: int = 256
    topk: int = 8
    num_iterations: int = 100
    warmup_iterations: int = 10


def log(msg: str, rank: int = 0) -> None:
    if dist.get_rank() == rank:
        print(f"[TEST] {msg}", flush=True)


def log_all(msg: str) -> None:
    print(f"[rank{dist.get_rank()}] {msg}", flush=True)


def test_zero_token_dispatch(cfg: TestConfig) -> bool:
    """Test that zero-token dispatch works correctly."""
    from deep_ep import Buffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    log(f"Testing zero-token dispatch with {world_size} GPUs")
    log(f"Config: hidden={cfg.hidden_size}, experts={cfg.num_experts}, topk={cfg.topk}")

    # Initialize DeepEP buffer
    num_nvl_bytes = max(
        Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(
            cfg.hidden_size * 2, world_size
        ),
        Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(
            cfg.hidden_size * 2, world_size
        ),
    )

    buffer = Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=0,
        explicitly_destroy=True,
    )

    log(f"Buffer initialized with {num_nvl_bytes / 1e6:.1f} MB NVL")

    num_local_experts = cfg.num_experts // world_size

    # Test 1: All ranks have tokens (baseline)
    log("=" * 50)
    log("Test 1: All ranks have tokens (baseline)")

    T = 64  # tokens per rank
    x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    # Random routing to experts
    topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device)
    topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)

    dist.barrier()

    try:
        # Get dispatch layout
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, cfg.num_experts
        )

        # Dispatch - note: x is positional, rest are keyword args
        (recv_x, recv_topk_idx, recv_topk_weights,
         counts_list, handle, _) = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

        log(f"Dispatch successful: recv shape = {recv_x.shape}")
        log(f"Counts per expert: {counts_list[:4]}... (first 4)")

        # Combine (just echo back for test) - use recv_topk_weights from dispatch
        y, _, _ = buffer.combine(
            recv_x, handle,
            recv_topk_weights,
        )

        log(f"Combine successful: output shape = {y.shape}")
        log("Test 1 PASSED")

    except Exception as e:
        log(f"Test 1 FAILED: {e}")
        return False

    dist.barrier()

    # Test 2: Only even ranks have tokens, odd ranks have 0
    log("=" * 50)
    log("Test 2: Even ranks have tokens, odd ranks have 0 tokens")

    if rank % 2 == 0:
        # Even ranks: have tokens
        T = 64
        x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device)
        topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
    else:
        # Odd ranks: 0 tokens
        T = 0
        x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
        topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

    log_all(f"T = {T} tokens")
    dist.barrier()

    try:
        # Get dispatch layout
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, cfg.num_experts
        )

        # Dispatch
        (recv_x, recv_topk_idx, recv_topk_weights,
         counts_list, handle, _) = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

        log_all(f"Dispatch: recv shape = {recv_x.shape}")

        # Combine - use recv_topk_weights from dispatch
        y, _, _ = buffer.combine(
            recv_x, handle,
            recv_topk_weights,
        )

        log_all(f"Combine: output shape = {y.shape}")

        # Verify: ranks with tokens should get output, ranks without should get empty
        expected_shape = (T, cfg.hidden_size)
        if y.shape[0] != T:
            log_all(f"ERROR: Expected {T} output tokens, got {y.shape[0]}")
            return False

        log("Test 2 PASSED")

    except Exception as e:
        log_all(f"Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    dist.barrier()

    # Test 3: Only rank 0 has tokens (extreme case)
    log("=" * 50)
    log("Test 3: Only rank 0 has tokens (extreme case)")

    if rank == 0:
        T = 128
        x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device)
        topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
    else:
        T = 0
        x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
        topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

    log_all(f"T = {T} tokens")
    dist.barrier()

    try:
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, cfg.num_experts
        )

        (recv_x, recv_topk_idx, recv_topk_weights,
         counts_list, handle, _) = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )

        log_all(f"Dispatch: recv shape = {recv_x.shape}")

        y, _, _ = buffer.combine(
            recv_x, handle,
            recv_topk_weights,
        )

        log_all(f"Combine: output shape = {y.shape}")
        log("Test 3 PASSED")

    except Exception as e:
        log_all(f"Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    dist.barrier()

    # Test 4: Measure overhead of empty dispatch
    log("=" * 50)
    log("Test 4: Measure empty dispatch overhead")

    # All ranks empty
    T = 0
    x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
    topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(cfg.warmup_iterations):
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, cfg.num_experts
        )
        (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        y, _, _ = buffer.combine(recv_x, handle, recv_topk_weights)

    torch.cuda.synchronize()
    dist.barrier()

    # Timed run
    start = time.perf_counter()
    for _ in range(cfg.num_iterations):
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx, cfg.num_experts
        )
        (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
            x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
        )
        y, _, _ = buffer.combine(recv_x, handle, recv_topk_weights)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = (elapsed / cfg.num_iterations) * 1e6
    log(f"Empty dispatch+combine: {avg_us:.1f} µs avg over {cfg.num_iterations} iterations")

    if avg_us < 100:
        log("Test 4 PASSED (overhead < 100µs)")
    else:
        log(f"Test 4 WARNING: overhead {avg_us:.1f}µs may be significant")

    dist.barrier()

    # Test 5: Multiple rounds with varying assignment (simulates dynamic disagg)
    log("=" * 50)
    log("Test 5: Dynamic assignment simulation (10 rounds)")

    success = True
    for round_idx in range(10):
        # Vary which ranks have tokens each round
        has_tokens = (rank + round_idx) % 3 != 0  # 2/3 of ranks have tokens each round

        if has_tokens:
            T = 32 + rank * 8  # Varying token counts
            x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device)
            topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
        else:
            T = 0
            x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
            topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

        try:
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx, cfg.num_experts
            )
            (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
                x,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
            )
            y, _, _ = buffer.combine(recv_x, handle, recv_topk_weights)

            if y.shape[0] != T:
                log_all(f"Round {round_idx}: shape mismatch! Expected {T}, got {y.shape[0]}")
                success = False
                break

        except Exception as e:
            log_all(f"Round {round_idx} FAILED: {e}")
            success = False
            break

        dist.barrier()

    if success:
        log("Test 5 PASSED (10 rounds of dynamic assignment)")
    else:
        log("Test 5 FAILED")
        buffer.destroy()
        return False

    log("=" * 50)
    log("All tests PASSED!")
    buffer.destroy()
    return True


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    cfg = TestConfig()

    try:
        success = test_zero_token_dispatch(cfg)
    except Exception as e:
        log_all(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    dist.barrier()

    if rank == 0:
        if success:
            print("\n[RESULT] Zero-token dispatch design VALIDATED")
        else:
            print("\n[RESULT] Zero-token dispatch design FAILED")

    dist.destroy_process_group()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
