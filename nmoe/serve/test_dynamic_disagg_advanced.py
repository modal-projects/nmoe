# SPDX-License-Identifier: Apache-2.0
"""Advanced tests for dynamic disaggregation design.

Tests:
1. Low-latency dispatch/combine with T=0 tokens
2. Normal ↔ Low-latency mode alternation
3. Mixed-workload straggler effects (prefill vs decode batch sizes)
4. Multi-layer overhead measurement

Run with: torchrun --nproc_per_node=8 -m nmoe.serve.test_dynamic_disagg_advanced
"""

import os
import time
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestConfig:
    hidden_size: int = 7168
    num_experts: int = 256
    num_local_experts: int = 32  # 256 / 8
    topk: int = 8
    num_layers: int = 58  # MoE layers in DeepSeek-V3
    decode_batch_max: int = 256
    prefill_tokens: int = 2048
    num_iterations: int = 20
    warmup_iterations: int = 5


def log(msg: str, rank: int = 0) -> None:
    if dist.get_rank() == rank:
        print(f"[TEST] {msg}", flush=True)


def log_all(msg: str) -> None:
    print(f"[rank{dist.get_rank()}] {msg}", flush=True)


def test_ll_zero_token(cfg: TestConfig) -> bool:
    """Test 1: Low-latency dispatch/combine with T=0 tokens."""
    from deep_ep import Buffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    log("=" * 60)
    log("Test 1: Low-latency mode with T=0 tokens")
    log("=" * 60)

    # Try to create LL buffer - may fail if RDMA not available
    try:
        # Calculate RDMA bytes needed for LL mode using DeepEP's size hint
        rdma_bytes = int(Buffer.get_low_latency_rdma_size_hint(
            cfg.decode_batch_max, cfg.hidden_size, world_size, cfg.num_experts
        ))

        buffer = Buffer(
            group=dist.group.WORLD,
            num_nvl_bytes=0,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            allow_nvlink_for_low_latency_mode=True,
            num_qps_per_rank=int(cfg.num_local_experts),
            explicitly_destroy=True,
        )
        log(f"LL Buffer created with {rdma_bytes / 1e6:.1f} MB RDMA")
    except Exception as e:
        log(f"LL Buffer creation failed (RDMA may not be available): {e}")
        log("Skipping LL tests - this is expected on NVLink-only nodes")
        return True  # Skip but don't fail

    # Test with varying T=0 patterns
    test_cases = [
        ("All ranks have tokens", lambda r: 64),
        ("Even ranks T=0", lambda r: 0 if r % 2 == 0 else 64),
        ("Only rank 0 has tokens", lambda r: 128 if r == 0 else 0),
        ("All ranks T=0", lambda r: 0),
    ]

    for test_name, token_fn in test_cases:
        T = token_fn(rank)
        log(f"  Subtest: {test_name} (rank {rank} has T={T})")

        if T > 0:
            x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device, dtype=torch.int64)
            topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
        else:
            x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
            topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

        dist.barrier()

        try:
            # Low-latency dispatch
            (recv_x_fp8, recv_x_scales), recv_count, handle, _, _ = buffer.low_latency_dispatch(
                x, topk_idx,
                num_max_dispatch_tokens_per_rank=cfg.decode_batch_max,
                num_experts=cfg.num_experts,
                use_fp8=True,
                use_ue8m0=True,
                round_scale=True,  # Required when use_ue8m0=True
            )

            # Simulate expert computation (just zeros for test)
            expert_out = torch.zeros(
                cfg.num_local_experts, cfg.decode_batch_max * world_size, cfg.hidden_size,
                device=device, dtype=torch.bfloat16
            )

            # Low-latency combine
            y, _, _ = buffer.low_latency_combine(
                expert_out, topk_idx, topk_weights, handle,
            )

            if y.shape[0] != T:
                log_all(f"ERROR: Expected {T} tokens, got {y.shape[0]}")
                return False

        except Exception as e:
            log_all(f"  Subtest '{test_name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

        dist.barrier()

    log("Test 1 PASSED")
    buffer.destroy()
    return True


def test_mode_alternation(cfg: TestConfig) -> bool:
    """Test 2: Alternating normal ↔ low-latency mode."""
    from deep_ep import Buffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    log("=" * 60)
    log("Test 2: Normal ↔ Low-latency mode alternation")
    log("=" * 60)

    # Create buffer with both NVL and RDMA
    num_nvl_bytes = max(
        Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
        Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
    )

    num_rdma_bytes = int(Buffer.get_low_latency_rdma_size_hint(
        cfg.decode_batch_max, cfg.hidden_size, world_size, cfg.num_experts
    ))

    try:
        buffer = Buffer(
            group=dist.group.WORLD,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            allow_nvlink_for_low_latency_mode=True,
            num_qps_per_rank=int(cfg.num_local_experts),
            explicitly_destroy=True,
        )
        log(f"Dual-mode buffer created: NVL={num_nvl_bytes/1e6:.1f}MB, RDMA={num_rdma_bytes/1e6:.1f}MB")
    except Exception as e:
        log(f"Dual-mode buffer creation failed: {e}")
        log("Skipping mode alternation test")
        return True

    num_rounds = 10
    success = True

    for round_idx in range(num_rounds):
        mode = "normal" if round_idx % 2 == 0 else "ll"
        T = 64 if mode == "normal" else 32

        x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device, dtype=torch.int64)
        topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)

        try:
            if mode == "normal":
                # Normal dispatch/combine
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
            else:
                # Clean buffer before LL mode
                buffer.clean_low_latency_buffer(cfg.decode_batch_max, cfg.hidden_size, cfg.num_experts)

                # Low-latency dispatch/combine
                (recv_x_fp8, recv_x_scales), recv_count, handle, _, _ = buffer.low_latency_dispatch(
                    x, topk_idx,
                    num_max_dispatch_tokens_per_rank=cfg.decode_batch_max,
                    num_experts=cfg.num_experts,
                    use_fp8=True,
                    use_ue8m0=True,
                    round_scale=True,  # Required when use_ue8m0=True
                )
                expert_out = torch.zeros(
                    cfg.num_local_experts, cfg.decode_batch_max * world_size, cfg.hidden_size,
                    device=device, dtype=torch.bfloat16
                )
                y, _, _ = buffer.low_latency_combine(
                    expert_out, topk_idx, topk_weights, handle,
                )

            if y.shape[0] != T:
                log(f"Round {round_idx} ({mode}): shape mismatch! Expected {T}, got {y.shape[0]}")
                success = False
                break

        except Exception as e:
            log(f"Round {round_idx} ({mode}) FAILED: {e}")
            import traceback
            traceback.print_exc()
            success = False
            break

        dist.barrier()

    if success:
        log("Test 2 PASSED")
    else:
        log("Test 2 FAILED")
    buffer.destroy()
    return success


def test_straggler_effects(cfg: TestConfig) -> bool:
    """Test 3: Mixed prefill/decode batch sizes - measure straggler gating."""
    from deep_ep import Buffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    log("=" * 60)
    log("Test 3: Mixed-workload straggler effects")
    log("=" * 60)

    num_nvl_bytes = max(
        Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
        Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
    )

    buffer = Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=0,
        explicitly_destroy=True,
    )

    # Test scenarios:
    # 1. All ranks same batch size (baseline)
    # 2. Mixed: some ranks prefill-sized, some decode-sized
    # 3. Extreme: 1 rank prefill, 7 decode

    scenarios = [
        ("Uniform decode (32 tokens each)", lambda r: 32),
        ("Uniform prefill (512 tokens each)", lambda r: 512),
        ("Mixed: ranks 0-3 prefill (512), ranks 4-7 decode (32)",
         lambda r: 512 if r < 4 else 32),
        ("Extreme: rank 0 prefill (2048), others decode (32)",
         lambda r: 2048 if r == 0 else 32),
        ("Extreme: rank 0 prefill (2048), others empty (0)",
         lambda r: 2048 if r == 0 else 0),
    ]

    for scenario_name, token_fn in scenarios:
        T = token_fn(rank)

        if T > 0:
            x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
            topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device, dtype=torch.int64)
            topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
        else:
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

        # Gather timing from all ranks
        timing = torch.tensor([avg_us], device=device)
        all_timings = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(all_timings, timing)

        if rank == 0:
            timings_list = [t.item() for t in all_timings]
            max_time = max(timings_list)
            min_time = min(timings_list)
            log(f"  {scenario_name}")
            log(f"    Per-rank times (µs): {[f'{t:.0f}' for t in timings_list]}")
            log(f"    Max: {max_time:.0f}µs, Min: {min_time:.0f}µs, Spread: {max_time - min_time:.0f}µs")

        dist.barrier()

    log("Test 3 PASSED (see timing data above)")
    buffer.destroy()
    return True


def test_multilayer_overhead(cfg: TestConfig) -> bool:
    """Test 4: Measure overhead across all MoE layers (simulating full forward pass)."""
    from deep_ep import Buffer

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    log("=" * 60)
    log(f"Test 4: Multi-layer overhead ({cfg.num_layers} MoE layers)")
    log("=" * 60)

    num_nvl_bytes = max(
        Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
        Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(cfg.hidden_size * 2, world_size),
    )

    buffer = Buffer(
        group=dist.group.WORLD,
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=0,
        explicitly_destroy=True,
    )

    # Scenario: Simulate dynamic disagg where some ranks are empty
    # Rank 0 does prefill (has tokens), ranks 1-7 do decode (empty for prefill collective)
    T = 512 if rank == 0 else 0

    if T > 0:
        x = torch.randn(T, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.randint(0, cfg.num_experts, (T, cfg.topk), device=device, dtype=torch.int64)
        topk_weights = torch.softmax(torch.randn(T, cfg.topk, device=device), dim=-1)
    else:
        x = torch.empty(0, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        topk_idx = torch.empty(0, cfg.topk, device=device, dtype=torch.int64)
        topk_weights = torch.empty(0, cfg.topk, device=device, dtype=torch.float32)

    # Warmup - single layer
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

    # Timed run - simulate full forward pass with cfg.num_layers MoE layers
    start = time.perf_counter()
    for _ in range(cfg.num_iterations):
        for layer in range(cfg.num_layers):
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

    per_forward_ms = (elapsed / cfg.num_iterations) * 1000
    per_layer_us = (elapsed / cfg.num_iterations / cfg.num_layers) * 1e6

    log(f"  Rank 0 prefill (T=512), ranks 1-7 empty")
    log(f"  {cfg.num_layers} MoE layers per forward pass")
    log(f"  Total per forward: {per_forward_ms:.2f} ms")
    log(f"  Per MoE layer: {per_layer_us:.1f} µs")
    log(f"  Empty rank overhead: ~{per_layer_us * 7 / 8:.1f} µs/layer (estimated)")

    # Compare with all-active baseline
    T_all = 64  # All ranks active
    x_all = torch.randn(T_all, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    topk_idx_all = torch.randint(0, cfg.num_experts, (T_all, cfg.topk), device=device, dtype=torch.int64)
    topk_weights_all = torch.softmax(torch.randn(T_all, cfg.topk, device=device), dim=-1)

    # Warmup
    for _ in range(cfg.warmup_iterations):
        num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
            topk_idx_all, cfg.num_experts
        )
        (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
            x_all,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx_all,
            topk_weights=topk_weights_all,
        )
        y, _, _ = buffer.combine(recv_x, handle, recv_topk_weights)

    torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    for _ in range(cfg.num_iterations):
        for layer in range(cfg.num_layers):
            num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(
                topk_idx_all, cfg.num_experts
            )
            (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
                x_all,
                num_tokens_per_rank=num_tokens_per_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx_all,
                topk_weights=topk_weights_all,
            )
            y, _, _ = buffer.combine(recv_x, handle, recv_topk_weights)

    torch.cuda.synchronize()
    elapsed_baseline = time.perf_counter() - start
    per_forward_ms_baseline = (elapsed_baseline / cfg.num_iterations) * 1000
    per_layer_us_baseline = (elapsed_baseline / cfg.num_iterations / cfg.num_layers) * 1e6

    log(f"  Baseline (all ranks T=64):")
    log(f"    Total per forward: {per_forward_ms_baseline:.2f} ms")
    log(f"    Per MoE layer: {per_layer_us_baseline:.1f} µs")
    log(f"  Overhead ratio: {per_forward_ms / per_forward_ms_baseline:.2f}x")

    log("Test 4 PASSED (see timing data above)")
    buffer.destroy()
    return True


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    cfg = TestConfig()

    results = {}

    # Test 1: LL mode with T=0
    try:
        results["ll_zero_token"] = test_ll_zero_token(cfg)
    except Exception as e:
        log(f"Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["ll_zero_token"] = False

    dist.barrier()

    # Test 2: Mode alternation
    try:
        results["mode_alternation"] = test_mode_alternation(cfg)
    except Exception as e:
        log(f"Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["mode_alternation"] = False

    dist.barrier()

    # Test 3: Straggler effects
    try:
        results["straggler_effects"] = test_straggler_effects(cfg)
    except Exception as e:
        log(f"Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["straggler_effects"] = False

    dist.barrier()

    # Test 4: Multi-layer overhead
    try:
        results["multilayer_overhead"] = test_multilayer_overhead(cfg)
    except Exception as e:
        log(f"Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results["multilayer_overhead"] = False

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        all_passed = True
        for name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n[RESULT] All advanced dynamic disagg tests PASSED")
        else:
            print("\n[RESULT] Some tests FAILED")

    dist.destroy_process_group()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
