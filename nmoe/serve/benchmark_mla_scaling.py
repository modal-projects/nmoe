# SPDX-License-Identifier: Apache-2.0
"""Scaling comparison between CuTeDSL and FlashMLA."""

import torch
from nmoe.serve.benchmark_mla_kernels import BenchmarkConfig, benchmark_cutedsl_mla, benchmark_flashmla


def main() -> int:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    configs = [
        (2, 256),
        (4, 1024),
        (8, 4096),
        (16, 8192),
        (32, 16384),
        (48, 24576),
    ]

    print("\nMLA Kernel Scaling Comparison")
    print("=" * 80)
    print(f"{'B':>4} {'Seq':>8} | {'CuTeDSL(ms)':>12} {'FlashMLA(ms)':>12} | {'Speedup':>10} {'Winner':>10}")
    print("-" * 80)

    for batch, seq in configs:
        config = BenchmarkConfig(batch_size=batch, num_heads=128, seq_len=seq, warmup_iters=5, bench_iters=20)

        torch.cuda.empty_cache()
        try:
            cute_result = benchmark_cutedsl_mla(config, device)
            cute_ms = cute_result["avg_latency_ms"]
        except Exception as e:
            cute_ms = float("inf")
            print(f"  CuTeDSL failed at B={batch}, seq={seq}: {e}")

        torch.cuda.empty_cache()
        try:
            flash_result = benchmark_flashmla(config, device)
            flash_ms = flash_result["avg_latency_ms"]
        except Exception as e:
            flash_ms = float("inf")
            print(f"  FlashMLA failed at B={batch}, seq={seq}: {e}")

        if cute_ms != float("inf") and flash_ms != float("inf"):
            speedup = flash_ms / cute_ms
            winner = "CuTeDSL" if speedup > 1.0 else "FlashMLA"
            print(f"{batch:>4} {seq:>8} | {cute_ms:>12.3f} {flash_ms:>12.3f} | {speedup:>10.2f}x {winner:>10}")
        else:
            print(f"{batch:>4} {seq:>8} | {'FAIL':>12} {'FAIL':>12} | {'N/A':>10} {'N/A':>10}")

    print("=" * 80)
    print("\nNote: Speedup > 1.0 means CuTeDSL is faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
