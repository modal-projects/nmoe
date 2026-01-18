#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Microbench: DeepGEMM UE8M0 scale packing (SM100).

Goal
  Validate and measure a production-grade alternative to DeepGEMM's internal
  transpose+pack of FP32 UE8M0 scaling factors:
    - Baseline: DeepGEMM packs FP32 scales internally (disable_ue8m0_cast=False)
    - Proposed: pack once via nmoe kernel, reuse packed INT scales (disable_ue8m0_cast=True)

Accuracy
  - Compare fp8 outputs (baseline vs packed) for exact match / tight tolerance.
  - Compare fp8 output vs BF16 reference (F.linear) for sanity.

Performance
  - Benchmark a 2-GEMM pattern (same input, different weights), matching
    MLP(w1,w3) or MLA(q,kv) projection reuse.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from nmoe.serve.kernels import quantize_fp8_ue8m0, pack_fp32_ue8m0_scales_to_int


def _sync() -> None:
    torch.cuda.synchronize()


def _ms_per_iter(fn, iters: int) -> float:
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256, help="M (tokens)")
    ap.add_argument("--k", type=int, default=7168, help="K (hidden)")
    ap.add_argument("--n", type=int, default=4096, help="N (out features)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    M, K, N = int(args.m), int(args.k), int(args.n)
    if K % 128 != 0:
        raise ValueError("K must be divisible by 128")

    import deep_gemm
    from deep_gemm import fp8_gemm_nt

    x_bf16 = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w1_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16)
    w3_bf16 = torch.randn((N, K), device=device, dtype=torch.bfloat16)

    # FP8 operands + FP32 UE8M0 scales (per-row granularity for the microbench).
    x_fp8, x_sf = quantize_fp8_ue8m0(x_bf16.contiguous())
    w1_fp8, w1_sf = quantize_fp8_ue8m0(w1_bf16.contiguous())
    w3_fp8, w3_sf = quantize_fp8_ue8m0(w3_bf16.contiguous())

    # Pre-pack weight scales once (weights are static).
    w1_sf_packed = pack_fp32_ue8m0_scales_to_int(w1_sf, mn=N, k=K, gran_mn_in=1)
    w3_sf_packed = pack_fp32_ue8m0_scales_to_int(w3_sf, mn=N, k=K, gran_mn_in=1)

    out0 = torch.empty((M, N), device=device, dtype=torch.bfloat16)
    out1 = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    # Baseline: DeepGEMM packs scales internally for each GEMM call.
    def run_baseline() -> None:
        fp8_gemm_nt((x_fp8, x_sf), (w1_fp8, w1_sf), out0, None, recipe=(1, 1, 128), compiled_dims="nk", disable_ue8m0_cast=False)
        fp8_gemm_nt((x_fp8, x_sf), (w3_fp8, w3_sf), out1, None, recipe=(1, 1, 128), compiled_dims="nk", disable_ue8m0_cast=False)

    # Proposed: pack activation scales once and reuse packed INT scales.
    x_sf_packed = pack_fp32_ue8m0_scales_to_int(x_sf, mn=M, k=K, gran_mn_in=1)

    def run_packed() -> None:
        fp8_gemm_nt((x_fp8, x_sf_packed), (w1_fp8, w1_sf_packed), out0, None, recipe=(1, 1, 128), compiled_dims="nk", disable_ue8m0_cast=True)
        fp8_gemm_nt((x_fp8, x_sf_packed), (w3_fp8, w3_sf_packed), out1, None, recipe=(1, 1, 128), compiled_dims="nk", disable_ue8m0_cast=True)

    # Warmup (build kernels).
    for _ in range(int(args.warmup)):
        run_baseline()
        run_packed()
    _sync()

    # Correctness: baseline fp8 vs packed fp8.
    run_baseline()
    o0_base = out0.clone()
    o1_base = out1.clone()
    run_packed()
    o0_new = out0.clone()
    o1_new = out1.clone()

    diff0 = (o0_new - o0_base).abs()
    diff1 = (o1_new - o1_base).abs()
    print("fp8 packed vs baseline (DeepGEMM internal pack):")
    print(f"  out0 max_abs={diff0.max().item():.3e} mean_abs={diff0.mean().item():.3e}")
    print(f"  out1 max_abs={diff1.max().item():.3e} mean_abs={diff1.mean().item():.3e}")

    # Sanity: fp8 output vs BF16 reference.
    ref0 = F.linear(x_bf16, w1_bf16)
    ref1 = F.linear(x_bf16, w3_bf16)
    err0 = (o0_new - ref0).abs()
    err1 = (o1_new - ref1).abs()
    denom0 = ref0.abs().clamp_min(1e-6)
    denom1 = ref1.abs().clamp_min(1e-6)
    rel0 = (err0 / denom0).flatten()
    rel1 = (err1 / denom1).flatten()
    k0 = max(1, int(0.99 * rel0.numel()))
    k1 = max(1, int(0.99 * rel1.numel()))
    print("fp8 packed vs BF16 F.linear:")
    print(f"  out0 max_abs={err0.max().item():.3e} mean_rel={rel0.mean().item():.3e} p99_rel={rel0.kthvalue(k0).values.item():.3e}")
    print(f"  out1 max_abs={err1.max().item():.3e} mean_rel={rel1.mean().item():.3e} p99_rel={rel1.kthvalue(k1).values.item():.3e}")

    # Performance.
    ms_base = _ms_per_iter(run_baseline, int(args.iters))
    ms_packed = _ms_per_iter(run_packed, int(args.iters))
    print("perf (2 GEMMs, same x):")
    print(f"  baseline_ms={ms_base:.4f} packed_ms={ms_packed:.4f} speedup={(ms_base / ms_packed):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
