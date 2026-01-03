# SPDX-License-Identifier: Apache-2.0
"""Custom CUDA kernels for nmoe.serve (SM100/B200 optimized)."""

from nmoe.serve.kernels.fp8_quant import quantize_fp8_ue8m0, weighted_scatter_add, silu_mul_fp8

__all__ = ["quantize_fp8_ue8m0", "weighted_scatter_add", "silu_mul_fp8"]
