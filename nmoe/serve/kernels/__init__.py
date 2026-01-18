# SPDX-License-Identifier: Apache-2.0
"""Custom CUDA kernels for nmoe.serve (SM100/B200 optimized)."""

from nmoe.serve.kernels.fp8_quant import (
  quantize_fp8_ue8m0,
  pack_fp32_ue8m0_scales_to_int,
  moe_pack_fp8_grouped,
  weighted_scatter_add,
  weighted_scatter_add_indexed,
  weighted_scatter_add_grouped,
  silu_mul_fp8,
  silu_mul_fp8_grouped,
)

__all__ = [
  "quantize_fp8_ue8m0",
  "pack_fp32_ue8m0_scales_to_int",
  "moe_pack_fp8_grouped",
  "weighted_scatter_add",
  "weighted_scatter_add_indexed",
  "weighted_scatter_add_grouped",
  "silu_mul_fp8",
  "silu_mul_fp8_grouped",
]
