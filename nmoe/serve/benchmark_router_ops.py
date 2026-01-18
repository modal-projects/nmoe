# SPDX-License-Identifier: Apache-2.0
"""Ported router kernels for isolated benchmarking (no external deps).

This module JIT-compiles (via `torch.utils.cpp_extension.load_inline`) two
third-party routing kernels so we can benchmark them in-tree:

- vLLM grouped_topk (CUDA) kernel: `grouped_topk_kernels.cu`
- SGLang sgl-kernel moe_fused_gate (CUDA) kernel: `moe_fused_gate.cu`

Both upstream projects are Apache-2.0 licensed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_VLLM_MODULE = None
_SGLANG_MODULE = None
_KERNEL_VERSION = 1


def _maybe_set_cutlass_path() -> None:
  """Best-effort CUTLASS_PATH discovery (container-first)."""
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return


def _cutlass_include_dir() -> str:
  _maybe_set_cutlass_path()
  cutlass = os.environ.get("CUTLASS_PATH")
  if not cutlass:
    raise RuntimeError("CUTLASS_PATH not set and could not be auto-discovered.")
  root = Path(cutlass)
  if (root / "include" / "cutlass").is_dir():
    return str(root / "include")
  if (root / "cutlass").is_dir():
    return str(root)
  raise RuntimeError(f"Invalid CUTLASS_PATH (missing cutlass headers): {root}")


def _read_src(name: str) -> str:
  path = Path(__file__).resolve().parent / name
  return path.read_text()


_VLLM_CPP_SOURCE = r"""
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> vllm_grouped_topk_cuda(
    torch::Tensor scores, int64_t n_group, int64_t topk_group, int64_t topk,
    bool renormalize, double routed_scaling_factor, torch::Tensor bias,
    int64_t scoring_func);
"""

_SGLANG_CPP_SOURCE = r"""
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> sglang_moe_fused_gate_cuda(
    torch::Tensor input, torch::Tensor bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t num_fused_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);
"""

_VLLM_WRAPPER_CUDA_SOURCE = r"""
#include <torch/extension.h>

// Defined in benchmark_router_vllm_grouped_topk.cu (ported from vLLM).
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);

std::tuple<torch::Tensor, torch::Tensor> vllm_grouped_topk_cuda(
    torch::Tensor scores, int64_t n_group, int64_t topk_group, int64_t topk,
    bool renormalize, double routed_scaling_factor, torch::Tensor bias,
    int64_t scoring_func) {
  return grouped_topk(scores, n_group, topk_group, topk, renormalize,
                      routed_scaling_factor, bias, scoring_func);
}
"""

_SGLANG_WRAPPER_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

// From sgl-kernel moe_fused_gate.cu.
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input, at::Tensor& bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t num_fused_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);

std::tuple<torch::Tensor, torch::Tensor> sglang_moe_fused_gate_cuda(
    torch::Tensor input, torch::Tensor bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t num_fused_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output) {
  at::Tensor input_ = input;
  at::Tensor bias_ = bias;
  auto out = moe_fused_gate(
      input_, bias_, num_expert_group, topk_group, topk,
      num_fused_shared_experts, routed_scaling_factor,
      apply_routed_scaling_factor_on_output);
  TORCH_CHECK(out.size() == 2, "moe_fused_gate must return 2 tensors.");
  return std::make_tuple(out[0], out[1]);
}
"""


def _get_vllm_module():
  global _VLLM_MODULE
  if _VLLM_MODULE is None:
    vllm_src = _read_src("benchmark_router_vllm_grouped_topk.cu")
    _VLLM_MODULE = load_inline(
      name=f"router_bench_vllm_ops_v{_KERNEL_VERSION}",
      cpp_sources=[_VLLM_CPP_SOURCE],
      cuda_sources=[vllm_src, _VLLM_WRAPPER_CUDA_SOURCE],
      functions=["vllm_grouped_topk_cuda"],
      extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        # Torch extensions compile with __CUDA_NO_* by default; the vLLM kernel
        # relies on half/bf16 operators and conversions.
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
      ],
      verbose=False,
    )
  return _VLLM_MODULE


def _get_sglang_module():
  global _SGLANG_MODULE
  if _SGLANG_MODULE is None:
    include_dir = _cutlass_include_dir()
    sglang_src = (Path(__file__).resolve().parent / "kernels" / "sglang_moe_fused_gate.cu").read_text()
    _SGLANG_MODULE = load_inline(
      name=f"router_bench_sglang_ops_v{_KERNEL_VERSION}",
      cpp_sources=[_SGLANG_CPP_SOURCE],
      cuda_sources=[sglang_src, _SGLANG_WRAPPER_CUDA_SOURCE],
      functions=["sglang_moe_fused_gate_cuda"],
      extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        f"-I{include_dir}",
      ],
      verbose=False,
    )
  return _SGLANG_MODULE


def vllm_grouped_topk(
  scores: torch.Tensor,
  *,
  num_expert_group: int,
  topk_group: int,
  topk: int,
  renormalize: bool,
  routed_scaling_factor: float,
  bias: torch.Tensor,
  scoring_func: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """vLLM grouped_topk port.

  scoring_func: 0=none (scores already computed), 1=sigmoid.
  Returns:
    topk_weights: float32 [T, topk]
    topk_ids: int32 [T, topk]
  """
  if not scores.is_contiguous():
    scores = scores.contiguous()
  if not bias.is_contiguous():
    bias = bias.contiguous()
  return _get_vllm_module().vllm_grouped_topk_cuda(
    scores,
    int(num_expert_group),
    int(topk_group),
    int(topk),
    bool(renormalize),
    float(routed_scaling_factor),
    bias,
    int(scoring_func),
  )


def sglang_moe_fused_gate(
  logits: torch.Tensor,
  *,
  bias: torch.Tensor,
  num_expert_group: int,
  topk_group: int,
  topk: int,
  num_fused_shared_experts: int,
  routed_scaling_factor: float,
  apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  """SGLang moe_fused_gate port (sigmoid + bias + grouped-topk in one CUDA op).

  Returns:
    topk_weights: float32 [T, topk]
    topk_ids: int32 [T, topk]
  """
  if not logits.is_contiguous():
    logits = logits.contiguous()
  if not bias.is_contiguous():
    bias = bias.contiguous()
  return _get_sglang_module().sglang_moe_fused_gate_cuda(
    logits,
    bias,
    int(num_expert_group),
    int(topk_group),
    int(topk),
    int(num_fused_shared_experts),
    float(routed_scaling_factor),
    bool(apply_routed_scaling_factor_on_output),
  )
