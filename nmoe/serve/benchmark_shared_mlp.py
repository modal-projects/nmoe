# SPDX-License-Identifier: Apache-2.0
"""Shared-expert MLP microbenchmark (DeepSeek-V3).

Benchmarks a single "shared expert" MLP forward with real checkpoint weights:
  - nmoe_ref_fp8: current nmoe shared MLP math (DeepGEMM FP8 GEMMs + float32 SwiGLU)
  - nmoe_fp8_fused: optimized FP8 path (fused W13 GEMM + silu_mul_fp8)
  - torch_bf16: BF16 weights via torch F.linear (W13 fused)
  - torch_fp8_scaled: FP8 GEMMs via torch scaled-mm (best-effort; torch API varies)
  - quack_bf16: QuACK GEMM for BF16 weights (W13 fused)
  - cublaslt_bf16: explicit cuBLASLt BF16 GEMM (W13 fused)

This is a single-GPU benchmark; it does not require torchrun / world_size=8.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nmoe.serve.model import weight_dequant


def _require_cuda() -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")


def _maybe_add_quack_to_syspath() -> None:
  # Prefer an in-container clone at <repo_root>/third_party/quack (container-first).
  here = Path(__file__).resolve()
  repo_root = here.parents[2]
  cand = repo_root / "third_party" / "quack"
  if (cand / "quack").is_dir():
    sys.path.insert(0, str(cand))


def _read_tensor(f, key: str) -> torch.Tensor:
  if key not in f.keys():
    raise KeyError(f"missing key: {key}")
  return f.get_tensor(key)


@dataclass(frozen=True)
class SharedMlpWeights:
  hidden: int
  inter: int
  # FP8 weights (DeepGEMM format) + block scales.
  w1_fp8: torch.Tensor
  w1_scale: torch.Tensor
  w3_fp8: torch.Tensor
  w3_scale: torch.Tensor
  w2_fp8: torch.Tensor
  w2_scale: torch.Tensor

  # BF16 dequantized weights (for BF16 baselines).
  w13_bf16: torch.Tensor  # [2*inter, hidden]
  w2_bf16: torch.Tensor  # [hidden, inter]


def _load_shared_mlp_weights(
  ckpt_path: str,
  *,
  mp_world_size: int,
  layer: int,
  device: torch.device,
) -> SharedMlpWeights:
  from safetensors.torch import safe_open

  shard = os.path.join(ckpt_path, f"model0-mp{mp_world_size}.safetensors")
  if not os.path.exists(shard):
    raise FileNotFoundError(f"missing mp shard: {shard}")

  prefix = f"layers.{int(layer)}.ffn.shared"
  k_w1 = f"{prefix}.w1.weight"
  k_w1s = f"{prefix}.w1.weight_scale_inv"
  k_w3 = f"{prefix}.w3.weight"
  k_w3s = f"{prefix}.w3.weight_scale_inv"
  k_w2 = f"{prefix}.w2.weight"
  k_w2s = f"{prefix}.w2.weight_scale_inv"

  with safe_open(shard, framework="pt", device="cpu") as f:
    w1_fp8 = _read_tensor(f, k_w1)
    w1_scale = _read_tensor(f, k_w1s)
    w3_fp8 = _read_tensor(f, k_w3)
    w3_scale = _read_tensor(f, k_w3s)
    w2_fp8 = _read_tensor(f, k_w2)
    w2_scale = _read_tensor(f, k_w2s)

  if w1_fp8.dim() != 2 or w3_fp8.dim() != 2 or w2_fp8.dim() != 2:
    raise RuntimeError("expected 2D weights")
  inter, hidden = int(w1_fp8.shape[0]), int(w1_fp8.shape[1])
  if tuple(w3_fp8.shape) != (inter, hidden):
    raise RuntimeError(f"w3 shape mismatch: {tuple(w3_fp8.shape)} vs {(inter, hidden)}")
  if tuple(w2_fp8.shape) != (hidden, inter):
    raise RuntimeError(f"w2 shape mismatch: {tuple(w2_fp8.shape)} vs {(hidden, inter)}")

  # Move to GPU once; do not time weight transfers.
  w1_fp8 = w1_fp8.to(device)
  w1_scale = w1_scale.to(device)
  w3_fp8 = w3_fp8.to(device)
  w3_scale = w3_scale.to(device)
  w2_fp8 = w2_fp8.to(device)
  w2_scale = w2_scale.to(device)

  # BF16 baselines use dequantized weights; keep this out of the timing region.
  w1_bf16 = weight_dequant(w1_fp8, w1_scale)
  w3_bf16 = weight_dequant(w3_fp8, w3_scale)
  w13_bf16 = torch.cat([w1_bf16, w3_bf16], dim=0).contiguous()
  w2_bf16 = weight_dequant(w2_fp8, w2_scale).contiguous()

  return SharedMlpWeights(
    hidden=hidden,
    inter=inter,
    w1_fp8=w1_fp8,
    w1_scale=w1_scale,
    w3_fp8=w3_fp8,
    w3_scale=w3_scale,
    w2_fp8=w2_fp8,
    w2_scale=w2_scale,
    w13_bf16=w13_bf16,
    w2_bf16=w2_bf16,
  )


def _bench_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
  y = None
  for _ in range(int(warmup)):
    y = fn()
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(int(iters)):
    y = fn()
  end.record()
  torch.cuda.synchronize()
  assert y is not None
  return float(start.elapsed_time(end)) / float(iters), y


def _match_frac(
  ref: torch.Tensor,
  out: torch.Tensor,
  *,
  atol: float,
  rtol: float,
) -> float:
  if ref.shape != out.shape:
    return 0.0
  ok = torch.isclose(out, ref, atol=float(atol), rtol=float(rtol))
  return float(ok.float().mean().item())


def _quantize_fp8_e4m3_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  # Best-effort FP8 quant for torch._scaled_mm baselines.
  # Use a single scale per tensor (not blockscaled).
  _FP8_E4M3_MAX = 448.0
  amax = x.abs().amax()
  # Avoid div0; keep scale on GPU for torch APIs.
  scale = (amax / _FP8_E4M3_MAX).clamp(min=1e-8).to(torch.float32)
  x_q = (x / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
  return x_q.contiguous(), scale


def _torch_scaled_mm(
  a_fp8: torch.Tensor,
  b_fp8: torch.Tensor,
  *,
  a_scale: torch.Tensor,
  b_scale: torch.Tensor,
  out_dtype: torch.dtype,
) -> torch.Tensor:
  # Torch FP8 APIs are not stable; support the common variants.
  fn = getattr(torch, "_scaled_mm", None)
  if fn is None:
    # Some builds only expose the aten op.
    fn = getattr(torch.ops.aten, "_scaled_mm", None)
  if fn is None:
    raise RuntimeError("torch._scaled_mm not available in this build.")

  # Common signature: (a, b, a_scale, b_scale, bias=None, out_dtype=..., use_fast_accum=...)
  try:
    out = fn(a_fp8, b_fp8, a_scale, b_scale, None, out_dtype)  # type: ignore[misc]
  except TypeError:
    try:
      out = fn(a_fp8, b_fp8, a_scale, b_scale, out_dtype=out_dtype)  # type: ignore[misc]
    except TypeError:
      out = fn(a_fp8, b_fp8, a_scale, b_scale)  # type: ignore[misc]

  if isinstance(out, (tuple, list)):
    out = out[0]
  if not isinstance(out, torch.Tensor):
    raise RuntimeError("unexpected torch._scaled_mm return type")
  if out_dtype is not None and out.dtype != out_dtype:
    out = out.to(out_dtype)
  return out


def _get_cublaslt_module():
  # Benchmark-only explicit cuBLASLt BF16 GEMM.
  from torch.utils.cpp_extension import load_inline

  name = "nmoe_bench_cublaslt_bf16_gemm_v1"
  cpp = r"""
#include <torch/extension.h>
torch::Tensor cublaslt_gemm_bf16_cuda(torch::Tensor a_bf16, torch::Tensor b_bf16);
"""
  cuda = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace {

inline cudaError_t cuda_from_cublas(cublasStatus_t s) {
  return (s == CUBLAS_STATUS_SUCCESS) ? cudaSuccess : cudaErrorUnknown;
}

struct LtState {
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 32ull << 20;
  std::mutex mu;
};

LtState& lt_state() {
  static LtState s;
  return s;
}

cudaError_t ensure_lt() {
  auto& s = lt_state();
  std::lock_guard<std::mutex> lock(s.mu);
  if (s.handle == nullptr) {
    auto st = cublasLtCreate(&s.handle);
    if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  }
  if (s.workspace == nullptr) {
    auto err = cudaMalloc(&s.workspace, s.workspace_bytes);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

struct Key {
  int m, n, k;
  int lda, ldb, ldc;
};

struct KeyHash {
  size_t operator()(const Key& x) const noexcept {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix((uint64_t)x.m);
    mix((uint64_t)x.n);
    mix((uint64_t)x.k);
    mix((uint64_t)x.lda);
    mix((uint64_t)x.ldb);
    mix((uint64_t)x.ldc);
    return h;
  }
};

inline bool operator==(const Key& a, const Key& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k &&
         a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc;
}

struct Plan {
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulAlgo_t algo{};
  ~Plan() {
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
  }
};

std::mutex plan_mu;
std::unordered_map<Key, std::unique_ptr<Plan>, KeyHash> plan_cache;

cudaError_t get_plan(
    cublasLtHandle_t handle,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    size_t workspace_bytes,
    Plan** out) {
  const Key key{m, n, k, lda, ldb, ldc};
  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto it = plan_cache.find(key);
    if (it != plan_cache.end()) {
      *out = it->second.get();
      return cudaSuccess;
    }
  }

  auto plan = std::make_unique<Plan>();

  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  const cublasOperation_t opA = CUBLAS_OP_N;
  const cublasOperation_t opB = CUBLAS_OP_T;  // treat B as [n,k] row-major

  auto st = cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, scale_type);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_16BF, m, k, lda);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_16BF, n, k, ldb);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_16BF, m, n, ldc);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  cublasLtMatmulPreference_t pref = nullptr;
  st = cublasLtMatmulPreferenceCreate(&pref);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &workspace_bytes, sizeof(workspace_bytes));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  constexpr int max_algos = 16;
  cublasLtMatmulHeuristicResult_t heur[max_algos];
  int returned = 0;
  // Use the same layout for C and D; even when beta==0 and C is not read,
  // cuBLASLt expects a valid C/D layout pair for heuristic selection.
  st = cublasLtMatmulAlgoGetHeuristic(handle, plan->matmul_desc,
                                      plan->a_layout, plan->b_layout,
                                      plan->c_layout, plan->c_layout,
                                      pref, max_algos, heur, &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  if (returned <= 0) return cudaErrorUnknown;
  plan->algo = heur[0].algo;

  Plan* raw = plan.get();
  {
    std::lock_guard<std::mutex> lock(plan_mu);
    plan_cache.emplace(key, std::move(plan));
  }
  *out = raw;
  return cudaSuccess;
}

} // namespace

torch::Tensor cublaslt_gemm_bf16_cuda(torch::Tensor a_bf16, torch::Tensor b_bf16) {
  TORCH_CHECK(a_bf16.is_cuda() && b_bf16.is_cuda(), "CUDA tensors required.");
  TORCH_CHECK(a_bf16.dtype() == torch::kBFloat16, "a must be bf16");
  TORCH_CHECK(b_bf16.dtype() == torch::kBFloat16, "b must be bf16");
  TORCH_CHECK(a_bf16.dim() == 2 && b_bf16.dim() == 2, "2D tensors required.");
  TORCH_CHECK(a_bf16.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b_bf16.is_contiguous(), "b must be contiguous");

  const int64_t m64 = a_bf16.size(0);
  const int64_t k64 = a_bf16.size(1);
  const int64_t n64 = b_bf16.size(0);
  TORCH_CHECK(b_bf16.size(1) == k64, "b must be [n,k] with same k as a");

  const int m = (int)m64;
  const int n = (int)n64;
  const int k = (int)k64;

  auto out = torch::empty({m, n}, torch::TensorOptions().device(a_bf16.device()).dtype(torch::kBFloat16));

  auto err = ensure_lt();
  TORCH_CHECK(err == cudaSuccess, "ensure_lt failed");
  auto& s = lt_state();

  const int lda = k;
  const int ldb = k;
  const int ldc = n;

  Plan* plan = nullptr;
  err = get_plan(s.handle, m, n, k, lda, ldb, ldc, s.workspace_bytes, &plan);
  TORCH_CHECK(err == cudaSuccess, "get_plan failed");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto st = cublasLtMatmul(s.handle,
                          plan->matmul_desc,
                          &alpha,
                          a_bf16.data_ptr(),
                          plan->a_layout,
                          b_bf16.data_ptr(),
                          plan->b_layout,
                          &beta,
                          out.data_ptr(),
                          plan->c_layout,
                          out.data_ptr(),
                          plan->c_layout,
                          &plan->algo,
                          s.workspace,
                          s.workspace_bytes,
                          stream);
  TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed");
  return out;
}
"""
  mod = load_inline(
    name=name,
    cpp_sources=[cpp],
    cuda_sources=[cuda],
    functions=["cublaslt_gemm_bf16_cuda"],
    extra_cuda_cflags=[
      "-O3",
      "--use_fast_math",
      "-U__CUDA_NO_HALF_OPERATORS__",
      "-U__CUDA_NO_HALF_CONVERSIONS__",
      "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
      "-U__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_ldflags=["-lcublasLt", "-lcublas"],
    verbose=False,
  )
  return mod


def _main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, help="Path to mp shard dir (contains model0-mp8.safetensors).")
  ap.add_argument("--mp-world-size", type=int, default=8)
  ap.add_argument("--layer", type=int, default=3, help="MoE layer index to load shared MLP weights from.")
  ap.add_argument("--device", type=int, default=0)
  ap.add_argument("--batch", type=str, default="32,256", help="Comma-separated T values (tokens) to benchmark.")
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--iters", type=int, default=500)
  ap.add_argument(
    "--impl",
    type=str,
    default="nmoe_ref_fp8,nmoe_fp8_fused,torch_bf16,torch_fp8_scaled,quack_bf16,cublaslt_bf16",
  )
  ap.add_argument("--atol", type=float, default=5e-2)
  ap.add_argument("--rtol", type=float, default=5e-2)
  args = ap.parse_args()

  _require_cuda()
  device = torch.device(f"cuda:{int(args.device)}")
  torch.cuda.set_device(device)

  weights = _load_shared_mlp_weights(
    args.ckpt,
    mp_world_size=int(args.mp_world_size),
    layer=int(args.layer),
    device=device,
  )

  impls = [s.strip() for s in str(args.impl).split(",") if s.strip()]
  batches = [int(x.strip()) for x in str(args.batch).split(",") if x.strip()]
  if not batches:
    raise ValueError("empty --batch list")

  # Prepare QuACK import (optional).
  _maybe_add_quack_to_syspath()
  quack_gemm = None
  if "quack_bf16" in impls:
    try:
      from quack.gemm_interface import gemm as quack_gemm  # type: ignore
    except Exception as e:
      raise RuntimeError(
        "quack import failed. If you cloned quack into third_party/quack, set PYTHONPATH accordingly."
      ) from e

  cublaslt_mod = None
  if "cublaslt_bf16" in impls:
    cublaslt_mod = _get_cublaslt_module()

  # Single fused BF16 weight for W13.
  w13_bf16 = weights.w13_bf16
  w2_bf16 = weights.w2_bf16

  # Fused FP8 weights for DeepGEMM FP8 path (W13 fused).
  w13_fp8 = None
  w13_scale_ue8 = None
  if "nmoe_fp8_fused" in impls:
    w13_fp8 = torch.cat([weights.w1_fp8, weights.w3_fp8], dim=0).contiguous()
    w13_scale_ue8 = torch.cat([weights.w1_scale, weights.w3_scale], dim=0).contiguous()

  # Pre-quantized weights for torch._scaled_mm baseline.
  w13_fp8_scaled = None
  w13_scale = None
  w2_fp8_scaled = None
  w2_scale = None
  if "torch_fp8_scaled" in impls:
    w13_fp8_scaled, w13_scale = _quantize_fp8_e4m3_per_tensor(w13_bf16)
    w2_fp8_scaled, w2_scale = _quantize_fp8_e4m3_per_tensor(w2_bf16)

  def ref_nmoe_fp8(x_bf16: torch.Tensor) -> torch.Tensor:
    from nmoe.serve.kernels.fp8_quant import quantize_fp8_ue8m0
    from deep_gemm import fp8_gemm_nt

    x_fp8, x_scale_ue8 = quantize_fp8_ue8m0(x_bf16)
    gate = torch.empty((x_bf16.size(0), weights.inter), device=device, dtype=torch.bfloat16)
    up = torch.empty((x_bf16.size(0), weights.inter), device=device, dtype=torch.bfloat16)
    fp8_gemm_nt((x_fp8, x_scale_ue8), (weights.w1_fp8, weights.w1_scale), gate, None)
    fp8_gemm_nt((x_fp8, x_scale_ue8), (weights.w3_fp8, weights.w3_scale), up, None)
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    act_fp8, act_scale_ue8 = quantize_fp8_ue8m0(act)
    out = torch.empty((x_bf16.size(0), weights.hidden), device=device, dtype=torch.bfloat16)
    fp8_gemm_nt((act_fp8, act_scale_ue8), (weights.w2_fp8, weights.w2_scale), out, None)
    return out

  def nmoe_fp8_fused(x_bf16: torch.Tensor) -> torch.Tensor:
    from nmoe.serve.kernels.fp8_quant import quantize_fp8_ue8m0, silu_mul_fp8
    from deep_gemm import fp8_gemm_nt

    assert w13_fp8 is not None and w13_scale_ue8 is not None

    x_fp8, x_scale_ue8 = quantize_fp8_ue8m0(x_bf16)
    gateup = torch.empty((x_bf16.size(0), 2 * weights.inter), device=device, dtype=torch.bfloat16)
    fp8_gemm_nt((x_fp8, x_scale_ue8), (w13_fp8, w13_scale_ue8), gateup, None)
    gate, up = gateup.chunk(2, dim=-1)
    down_in_fp8, down_in_scale = silu_mul_fp8(gate, up)
    out = torch.empty((x_bf16.size(0), weights.hidden), device=device, dtype=torch.bfloat16)
    fp8_gemm_nt((down_in_fp8, down_in_scale), (weights.w2_fp8, weights.w2_scale), out, None)
    return out

  def torch_bf16(x_bf16: torch.Tensor) -> torch.Tensor:
    y13 = F.linear(x_bf16, w13_bf16, None)
    gate = y13[:, : weights.inter]
    up = y13[:, weights.inter :]
    act = F.silu(gate) * up
    return F.linear(act, w2_bf16, None)

  def torch_fp8_scaled(x_bf16: torch.Tensor) -> torch.Tensor:
    assert w13_fp8_scaled is not None and w13_scale is not None
    assert w2_fp8_scaled is not None and w2_scale is not None
    x_fp8, x_scale = _quantize_fp8_e4m3_per_tensor(x_bf16)
    y13 = _torch_scaled_mm(
      x_fp8,
      w13_fp8_scaled.T,
      a_scale=x_scale,
      b_scale=w13_scale,
      out_dtype=torch.bfloat16,
    )
    gate = y13[:, : weights.inter]
    up = y13[:, weights.inter :]
    act = F.silu(gate) * up
    act_fp8, act_scale = _quantize_fp8_e4m3_per_tensor(act)
    return _torch_scaled_mm(
      act_fp8,
      w2_fp8_scaled.T,
      a_scale=act_scale,
      b_scale=w2_scale,
      out_dtype=torch.bfloat16,
    )

  def quack_bf16(x_bf16: torch.Tensor) -> torch.Tensor:
    assert quack_gemm is not None
    y13 = quack_gemm(x_bf16, w13_bf16.T.contiguous(), out_dtype=torch.bfloat16)
    gate = y13[:, : weights.inter]
    up = y13[:, weights.inter :]
    act = F.silu(gate) * up
    return quack_gemm(act, w2_bf16.T.contiguous(), out_dtype=torch.bfloat16)

  def cublaslt_bf16(x_bf16: torch.Tensor) -> torch.Tensor:
    assert cublaslt_mod is not None
    y13 = cublaslt_mod.cublaslt_gemm_bf16_cuda(x_bf16, w13_bf16)  # [T, 2*inter]
    gate = y13[:, : weights.inter]
    up = y13[:, weights.inter :]
    act = F.silu(gate) * up
    return cublaslt_mod.cublaslt_gemm_bf16_cuda(act.contiguous(), w2_bf16)  # [T, hidden]

  fn_by_name: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "nmoe_ref_fp8": ref_nmoe_fp8,
    "nmoe_fp8_fused": nmoe_fp8_fused,
    "torch_bf16": torch_bf16,
    "torch_fp8_scaled": torch_fp8_scaled,
    "quack_bf16": quack_bf16,
    "cublaslt_bf16": cublaslt_bf16,
  }

  for T in batches:
    x = (torch.randn((int(T), weights.hidden), device=device, dtype=torch.bfloat16) * 0.01).contiguous()
    print(f"\n== T={T} hidden={weights.hidden} inter={weights.inter} layer={int(args.layer)} ==", flush=True)

    # Compute reference once for correctness comparisons.
    with torch.no_grad():
      ref = ref_nmoe_fp8(x)

    for name in impls:
      if name not in fn_by_name:
        raise ValueError(f"unknown impl: {name!r}")
      fn = fn_by_name[name]
      with torch.no_grad():
        try:
          ms, y = _bench_ms(lambda: fn(x), warmup=int(args.warmup), iters=int(args.iters))
        except Exception as e:
          print(f"{name:>16s}: ERROR: {e}", flush=True)
          continue
        max_abs = float((y - ref).abs().max().item()) if y.shape == ref.shape else float("nan")
        match = _match_frac(ref, y, atol=float(args.atol), rtol=float(args.rtol)) if y.shape == ref.shape else 0.0
        print(f"{name:>16s}: {ms:8.4f} ms   max_abs={max_abs:9.3e}   match={match*100:6.2f}%", flush=True)


if __name__ == "__main__":
  _main()
