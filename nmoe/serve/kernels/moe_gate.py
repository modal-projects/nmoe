# SPDX-License-Identifier: Apache-2.0
"""Fused MoE gate (router) kernels (SM100/B200).

This ports SGLang's `moe_fused_gate` CUDA kernel so we can run DeepSeek-V3
routing (sigmoid + grouped-topk) with minimal launch overhead.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_KERNEL_VERSION = 2
_module = None


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


_CPP_SOURCE = r"""
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> sglang_moe_fused_gate_cuda(
    torch::Tensor input, torch::Tensor bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t num_fused_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);

std::tuple<torch::Tensor, torch::Tensor> sglang_router_bf16_gemm_moe_fused_gate_cuda(
    torch::Tensor x_bf16, torch::Tensor weight_bf16, torch::Tensor bias_f32,
    int64_t num_expert_group, int64_t topk_group, int64_t topk,
    int64_t num_fused_shared_experts, double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);
"""

_WRAPPER_CUDA_SOURCE = r"""
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

std::tuple<torch::Tensor, torch::Tensor> sglang_router_bf16_gemm_moe_fused_gate_cuda(
    torch::Tensor x_bf16, torch::Tensor weight_bf16, torch::Tensor bias_f32,
    int64_t num_expert_group, int64_t topk_group, int64_t topk,
    int64_t num_fused_shared_experts, double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output);
"""

_ROUTER_BF16_GEMM_CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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

cudaError_t ensure_lt(cudaStream_t stream) {
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
  (void)stream;
  return cudaSuccess;
}

struct MatmulKey {
  int opA;
  int opB;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
};

struct MatmulKeyHash {
  size_t operator()(const MatmulKey& x) const noexcept {
    size_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix(static_cast<uint64_t>(x.opA));
    mix(static_cast<uint64_t>(x.opB));
    mix(static_cast<uint64_t>(x.m));
    mix(static_cast<uint64_t>(x.n));
    mix(static_cast<uint64_t>(x.k));
    mix(static_cast<uint64_t>(x.lda));
    mix(static_cast<uint64_t>(x.ldb));
    mix(static_cast<uint64_t>(x.ldc));
    return h;
  }
};

inline bool operator==(const MatmulKey& a, const MatmulKey& b) {
  return a.opA == b.opA && a.opB == b.opB && a.m == b.m && a.n == b.n && a.k == b.k &&
         a.lda == b.lda && a.ldb == b.ldb && a.ldc == b.ldc;
}

struct MatmulPlan {
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulAlgo_t algo{};

  MatmulPlan() = default;
  MatmulPlan(const MatmulPlan&) = delete;
  MatmulPlan& operator=(const MatmulPlan&) = delete;

  ~MatmulPlan() {
    if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
  }
};

std::mutex plan_mu;
std::unordered_map<MatmulKey, std::unique_ptr<MatmulPlan>, MatmulKeyHash> plan_cache;

cudaError_t get_plan(cublasLtHandle_t handle,
                     cublasOperation_t opA,
                     cublasOperation_t opB,
                     int m,
                     int n,
                     int k,
                     int lda,
                     int ldb,
                     int ldc,
                     size_t workspace_bytes,
                     MatmulPlan** out) {
  const MatmulKey key{
      static_cast<int>(opA),
      static_cast<int>(opB),
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
  };

  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto it = plan_cache.find(key);
    if (it != plan_cache.end()) {
      *out = it->second.get();
      return cudaSuccess;
    }
  }

  auto plan = std::make_unique<MatmulPlan>();

  const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type = CUDA_R_32F;
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

  cublasStatus_t st = cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, scale_type);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulDescSetAttribute(
      plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  const int a_rows = (opA == CUBLAS_OP_N) ? m : k;
  const int a_cols = (opA == CUBLAS_OP_N) ? k : m;
  const int b_rows = (opB == CUBLAS_OP_N) ? k : n;
  const int b_cols = (opB == CUBLAS_OP_N) ? n : k;

  st = cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_16BF, a_rows, a_cols, lda);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_16BF, b_rows, b_cols, ldb);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  st = cublasLtMatrixLayoutCreate(&plan->c_layout, CUDA_R_32F, m, n, ldc);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatrixLayoutSetAttribute(
      plan->c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);

  cublasLtMatmulPreference_t pref = nullptr;
  st = cublasLtMatmulPreferenceCreate(&pref);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  st = cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(pref);
    return cuda_from_cublas(st);
  }

  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(
      handle, plan->matmul_desc,
      plan->a_layout, plan->b_layout,
      plan->c_layout, plan->c_layout,
      pref,
      1,
      &heur,
      &returned);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS) return cuda_from_cublas(st);
  if (returned == 0) return cudaErrorUnknown;

  plan->algo = heur.algo;

  {
    std::lock_guard<std::mutex> lock(plan_mu);
    auto [it, inserted] = plan_cache.emplace(key, std::move(plan));
    if (!inserted) {
      *out = it->second.get();
      return cudaSuccess;
    }
    *out = it->second.get();
  }
  return cudaSuccess;
}

cudaError_t lt_gemm_bf16_f32accum_f32out(
    const __nv_bfloat16* A,
    cublasOperation_t opA,
    const __nv_bfloat16* B,
    cublasOperation_t opB,
    float* C,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    float alpha,
    float beta,
    cudaStream_t stream) {
  auto err = ensure_lt(stream);
  if (err != cudaSuccess) return err;

  auto& s = lt_state();

  MatmulPlan* plan = nullptr;
  err = get_plan(s.handle, opA, opB, m, n, k, lda, ldb, ldc, s.workspace_bytes, &plan);
  if (err != cudaSuccess) return err;

  cublasStatus_t st = cublasLtMatmul(
      s.handle,
      plan->matmul_desc,
      &alpha,
      A,
      plan->a_layout,
      B,
      plan->b_layout,
      &beta,
      C,
      plan->c_layout,
      C,
      plan->c_layout,
      &plan->algo,
      s.workspace,
      s.workspace_bytes,
      stream);
  return cuda_from_cublas(st);
}

}  // namespace

// From sgl-kernel moe_fused_gate.cu.
std::vector<at::Tensor> moe_fused_gate(
    at::Tensor& input, at::Tensor& bias, int64_t num_expert_group,
    int64_t topk_group, int64_t topk, int64_t num_fused_shared_experts,
    double routed_scaling_factor, bool apply_routed_scaling_factor_on_output);

std::tuple<torch::Tensor, torch::Tensor> sglang_router_bf16_gemm_moe_fused_gate_cuda(
    torch::Tensor x_bf16, torch::Tensor weight_bf16, torch::Tensor bias_f32,
    int64_t num_expert_group, int64_t topk_group, int64_t topk,
    int64_t num_fused_shared_experts, double routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
  TORCH_CHECK(x_bf16.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight_bf16.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(bias_f32.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(x_bf16.dtype() == torch::kBFloat16, "x must be bf16");
  TORCH_CHECK(weight_bf16.dtype() == torch::kBFloat16, "weight must be bf16");
  TORCH_CHECK(bias_f32.dtype() == torch::kFloat32, "bias must be fp32");
  TORCH_CHECK(x_bf16.dim() == 2, "x must be [T,H]");
  TORCH_CHECK(weight_bf16.dim() == 2, "weight must be [E,H]");
  TORCH_CHECK(bias_f32.dim() == 1, "bias must be [E]");

  const int64_t T = x_bf16.size(0);
  const int64_t H = x_bf16.size(1);
  const int64_t E = weight_bf16.size(0);
  TORCH_CHECK(weight_bf16.size(1) == H, "weight second dim must match x.hidden");
  TORCH_CHECK(bias_f32.size(0) == E, "bias must match num_experts");
  if (T == 0) {
    auto empty_w = torch::empty({0, topk}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto empty_i = torch::empty({0, topk}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    return std::make_tuple(empty_w, empty_i);
  }

  auto logits = torch::empty({T, E}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  auto x_c = x_bf16.contiguous();
  auto w_c = weight_bf16.contiguous();
  auto b_c = bias_f32.contiguous();

  const int m = static_cast<int>(T);
  const int n = static_cast<int>(E);
  const int k = static_cast<int>(H);
  const int lda = k;  // A [m,k] row-major
  const int ldb = k;  // B [n,k] row-major, transposed via opB=T
  const int ldc = n;  // C [m,n] row-major

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto err = lt_gemm_bf16_f32accum_f32out(
      reinterpret_cast<const __nv_bfloat16*>(x_c.data_ptr()),
      CUBLAS_OP_N,
      reinterpret_cast<const __nv_bfloat16*>(w_c.data_ptr()),
      CUBLAS_OP_T,
      reinterpret_cast<float*>(logits.data_ptr()),
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      alpha,
      beta,
      stream);
  TORCH_CHECK(err == cudaSuccess, "cublasLt matmul failed: ", static_cast<int>(err));

  at::Tensor logits_ = logits;
  at::Tensor bias_ = b_c;
  auto out = moe_fused_gate(
      logits_, bias_, num_expert_group, topk_group, topk,
      num_fused_shared_experts, routed_scaling_factor,
      apply_routed_scaling_factor_on_output);
  TORCH_CHECK(out.size() == 2, "moe_fused_gate must return 2 tensors.");
  return std::make_tuple(out[0], out[1]);
}
"""


def _get_module():
  global _module
  if _module is None:
    include_dir = _cutlass_include_dir()
    sglang_src = (Path(__file__).resolve().parent / "sglang_moe_fused_gate.cu").read_text()
    _module = load_inline(
      name=f"sglang_moe_gate_v{_KERNEL_VERSION}",
      cpp_sources=[_CPP_SOURCE],
      cuda_sources=[sglang_src, _ROUTER_BF16_GEMM_CUDA_SOURCE, _WRAPPER_CUDA_SOURCE],
      functions=["sglang_moe_fused_gate_cuda", "sglang_router_bf16_gemm_moe_fused_gate_cuda"],
      extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        f"-I{include_dir}",
      ],
      extra_ldflags=[
        "-lcublasLt",
        "-lcublas",
      ],
      verbose=False,
    )
  return _module


def moe_fused_gate(
  logits: torch.Tensor,
  *,
  bias: torch.Tensor,
  num_expert_group: int,
  topk_group: int,
  topk: int,
  num_fused_shared_experts: int = 0,
  routed_scaling_factor: float,
  apply_routed_scaling_factor_on_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """SGLang moe_fused_gate (sigmoid + bias + grouped-topk in one CUDA op).

  Args:
    logits: [T, E] CUDA tensor (float/bf16/fp16); this is *pre-sigmoid* gate logits.
    bias: [E] CUDA tensor, same dtype as logits.
  Returns:
    topk_weights: float32 [T, topk]
    topk_ids: int32 [T, topk]
  """
  if logits.numel() == 0:
    # Avoid JIT dispatch for empty batches (dynamic disagg / T=0 participation).
    empty_w = torch.empty((logits.size(0), int(topk)), device=logits.device, dtype=torch.float32)
    empty_i = torch.empty((logits.size(0), int(topk)), device=logits.device, dtype=torch.int32)
    return empty_w, empty_i
  if not logits.is_contiguous():
    logits = logits.contiguous()
  if not bias.is_contiguous():
    bias = bias.contiguous()
  return _get_module().sglang_moe_fused_gate_cuda(
    logits,
    bias,
    int(num_expert_group),
    int(topk_group),
    int(topk),
    int(num_fused_shared_experts),
    float(routed_scaling_factor),
    bool(apply_routed_scaling_factor_on_output),
  )


def moe_fused_gate_bf16_gemm(
  x_bf16: torch.Tensor,
  *,
  weight_bf16: torch.Tensor,
  bias: torch.Tensor,
  num_expert_group: int,
  topk_group: int,
  topk: int,
  num_fused_shared_experts: int = 0,
  routed_scaling_factor: float,
  apply_routed_scaling_factor_on_output: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Router GEMM (BF16 inputs, FP32 output) + fused selection.

  This is a router end-to-end CUDA op for DeepSeek-V3-shaped models:
    logits = x @ weight.T   (BF16 inputs, FP32 accumulate/output via cuBLASLt)
    (weights, ids) = moe_fused_gate(logits, bias, ...)

  Args:
    x_bf16: [T, H] bf16 CUDA tensor.
    weight_bf16: [E, H] bf16 CUDA tensor.
    bias: [E] fp32 CUDA tensor (DeepSeek-V3 gate bias).
  Returns:
    topk_weights: float32 [T, topk]
    topk_ids: int32 [T, topk]
  """
  if x_bf16.numel() == 0:
    empty_w = torch.empty((x_bf16.size(0), int(topk)), device=x_bf16.device, dtype=torch.float32)
    empty_i = torch.empty((x_bf16.size(0), int(topk)), device=x_bf16.device, dtype=torch.int32)
    return empty_w, empty_i
  if not x_bf16.is_contiguous():
    x_bf16 = x_bf16.contiguous()
  if not weight_bf16.is_contiguous():
    weight_bf16 = weight_bf16.contiguous()
  if not bias.is_contiguous():
    bias = bias.contiguous()
  return _get_module().sglang_router_bf16_gemm_moe_fused_gate_cuda(
    x_bf16,
    weight_bf16,
    bias,
    int(num_expert_group),
    int(topk_group),
    int(topk),
    int(num_fused_shared_experts),
    float(routed_scaling_factor),
    bool(apply_routed_scaling_factor_on_output),
  )
