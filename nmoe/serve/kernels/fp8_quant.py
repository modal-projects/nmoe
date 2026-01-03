# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 quantization kernel with UE8M0 scales.

This kernel fuses the following operations into a single pass:
1. Compute per-block (128 elements) max absolute value
2. Compute UE8M0 scale (power-of-2)
3. Quantize to FP8 e4m3fn

Replaces the naive PyTorch implementation which was 60% of MoE time.
"""

import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel source - TransformerEngine-style optimized
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Block size for quantization (DeepGEMM requirement)
constexpr int BLOCK_SIZE = 128;
constexpr int THREADS_PER_BLOCK = 128;
constexpr int ELEMS_PER_THREAD = 1;  // Each thread handles 1 element in the chunk

// Warp reduction for max (optimized)
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Force scale to power-of-2 using bit masking (TransformerEngine pattern)
__device__ __forceinline__ float force_pow2_scale(float scale) {
    // Mask: 0xFF800000 keeps sign + exponent, zeros mantissa
    // This forces scale to nearest power-of-2
    uint32_t bits = *reinterpret_cast<uint32_t*>(&scale);
    // Round up: if mantissa was non-zero, increment exponent
    if (bits & 0x007FFFFF) {
        bits = (bits & 0xFF800000) + 0x00800000;  // Add 1 to exponent
    } else {
        bits = bits & 0xFF800000;
    }
    return *reinterpret_cast<float*>(&bits);
}

// Kernel: quantize BF16 to FP8 with per-128-element UE8M0 scales
// 128 threads per block, each block handles one 128-element chunk
// Simple and efficient - no complex grid-stride needed
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void quantize_fp8_ue8m0_kernel(
    const __nv_bfloat16* __restrict__ input,  // [T, K]
    __nv_fp8_e4m3* __restrict__ output,       // [T, K]
    float* __restrict__ scales,               // [T * K/128]
    int total_elements
) {
    const int chunk_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int global_idx = chunk_idx * BLOCK_SIZE + lane;

    // Load input (coalesced access)
    float val = 0.0f;
    if (global_idx < total_elements) {
        val = __bfloat162float(input[global_idx]);
    }
    float abs_val = fabsf(val);

    // Warp-level reduction for max (4 warps = 128 threads)
    float warp_max = warp_reduce_max(abs_val);

    // Cross-warp reduction via shared memory
    __shared__ float smem[4];
    if ((lane & 31) == 0) {
        smem[lane >> 5] = warp_max;
    }
    __syncthreads();

    // Final reduction in first warp
    float block_max;
    if (lane < 4) {
        block_max = smem[lane];
    } else {
        block_max = 0.0f;
    }
    block_max = warp_reduce_max(block_max);

    // Broadcast block_max to ALL threads (not just warp 0)
    if (lane == 0) {
        smem[0] = block_max;
    }
    __syncthreads();
    block_max = smem[0];

    // Compute UE8M0 scale matching PyTorch reference:
    // 1. clamp(amax, min=1e-4)
    // 2. scale = amax / 448.0
    // 3. scale = pow2_ceil(scale) via bit manipulation
    block_max = fmaxf(block_max, 1e-4f);  // Clamp FIRST
    float raw_scale = block_max / 448.0f;
    float scale = force_pow2_scale(raw_scale);

    // Write scale (one thread per chunk)
    if (lane == 0) {
        scales[chunk_idx] = scale;
    }

    // Quantize and write output
    if (global_idx < total_elements) {
        float quantized = val / scale;
        // Clamp to FP8 E4M3 range and convert using proper CUDA intrinsic
        quantized = fminf(fmaxf(quantized, -448.0f), 448.0f);
        // Use __nv_cvt_float_to_fp8 with saturation for proper conversion
        output[global_idx].__x = __nv_cvt_float_to_fp8(quantized, __NV_SATFINITE, __NV_E4M3);
    }
}

// Wrapper function
std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_ue8m0_cuda(
    torch::Tensor input  // [T, K] bfloat16
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kBFloat16, "Input must be bfloat16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    const int T = input.size(0);
    const int K = input.size(1);
    TORCH_CHECK(K % BLOCK_SIZE == 0, "K must be divisible by 128");

    auto output = torch::empty({T, K}, torch::dtype(torch::kFloat8_e4m3fn).device(input.device()));
    auto scales = torch::empty({T, K / BLOCK_SIZE}, torch::dtype(torch::kFloat32).device(input.device()));

    const int total_chunks = T * (K / BLOCK_SIZE);
    const int total_elements = T * K;

    // Simple launch: one block per 128-element chunk
    quantize_fp8_ue8m0_kernel<<<total_chunks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        scales.data_ptr<float>(),
        total_elements
    );

    return std::make_tuple(output, scales);
}
"""

_CPP_SOURCE = r"""
std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_ue8m0_cuda(torch::Tensor input);
"""

# JIT compile the kernel
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="fp8_quant_kernel",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["quantize_fp8_ue8m0_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def quantize_fp8_ue8m0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 tensor to FP8 e4m3fn with UE8M0 block scales.

    Args:
        x: [T, K] bfloat16 tensor, K must be divisible by 128

    Returns:
        x_fp8: [T, K] float8_e4m3fn tensor
        scales: [T, K//128] float32 tensor (power-of-2 scales)
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return _get_module().quantize_fp8_ue8m0_cuda(x)


# Fallback PyTorch implementation for testing
def quantize_fp8_ue8m0_pytorch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    T, K = x.shape
    assert K % 128 == 0
    x_view = x.view(T, K // 128, 128)
    scales = x_view.float().abs().amax(dim=-1).clamp(min=1e-4) / 448.0
    scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))
    x_q = (x_view.float() / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return x_q.view(T, K), scales


# =============================================================================
# Fused weighted scatter-add kernel
# =============================================================================
# Replaces: y.index_add_(0, token_ids, expert_out * weights.unsqueeze(1))
# This was 21% of MoE time (1.51ms per layer)

_SCATTER_ADD_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// SM100 (B200) optimized weighted scatter-add
// Uses native bf16 atomics and vectorized memory access
// For each token i: output[token_ids[i]] += expert_out[i] * weights[i]

__global__ void weighted_scatter_add_sm100_kernel(
    const __nv_bfloat162* __restrict__ expert_out,  // [N, hidden/2] as bf16x2
    const int64_t* __restrict__ token_ids,          // [N]
    const float* __restrict__ weights,              // [N]
    __nv_bfloat162* __restrict__ output,            // [T, hidden/2] as bf16x2
    int N,
    int hidden_half  // hidden / 2
) {
    // Grid-stride loop for better occupancy
    for (int token_idx = blockIdx.x; token_idx < N; token_idx += gridDim.x) {
        const int dst_row = token_ids[token_idx];
        const __nv_bfloat162 w2 = __float2bfloat162_rn(weights[token_idx]);

        // Vectorized: process 2 bf16 elements at a time
        for (int h = threadIdx.x; h < hidden_half; h += blockDim.x) {
            __nv_bfloat162 val = expert_out[token_idx * hidden_half + h];
            val = __hmul2(val, w2);  // Multiply both elements by weight

            // SM100 native bf16x2 atomic add
            __nv_bfloat162* dst = &output[dst_row * hidden_half + h];
            atomicAdd(dst, val);
        }
    }
}

void weighted_scatter_add_cuda(
    torch::Tensor expert_out,   // [N, hidden] bfloat16
    torch::Tensor token_ids,    // [N] int64
    torch::Tensor weights,      // [N] float32
    torch::Tensor output        // [T, hidden] bfloat16 (modified in place)
) {
    TORCH_CHECK(expert_out.is_cuda(), "expert_out must be CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");

    TORCH_CHECK(expert_out.dtype() == torch::kBFloat16, "expert_out must be bfloat16");
    TORCH_CHECK(token_ids.dtype() == torch::kInt64, "token_ids must be int64");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");

    TORCH_CHECK(expert_out.is_contiguous(), "expert_out must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    const int N = expert_out.size(0);
    const int hidden = expert_out.size(1);

    TORCH_CHECK(hidden % 2 == 0, "hidden must be even for vectorized access");

    if (N == 0) return;

    const int threads = 256;
    const int blocks = std::min(N, 65535);

    weighted_scatter_add_sm100_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat162*>(expert_out.data_ptr()),
        token_ids.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr()),
        N,
        hidden / 2
    );
}
"""

_SCATTER_ADD_CPP_SOURCE = r"""
void weighted_scatter_add_cuda(
    torch::Tensor expert_out,
    torch::Tensor token_ids,
    torch::Tensor weights,
    torch::Tensor output
);
"""

_scatter_module = None

def _get_scatter_module():
    global _scatter_module
    if _scatter_module is None:
        _scatter_module = load_inline(
            name="scatter_add_kernel",
            cpp_sources=[_SCATTER_ADD_CPP_SOURCE],
            cuda_sources=[_SCATTER_ADD_CUDA_SOURCE],
            functions=["weighted_scatter_add_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _scatter_module


def weighted_scatter_add(
    expert_out: torch.Tensor,  # [N, hidden] bfloat16
    token_ids: torch.Tensor,   # [N] int64
    weights: torch.Tensor,     # [N] float32
    output: torch.Tensor,      # [T, hidden] bfloat16 (modified in place)
) -> None:
    """Fused weighted scatter-add: output[token_ids[i]] += expert_out[i] * weights[i]

    This replaces the slow pattern:
        y.index_add_(0, token_ids, expert_out * weights.unsqueeze(1))
    """
    _get_scatter_module().weighted_scatter_add_cuda(expert_out, token_ids, weights, output)


# =============================================================================
# Fused SiLU * UP + FP8 Quantization kernel
# =============================================================================
# Replaces: (F.silu(gate.float()) * up.float()).to(bf16) + quantize_fp8_ue8m0()
# This was ~400ms per forward (silu+mul+cast+quant called 116 times)

_SILU_MUL_QUANT_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

constexpr int BLOCK_SIZE = 128;
constexpr int THREADS_PER_BLOCK = 128;

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float force_pow2_scale(float scale) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&scale);
    if (bits & 0x007FFFFF) {
        bits = (bits & 0xFF800000) + 0x00800000;
    } else {
        bits = bits & 0xFF800000;
    }
    return *reinterpret_cast<float*>(&bits);
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Fused kernel: silu(gate) * up -> FP8 with UE8M0 scales
// Input: gate [T, K], up [T, K] both bfloat16
// Output: out_fp8 [T, K], scales [T, K/128]
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void silu_mul_fp8_kernel(
    const __nv_bfloat16* __restrict__ gate,   // [T, K]
    const __nv_bfloat16* __restrict__ up,     // [T, K]
    __nv_fp8_e4m3* __restrict__ output,       // [T, K]
    float* __restrict__ scales,               // [T * K/128]
    int total_elements
) {
    const int chunk_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int global_idx = chunk_idx * BLOCK_SIZE + lane;

    // Load and compute silu(gate) * up
    float val = 0.0f;
    if (global_idx < total_elements) {
        float g = __bfloat162float(gate[global_idx]);
        float u = __bfloat162float(up[global_idx]);
        val = silu(g) * u;
    }
    float abs_val = fabsf(val);

    // Warp reduction for max
    float warp_max = warp_reduce_max(abs_val);

    // Cross-warp reduction
    __shared__ float smem[4];
    if ((lane & 31) == 0) {
        smem[lane >> 5] = warp_max;
    }
    __syncthreads();

    float block_max;
    if (lane < 4) {
        block_max = smem[lane];
    } else {
        block_max = 0.0f;
    }
    block_max = warp_reduce_max(block_max);

    // Broadcast to all threads
    if (lane == 0) {
        smem[0] = block_max;
    }
    __syncthreads();
    block_max = smem[0];

    // Compute UE8M0 scale
    block_max = fmaxf(block_max, 1e-4f);
    float raw_scale = block_max / 448.0f;
    float scale = force_pow2_scale(raw_scale);

    if (lane == 0) {
        scales[chunk_idx] = scale;
    }

    // Quantize to FP8
    if (global_idx < total_elements) {
        float quantized = val / scale;
        quantized = fminf(fmaxf(quantized, -448.0f), 448.0f);
        output[global_idx].__x = __nv_cvt_float_to_fp8(quantized, __NV_SATFINITE, __NV_E4M3);
    }
}

std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_cuda(
    torch::Tensor gate,  // [T, K] bfloat16
    torch::Tensor up     // [T, K] bfloat16
) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gate.dtype() == torch::kBFloat16 && up.dtype() == torch::kBFloat16, "Inputs must be bfloat16");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have same shape");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "Inputs must be contiguous");

    const int T = gate.size(0);
    const int K = gate.size(1);
    TORCH_CHECK(K % BLOCK_SIZE == 0, "K must be divisible by 128");

    auto output = torch::empty({T, K}, torch::dtype(torch::kFloat8_e4m3fn).device(gate.device()));
    auto scales = torch::empty({T, K / BLOCK_SIZE}, torch::dtype(torch::kFloat32).device(gate.device()));

    const int total_chunks = T * (K / BLOCK_SIZE);
    const int total_elements = T * K;

    silu_mul_fp8_kernel<<<total_chunks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        scales.data_ptr<float>(),
        total_elements
    );

    return std::make_tuple(output, scales);
}
"""

_SILU_MUL_QUANT_CPP_SOURCE = r"""
std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_cuda(torch::Tensor gate, torch::Tensor up);
"""

_silu_mul_module = None

def _get_silu_mul_module():
    global _silu_mul_module
    if _silu_mul_module is None:
        _silu_mul_module = load_inline(
            name="silu_mul_fp8_kernel",
            cpp_sources=[_SILU_MUL_QUANT_CPP_SOURCE],
            cuda_sources=[_SILU_MUL_QUANT_CUDA_SOURCE],
            functions=["silu_mul_fp8_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _silu_mul_module


def silu_mul_fp8(gate: torch.Tensor, up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU(gate) * up -> FP8 with UE8M0 scales.

    Replaces the slow pattern:
        down_in = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        down_in_q, down_in_scale = quantize_fp8_ue8m0(down_in)

    Args:
        gate: [T, K] bfloat16 tensor
        up: [T, K] bfloat16 tensor, K must be divisible by 128

    Returns:
        out_fp8: [T, K] float8_e4m3fn tensor
        scales: [T, K//128] float32 tensor (power-of-2 scales)
    """
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not up.is_contiguous():
        up = up.contiguous()
    return _get_silu_mul_module().silu_mul_fp8_cuda(gate, up)
