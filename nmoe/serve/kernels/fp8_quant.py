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
#include <ATen/cuda/CUDAContext.h>

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
    quantize_fp8_ue8m0_kernel<<<total_chunks, THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        scales.data_ptr<float>(),
        total_elements
    );

    return std::make_tuple(output, scales);
}

// Pack FP32 UE8M0 scaling factors into DeepGEMM's required (INT, MN-major, TMA-aligned) layout.
//
// Input:
//   sf: [ceil_div(mn, gran_mn_in), ceil_div(k, 128)] float32, row-major or contiguous.
//
// Output:
//   packed: [mn, ceil_div(ceil_div(k,128), 4)] int32 with stride(-2)=1 and stride(-1)=aligned_mn,
//   where aligned_mn = align(mn, 4) (16B TMA alignment for int32 elements).
//
// Semantics match DeepGEMM's get_mn_major_tma_aligned_packed_ue8m0_tensor_torch:
//   ue8m0_byte = (reinterpret_cast<int32_t&>(sf) >> 23) & 0xFF
// packed int stores 4 consecutive ue8m0 bytes in little-endian order.
__global__ void pack_fp32_sf_to_ue8m0_int_kernel(
    const float* __restrict__ sf,     // [num_groups, mn_in, sf_k] float32 (row-major)
    int32_t* __restrict__ out,        // [num_groups, aligned_mn, packed_sf_k] int32 (MN-major)
    int num_groups,
    int mn,
    int mn_in,
    int sf_k,
    int packed_sf_k,
    int gran_mn_in,
    int aligned_mn
) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int per_group = mn * packed_sf_k;
    const int total = num_groups * per_group;
    if (idx >= total) {
        return;
    }

    const int g = idx / per_group;
    const int rem = idx - g * per_group;
    const int row = rem % mn;
    const int col = rem / mn;

    const int src_row = row / gran_mn_in;
    const int base_k = col * 4;

    uint32_t packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k_idx = base_k + i;
        uint32_t byte = 0;
        if (k_idx < sf_k) {
            const float v = sf[(g * mn_in + src_row) * sf_k + k_idx];
            const uint32_t bits = __float_as_uint(v);
            byte = (bits >> 23) & 0xFFu;
        }
        packed |= (byte << (8 * i));
    }

    out[g * (aligned_mn * packed_sf_k) + row + col * aligned_mn] = static_cast<int32_t>(packed);
}

torch::Tensor pack_fp32_sf_to_ue8m0_int_cuda(
    torch::Tensor sf,  // [mn_in, sf_k] float32
    int64_t mn,
    int64_t k,
    int64_t gran_mn_in
) {
    TORCH_CHECK(sf.is_cuda(), "sf must be CUDA tensor");
    TORCH_CHECK(sf.dtype() == torch::kFloat32, "sf must be float32");
    TORCH_CHECK(sf.dim() == 2 || sf.dim() == 3, "sf must be 2D [mn_in, sf_k] or 3D [num_groups, mn_in, sf_k]");
    TORCH_CHECK(sf.is_contiguous(), "sf must be contiguous");
    TORCH_CHECK(mn > 0 && k > 0, "mn and k must be > 0");
    TORCH_CHECK(gran_mn_in == 1 || gran_mn_in == 128, "gran_mn_in must be 1 or 128");

    const int mn_i = static_cast<int>(mn);
    const int k_i = static_cast<int>(k);
    const int sf_k = (k_i + 128 - 1) / 128;
    TORCH_CHECK(sf.size(-1) == sf_k, "sf.size(-1) must equal ceil_div(k,128)");

    const int num_groups = (sf.dim() == 3) ? static_cast<int>(sf.size(0)) : 1;
    const int mn_in = (sf.dim() == 3) ? static_cast<int>(sf.size(1)) : static_cast<int>(sf.size(0));
    TORCH_CHECK(mn_in * static_cast<int>(gran_mn_in) >= mn_i, "sf mn_in * gran_mn_in must cover mn");

    const int packed_sf_k = (sf_k + 4 - 1) / 4;
    const int aligned_mn = ((mn_i + 3) / 4) * 4;  // 16B TMA alignment for int32 elements

    torch::Tensor out;
    if (sf.dim() == 2) {
        out = torch::empty_strided(
            {mn_i, packed_sf_k},
            {1, aligned_mn},
            torch::dtype(torch::kInt32).device(sf.device())
        );
    } else {
        out = torch::empty_strided(
            {num_groups, mn_i, packed_sf_k},
            {aligned_mn * packed_sf_k, 1, aligned_mn},
            torch::dtype(torch::kInt32).device(sf.device())
        );
    }

    const int threads = 256;
    const int total = num_groups * mn_i * packed_sf_k;
    const int blocks = (total + threads - 1) / threads;
    pack_fp32_sf_to_ue8m0_int_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        sf.data_ptr<float>(),
        out.data_ptr<int32_t>(),
        num_groups,
        mn_i,
        mn_in,
        sf_k,
        packed_sf_k,
        static_cast<int>(gran_mn_in),
        aligned_mn
    );

    return out;
}
"""

_CPP_SOURCE = r"""
std::tuple<torch::Tensor, torch::Tensor> quantize_fp8_ue8m0_cuda(torch::Tensor input);
torch::Tensor pack_fp32_sf_to_ue8m0_int_cuda(torch::Tensor sf, int64_t mn, int64_t k, int64_t gran_mn_in);
"""

# JIT compile the kernel
_KERNEL_VERSION = 3
_MOE_FUSED_KERNEL_VERSION = 7
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name=f"fp8_quant_kernel_v{_KERNEL_VERSION}",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=["quantize_fp8_ue8m0_cuda", "pack_fp32_sf_to_ue8m0_int_cuda"],
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


def pack_fp32_ue8m0_scales_to_int(sf: torch.Tensor, *, mn: int, k: int, gran_mn_in: int) -> torch.Tensor:
    """Pack FP32 UE8M0 scales into DeepGEMM's required INT layout (SM100 fast-path).

    This matches DeepGEMM's internal pack: float bits >> 23 to get UE8M0 exponent bytes,
    then pack 4 bytes into one int32, and store as an MN-major, TMA-aligned matrix.

    Args:
      sf: FP32 UE8M0 scaling factors:
        - 2D: [mn_in, ceil_div(k,128)]
        - 3D: [num_groups, mn_in, ceil_div(k,128)]
        where mn_in = ceil_div(mn, gran_mn_in).
      mn: logical number of rows in the corresponding value matrix (A or B operand).
      k:  logical K dimension of the corresponding value matrix.
      gran_mn_in: row granularity of `sf` (1 for per-row scales, 128 for per-128-row scales).

    Returns:
      packed INT UE8M0 scales in DeepGEMM-required MN-major, TMA-aligned layout:
        - 2D input -> [mn, ceil_div(ceil_div(k,128),4)] int32 with stride(-2)=1 and stride(-1)=align(mn,4).
        - 3D input -> [num_groups, mn, ceil_div(ceil_div(k,128),4)] int32 with strides
          (align(mn,4)*packed_k, 1, align(mn,4)).
    """
    if sf.dtype != torch.float32:
        raise TypeError(f"sf must be float32, got {sf.dtype}")
    if not sf.is_cuda:
        raise TypeError("sf must be a CUDA tensor")
    if int(mn) == 0:
        # Empty fast-path (dynamic disagg / T=0 participation). Avoid calling into
        # the extension which asserts mn>0.
        sf_k = (int(k) + 128 - 1) // 128
        packed_sf_k = (sf_k + 4 - 1) // 4
        return torch.empty((0, packed_sf_k), device=sf.device, dtype=torch.int32)
    if not sf.is_contiguous():
        sf = sf.contiguous()
    return _get_module().pack_fp32_sf_to_ue8m0_int_cuda(sf, int(mn), int(k), int(gran_mn_in))


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
#include <ATen/cuda/CUDAContext.h>

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

    weighted_scatter_add_sm100_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat162*>(expert_out.data_ptr()),
        token_ids.data_ptr<int64_t>(),
        weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr()),
        N,
        hidden / 2
    );
}

// Indexed variant: output[token_ids[i]] += src[src_indices[i]] * weights[i]
__global__ void weighted_scatter_add_indexed_sm100_kernel(
    const __nv_bfloat162* __restrict__ src,          // [M, hidden/2] as bf16x2
    const int64_t* __restrict__ src_indices,         // [N]
    const int64_t* __restrict__ token_ids,           // [N]
    const float* __restrict__ weights,               // [N]
    __nv_bfloat162* __restrict__ output,             // [T, hidden/2] as bf16x2
    int N,
    int hidden_half  // hidden / 2
) {
    for (int pair_idx = blockIdx.x; pair_idx < N; pair_idx += gridDim.x) {
        const int64_t src_row = src_indices[pair_idx];
        const int dst_row = static_cast<int>(token_ids[pair_idx]);
        const __nv_bfloat162 w2 = __float2bfloat162_rn(weights[pair_idx]);

        const __nv_bfloat162* src_row_ptr = src + src_row * static_cast<int64_t>(hidden_half);
        __nv_bfloat162* dst_row_ptr = output + static_cast<int64_t>(dst_row) * static_cast<int64_t>(hidden_half);

        for (int h = threadIdx.x; h < hidden_half; h += blockDim.x) {
            __nv_bfloat162 val = src_row_ptr[h];
            val = __hmul2(val, w2);
            atomicAdd(&dst_row_ptr[h], val);
        }
    }
}

void weighted_scatter_add_indexed_cuda(
    torch::Tensor src,          // [M, hidden] bfloat16
    torch::Tensor src_indices,  // [N] int64
    torch::Tensor token_ids,    // [N] int64
    torch::Tensor weights,      // [N] float32
    torch::Tensor output        // [T, hidden] bfloat16 (modified in place)
) {
    TORCH_CHECK(src.is_cuda(), "src must be CUDA tensor");
    TORCH_CHECK(src_indices.is_cuda(), "src_indices must be CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");

    TORCH_CHECK(src.dtype() == torch::kBFloat16, "src must be bfloat16");
    TORCH_CHECK(src_indices.dtype() == torch::kInt64, "src_indices must be int64");
    TORCH_CHECK(token_ids.dtype() == torch::kInt64, "token_ids must be int64");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");

    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(src_indices.is_contiguous(), "src_indices must be contiguous");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(src.dim() == 2, "src must be 2D");
    TORCH_CHECK(output.dim() == 2, "output must be 2D");
    TORCH_CHECK(src_indices.dim() == 1, "src_indices must be 1D");
    TORCH_CHECK(token_ids.dim() == 1, "token_ids must be 1D");
    TORCH_CHECK(weights.dim() == 1, "weights must be 1D");

    const int N = token_ids.size(0);
    TORCH_CHECK(src_indices.size(0) == N, "src_indices must match token_ids length");
    TORCH_CHECK(weights.size(0) == N, "weights must match token_ids length");
    if (N == 0) return;

    const int hidden = src.size(1);
    TORCH_CHECK(hidden == output.size(1), "src/output hidden mismatch");
    TORCH_CHECK(hidden % 2 == 0, "hidden must be even for vectorized access");

    const int threads = 256;
    const int blocks = std::min(N, 65535);

    weighted_scatter_add_indexed_sm100_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat162*>(src.data_ptr()),
        src_indices.data_ptr<int64_t>(),
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

void weighted_scatter_add_indexed_cuda(
    torch::Tensor src,
    torch::Tensor src_indices,
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
            name=f"scatter_add_kernel_v{_KERNEL_VERSION}",
            cpp_sources=[_SCATTER_ADD_CPP_SOURCE],
            cuda_sources=[_SCATTER_ADD_CUDA_SOURCE],
            functions=["weighted_scatter_add_cuda", "weighted_scatter_add_indexed_cuda"],
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


def weighted_scatter_add_indexed(
    src: torch.Tensor,          # [M, hidden] bfloat16
    src_indices: torch.Tensor,  # [N] int64
    token_ids: torch.Tensor,    # [N] int64
    weights: torch.Tensor,      # [N] float32
    output: torch.Tensor,       # [T, hidden] bfloat16 (modified in place)
) -> None:
    """Indexed weighted scatter-add: output[token_ids[i]] += src[src_indices[i]] * weights[i]."""
    _get_scatter_module().weighted_scatter_add_indexed_cuda(src, src_indices, token_ids, weights, output)


# =============================================================================
# Fused SiLU * UP + FP8 Quantization kernel
# =============================================================================
# Replaces: (F.silu(gate.float()) * up.float()).to(bf16) + quantize_fp8_ue8m0()
# This was ~400ms per forward (silu+mul+cast+quant called 116 times)

_SILU_MUL_QUANT_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>

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

    silu_mul_fp8_kernel<<<total_chunks, THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
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
            name=f"silu_mul_fp8_kernel_v{_KERNEL_VERSION}",
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


# =============================================================================
# MoE fused pack + grouped SiLU + grouped scatter-add
# =============================================================================

_MOE_FUSED_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

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

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// One block per (token, k) pair.
// Copies FP8 activation bytes + float32 scales into per-expert grouped layout.
template <typename IdxT>
__global__ void moe_pack_fp8_grouped_sm100_kernel(
    const uint8_t* __restrict__ recv_x_fp8,         // [num_recv, hidden] FP8 bytes
    const float* __restrict__ recv_x_scale,         // [num_recv, scale_blocks] float32
    const IdxT* __restrict__ recv_topk_idx,          // [num_recv, topk] local expert ids or -1
    const float* __restrict__ recv_topk_weights,     // [num_recv, topk] float32
    uint8_t* __restrict__ x_grouped,                // [num_local, expected_m, hidden] FP8 bytes
    float* __restrict__ scale_grouped,              // [num_local, expected_m, scale_blocks] float32
    int32_t* __restrict__ token_ids_grouped,         // [num_local, expected_m] int32
    float* __restrict__ weights_grouped,             // [num_local, expected_m] float32
    int32_t* __restrict__ masked_m,                  // [num_local] int32 (initialized to 0)
    int32_t* __restrict__ overflow_flag,             // [1] int32 (0/1)
    int num_recv,
    int hidden,
    int scale_blocks,
    int topk,
    int expected_m
) {
    const int pair = static_cast<int>(blockIdx.x);
    const int token = pair / topk;
    if (token >= num_recv) {
        return;
    }

    const int expert = static_cast<int>(recv_topk_idx[pair]);
    if (expert < 0) {
        return;
    }

    // Reserve a row for this expert exactly once per (token, k) pair.
    // NOTE: One block corresponds to one pair, so only lane0 should mutate
    // masked_m. Other threads consume the slot via shared memory.
    __shared__ int slot_s;
    __shared__ int ok_s;
    if (threadIdx.x == 0) {
        // Saturating counter: clip per-expert rows to expected_m.
        // This enforces masked_m[expert] <= expected_m without a separate clamp kernel,
        // and drops overflowed pairs (counted in overflow_flag).
        int slot = -1;
        int ok = 0;
        while (true) {
            const int cur = masked_m[expert];
            if (cur >= expected_m) {
                atomicAdd(overflow_flag, 1);
                ok = 0;
                break;
            }
            const int prev = atomicCAS(&masked_m[expert], cur, cur + 1);
            if (prev == cur) {
                slot = cur;
                ok = 1;
                break;
            }
        }
        slot_s = slot;
        ok_s = ok;
    }
    __syncthreads();
    if (!ok_s) {
        return;
    }
    const int slot = slot_s;

    const int row = expert * expected_m + slot;

    // Copy FP8 activation bytes.
    const uint8_t* __restrict__ x_src = recv_x_fp8 + static_cast<size_t>(token) * static_cast<size_t>(hidden);
    uint8_t* __restrict__ x_dst = x_grouped + static_cast<size_t>(row) * static_cast<size_t>(hidden);
    for (int i = static_cast<int>(threadIdx.x); i < hidden; i += static_cast<int>(blockDim.x)) {
        x_dst[i] = x_src[i];
    }

    // Copy per-block scales.
    const float* __restrict__ s_src = recv_x_scale + static_cast<size_t>(token) * static_cast<size_t>(scale_blocks);
    float* __restrict__ s_dst = scale_grouped + static_cast<size_t>(row) * static_cast<size_t>(scale_blocks);
    for (int j = static_cast<int>(threadIdx.x); j < scale_blocks; j += static_cast<int>(blockDim.x)) {
        s_dst[j] = s_src[j];
    }

    if (threadIdx.x == 0) {
        token_ids_grouped[row] = static_cast<int32_t>(token);
        weights_grouped[row] = recv_topk_weights[pair];
    }
}

// Grouped SiLU(gate) * up -> FP8 with UE8M0 scales.
// gateup_out: [num_local, expected_m, 2*K] BF16
// output:     [num_local, expected_m, K] FP8
// scales:     [num_local, expected_m, K/128] float32
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void silu_mul_fp8_grouped_kernel(
    const __nv_bfloat16* __restrict__ gateup,  // [rows, 2*K]
    const int32_t* __restrict__ masked_m,      // [num_local]
    __nv_fp8_e4m3* __restrict__ output,        // [rows, K]
    float* __restrict__ scales,                // [rows * (K/128)]
    int expected_m,
    int K,
    int chunks_per_row
) {
    const int chunk_idx = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    const int row = chunk_idx / chunks_per_row;
    const int chunk_in_row = chunk_idx - row * chunks_per_row;

    const int expert = row / expected_m;
    const int slot = row - expert * expected_m;
    if (slot >= masked_m[expert]) {
        return;
    }

    const int col = chunk_in_row * BLOCK_SIZE + lane;
    const int gateup_row_stride = 2 * K;
    const int base = row * gateup_row_stride + col;

    float val = 0.0f;
    // K is divisible by 128; col < K always holds for this kernel launch.
    {
        float g = __bfloat162float(gateup[base]);
        float u = __bfloat162float(gateup[base + K]);
        val = silu(g) * u;
    }

    float abs_val = fabsf(val);

    float warp_max = warp_reduce_max(abs_val);

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

    if (lane == 0) {
        smem[0] = block_max;
    }
    __syncthreads();
    block_max = smem[0];

    block_max = fmaxf(block_max, 1e-4f);
    float raw_scale = block_max / 448.0f;
    float scale = force_pow2_scale(raw_scale);

    if (lane == 0) {
        scales[chunk_idx] = scale;
    }

    float quantized = val / scale;
    quantized = fminf(fmaxf(quantized, -448.0f), 448.0f);
    output[row * K + col].__x = __nv_cvt_float_to_fp8(quantized, __NV_SATFINITE, __NV_E4M3);
}

__global__ void weighted_scatter_add_grouped_sm100_kernel(
    const __nv_bfloat162* __restrict__ expert_out,  // [num_local * expected_m, hidden/2]
    const int32_t* __restrict__ token_ids,          // [num_local * expected_m]
    const float* __restrict__ weights,              // [num_local * expected_m]
    const int32_t* __restrict__ masked_m,           // [num_local]
    __nv_bfloat162* __restrict__ output,            // [num_recv, hidden/2]
    int expected_m,
    int hidden_half
) {
    const int row = static_cast<int>(blockIdx.x);
    const int expert = row / expected_m;
    const int slot = row - expert * expected_m;
    if (slot >= masked_m[expert]) {
        return;
    }
    const int dst_row = static_cast<int>(token_ids[row]);
    const __nv_bfloat162 w2 = __float2bfloat162_rn(weights[row]);

    for (int h = static_cast<int>(threadIdx.x); h < hidden_half; h += static_cast<int>(blockDim.x)) {
        __nv_bfloat162 val = expert_out[row * hidden_half + h];
        val = __hmul2(val, w2);
        __nv_bfloat162* dst = &output[dst_row * hidden_half + h];
        atomicAdd(dst, val);
    }
}

// Grouped SiLU(gate) * up -> FP8 with UE8M0 scales packed into DeepGEMM int layout.
//
// gateup_out:     [num_local, expected_m, 2*K] BF16
// output:         [num_local, expected_m, K] FP8
// scales_packed:  [num_local, expected_m, packed_k] int32 with DeepGEMM MN-major layout:
//   strides = (align(expected_m,4)*packed_k, 1, align(expected_m,4))
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void silu_mul_fp8_grouped_packed_kernel(
    const __nv_bfloat16* __restrict__ gateup,  // [rows, 2*K]
    const int32_t* __restrict__ masked_m,      // [num_local]
    __nv_fp8_e4m3* __restrict__ output,        // [rows, K]
    uint32_t* __restrict__ scales_packed,      // [num_local * align_mn * packed_k]
    int expected_m,
    int K,
    int chunks_per_row,
    int packed_k,
    int align_mn
) {
    const int chunk_idx = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    const int row = chunk_idx / chunks_per_row;
    const int chunk_in_row = chunk_idx - row * chunks_per_row;

    const int expert = row / expected_m;
    const int slot = row - expert * expected_m;
    if (slot >= masked_m[expert]) {
        return;
    }

    const int col = chunk_in_row * BLOCK_SIZE + lane;
    const int gateup_row_stride = 2 * K;
    const int base = row * gateup_row_stride + col;

    float val = 0.0f;
    {
        float g = __bfloat162float(gateup[base]);
        float u = __bfloat162float(gateup[base + K]);
        val = silu(g) * u;
    }

    float abs_val = fabsf(val);
    float warp_max = warp_reduce_max(abs_val);

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

    if (lane == 0) {
        smem[0] = block_max;
    }
    __syncthreads();
    block_max = smem[0];

    block_max = fmaxf(block_max, 1e-4f);
    float raw_scale = block_max / 448.0f;
    float scale = force_pow2_scale(raw_scale);

    // Pack UE8M0 exponent bytes into DeepGEMM's int layout on lane0.
    if (lane == 0) {
        // Convert scale float bits to UE8M0 exponent byte.
        // pack_fp32_ue8m0_scales_to_int uses (bits >> 23) to extract the exponent byte.
        const uint32_t bits = reinterpret_cast<const uint32_t&>(scale);
        const uint32_t exp_byte = (bits >> 23) & 0xFFu;

        const int pack_idx = chunk_in_row >> 2;          // /4
        const int byte_idx = chunk_in_row & 3;           // %4
        const uint32_t shifted = exp_byte << (byte_idx * 8);

        // scales_packed address for [expert, slot, pack_idx] in MN-major layout.
        const int64_t base_off = static_cast<int64_t>(expert) * static_cast<int64_t>(align_mn) * static_cast<int64_t>(packed_k);
        const int64_t off = base_off + static_cast<int64_t>(pack_idx) * static_cast<int64_t>(align_mn) + static_cast<int64_t>(slot);
        atomicOr(&scales_packed[off], shifted);
    }

    float quantized = val / scale;
    quantized = fminf(fmaxf(quantized, -448.0f), 448.0f);
    output[row * K + col].__x = __nv_cvt_float_to_fp8(quantized, __NV_SATFINITE, __NV_E4M3);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_pack_fp8_cuda(
    torch::Tensor recv_x_fp8,
    torch::Tensor recv_x_scale,
    torch::Tensor recv_topk_idx,
    torch::Tensor recv_topk_weights,
    int64_t num_local,
    int64_t expected_m
) {
    TORCH_CHECK(recv_x_fp8.is_cuda(), "recv_x_fp8 must be CUDA tensor");
    TORCH_CHECK(recv_x_scale.is_cuda(), "recv_x_scale must be CUDA tensor");
    TORCH_CHECK(recv_topk_idx.is_cuda(), "recv_topk_idx must be CUDA tensor");
    TORCH_CHECK(recv_topk_weights.is_cuda(), "recv_topk_weights must be CUDA tensor");

    TORCH_CHECK(recv_x_fp8.dtype() == torch::kFloat8_e4m3fn, "recv_x_fp8 must be float8_e4m3fn");
    TORCH_CHECK(recv_x_scale.dtype() == torch::kFloat32, "recv_x_scale must be float32");
    TORCH_CHECK(
        recv_topk_idx.dtype() == torch::kInt64 || recv_topk_idx.dtype() == torch::kInt32,
        "recv_topk_idx must be int64 or int32"
    );
    TORCH_CHECK(recv_topk_weights.dtype() == torch::kFloat32, "recv_topk_weights must be float32");

    TORCH_CHECK(recv_x_fp8.is_contiguous(), "recv_x_fp8 must be contiguous");
    TORCH_CHECK(recv_x_scale.is_contiguous(), "recv_x_scale must be contiguous");
    TORCH_CHECK(recv_topk_idx.is_contiguous(), "recv_topk_idx must be contiguous");
    TORCH_CHECK(recv_topk_weights.is_contiguous(), "recv_topk_weights must be contiguous");

    TORCH_CHECK(recv_x_fp8.dim() == 2, "recv_x_fp8 must be 2D");
    TORCH_CHECK(recv_x_scale.dim() == 2, "recv_x_scale must be 2D");
    TORCH_CHECK(recv_topk_idx.dim() == 2, "recv_topk_idx must be 2D");
    TORCH_CHECK(recv_topk_weights.dim() == 2, "recv_topk_weights must be 2D");

    const int64_t num_recv = recv_x_fp8.size(0);
    const int64_t hidden = recv_x_fp8.size(1);
    const int64_t scale_blocks = recv_x_scale.size(1);
    const int64_t topk = recv_topk_idx.size(1);

    TORCH_CHECK(recv_x_scale.size(0) == num_recv, "recv_x_scale must match recv_x_fp8 rows");
    TORCH_CHECK(recv_topk_idx.size(0) == num_recv, "recv_topk_idx must match recv_x_fp8 rows");
    TORCH_CHECK(recv_topk_weights.size(0) == num_recv, "recv_topk_weights must match recv_x_fp8 rows");
    TORCH_CHECK(recv_topk_weights.size(1) == topk, "recv_topk_weights must match recv_topk_idx shape");

    auto x_grouped = torch::empty({num_local, expected_m, hidden}, recv_x_fp8.options());
    auto scale_grouped = torch::empty({num_local, expected_m, scale_blocks}, recv_x_scale.options());
    auto token_ids = torch::empty({num_local, expected_m}, torch::dtype(torch::kInt32).device(recv_x_fp8.device()));
    auto weights = torch::empty({num_local, expected_m}, recv_topk_weights.options());
    auto masked_m = torch::empty({num_local}, torch::dtype(torch::kInt32).device(recv_x_fp8.device()));
    auto overflow_flag = torch::empty({1}, torch::dtype(torch::kInt32).device(recv_x_fp8.device()));

    // Avoid per-call fill kernels by zero-initializing on the current CUDA stream.
    // These outputs are consumed via atomicAdd in the pack kernel and must start at 0.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(masked_m.data_ptr<int32_t>(), 0, static_cast<size_t>(num_local) * sizeof(int32_t), stream);
    cudaMemsetAsync(overflow_flag.data_ptr<int32_t>(), 0, sizeof(int32_t), stream);

    const int blocks = static_cast<int>(num_recv * topk);
    const int threads = 256;
    if (recv_topk_idx.dtype() == torch::kInt64) {
        moe_pack_fp8_grouped_sm100_kernel<int64_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint8_t*>(recv_x_fp8.data_ptr()),
            recv_x_scale.data_ptr<float>(),
            recv_topk_idx.data_ptr<int64_t>(),
            recv_topk_weights.data_ptr<float>(),
            reinterpret_cast<uint8_t*>(x_grouped.data_ptr()),
            scale_grouped.data_ptr<float>(),
            token_ids.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            masked_m.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            static_cast<int>(num_recv),
            static_cast<int>(hidden),
            static_cast<int>(scale_blocks),
            static_cast<int>(topk),
            static_cast<int>(expected_m)
        );
    } else {
        moe_pack_fp8_grouped_sm100_kernel<int32_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint8_t*>(recv_x_fp8.data_ptr()),
            recv_x_scale.data_ptr<float>(),
            recv_topk_idx.data_ptr<int32_t>(),
            recv_topk_weights.data_ptr<float>(),
            reinterpret_cast<uint8_t*>(x_grouped.data_ptr()),
            scale_grouped.data_ptr<float>(),
            token_ids.data_ptr<int32_t>(),
            weights.data_ptr<float>(),
            masked_m.data_ptr<int32_t>(),
            overflow_flag.data_ptr<int32_t>(),
            static_cast<int>(num_recv),
            static_cast<int>(hidden),
            static_cast<int>(scale_blocks),
            static_cast<int>(topk),
            static_cast<int>(expected_m)
        );
    }

    return std::make_tuple(x_grouped, scale_grouped, token_ids, weights, masked_m, overflow_flag);
}

std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_grouped_cuda(torch::Tensor gateup_out, torch::Tensor masked_m) {
    TORCH_CHECK(gateup_out.is_cuda(), "gateup_out must be CUDA tensor");
    TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA tensor");
    TORCH_CHECK(gateup_out.dtype() == torch::kBFloat16, "gateup_out must be bfloat16");
    TORCH_CHECK(masked_m.dtype() == torch::kInt32, "masked_m must be int32");
    TORCH_CHECK(gateup_out.is_contiguous(), "gateup_out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(gateup_out.dim() == 3, "gateup_out must be [num_local, expected_m, 2*K]");
    TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [num_local]");

    const int64_t num_local = gateup_out.size(0);
    const int64_t expected_m = gateup_out.size(1);
    const int64_t two_k = gateup_out.size(2);
    TORCH_CHECK(two_k % 2 == 0, "gateup_out last dim must be even");
    const int64_t K = two_k / 2;
    TORCH_CHECK(K % BLOCK_SIZE == 0, "K must be divisible by 128");
    TORCH_CHECK(masked_m.size(0) == num_local, "masked_m must match num_local");

    const int64_t chunks_per_row = K / BLOCK_SIZE;
    auto output = torch::empty({num_local, expected_m, K}, torch::dtype(torch::kFloat8_e4m3fn).device(gateup_out.device()));
    auto scales = torch::empty({num_local, expected_m, chunks_per_row}, torch::dtype(torch::kFloat32).device(gateup_out.device()));

    const int64_t rows = num_local * expected_m;
    const int64_t total_chunks = rows * chunks_per_row;

    silu_mul_fp8_grouped_kernel<<<static_cast<int>(total_chunks), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(gateup_out.data_ptr()),
        masked_m.data_ptr<int32_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        scales.data_ptr<float>(),
        static_cast<int>(expected_m),
        static_cast<int>(K),
        static_cast<int>(chunks_per_row)
    );

    return std::make_tuple(output, scales);
}

std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_grouped_packed_cuda(torch::Tensor gateup_out, torch::Tensor masked_m) {
    TORCH_CHECK(gateup_out.is_cuda(), "gateup_out must be CUDA tensor");
    TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA tensor");
    TORCH_CHECK(gateup_out.dtype() == torch::kBFloat16, "gateup_out must be bfloat16");
    TORCH_CHECK(masked_m.dtype() == torch::kInt32, "masked_m must be int32");
    TORCH_CHECK(gateup_out.is_contiguous(), "gateup_out must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(gateup_out.dim() == 3, "gateup_out must be [num_local, expected_m, 2*K]");
    TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [num_local]");

    const int64_t num_local = gateup_out.size(0);
    const int64_t expected_m = gateup_out.size(1);
    const int64_t two_k = gateup_out.size(2);
    TORCH_CHECK(two_k % 2 == 0, "gateup_out last dim must be even");
    const int64_t K = two_k / 2;
    TORCH_CHECK(K % BLOCK_SIZE == 0, "K must be divisible by 128");
    TORCH_CHECK(masked_m.size(0) == num_local, "masked_m must match num_local");

    const int64_t chunks_per_row = K / BLOCK_SIZE;
    // DeepGEMM scale packing packs 4 exponent bytes per int32.
    const int64_t packed_k = (chunks_per_row + 4 - 1) / 4;
    const int64_t align_mn = (expected_m + 4 - 1) / 4 * 4;

    auto output = torch::empty(
        {num_local, expected_m, K},
        torch::dtype(torch::kFloat8_e4m3fn).device(gateup_out.device())
    );
    // DeepGEMM MN-major layout: stride(-2)=1, stride(-1)=align(mn,4).
    auto scales_packed = torch::empty_strided(
        {num_local, expected_m, packed_k},
        {align_mn * packed_k, 1, align_mn},
        torch::dtype(torch::kInt32).device(gateup_out.device())
    );

    // Zero-initialize the packed scale buffer (atomic OR writes into it).
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(
        scales_packed.data_ptr<int32_t>(),
        0,
        static_cast<size_t>(num_local * align_mn * packed_k) * sizeof(int32_t),
        stream
    );

    const int64_t rows = num_local * expected_m;
    const int64_t total_chunks = rows * chunks_per_row;

    silu_mul_fp8_grouped_packed_kernel<<<static_cast<int>(total_chunks), THREADS_PER_BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gateup_out.data_ptr()),
        masked_m.data_ptr<int32_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        reinterpret_cast<uint32_t*>(scales_packed.data_ptr<int32_t>()),
        static_cast<int>(expected_m),
        static_cast<int>(K),
        static_cast<int>(chunks_per_row),
        static_cast<int>(packed_k),
        static_cast<int>(align_mn)
    );

    return std::make_tuple(output, scales_packed);
}

void weighted_scatter_add_grouped_cuda(
    torch::Tensor expert_out,  // [num_local, expected_m, hidden] bfloat16
    torch::Tensor token_ids,   // [num_local, expected_m] int32
    torch::Tensor weights,     // [num_local, expected_m] float32
    torch::Tensor masked_m,    // [num_local] int32
    torch::Tensor output       // [num_recv, hidden] bfloat16 (modified in place)
) {
    TORCH_CHECK(expert_out.is_cuda(), "expert_out must be CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");

    TORCH_CHECK(expert_out.dtype() == torch::kBFloat16, "expert_out must be bfloat16");
    TORCH_CHECK(token_ids.dtype() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(masked_m.dtype() == torch::kInt32, "masked_m must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");

    TORCH_CHECK(expert_out.is_contiguous(), "expert_out must be contiguous");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(expert_out.dim() == 3, "expert_out must be [num_local, expected_m, hidden]");
    TORCH_CHECK(token_ids.dim() == 2, "token_ids must be [num_local, expected_m]");
    TORCH_CHECK(weights.dim() == 2, "weights must be [num_local, expected_m]");
    TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [num_local]");

    const int64_t num_local = expert_out.size(0);
    const int64_t expected_m = expert_out.size(1);
    const int64_t hidden = expert_out.size(2);
    TORCH_CHECK(hidden % 2 == 0, "hidden must be even for bf16x2 atomic");
    TORCH_CHECK(token_ids.size(0) == num_local && token_ids.size(1) == expected_m, "token_ids shape mismatch");
    TORCH_CHECK(weights.size(0) == num_local && weights.size(1) == expected_m, "weights shape mismatch");
    TORCH_CHECK(masked_m.size(0) == num_local, "masked_m shape mismatch");
    TORCH_CHECK(output.size(1) == hidden, "output hidden mismatch");

    const int blocks = static_cast<int>(num_local * expected_m);
    const int threads = 256;
    weighted_scatter_add_grouped_sm100_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat162*>(expert_out.data_ptr()),
        token_ids.data_ptr<int32_t>(),
        weights.data_ptr<float>(),
        masked_m.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr()),
        static_cast<int>(expected_m),
        static_cast<int>(hidden / 2)
    );
}

void weighted_scatter_add_grouped_zeroed_cuda(
    torch::Tensor expert_out,  // [num_local, expected_m, hidden] bfloat16
    torch::Tensor token_ids,   // [num_local, expected_m] int32
    torch::Tensor weights,     // [num_local, expected_m] float32
    torch::Tensor masked_m,    // [num_local] int32
    torch::Tensor output       // [num_recv, hidden] bfloat16 (overwritten)
) {
    TORCH_CHECK(expert_out.is_cuda(), "expert_out must be CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA tensor");
    TORCH_CHECK(masked_m.is_cuda(), "masked_m must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");

    TORCH_CHECK(expert_out.dtype() == torch::kBFloat16, "expert_out must be bfloat16");
    TORCH_CHECK(token_ids.dtype() == torch::kInt32, "token_ids must be int32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(masked_m.dtype() == torch::kInt32, "masked_m must be int32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bfloat16");

    TORCH_CHECK(expert_out.is_contiguous(), "expert_out must be contiguous");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(masked_m.is_contiguous(), "masked_m must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(expert_out.dim() == 3, "expert_out must be [num_local, expected_m, hidden]");
    TORCH_CHECK(token_ids.dim() == 2, "token_ids must be [num_local, expected_m]");
    TORCH_CHECK(weights.dim() == 2, "weights must be [num_local, expected_m]");
    TORCH_CHECK(masked_m.dim() == 1, "masked_m must be [num_local]");

    const int64_t num_local = expert_out.size(0);
    const int64_t expected_m = expert_out.size(1);
    const int64_t hidden = expert_out.size(2);
    TORCH_CHECK(hidden % 2 == 0, "hidden must be even for bf16x2 atomic");
    TORCH_CHECK(token_ids.size(0) == num_local && token_ids.size(1) == expected_m, "token_ids shape mismatch");
    TORCH_CHECK(weights.size(0) == num_local && weights.size(1) == expected_m, "weights shape mismatch");
    TORCH_CHECK(masked_m.size(0) == num_local, "masked_m shape mismatch");
    TORCH_CHECK(output.size(1) == hidden, "output hidden mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(output.data_ptr(), 0, static_cast<size_t>(output.numel()) * output.element_size(), stream);

    const int blocks = static_cast<int>(num_local * expected_m);
    const int threads = 256;
    weighted_scatter_add_grouped_sm100_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(expert_out.data_ptr()),
        token_ids.data_ptr<int32_t>(),
        weights.data_ptr<float>(),
        masked_m.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr()),
        static_cast<int>(expected_m),
        static_cast<int>(hidden / 2)
    );
}
"""

_MOE_FUSED_CPP_SOURCE = r"""
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> moe_pack_fp8_cuda(
    torch::Tensor recv_x_fp8,
    torch::Tensor recv_x_scale,
    torch::Tensor recv_topk_idx,
    torch::Tensor recv_topk_weights,
    int64_t num_local,
    int64_t expected_m
);

std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_grouped_cuda(torch::Tensor gateup_out, torch::Tensor masked_m);

std::tuple<torch::Tensor, torch::Tensor> silu_mul_fp8_grouped_packed_cuda(torch::Tensor gateup_out, torch::Tensor masked_m);

void weighted_scatter_add_grouped_cuda(
    torch::Tensor expert_out,
    torch::Tensor token_ids,
    torch::Tensor weights,
    torch::Tensor masked_m,
    torch::Tensor output
);

void weighted_scatter_add_grouped_zeroed_cuda(
    torch::Tensor expert_out,
    torch::Tensor token_ids,
    torch::Tensor weights,
    torch::Tensor masked_m,
    torch::Tensor output
);
"""

_moe_fused_module = None

def _get_moe_fused_module():
    global _moe_fused_module
    if _moe_fused_module is None:
        _moe_fused_module = load_inline(
            name=f"moe_fused_kernel_v{_MOE_FUSED_KERNEL_VERSION}",
            cpp_sources=[_MOE_FUSED_CPP_SOURCE],
            cuda_sources=[_MOE_FUSED_CUDA_SOURCE],
            functions=[
                "moe_pack_fp8_cuda",
                "silu_mul_fp8_grouped_cuda",
                "silu_mul_fp8_grouped_packed_cuda",
                "weighted_scatter_add_grouped_cuda",
                "weighted_scatter_add_grouped_zeroed_cuda",
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _moe_fused_module


def moe_pack_fp8_grouped(
    recv_x_fp8: torch.Tensor,         # [num_recv, hidden] float8_e4m3fn
    recv_x_scale: torch.Tensor,       # [num_recv, hidden/128] float32
    recv_topk_idx: torch.Tensor,      # [num_recv, topk] int64 (local expert ids or -1)
    recv_topk_weights: torch.Tensor,  # [num_recv, topk] float32
    *,
    num_local: int,
    expected_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack received tokens into a per-expert grouped layout (decode-only fast path).

    Returns:
      x_grouped:      [num_local, expected_m, hidden] float8_e4m3fn
      scale_grouped:  [num_local, expected_m, hidden/128] float32
      token_ids:      [num_local, expected_m] int32 (valid for slots < masked_m[expert])
      weights:        [num_local, expected_m] float32
      masked_m:       [num_local] int32 counts (rows per expert)
      overflow_flag:  [1] int32 count of dropped (overflowed) pairs
    """
    if not recv_x_fp8.is_contiguous():
        recv_x_fp8 = recv_x_fp8.contiguous()
    if not recv_x_scale.is_contiguous():
        recv_x_scale = recv_x_scale.contiguous()
    if not recv_topk_idx.is_contiguous():
        recv_topk_idx = recv_topk_idx.contiguous()
    if not recv_topk_weights.is_contiguous():
        recv_topk_weights = recv_topk_weights.contiguous()
    return _get_moe_fused_module().moe_pack_fp8_cuda(recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_weights, num_local, expected_m)


def silu_mul_fp8_grouped(gateup_out: torch.Tensor, masked_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SiLU(gate) * up -> FP8 (for masked grouped GEMMs).

    Args:
      gateup_out: [num_local, expected_m, 2*inter] bfloat16
      masked_m:   [num_local] int32 (counts)
    """
    if not gateup_out.is_contiguous():
        gateup_out = gateup_out.contiguous()
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()
    return _get_moe_fused_module().silu_mul_fp8_grouped_cuda(gateup_out, masked_m)


def silu_mul_fp8_grouped_packed(gateup_out: torch.Tensor, masked_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped SiLU(gate) * up -> FP8, returning packed UE8M0 scales in DeepGEMM layout.

    Args:
      gateup_out: [num_local, expected_m, 2*inter] bfloat16
      masked_m:   [num_local] int32 (counts)

    Returns:
      down_in_q:          [num_local, expected_m, inter] float8_e4m3fn
      down_in_scale_int: [num_local, expected_m, packed_k] int32 in DeepGEMM MN-major layout
    """
    if not gateup_out.is_contiguous():
        gateup_out = gateup_out.contiguous()
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()
    return _get_moe_fused_module().silu_mul_fp8_grouped_packed_cuda(gateup_out, masked_m)


def weighted_scatter_add_grouped(
    expert_out: torch.Tensor,  # [num_local, expected_m, hidden] bfloat16
    token_ids: torch.Tensor,   # [num_local, expected_m] int32
    weights: torch.Tensor,     # [num_local, expected_m] float32
    masked_m: torch.Tensor,    # [num_local] int32
    output: torch.Tensor,      # [num_recv, hidden] bfloat16 (modified in place)
) -> None:
    """Accumulate grouped expert outputs back into per-token outputs (decode-only)."""
    if not expert_out.is_contiguous():
        expert_out = expert_out.contiguous()
    if not token_ids.is_contiguous():
        token_ids = token_ids.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()
    _get_moe_fused_module().weighted_scatter_add_grouped_cuda(expert_out, token_ids, weights, masked_m, output)


def weighted_scatter_add_grouped_zeroed(
    expert_out: torch.Tensor,  # [num_local, expected_m, hidden] bfloat16
    token_ids: torch.Tensor,   # [num_local, expected_m] int32
    weights: torch.Tensor,     # [num_local, expected_m] float32
    masked_m: torch.Tensor,    # [num_local] int32
    output: torch.Tensor,      # [num_recv, hidden] bfloat16 (overwritten)
) -> None:
    """Like weighted_scatter_add_grouped, but zero-initializes `output` on-stream first."""
    if not expert_out.is_contiguous():
        expert_out = expert_out.contiguous()
    if not token_ids.is_contiguous():
        token_ids = token_ids.contiguous()
    if not weights.is_contiguous():
        weights = weights.contiguous()
    if not masked_m.is_contiguous():
        masked_m = masked_m.contiguous()
    _get_moe_fused_module().weighted_scatter_add_grouped_zeroed_cuda(expert_out, token_ids, weights, masked_m, output)
