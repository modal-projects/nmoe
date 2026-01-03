# SPDX-License-Identifier: Apache-2.0
"""Debug MLA standalone - test MLA module directly without full model.

This test creates a minimal MLA module and compares prefill vs decode.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def _maybe_set_cutlass_path() -> None:
    if os.environ.get("CUTLASS_PATH"):
        return
    candidates = ["/opt/nvidia/cutlass", "/workspace/nmoe/third_party/cutlass/python"]
    for p in candidates:
        if os.path.isdir(p):
            os.environ["CUTLASS_PATH"] = p
            return

_maybe_set_cutlass_path()


def pytorch_absorbed_attn_reference(q_latent, q_pe, kv_latent, k_rope, scale):
    """Reference PyTorch implementation."""
    B, S_q, H, L = q_latent.shape
    S_kv = kv_latent.shape[0]

    # Score = q_latent @ kv_latent^T + q_pe @ k_rope^T
    score_nope = torch.einsum("bqhl,kl->bqhk", q_latent.float(), kv_latent.float())
    score_pe = torch.einsum("bqhr,kr->bqhk", q_pe.float(), k_rope.float())
    score = (score_nope + score_pe) * scale

    # Causal mask
    if S_q > 1:
        mask = torch.triu(torch.ones(S_q, S_kv, device=score.device, dtype=torch.bool), diagonal=S_kv-S_q+1)
        score = score.masked_fill(mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    attn = F.softmax(score, dim=-1)
    out = torch.einsum("bqhk,kl->bqhl", attn, kv_latent.float())
    return out


def test_mla_standalone():
    """Test MLA prefill vs decode directly."""
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print("=" * 60)
        print("MLA Standalone Debug Test")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # Import MLA after setting up environment
    from nmoe.serve.mla import MLA, _CompiledMlaKernel
    from nmoe.serve.model import ModelConfig

    # Config
    cfg = ModelConfig(
        attention_type="mla",
    )

    # Test parameters
    B = 1
    S_prefill = 5
    H = cfg.num_heads // world_size  # Local heads
    latent_dim = cfg.kv_lora_rank  # 512
    rope_dim = cfg.qk_rope_head_dim  # 64
    qk_nope_dim = cfg.qk_nope_head_dim  # 128
    v_dim = cfg.v_head_dim  # 128
    page_size = 64
    num_pages = 4
    softmax_scale = (qk_nope_dim + rope_dim) ** -0.5  # 1/sqrt(192)

    if rank == 0:
        print(f"\nConfig:")
        print(f"  local_heads: {H}")
        print(f"  latent_dim: {latent_dim}")
        print(f"  rope_dim: {rope_dim}")
        print(f"  qk_nope_dim: {qk_nope_dim}")
        print(f"  v_dim: {v_dim}")
        print(f"  softmax_scale: {softmax_scale:.6f}")

    # Random test data (all ranks same seed for debugging)
    torch.manual_seed(42)

    # KV caches
    kv_cache_latent = torch.zeros(num_pages, page_size, latent_dim, dtype=torch.bfloat16, device=device)
    kv_cache_rope = torch.zeros(num_pages, page_size, rope_dim, dtype=torch.bfloat16, device=device)

    # Create random absorbed Q and KV for testing (bypass model projections)
    # This tests the kernel directly
    q_latent_prefill = torch.randn(B, S_prefill, H, latent_dim, dtype=torch.bfloat16, device=device)
    q_pe_prefill = torch.randn(B, S_prefill, H, rope_dim, dtype=torch.bfloat16, device=device)
    kv_latent_prefill = torch.randn(B, S_prefill, latent_dim, dtype=torch.bfloat16, device=device)
    k_rope_prefill = torch.randn(B, S_prefill, rope_dim, dtype=torch.bfloat16, device=device)

    # Write prefill KV to cache (positions 0..S_prefill-1)
    for i in range(S_prefill):
        page_idx = i // page_size
        offset = i % page_size
        kv_cache_latent[page_idx, offset] = kv_latent_prefill[0, i]
        kv_cache_rope[page_idx, offset] = k_rope_prefill[0, i]

    # === TEST 1: PyTorch reference for prefill ===
    if rank == 0:
        print("\n=== TEST 1: PyTorch Reference (Prefill S=5) ===")

    out_ref_prefill = pytorch_absorbed_attn_reference(
        q_latent_prefill, q_pe_prefill,
        kv_latent_prefill[0],  # [S, L]
        k_rope_prefill[0],      # [S, R]
        softmax_scale,
    )

    if rank == 0:
        print(f"out_ref_prefill shape: {out_ref_prefill.shape}")
        print(f"out_ref_prefill range: [{out_ref_prefill.min().item():.4f}, {out_ref_prefill.max().item():.4f}]")
        print(f"out_ref_prefill[0, -1, 0, :5]: {out_ref_prefill[0, -1, 0, :5].tolist()}")

    # === TEST 2: CuTe kernel for decode (single token) ===
    if rank == 0:
        print("\n=== TEST 2: CuTe Kernel (Decode S=1) ===")

    # Decode query at position S_prefill (attending to 0..S_prefill-1)
    q_latent_decode = torch.randn(B, 1, H, latent_dim, dtype=torch.bfloat16, device=device)
    q_pe_decode = torch.randn(B, 1, H, rope_dim, dtype=torch.bfloat16, device=device)
    kv_latent_decode = torch.randn(B, 1, latent_dim, dtype=torch.bfloat16, device=device)
    k_rope_decode = torch.randn(B, 1, rope_dim, dtype=torch.bfloat16, device=device)

    # Write decode KV at position S_prefill
    decode_pos = S_prefill
    page_idx = decode_pos // page_size
    offset = decode_pos % page_size
    kv_cache_latent[page_idx, offset] = kv_latent_decode[0, 0]
    kv_cache_rope[page_idx, offset] = k_rope_decode[0, 0]

    # Setup CuTe kernel
    kernel = _CompiledMlaKernel(
        num_heads=H,
        max_batch=B,
        max_seq_len=num_pages * page_size,
        page_size=page_size,
        device=device,
    )

    # Block table and seqlens
    block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([decode_pos + 1], dtype=torch.int32, device=device)  # Attend to 0..S_prefill

    # Prepare kernel inputs
    q_latent_k = q_latent_decode.view(B, H, latent_dim).permute(1, 2, 0).contiguous().half()  # [H, L, B]
    q_pe_k = q_pe_decode.view(B, H, rope_dim).permute(1, 2, 0).contiguous().half()  # [H, R, B]

    # Run CuTe kernel
    out_cute, _ = kernel(
        q_latent_k,
        q_pe_k,
        kv_cache_latent.half(),
        kv_cache_rope.half(),
        block_table.T.contiguous(),
        cache_seqlens,
        softmax_scale,
    )

    # Convert back to [B, 1, H, L]
    out_cute = out_cute.permute(2, 0, 1).view(B, 1, H, latent_dim).to(torch.bfloat16)

    if rank == 0:
        print(f"out_cute shape: {out_cute.shape}")
        print(f"out_cute range: [{out_cute.min().item():.4f}, {out_cute.max().item():.4f}]")
        print(f"out_cute[0, 0, 0, :5]: {out_cute[0, 0, 0, :5].tolist()}")

    # === TEST 3: PyTorch reference for same decode ===
    if rank == 0:
        print("\n=== TEST 3: PyTorch Reference (Decode S=1) ===")

    # Collect all KV from cache for reference (positions 0..S_prefill)
    all_kv_latent = kv_cache_latent.view(-1, latent_dim)[:decode_pos + 1]  # [S_prefill+1, L]
    all_k_rope = kv_cache_rope.view(-1, rope_dim)[:decode_pos + 1]  # [S_prefill+1, R]

    out_ref_decode = pytorch_absorbed_attn_reference(
        q_latent_decode, q_pe_decode,
        all_kv_latent,
        all_k_rope,
        softmax_scale,
    )

    if rank == 0:
        print(f"out_ref_decode shape: {out_ref_decode.shape}")
        print(f"out_ref_decode range: [{out_ref_decode.min().item():.4f}, {out_ref_decode.max().item():.4f}]")
        print(f"out_ref_decode[0, 0, 0, :5]: {out_ref_decode[0, 0, 0, :5].tolist()}")

    # === COMPARE ===
    if rank == 0:
        print("\n=== COMPARISON: CuTe vs PyTorch Reference ===")
        diff = (out_cute.float() - out_ref_decode).abs()
        rel_diff = diff / (out_ref_decode.abs() + 1e-6)
        print(f"Max abs diff: {diff.max().item():.6f}")
        print(f"Mean abs diff: {diff.mean().item():.6f}")
        print(f"Max rel diff: {rel_diff.max().item():.6f}")

        if diff.max().item() < 0.1:
            print("\n✓ CuTe kernel matches PyTorch reference!")
        else:
            print("\n✗ CuTe kernel does NOT match PyTorch reference!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_mla_standalone()
