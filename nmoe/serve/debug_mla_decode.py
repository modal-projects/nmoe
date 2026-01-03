# SPDX-License-Identifier: Apache-2.0
"""Debug MLA decode path - compare CuTe kernel vs PyTorch reference."""
import os
import math
import torch
import torch.nn.functional as F

# Set cutlass path before imports
def _maybe_set_cutlass_path() -> None:
    if os.environ.get("CUTLASS_PATH"):
        return
    candidates = [
        "/opt/nvidia/cutlass",
        "/workspace/nmoe/third_party/cutlass",
    ]
    for p in candidates:
        if os.path.isdir(p):
            os.environ["CUTLASS_PATH"] = p
            return

_maybe_set_cutlass_path()

def pytorch_mla_attention(
    q_latent: torch.Tensor,  # [B, S_q, H, latent_dim]
    q_pe: torch.Tensor,      # [B, S_q, H, rope_dim]
    kv_latent: torch.Tensor, # [S_kv, latent_dim]
    k_rope: torch.Tensor,    # [S_kv, rope_dim]
    softmax_scale: float,
) -> torch.Tensor:
    """Reference PyTorch implementation of absorbed MLA attention.

    Returns: out_latent [B, S_q, H, latent_dim]
    """
    B, S_q, H, L = q_latent.shape
    S_kv = kv_latent.shape[0]
    rope_dim = q_pe.shape[-1]

    # Score = q_latent @ kv_latent.T + q_pe @ k_rope.T
    # q_latent: [B, S_q, H, L]
    # kv_latent: [S_kv, L]
    # score_nope: [B, S_q, H, S_kv]
    score_nope = torch.einsum("bqhl,kl->bqhk", q_latent, kv_latent)

    # q_pe: [B, S_q, H, rope_dim]
    # k_rope: [S_kv, rope_dim]
    # score_pe: [B, S_q, H, S_kv]
    score_pe = torch.einsum("bqhr,kr->bqhk", q_pe, k_rope)

    score = (score_nope + score_pe) * softmax_scale

    # Causal mask (only attend to positions <= current position)
    # For decode (S_q=1), we attend to all S_kv positions
    # For prefill, we need causal mask
    if S_q > 1:
        mask = torch.triu(torch.ones(S_q, S_kv, device=score.device, dtype=torch.bool), diagonal=S_kv-S_q+1)
        score = score.masked_fill(mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    attn = F.softmax(score, dim=-1)

    # Output = attn @ kv_latent
    # attn: [B, S_q, H, S_kv]
    # kv_latent: [S_kv, L]
    # out: [B, S_q, H, L]
    out_latent = torch.einsum("bqhk,kl->bqhl", attn, kv_latent)

    return out_latent


def test_cute_vs_pytorch():
    """Test CuTe kernel against PyTorch reference."""
    device = torch.device("cuda:0")

    # Small test case
    B = 1  # batch size
    H = 16  # num heads (local)
    S_kv = 8  # context length
    S_q = 1  # decode mode
    latent_dim = 512
    rope_dim = 64
    page_size = 64
    num_pages = (S_kv + page_size - 1) // page_size + 1

    softmax_scale = (latent_dim + rope_dim) ** -0.5  # MLA scale

    # Create test inputs
    torch.manual_seed(42)

    # Query (absorbed)
    q_latent = torch.randn(B, S_q, H, latent_dim, device=device, dtype=torch.bfloat16)
    q_pe = torch.randn(B, S_q, H, rope_dim, device=device, dtype=torch.bfloat16)

    # KV cache (flat storage)
    kv_latent_flat = torch.randn(S_kv, latent_dim, device=device, dtype=torch.bfloat16)
    k_rope_flat = torch.randn(S_kv, rope_dim, device=device, dtype=torch.bfloat16)

    # === PyTorch Reference ===
    out_ref = pytorch_mla_attention(
        q_latent.float(), q_pe.float(),
        kv_latent_flat.float(), k_rope_flat.float(),
        softmax_scale,
    ).to(torch.bfloat16)

    print("=== PyTorch Reference ===")
    print(f"q_latent: {q_latent.shape}, q_pe: {q_pe.shape}")
    print(f"kv_latent: {kv_latent_flat.shape}, k_rope: {k_rope_flat.shape}")
    print(f"out_ref: {out_ref.shape}")
    print(f"out_ref range: [{out_ref.min().item():.4f}, {out_ref.max().item():.4f}]")
    print(f"out_ref mean: {out_ref.mean().item():.4f}")

    # === Prepare paged KV cache ===
    kv_cache_latent = torch.zeros(num_pages, page_size, latent_dim, device=device, dtype=torch.bfloat16)
    kv_cache_rope = torch.zeros(num_pages, page_size, rope_dim, device=device, dtype=torch.bfloat16)

    # Store KV at positions 0..S_kv-1
    for i in range(S_kv):
        page_idx = i // page_size
        offset = i % page_size
        kv_cache_latent[page_idx, offset] = kv_latent_flat[i]
        kv_cache_rope[page_idx, offset] = k_rope_flat[i]

    # Page table: batch 0 uses pages [0, 1, 2, ...]
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).unsqueeze(0)
    cache_seqlens = torch.tensor([S_kv], device=device, dtype=torch.int32)

    print("\n=== Paged KV Cache ===")
    print(f"kv_cache_latent: {kv_cache_latent.shape}")
    print(f"kv_cache_rope: {kv_cache_rope.shape}")
    print(f"block_table: {block_table.shape}")
    print(f"cache_seqlens: {cache_seqlens}")

    # === CuTe Kernel ===
    from nmoe.serve.mla import _CompiledMlaKernel

    kernel = _CompiledMlaKernel(
        num_heads=H,
        max_batch=B,
        max_seq_len=num_pages * page_size,
        page_size=page_size,
        device=device,
    )

    # Prepare inputs for kernel: [H, D, B]
    q_latent_k = q_latent.view(B, H, latent_dim).permute(1, 2, 0).contiguous().half()
    q_rope_k = q_pe.view(B, H, rope_dim).permute(1, 2, 0).contiguous().half()

    print("\n=== CuTe Kernel Input ===")
    print(f"q_latent_k: {q_latent_k.shape} (should be [H, L, B])")
    print(f"q_rope_k: {q_rope_k.shape} (should be [H, R, B])")
    print(f"block_table.T: {block_table.T.shape} (should be [max_blocks, B])")

    out_latent_k, lse = kernel(
        q_latent_k,
        q_rope_k,
        kv_cache_latent.half(),
        kv_cache_rope.half(),
        block_table.T.contiguous(),
        cache_seqlens,
        softmax_scale,
    )

    # Convert output back: [H, L, B] -> [B, 1, H, L]
    out_cute = out_latent_k.permute(2, 0, 1).view(B, S_q, H, latent_dim).to(torch.bfloat16)

    print("\n=== CuTe Kernel Output ===")
    print(f"out_cute: {out_cute.shape}")
    print(f"out_cute range: [{out_cute.min().item():.4f}, {out_cute.max().item():.4f}]")
    print(f"out_cute mean: {out_cute.mean().item():.4f}")

    # === Compare ===
    diff = (out_ref - out_cute).abs()
    rel_diff = diff / (out_ref.abs() + 1e-6)

    print("\n=== Comparison ===")
    print(f"Max abs diff: {diff.max().item():.6f}")
    print(f"Mean abs diff: {diff.mean().item():.6f}")
    print(f"Max rel diff: {rel_diff.max().item():.6f}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6f}")

    # Check if they match (allowing for fp16 precision)
    if diff.max().item() < 0.1:
        print("\n✓ CuTe kernel matches PyTorch reference!")
    else:
        print("\n✗ CuTe kernel does NOT match PyTorch reference!")
        print("\nDetailed comparison (first head, first batch):")
        print(f"out_ref[0,0,0,:10]: {out_ref[0,0,0,:10]}")
        print(f"out_cute[0,0,0,:10]: {out_cute[0,0,0,:10]}")


if __name__ == "__main__":
    test_cute_vs_pytorch()
