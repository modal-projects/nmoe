# SPDX-License-Identifier: Apache-2.0
"""Test MLA three-path forward: dense prefill, paged prefill, decode."""

from __future__ import annotations

import torch


def test_mla_paths():
    """Test all three MLA forward paths."""
    from nmoe.serve.model import ModelConfig, init_distributed
    from nmoe.serve.mla import MLA

    print("Testing MLA three-path forward...\n")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    init_distributed(0, 1)

    # Config
    cfg = ModelConfig(attention_type="mla")
    B, S_prefill, S_decode = 2, 64, 1
    num_pages, page_size = 32, 64
    max_blocks = 8

    # Create MLA module (use layer 0 config)
    mla = MLA(cfg, layer_idx=0).to(device).eval()

    # Allocate caches
    # CuTeDSL layout: (page_size, D, num_pages) with D stride=1, backed by a
    # contiguous (num_pages, page_size, D) allocation (pages are contiguous blocks).
    kv_cache_latent = torch.zeros(num_pages, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
    kv_cache_rope = torch.zeros(num_pages, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)

    # Block table: each sequence uses sequential pages
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for p in range(max_blocks):
            block_table[b, p] = b * max_blocks + p

    # Precompute freqs_cis for rope
    from nmoe.serve.model import precompute_freqs_cis
    freqs_cis_full = precompute_freqs_cis(cfg, device)  # [max_seq_len, rope_dim/2] complex

    def get_freqs(positions):
        # Index into precomputed freqs by position
        # apply_rotary_emb expects [S, rope_dim/2] (shared across batch)
        # positions: [B, S] - use first batch's positions
        S = positions.shape[1]
        pos = positions[0]  # [S]
        freqs = freqs_cis_full[pos]  # [S, rope_dim/2]
        return freqs

    # =========================================================================
    # PATH 1: Dense prefill (S>1, cached_len=0)
    # =========================================================================
    print("=" * 60)
    print("PATH 1: Dense prefill (S>1, cached_len=0)")
    print("=" * 60)

    x_prefill = torch.randn(B, S_prefill, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    positions_prefill = torch.arange(S_prefill, device=device).unsqueeze(0).expand(B, -1)
    freqs_prefill = get_freqs(positions_prefill)

    # out_loc: where to store KV in cache
    out_loc_prefill = torch.zeros(B, S_prefill, dtype=torch.int32, device=device)
    for b in range(B):
        for s in range(S_prefill):
            page_idx = s // page_size
            slot_idx = s % page_size
            out_loc_prefill[b, s] = block_table[b, page_idx] * page_size + slot_idx

    # cache_seqlens AFTER this forward = S_prefill (starting from 0)
    cache_seqlens_prefill = torch.full((B,), S_prefill, dtype=torch.int32, device=device)

    with torch.no_grad():
        out_dense = mla(
            x_prefill,
            freqs_prefill,
            kv_cache_latent=kv_cache_latent,
            kv_cache_rope=kv_cache_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens_prefill,
            out_loc=out_loc_prefill,
            positions=positions_prefill,
            prefill_mode="dense",
        )

    print(f"  Input:  x {x_prefill.shape}")
    print(f"  Output: {out_dense.shape}")
    print(f"  Output dtype: {out_dense.dtype}")
    print(f"  Output range: [{out_dense.min():.4f}, {out_dense.max():.4f}]")
    print(f"  ✓ Dense prefill passed\n")

    # =========================================================================
    # PATH 3: Decode (S=1) - test this before paged prefill
    # =========================================================================
    print("=" * 60)
    print("PATH 3: Decode (S=1)")
    print("=" * 60)

    x_decode = torch.randn(B, S_decode, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    positions_decode = torch.full((B, S_decode), S_prefill, dtype=torch.int64, device=device)
    freqs_decode = get_freqs(positions_decode)

    # out_loc for decode token
    out_loc_decode = torch.zeros(B, S_decode, dtype=torch.int32, device=device)
    for b in range(B):
        pos = S_prefill
        page_idx = pos // page_size
        slot_idx = pos % page_size
        out_loc_decode[b, 0] = block_table[b, page_idx] * page_size + slot_idx

    # cache_seqlens AFTER decode = S_prefill + 1
    cache_seqlens_decode = torch.full((B,), S_prefill + 1, dtype=torch.int32, device=device)

    with torch.no_grad():
        out_decode = mla(
            x_decode,
            freqs_decode,
            kv_cache_latent=kv_cache_latent,
            kv_cache_rope=kv_cache_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens_decode,
            out_loc=out_loc_decode,
            positions=positions_decode,
            prefill_mode=None,  # decode
        )

    print(f"  Input:  x {x_decode.shape}")
    print(f"  Output: {out_decode.shape}")
    print(f"  Output dtype: {out_decode.dtype}")
    print(f"  Output range: [{out_decode.min():.4f}, {out_decode.max():.4f}]")
    print(f"  ✓ Decode passed\n")

    # =========================================================================
    # PATH 2: Paged prefill (S>1, cached_len>0)
    # =========================================================================
    print("=" * 60)
    print("PATH 2: Paged prefill (S>1, cached_len>0)")
    print("=" * 60)

    # Simulate: we have S_prefill+1 tokens cached, now adding S_chunk more
    S_chunk = 16
    cached_len = S_prefill + 1  # From previous prefill + decode

    x_paged = torch.randn(B, S_chunk, cfg.hidden_size, dtype=torch.bfloat16, device=device)
    positions_paged = torch.arange(cached_len, cached_len + S_chunk, device=device).unsqueeze(0).expand(B, -1)
    freqs_paged = get_freqs(positions_paged)

    # out_loc for new chunk
    out_loc_paged = torch.zeros(B, S_chunk, dtype=torch.int32, device=device)
    for b in range(B):
        for s in range(S_chunk):
            pos = cached_len + s
            page_idx = pos // page_size
            slot_idx = pos % page_size
            out_loc_paged[b, s] = block_table[b, page_idx] * page_size + slot_idx

    # cache_seqlens AFTER this forward
    cache_seqlens_paged = torch.full((B,), cached_len + S_chunk, dtype=torch.int32, device=device)

    with torch.no_grad():
        out_paged = mla(
            x_paged,
            freqs_paged,
            kv_cache_latent=kv_cache_latent,
            kv_cache_rope=kv_cache_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens_paged,
            out_loc=out_loc_paged,
            positions=positions_paged,
            prefill_mode="paged",
        )

    print(f"  Input:  x {x_paged.shape}")
    print(f"  Cached len: {cached_len}")
    print(f"  Output: {out_paged.shape}")
    print(f"  Output dtype: {out_paged.dtype}")
    print(f"  Output range: [{out_paged.min():.4f}, {out_paged.max():.4f}]")
    print(f"  ✓ Paged prefill passed\n")

    print("=" * 60)
    print("All MLA paths passed!")
    print("=" * 60)


def main() -> int:
    try:
        test_mla_paths()
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
