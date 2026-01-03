# SPDX-License-Identifier: Apache-2.0
"""DeepSeek Sparse Attention (DSA) + FlashMLA for V3-exp inference.

This module implements the DSA indexer with FlashMLA sparse attention kernel.
DSA computes learned sparse indices for each query position, enabling
sub-linear attention complexity for long sequences.

Key components:
- DSA indexer: q_idx(x), k_idx(x), w_idx(x) projections
- Sparse index selection via streaming top-k
- FlashMLA with FP8 KV cache and sparse indices
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from nmoe.triton.dsa import compute_indexer_scores
from nmoe.triton.flashmla_kv import flashmla_pack_kv_fp8_ue8m0_scatter

from nmoe.serve.model import (
    ModelConfig,
    RMSNorm,
    Linear,
    FP8Linear,
    FP8ColumnParallelLinear,
    FP8RowParallelLinear,
    _require,
    _sm100_only,
    apply_rotary_emb,
    weight_dequant,
    _world_size,
)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard transform to activations (required for DSA indexer).

    Reference: DeepSeek-V3.2 inference code uses fast_hadamard_transform.
    This is critical for the indexer to work correctly.
    """
    _require(x.dtype == torch.bfloat16, "rotate_activation expects BF16 input")
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


def _phys_token_ids(block_table_1d: torch.Tensor, context_len: int) -> torch.Tensor:
    """Build physical token ids [context_len] for one sequence: block_id*64 + offset."""
    _require(
        block_table_1d.is_cuda and block_table_1d.dtype == torch.int32 and block_table_1d.ndim == 1,
        "block_table must be CUDA int32 [num_blocks].",
    )
    if context_len <= 0:
        return torch.empty((0,), device=block_table_1d.device, dtype=torch.int32)
    pos = torch.arange(context_len, device=block_table_1d.device, dtype=torch.int64)
    page = torch.div(pos, 64, rounding_mode="floor").to(torch.int64)
    off = (pos % 64).to(torch.int64)
    blk = block_table_1d.index_select(0, page).to(torch.int64)
    return (blk * 64 + off).to(torch.int32)


class DSA(nn.Module):
    """One attention layer: DSA indexer + FlashMLA sparse attention."""

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_heads)
        self.num_local_heads = int(cfg.num_heads // _world_size)

        self.q_lora_rank = int(cfg.q_lora_rank)
        self.kv_lora_rank = int(cfg.kv_lora_rank)
        self.qk_nope_head_dim = int(cfg.qk_nope_head_dim)
        self.qk_rope_head_dim = int(cfg.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(cfg.v_head_dim)
        # IMPORTANT: Match DeepSeek reference scaling.
        # The MLA kernel consumes a 576-d "latent query" (kv_lora_rank + rope_dim),
        # but the reference model scales attention by 1/sqrt(qk_head_dim=192).
        # Using 1/sqrt(576) changes numerics enough to destroy generation while
        # staying finite. Keep scaling anchored to qk_head_dim.
        self.softmax_scale = self.qk_head_dim ** -0.5
        if int(cfg.max_seq_len) > int(cfg.original_seq_len):
            mscale = 0.1 * float(cfg.mscale) * math.log(float(cfg.rope_factor)) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.q_lora_rank > 0:
            self.wq_a = FP8Linear(self.hidden_size, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = FP8ColumnParallelLinear(self.q_lora_rank, self.num_heads * self.qk_head_dim)
        else:
            self.wq = FP8ColumnParallelLinear(self.hidden_size, self.num_heads * self.qk_head_dim)

        self.wkv_a = FP8Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = FP8ColumnParallelLinear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        self.wo = FP8RowParallelLinear(self.num_heads * self.v_head_dim, self.hidden_size)
        # Cached dequantized wkv_b for absorbed attention (populated on first forward)
        self._wkv_b_dequant: Optional[torch.Tensor] = None

        # DSA indexer projections
        # NOTE: Q_idx shares wq_a with attention Q, then has its own wq_idx (wq_b in checkpoint)
        self.n_idx_heads = int(cfg.dsa_n_idx_heads)
        self.idx_dim = int(cfg.dsa_idx_dim)
        self.topk = int(cfg.dsa_topk)
        _require(self.topk > 0 and self.topk % 64 == 0, f"DSA topk must be >0 and divisible by 64 (got {self.topk}).")
        # wq_idx: from LoRA latent (q_lora_rank) to indexer output (n_idx_heads * idx_dim)
        self.wq_idx = FP8Linear(self.q_lora_rank, self.n_idx_heads * self.idx_dim)
        self.wk_idx = FP8Linear(self.hidden_size, self.idx_dim)
        self.k_norm = nn.LayerNorm(self.idx_dim, dtype=torch.bfloat16)  # LayerNorm for K indexer (has bias)
        # Reference stores weights_proj in BF16 but runs it in FP32 for numerical stability / convenience.
        self.w_idx = Linear(self.hidden_size, self.n_idx_heads, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,                # [B,S,H]
        freqs_cis: torch.Tensor,        # [B,S,rope_dim/2] complex
        *,
        kv_cache: torch.Tensor,         # [num_blocks,64,1,656] uint8
        idx_k_cache: torch.Tensor,      # [num_blocks,64,idx_dim] bf16
        block_table: torch.Tensor,      # [B,max_blocks] int32
        cache_seqlens: torch.Tensor,    # [B] int32
        cache_seqlens_cpu: Optional[list[int]] = None,
        out_loc: torch.Tensor,          # [B,S] int32 physical slot ids (block*64 + off)
        positions: torch.Tensor,        # [B,S] int64 absolute positions (0..cache_len-1)
    ) -> torch.Tensor:
        _sm100_only(x.device)
        _require(x.is_cuda and x.dtype == torch.bfloat16 and x.ndim == 3, "x must be CUDA BF16 [B,S,H].")
        B, S, _ = x.shape
        _require(
            block_table.is_cuda and block_table.dtype == torch.int32 and block_table.shape[0] == B,
            "block_table must be CUDA int32 [B,max_blocks].",
        )
        _require(
            cache_seqlens.is_cuda and cache_seqlens.dtype == torch.int32 and cache_seqlens.numel() == B,
            "cache_seqlens must be CUDA int32 [B].",
        )
        _require(out_loc.is_cuda and out_loc.dtype == torch.int32 and out_loc.shape == (B, S), "out_loc must be CUDA int32 [B,S].")
        # Contract: out_loc must be non-negative (no padding). Enforce without host sync.
        torch._assert_async((out_loc >= 0).all(), "out_loc must be non-negative (no padding in this layer contract).")
        _require(positions.is_cuda and positions.shape == (B, S), "positions must be CUDA [B,S].")
        _require(
            kv_cache.is_cuda
            and kv_cache.dtype == torch.uint8
            and kv_cache.ndim == 4
            and kv_cache.size(1) == 64
            and kv_cache.size(2) == 1
            and kv_cache.size(3) == 656,
            "kv_cache must be [num_blocks,64,1,656] uint8.",
        )
        _require(
            idx_k_cache.is_cuda
            and idx_k_cache.dtype == torch.bfloat16
            and idx_k_cache.ndim == 3
            and idx_k_cache.size(1) == 64
            and idx_k_cache.size(2) == self.idx_dim,
            "idx_k_cache must be [num_blocks,64,idx_dim] bf16.",
        )

        # === MLA projections ===
        # Compute Q LoRA latent (shared with DSA indexer Q)
        if self.q_lora_rank > 0:
            q_latent = self.q_norm(self.wq_a(x))  # [B,S,q_lora_rank]
            q = self.wq_b(q_latent)
        else:
            q = self.wq(x)
            q_latent = None
        q = q.view(B, S, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv_latent, k_pe = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)
        k_rope = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2).contiguous()  # [B,S,64]

        # === Store KV + indexer K into caches ===
        kv_latent2 = kv_latent.reshape(B * S, 512).contiguous()
        k_rope2 = k_rope.reshape(B * S, 64).contiguous()
        loc = out_loc.reshape(B * S).to(torch.int64)
        flashmla_pack_kv_fp8_ue8m0_scatter(
            kv_latent2,
            k_rope2,
            loc,
            kv_cache.view(-1, 656),
        )

        # === Compute sparse indices (DSA) using triton kernel ===
        # Q indexer uses shared q_latent from MLA LoRA
        if q_latent is not None:
            q_idx_all = self.wq_idx(q_latent).view(B, S, self.n_idx_heads, self.idx_dim)
        else:
            # Fallback if no LoRA (shouldn't happen for V3.2)
            q_idx_all = self.wq_idx(x).view(B, S, self.n_idx_heads, self.idx_dim)

        # DSA indexer RoPE: non-interleaved format (reference uses interleaved=False)
        # Split into rope and non-rope parts, apply RoPE to rope part only
        q_idx_pe, q_idx_nope = torch.split(q_idx_all, [self.qk_rope_head_dim, self.idx_dim - self.qk_rope_head_dim], dim=-1)
        q_idx_pe = apply_rotary_emb(q_idx_pe, freqs_cis, interleaved=False)
        q_idx_all = torch.cat([q_idx_pe, q_idx_nope], dim=-1)

        # K indexer: project, normalize, apply RoPE (non-interleaved)
        k_idx_raw = self.k_norm(self.wk_idx(x))  # [B, S, idx_dim]
        k_idx_pe, k_idx_nope = torch.split(k_idx_raw, [self.qk_rope_head_dim, self.idx_dim - self.qk_rope_head_dim], dim=-1)
        k_idx_pe = apply_rotary_emb(k_idx_pe.unsqueeze(2), freqs_cis, interleaved=False).squeeze(2)
        k_idx_processed = torch.cat([k_idx_pe, k_idx_nope], dim=-1)

        # Apply Hadamard transform (critical for DSA to work correctly)
        q_idx_all = rotate_activation(q_idx_all.to(torch.bfloat16))
        k_idx_processed = rotate_activation(k_idx_processed.to(torch.bfloat16))

        # Store processed K indexer in cache (after RoPE and Hadamard)
        k_idx_new = k_idx_processed.reshape(B * S, self.idx_dim).contiguous()
        idx_k_cache.view(-1, self.idx_dim).index_copy_(0, loc, k_idx_new)

        # Reference: weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        # Reference: weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        # We use BF16 (no q_scale needed), but softmax_scale is critical for correct scores.
        idx_softmax_scale = self.idx_dim ** -0.5  # 128 ** -0.5 = 0.0884
        w_idx_all = self.w_idx(x.float()).view(B, S, self.n_idx_heads) * (self.n_idx_heads ** -0.5) * idx_softmax_scale

        # FlashMLA sparse path accepts invalid indices as -1 (or >= total_seq_len_kv).
        # Follow sglang/vLLM convention: fill unused entries with -1, never duplicate
        # a real token to pad (duplicates bias the softmax).
        indices = torch.full((B, S, self.topk), -1, device=x.device, dtype=torch.int32)
        _require(
            cache_seqlens_cpu is not None and len(cache_seqlens_cpu) == B,
            "cache_seqlens_cpu must be provided as a CPU list[int] of length B (no GPU scalar reads in hot path).",
        )
        for b in range(B):
            ctx_len = int(cache_seqlens_cpu[b])
            _require(ctx_len > 0, "cache_seqlens must be > 0.")

            # Gather idx_k for this batch's context from paged cache
            bt = block_table[b]
            phys_ids = _phys_token_ids(bt, ctx_len).to(torch.int64)
            k_ctx = idx_k_cache.view(-1, self.idx_dim).index_select(0, phys_ids)  # [ctx_len, D]

            # Prepare inputs for triton kernel: [1, S, H, D], [1, N, D], [1, S, H]
            q_idx = q_idx_all[b : b + 1]
            k_idx = k_ctx.unsqueeze(0)
            w_idx = w_idx_all[b : b + 1]

            # Compute scores using triton kernel (causal=False, we apply our own mask)
            scores = compute_indexer_scores(q_idx, k_idx, w_idx, causal=False)  # [1, S, ctx_len]
            scores = scores.squeeze(0)  # [S, ctx_len]

            # Apply position-based causal mask: key_pos <= query_pos
            q_pos = positions[b].to(torch.int64)  # [S]
            k_pos = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
            causal_mask = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)  # [S, ctx_len]
            scores = scores.masked_fill(causal_mask, float("-inf"))

            # Select top-k from the masked scores. For early query positions where the
            # causal mask excludes most keys, scores will contain -inf. We must not
            # pass those masked keys to FlashMLA; mark them as -1 (invalid) instead.
            k_sel = min(self.topk, ctx_len)
            vals, topk_logical = scores.topk(k_sel, dim=-1)  # [S, k_sel]

            # FlashMLA expects indices in *physical* KV coordinates (flattened):
            # global_block_id * 64 + offset. Our scores are computed against the
            # gathered context `k_ctx`, whose rows correspond 1:1 with `phys_ids`.
            topk_physical = phys_ids.index_select(0, topk_logical.to(torch.int64).reshape(-1)).view(S, k_sel)
            topk_physical_i32 = topk_physical.to(torch.int32)
            topk_physical_i32 = torch.where(torch.isfinite(vals), topk_physical_i32, torch.full_like(topk_physical_i32, -1))

            indices[b, :, :k_sel] = topk_physical_i32

        # === FlashMLA sparse attention ===
        # Absorb q_nope via wkv_b (first 128 rows) to produce 512-d latent query.
        # Dequantize wkv_b weight once and cache (FP8 -> BF16)
        if self._wkv_b_dequant is None:
            self._wkv_b_dequant = weight_dequant(self.wkv_b.weight, self.wkv_b.weight_scale_inv)
        wkv_b_w = self._wkv_b_dequant.view(self.num_local_heads, -1, self.kv_lora_rank)
        q_nope_abs = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_w[:, : self.qk_nope_head_dim])
        q_for_attn = torch.cat([q_nope_abs, q_pe], dim=-1).contiguous()  # [B,S,H,576]

        from flash_mla import get_mla_metadata, flash_mla_with_kvcache  # type: ignore

        # FlashMLA sparse decode kernel schedules work over `topk` indices, not the
        # true cache length. Follow vLLM/sglang convention: use cache_seqlens=topk
        # for metadata construction (and keep invalid indices as -1).
        cache_seqlens_meta = torch.full_like(cache_seqlens, self.topk)
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens_meta,
            num_q_tokens_per_head_k=int(S * self.num_local_heads),
            num_heads_k=1,
            num_heads_q=self.num_local_heads,
            is_fp8_kvcache=True,
            topk=self.topk,
        )
        out_latent, _lse = flash_mla_with_kvcache(
            q_for_attn,
            kv_cache,
            block_table,
            cache_seqlens,
            head_dim_v=512,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=float(self.softmax_scale),
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )

        # Project latent -> v_head_dim and output projection.
        out = torch.einsum("bshc,hdc->bshd", out_latent, wkv_b_w[:, -self.v_head_dim :])
        return self.wo(out.flatten(2))
