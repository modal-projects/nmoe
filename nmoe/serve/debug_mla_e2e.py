# SPDX-License-Identifier: Apache-2.0
"""Debug MLA end-to-end: trace through prefill and decode to find the bug."""
import os
import math
import torch
import torch.distributed as dist
from pathlib import Path

def _maybe_set_cutlass_path() -> None:
    if os.environ.get("CUTLASS_PATH"):
        return
    candidates = ["/opt/nvidia/cutlass", "/workspace/nmoe/third_party/cutlass/python"]
    for p in candidates:
        if os.path.isdir(p):
            os.environ["CUTLASS_PATH"] = p
            return

_maybe_set_cutlass_path()

def pytorch_absorbed_attn(q_latent, q_pe, kv_latent, k_rope, scale, causal=True):
    """Reference attention for comparison."""
    import torch.nn.functional as F
    # Score = q_latent @ kv_latent^T + q_pe @ k_rope^T
    # q_latent: [B, S_q, H, L], kv_latent: [S_kv, L]
    score_nope = torch.einsum("bqhl,kl->bqhk", q_latent.float(), kv_latent.float())
    score_pe = torch.einsum("bqhr,kr->bqhk", q_pe.float(), k_rope.float())
    score = (score_nope + score_pe) * scale

    S_q = q_latent.shape[1]
    S_kv = kv_latent.shape[0]
    if causal and S_q > 1:
        mask = torch.triu(torch.ones(S_q, S_kv, device=score.device, dtype=torch.bool), diagonal=S_kv-S_q+1)
        score = score.masked_fill(mask.unsqueeze(0).unsqueeze(2), float('-inf'))

    attn = F.softmax(score, dim=-1)
    out = torch.einsum("bqhk,kl->bqhl", attn, kv_latent.float())
    return out


def test_mla_e2e():
    """Test MLA through model forward to find the decode bug."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
    from deep_ep import Buffer

    init_distributed(rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("MLA E2E Debug Test")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # Model config - MLA attention
    cfg = ModelConfig(
        num_layers=2,  # Just 2 layers for debugging
        num_dense_layers=1,
        attention_type="mla",
    )

    # Create model
    # Buffer needs process group and NVL bytes for MoE dispatch
    num_nvl_bytes = cfg.hidden_size * 2 * 256  # Enough for testing
    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
    model = DeepSeekV3(cfg, buffer).to(device)
    model.eval()

    # Don't load checkpoint - use random weights for debugging
    if rank == 0:
        print(f"\nUsing random weights for debugging")
        print(f"Attention type: {cfg.attention_type}")
        print(f"num_local_heads: {cfg.num_heads // world_size}")

    dist.barrier()

    # Test setup
    B, S = 1, 5
    page_size = 64
    num_blocks = (S + 64) // page_size + 2

    # KV caches
    kv_page_size = int(page_size)
    kv_caches_latent = [
        torch.zeros(num_blocks, kv_page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
        for _ in range(cfg.num_layers)
    ]
    kv_caches_rope = [
        torch.zeros(num_blocks, kv_page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
        for _ in range(cfg.num_layers)
    ]

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

    # Simple test input
    torch.manual_seed(42 + rank)
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    positions = torch.arange(S, device=device).unsqueeze(0)
    out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

    # === PREFILL ===
    if rank == 0:
        print(f"\n=== PREFILL (S={S}) ===")

    with torch.inference_mode():
        logits_prefill = model(
            input_ids, positions,
            kv_caches_latent=kv_caches_latent, kv_caches_rope=kv_caches_rope,
            block_table=block_table, cache_seqlens=cache_seqlens,
            out_loc=out_loc,
            prefill_mode="dense",
        )

    if rank == 0:
        print(f"Prefill logits shape: {logits_prefill.shape}")
        print(f"Prefill logits range: [{logits_prefill.min().item():.4f}, {logits_prefill.max().item():.4f}]")
        print(f"Prefill logits mean: {logits_prefill.mean().item():.4f}")
        print(f"Has NaN: {logits_prefill.isnan().any().item()}")

        # Check KV cache was written
        kv_nonzero = (kv_caches_latent[0] != 0).sum().item()
        print(f"KV cache nonzero entries: {kv_nonzero}")

    dist.barrier()

    # Sample next token (just take argmax from last position)
    next_token = logits_prefill[0, -1].argmax().item()

    # === DECODE (1 token) ===
    if rank == 0:
        print(f"\n=== DECODE (token {next_token} at position {S}) ===")

    decode_input = torch.tensor([[next_token]], dtype=torch.int64, device=device)
    decode_pos = torch.tensor([[S]], dtype=torch.int64, device=device)
    decode_out_loc = torch.tensor([[S]], dtype=torch.int32, device=device)
    decode_cache_seqlens = torch.tensor([S + 1], dtype=torch.int32, device=device)

    with torch.inference_mode():
        logits_decode = model(
            decode_input, decode_pos,
            kv_caches_latent=kv_caches_latent, kv_caches_rope=kv_caches_rope,
            block_table=block_table, cache_seqlens=decode_cache_seqlens,
            out_loc=decode_out_loc,
            prefill_mode=None,  # decode mode
        )

    if rank == 0:
        print(f"Decode logits shape: {logits_decode.shape}")
        print(f"Decode logits range: [{logits_decode.min().item():.4f}, {logits_decode.max().item():.4f}]")
        print(f"Decode logits mean: {logits_decode.mean().item():.4f}")
        print(f"Has NaN: {logits_decode.isnan().any().item()}")

    dist.barrier()

    # === DEBUG: Run prefill again at position S to compare ===
    if rank == 0:
        print(f"\n=== COMPARISON: Prefill mode at position {S} ===")

    # Add the token to sequence and run as prefill
    input_ids_ext = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    positions_ext = torch.arange(S + 1, device=device).unsqueeze(0)
    out_loc_ext = torch.arange(S + 1, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens_ext = torch.tensor([S + 1], dtype=torch.int32, device=device)

    # Reset KV cache
    for kv in kv_caches_latent:
        kv.zero_()
    for kv in kv_caches_rope:
        kv.zero_()

    with torch.inference_mode():
        logits_prefill_ext = model(
            input_ids_ext, positions_ext,
            kv_caches_latent=kv_caches_latent, kv_caches_rope=kv_caches_rope,
            block_table=block_table, cache_seqlens=cache_seqlens_ext,
            out_loc=out_loc_ext,
            prefill_mode="dense",
        )

    if rank == 0:
        print(f"Prefill-ext logits shape: {logits_prefill_ext.shape}")
        print(f"Prefill-ext last position: [{logits_prefill_ext[0, -1].min().item():.4f}, {logits_prefill_ext[0, -1].max().item():.4f}]")
        print(f"Decode logits: [{logits_decode[0, 0].min().item():.4f}, {logits_decode[0, 0].max().item():.4f}]")

        # Compare the outputs
        diff = (logits_prefill_ext[0, -1] - logits_decode[0, 0]).abs()
        print(f"\nMax diff between decode and prefill-ext at position {S}: {diff.max().item():.6f}")
        print(f"Mean diff: {diff.mean().item():.6f}")

        if diff.max().item() < 1.0:
            print("✓ Decode matches prefill!")
        else:
            print("✗ Decode does NOT match prefill!")
            # Show top-k comparison
            prefill_topk = logits_prefill_ext[0, -1].topk(5)
            decode_topk = logits_decode[0, 0].topk(5)
            print(f"\nPrefill-ext top5 indices: {prefill_topk.indices.tolist()}")
            print(f"Decode top5 indices: {decode_topk.indices.tolist()}")
            print(f"Prefill-ext top5 values: {prefill_topk.values.tolist()}")
            print(f"Decode top5 values: {decode_topk.values.tolist()}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_mla_e2e()
