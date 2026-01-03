# SPDX-License-Identifier: Apache-2.0
"""Quick test of raw model forward pass timing."""
import torch
import torch.distributed as dist
import time

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
    from deep_ep import Buffer

    init_distributed(rank, world_size)

    # 2 layers for quick test
    cfg = ModelConfig(num_layers=2, num_dense_layers=1, attention_type="mla")

    hidden_bytes = cfg.hidden_size * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )

    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
    model = DeepSeekV3(cfg, buffer).to(device)
    model.eval()

    if rank == 0:
        print(f"Model: {cfg.num_layers} layers, MLA attention")

    B, S = 4, 512
    num_blocks = (S + 63) // 64 + 1
    total_blocks = B * num_blocks

    input_ids = torch.randint(0, 1000, (B, S), device=device)
    positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    kv_caches_latent = [
        torch.zeros(total_blocks, 64, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device)
        for _ in range(cfg.num_layers)
    ]
    kv_caches_rope = [
        torch.zeros(total_blocks, 64, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(cfg.num_layers)
    ]
    block_table = torch.zeros(B, num_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.arange(b * num_blocks, (b + 1) * num_blocks)
    cache_seqlens = torch.full((B,), S, dtype=torch.int32, device=device)
    out_loc = torch.zeros(B, S, dtype=torch.int32, device=device)
    for b in range(B):
        out_loc[b] = torch.arange(b * num_blocks * 64, b * num_blocks * 64 + S)

    # Warmup
    if rank == 0:
        print("Warmup...")
    with torch.inference_mode():
        logits = model(
            input_ids, positions,
            kv_caches_latent=kv_caches_latent,
            kv_caches_rope=kv_caches_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            out_loc=out_loc,
            prefill_mode="dense",
        )
    torch.cuda.synchronize()

    if rank == 0:
        print(f"Warmup done, logits: {logits.shape}")

    # Timed runs
    num_iters = 10
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_iters):
            logits = model(
                input_ids, positions,
                kv_caches_latent=kv_caches_latent,
                kv_caches_rope=kv_caches_rope,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                out_loc=out_loc,
                prefill_mode="dense",
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = B * S * num_iters
    tok_per_sec = total_tokens / elapsed

    if rank == 0:
        print(f"\n=== Results (2 layers only) ===")
        print(f"Config: B={B}, S={S}")
        print(f"{num_iters} iterations: {elapsed:.3f}s")
        print(f"Throughput: {tok_per_sec:,.0f} tok/s")
        print(f"Per-forward: {elapsed/num_iters*1000:.1f}ms")
        print(f"\nNote: Full model is 61 layers, expect ~30x slower")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
