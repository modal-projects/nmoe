# SPDX-License-Identifier: Apache-2.0
"""Test V3-0324 checkpoint loading with MLA attention."""

import os
from pathlib import Path


def _maybe_set_cutlass_path() -> None:
    if os.environ.get("CUTLASS_PATH"):
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
        if cand.is_dir():
            os.environ["CUTLASS_PATH"] = str(cand)
            return


_maybe_set_cutlass_path()

import torch
import torch.distributed as dist
from safetensors.torch import safe_open


def load_mp8_checkpoint(model: torch.nn.Module, ckpt_path: str, rank: int, world_size: int) -> tuple[set, set]:
    """Load pre-converted mp8 checkpoint (one file per rank)."""
    fpath = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Checkpoint not found: {fpath}")

    state_dict = {}
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # Load into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return set(missing), set(unexpected)


def test_load_v3_0324():
    """Test loading V3-0324 checkpoint with MLA."""
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
        print("V3-0324 Checkpoint Loading Test (mp8 format)")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # V3-0324 uses standard MLA (no DSA indexer)
    cfg = ModelConfig(
        num_layers=61,
        num_dense_layers=3,
        attention_type="mla",  # Standard MLA for V3-0324
    )

    if rank == 0:
        print(f"\nModelConfig:")
        print(f"  attention_type: {cfg.attention_type}")
        print(f"  num_layers: {cfg.num_layers}")
        print(f"  kv_lora_rank: {cfg.kv_lora_rank}")
        print(f"  q_lora_rank: {cfg.q_lora_rank}")

    # DeepEP buffer
    hidden_bytes = cfg.hidden_size * 2
    dispatch_config = Buffer.get_dispatch_config(world_size)
    combine_config = Buffer.get_combine_config(world_size)
    num_nvl_bytes = max(
        dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
        combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    )

    if rank == 0:
        print(f"\nCreating model with MLA attention...")

    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
    model = DeepSeekV3(cfg, buffer).to(device)
    model.eval()

    # Load converted mp8 checkpoint
    ckpt_path = "/data/models/DeepSeek-V3-0324-mp8"
    if rank == 0:
        print(f"\nLoading mp8 checkpoint from {ckpt_path}...")

    missing, unexpected = load_mp8_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"\nCheckpoint loaded!")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            for k in sorted(missing)[:10]:
                print(f"    - {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")

        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for k in sorted(unexpected)[:10]:
                print(f"    - {k}")
            if len(unexpected) > 10:
                print(f"    ... and {len(unexpected) - 10} more")

    dist.barrier()

    # Quick sanity check - verify attention weights loaded
    if rank == 0:
        layer = model.layers[0]
        attn = layer.attn

        print(f"\nAttention weight shapes (layer 0):")
        if hasattr(attn, 'wq_a'):
            print(f"  wq_a: {attn.wq_a.weight.shape}")
            print(f"  wq_b: {attn.wq_b.weight.shape}")
            print(f"  wkv_a: {attn.wkv_a.weight.shape}")
            print(f"  wkv_b: {attn.wkv_b.weight.shape}")
            print(f"  wo: {attn.wo.weight.shape}")

    dist.barrier()

    # Simple forward pass test
    if rank == 0:
        print(f"\nRunning forward pass test...")

    batch_size = 1
    seq_len = 16
    num_blocks = (seq_len + 63) // 64 + 1

    input_ids = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    # MLA KV caches:
    # - kv_caches_latent: [num_pages, page_size, kv_lora_rank] bfloat16
    # - kv_caches_rope: [num_pages, page_size, qk_rope_head_dim] bfloat16
    kv_caches_latent = [
        torch.zeros(num_blocks, 64, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device)
        for _ in range(cfg.num_layers)
    ]
    kv_caches_rope = [
        torch.zeros(num_blocks, 64, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(cfg.num_layers)
    ]

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    out_loc = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)

    with torch.inference_mode():
        logits = model(
            input_ids, positions,
            kv_caches_latent=kv_caches_latent, kv_caches_rope=kv_caches_rope,
            block_table=block_table, cache_seqlens=cache_seqlens,
            out_loc=out_loc,
            prefill_mode="dense",  # First prefill uses FA4
        )

    if rank == 0:
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"\n✓ V3-0324 checkpoint loaded and forward pass succeeded!")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_load_v3_0324()
