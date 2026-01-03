# SPDX-License-Identifier: Apache-2.0
"""Test V3-0324 text generation with MLA attention."""

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
from transformers import AutoTokenizer


def _all_gather_vocab_shards(local_logits: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_logits
    parts = [torch.empty_like(local_logits) for _ in range(world_size)]
    dist.all_gather(parts, local_logits.contiguous())
    return torch.cat(parts, dim=-1)


def _tp_greedy_argmax(local_logits: torch.Tensor, vocab_size: int, rank: int, world_size: int) -> int:
    """Greedy argmax over vocab-parallel shards, returning global token id."""
    if world_size == 1:
        return int(torch.argmax(local_logits, dim=-1).item())

    v_shard = int(local_logits.numel())
    if v_shard * world_size != int(vocab_size):
        raise ValueError(f"Vocab sharding mismatch: {v_shard}*{world_size} != {vocab_size}")

    start = rank * v_shard
    local_max, local_idx = torch.max(local_logits, dim=-1)
    local_gid = local_idx.to(torch.int64) + int(start)

    gathered_vals = [torch.empty_like(local_max) for _ in range(world_size)]
    gathered_gids = [torch.empty_like(local_gid) for _ in range(world_size)]
    dist.all_gather(gathered_vals, local_max.contiguous())
    dist.all_gather(gathered_gids, local_gid.contiguous())

    vals = torch.stack(gathered_vals, dim=0)
    gids = torch.stack(gathered_gids, dim=0)
    gmax = torch.max(vals, dim=0).values

    mask = vals == gmax.unsqueeze(0)
    big = torch.full_like(gids, int(vocab_size) + 1)
    candidates = torch.where(mask, gids, big)
    return int(torch.min(candidates, dim=0).values.item())


def load_mp8_checkpoint(model: torch.nn.Module, ckpt_path: str, rank: int, world_size: int):
    """Load pre-converted mp8 checkpoint."""
    fpath = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    state_dict = {}
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
    from deep_ep import Buffer

    init_distributed(rank, world_size)

    # V3-0324 config with MLA attention
    cfg = ModelConfig(
        num_layers=61,
        num_dense_layers=3,
        attention_type="mla",
    )

    if rank == 0:
        print("=" * 60)
        print("V3-0324 Text Generation Test")
        print(f"GPUs: {world_size}")
        print("=" * 60)

    # DeepEP buffer
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

    # Load mp8 checkpoint
    ckpt_path = "/data/models/DeepSeek-V3-0324-mp8"
    hf_path = "/data/models/DeepSeek-V3-0324"  # For tokenizer
    if rank == 0:
        print(f"\nLoading checkpoint from {ckpt_path}...")

    missing, unexpected = load_mp8_checkpoint(model, ckpt_path, rank, world_size)
    if rank == 0:
        print(f"Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    dist.barrier()

    # Load tokenizer from HF checkpoint
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        prompts = [
            "The capital of France is",
            "2 + 2 =",
            "Hello, my name is",
            "The meaning of life is",
        ]
    else:
        tokenizer = None
        prompts = [""] * 4

    for pi, prompt in enumerate(prompts):
        # Broadcast input_ids from rank 0
        if rank == 0:
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            seq_len = torch.tensor([input_ids.size(1)], dtype=torch.int64, device=device)
        else:
            seq_len = torch.tensor([0], dtype=torch.int64, device=device)
            input_ids = None

        dist.broadcast(seq_len, src=0)
        S = int(seq_len.item())

        if rank != 0:
            input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
        dist.broadcast(input_ids, src=0)

        # Setup MLA KV caches
        num_blocks = 8
        kv_caches_latent = [
            torch.zeros(num_blocks, 64, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device)
            for _ in range(cfg.num_layers)
        ]
        kv_caches_rope = [
            torch.zeros(num_blocks, 64, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device)
            for _ in range(cfg.num_layers)
        ]
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
        positions = torch.arange(S, device=device).unsqueeze(0)
        out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
        cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

        generated = input_ids[0].tolist() if rank == 0 else []

        # Prefill
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

        # Get next token from vocab-sharded logits
        next_token = _tp_greedy_argmax(
            logits[0, -1, :],
            vocab_size=int(cfg.vocab_size),
            rank=rank,
            world_size=world_size,
        )

        # All ranks must participate in all_gather
        full_last = _all_gather_vocab_shards(logits[0, -1, :], world_size)
        if rank == 0:
            top5 = full_last.topk(5)
            print(f"\n{'='*60}")
            print(f"Prompt: '{prompt}'")
            print(f"Top5: {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
            generated.append(next_token)

        # Decode 20 tokens
        eos_id = tokenizer.eos_token_id if rank == 0 else 0
        for i in range(20):
            cur_pos = S + i + 1
            cache_seqlens = torch.tensor([cur_pos], dtype=torch.int32, device=device)
            inp = torch.tensor([[next_token]], dtype=torch.int64, device=device)
            pos = torch.tensor([[cur_pos - 1]], dtype=torch.int64, device=device)
            out_loc_decode = torch.tensor([[cur_pos - 1]], dtype=torch.int32, device=device)

            with torch.inference_mode():
                logits = model(
                    inp, pos,
                    kv_caches_latent=kv_caches_latent,
                    kv_caches_rope=kv_caches_rope,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    out_loc=out_loc_decode,
                    prefill_mode="paged",  # Use paged path for decode (CuTeDSL decode kernel has issues)
                )

            next_token = _tp_greedy_argmax(
                logits[0, 0, :],
                vocab_size=int(cfg.vocab_size),
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                generated.append(next_token)

            stop = torch.tensor([1 if (rank == 0 and next_token == eos_id) else 0], device=device)
            dist.broadcast(stop, src=0)
            if stop.item():
                break

        if rank == 0:
            print(f"Generated: {tokenizer.decode(generated)}")

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("V3-0324 text generation test complete!")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
