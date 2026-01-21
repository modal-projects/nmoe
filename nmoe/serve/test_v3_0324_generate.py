# SPDX-License-Identifier: Apache-2.0
"""Test V3-0324 text generation with MLA attention.

This test assumes TP=1 (EP-only): lm_head is replicated, so each rank has
full-vocab logits and sampling is local (no vocab all-gather).
"""

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
from transformers import AutoTokenizer


def _greedy_argmax(logits: torch.Tensor) -> int:
    """Greedy argmax over full vocab logits (TP=1 mode)."""
    return int(torch.argmax(logits, dim=-1).item())


def _assert_all_ranks_equal(name: str, value: int, device: torch.device) -> None:
    """Assert that all ranks computed the same integer value."""
    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    vals = [int(x.item()) for x in gathered]
    if len(set(vals)) != 1:
        raise RuntimeError(f"{name}: mismatch across ranks: {vals}")


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import DeepSeekV3, init_distributed
    from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
    from deep_ep import Buffer

    # TP=1 for EP-only dynamic disagg bringup.
    init_distributed(rank, world_size, tp_size=1)

    ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1")
    cfg = load_model_config(ckpt_path)
    if cfg.attention_type != "mla":
        raise ValueError(f"Expected MLA checkpoint, got attention_type={cfg.attention_type!r}")

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

    # Auto-detect sharded checkpoint format.
    sharded_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if rank == 0:
        print(f"\nLoading checkpoint from {ckpt_path}...")
    if os.path.exists(sharded_file):
        missing, unexpected = load_sharded_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)
    else:
        missing, unexpected = load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
    if rank == 0:
        print(f"Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    dist.barrier()

    # Load tokenizer from HF checkpoint
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
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
        page_size = 64
        kv_caches_latent = [
            torch.zeros(num_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(cfg.num_layers)
        ]
        kv_caches_rope = [
            torch.zeros(num_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
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

        next_token = _greedy_argmax(logits[0, -1, :])
        _assert_all_ranks_equal("prefill_next_token", next_token, device)
        if rank == 0:
            top5 = logits[0, -1, :].topk(5)
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
                    prefill_mode=None,  # decode
                )

            next_token = _greedy_argmax(logits[0, 0, :])
            _assert_all_ranks_equal("decode_next_token", next_token, device)

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
