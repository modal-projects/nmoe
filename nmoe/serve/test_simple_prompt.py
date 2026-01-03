# SPDX-License-Identifier: Apache-2.0
"""Test generation with simple prompt (no </think> token)."""

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

  vals = torch.stack(gathered_vals, dim=0)  # [W]
  gids = torch.stack(gathered_gids, dim=0)  # [W]
  gmax = torch.max(vals, dim=0).values

  # Tie-break deterministically by smallest global token id among max logits.
  mask = vals == gmax.unsqueeze(0)
  big = torch.full_like(gids, int(vocab_size) + 1)
  candidates = torch.where(mask, gids, big)
  return int(torch.min(candidates, dim=0).values.item())


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=61, num_dense_layers=3)

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

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

    # Test with simple prompts (no special tokens)
    prompts = [
      "The capital of France is",  # Simple completion
      "2 + 2 =",  # Math
      "Hello, my name is",  # Name completion
    ]

    for prompt in prompts:
      print(f"\n{'='*60}")
      print(f"Prompt: '{prompt}'")

      # Use simple encoding without special tokens
      input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
  else:
    input_ids = None
    prompts = ["", "", ""]

  for pi, prompt in enumerate(prompts):
    # Broadcast input_ids
    seq_len = torch.tensor([input_ids.size(1) if input_ids is not None else 0], dtype=torch.int64, device=device)
    dist.broadcast(seq_len, src=0)
    S = seq_len.item()

    if rank != 0:
      input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
    dist.broadcast(input_ids, src=0)

    # Setup caches
    num_blocks = 4
    kv_caches = [
      torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
      for _ in range(cfg.num_layers)
    ]
    idx_k_caches = [
      torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
      for _ in range(cfg.num_layers)
    ]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    positions = torch.arange(S, device=device).unsqueeze(0)
    out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

    generated = input_ids[0].tolist() if rank == 0 else []

    # Prefill
    with torch.no_grad():
      logits = model(
        input_ids, positions,
        kv_caches=kv_caches, idx_k_caches=idx_k_caches,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[S], out_loc=out_loc,
      )

    # Decode token from vocab-sharded logits.
    next_token = _tp_greedy_argmax(
      logits[0, -1, :],
      vocab_size=int(cfg.vocab_size),
      rank=rank,
      world_size=world_size,
    )

    if rank == 0:
      full_last = _all_gather_vocab_shards(logits[0, -1, :], world_size)
      top5 = full_last.topk(5)
      print(f"Prefill top5: {top5.indices.tolist()} = {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
      generated.append(next_token)

    # Decode 10 tokens
    eos_id = tokenizer.eos_token_id if rank == 0 else 0
    for i in range(10):
      cur_pos = S + i + 1
      cache_seqlens = torch.tensor([cur_pos], dtype=torch.int32, device=device)
      inp = torch.tensor([[next_token]], dtype=torch.int64, device=device)
      pos = torch.tensor([[cur_pos - 1]], dtype=torch.int64, device=device)
      out_loc_decode = torch.tensor([[cur_pos - 1]], dtype=torch.int32, device=device)

      with torch.no_grad():
        logits = model(
          inp, pos,
          kv_caches=kv_caches, idx_k_caches=idx_k_caches,
          block_table=block_table, cache_seqlens=cache_seqlens,
          cache_seqlens_cpu=[cur_pos], out_loc=out_loc_decode,
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

    # Re-encode for next prompt
    if rank == 0 and pi < len(prompts) - 1:
      input_ids = tokenizer.encode(prompts[pi + 1], return_tensors="pt", add_special_tokens=False).to(device)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
