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

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  # TP=1 for EP-only bringup (replicated lm_head; no vocab sharding).
  init_distributed(rank, world_size, tp_size=1)

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
    next_token = _greedy_argmax(logits[0, -1, :])
    _assert_all_ranks_equal("prefill_next_token", next_token, device)

    if rank == 0:
      top5 = logits[0, -1, :].topk(5)
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

    # Re-encode for next prompt
    if rank == 0 and pi < len(prompts) - 1:
      input_ids = tokenizer.encode(prompts[pi + 1], return_tensors="pt", add_special_tokens=False).to(device)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
