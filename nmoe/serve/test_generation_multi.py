# SPDX-License-Identifier: Apache-2.0
"""Test text generation on 8 GPUs to validate MoE alignment fix."""

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


def generate(
  model,
  tokenizer,
  prompt: str,
  max_new_tokens: int = 32,
  temperature: float = 0.0,  # Greedy for reproducibility
  device: torch.device = None,
  rank: int = 0,
):
  """Simple autoregressive generation."""
  device = device or next(model.parameters()).device
  cfg = model.cfg

  # Tokenize
  input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
  B, S = input_ids.shape

  # Allocate KV caches
  max_seq = S + max_new_tokens + 64
  num_blocks = (max_seq + 63) // 64
  num_layers = cfg.num_layers

  kv_caches = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(num_layers)
  ]
  idx_k_caches = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

  generated_ids = input_ids.tolist()[0]

  # Prefill
  positions = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

  with torch.inference_mode():
    logits = model(
      input_ids,
      positions,
      kv_caches=kv_caches,
      idx_k_caches=idx_k_caches,
      block_table=block_table,
      cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S],
      out_loc=out_loc,
    )

    # Greedy over vocab-sharded logits.
    next_token = _tp_greedy_argmax(
      logits[0, -1, :],
      vocab_size=int(cfg.vocab_size),
      rank=rank,
      world_size=dist.get_world_size(),
    )
    generated_ids.append(next_token)

    if rank == 0:
      print(f"  Prefill done. First token: {tokenizer.decode([next_token])!r}")

    # Decode loop
    for step in range(max_new_tokens - 1):
      cur_pos = S + step + 1
      cache_seqlens = torch.tensor([cur_pos], dtype=torch.int32, device=device)

      input_ids = torch.tensor([[next_token]], dtype=torch.int64, device=device)
      positions = torch.tensor([[cur_pos - 1]], dtype=torch.int64, device=device)
      out_loc = torch.tensor([[cur_pos - 1]], dtype=torch.int32, device=device)

      logits = model(
        input_ids,
        positions,
        kv_caches=kv_caches,
        idx_k_caches=idx_k_caches,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[cur_pos],
        out_loc=out_loc,
      )

      next_token = _tp_greedy_argmax(
        logits[0, 0, :],
        vocab_size=int(cfg.vocab_size),
        rank=rank,
        world_size=dist.get_world_size(),
      )
      generated_ids.append(next_token)

      # Check for EOS
      if next_token == tokenizer.eos_token_id:
        break

  return tokenizer.decode(generated_ids, skip_special_tokens=True)


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

  if rank == 0:
    print("=" * 60)
    print("Multi-GPU Generation Test")
    print("=" * 60)

  # Use more layers for quality
  num_layers = 61
  num_dense = 3
  cfg = ModelConfig(num_layers=num_layers, num_dense_layers=num_dense)

  if rank == 0:
    print(f"\nLoading model with {num_layers} layers on {world_size} GPUs...")

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
    print("Model loaded. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
  else:
    tokenizer = None

  # Broadcast tokenizer vocab size to verify
  if rank == 0:
    vocab_size = torch.tensor([len(tokenizer)], dtype=torch.int64, device=device)
  else:
    vocab_size = torch.empty(1, dtype=torch.int64, device=device)
  dist.broadcast(vocab_size, src=0)

  # Only rank 0 needs tokenizer for decode
  if rank != 0:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

  prompts = [
    "The capital of France is",
    "def fibonacci(n):",
  ]

  if rank == 0:
    print("\n" + "=" * 60)
    print("Generation Tests (greedy decoding)")
    print("=" * 60)

  for prompt in prompts:
    dist.barrier()
    if rank == 0:
      print(f"\n--- Prompt: {repr(prompt)} ---")

    try:
      output = generate(model, tokenizer, prompt, max_new_tokens=30, device=device, rank=rank)
      if rank == 0:
        print(f"Output: {output}")
    except Exception as e:
      if rank == 0:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
