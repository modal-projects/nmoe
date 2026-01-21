# SPDX-License-Identifier: Apache-2.0
"""Debug decode path - check DSA indices and attention."""

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
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    S = input_ids.size(1)
    print(f"Prompt: '{prompt}' ({S} tokens)")
    print(f"Tokens: {input_ids[0].tolist()}")
  else:
    input_ids = None
    S = 0

  # Broadcast
  seq_len_t = torch.tensor([S] if rank == 0 else [0], dtype=torch.int64, device=device)
  dist.broadcast(seq_len_t, src=0)
  S = seq_len_t.item()

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

  # === PREFILL ===
  if rank == 0:
    print(f"\n=== PREFILL ({S} tokens) ===")
  positions = torch.arange(S, device=device).unsqueeze(0)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  if rank == 0:
    top5 = logits[0, -1, :].topk(5)
    print(f"Prefill logits[-1] top5: {top5.indices.tolist()}")
    print(f"  = {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
    next_token = int(logits[0, -1, :].argmax().item())
    print(f"Next token: {next_token} = '{tokenizer.decode([next_token])}'")
  else:
    next_token = 0

  next_token_t = torch.tensor([next_token], dtype=torch.int64, device=device)
  dist.broadcast(next_token_t, src=0)
  next_token = int(next_token_t.item())

  # === DECODE STEP 1 ===
  if rank == 0:
    print(f"\n=== DECODE Step 1 (token {next_token} at position {S}) ===")

  # After prefill, cache has S tokens (positions 0..S-1)
  # Now we process position S
  cache_seqlens = torch.tensor([S + 1], dtype=torch.int32, device=device)
  inp = torch.tensor([[next_token]], dtype=torch.int64, device=device)
  pos = torch.tensor([[S]], dtype=torch.int64, device=device)
  out_loc_decode = torch.tensor([[S]], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits2 = model(
      inp, pos,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S + 1], out_loc=out_loc_decode,
    )

  if rank == 0:
    top5 = logits2[0, 0, :].topk(5)
    print(f"Decode logits[0] top5: {top5.indices.tolist()}")
    print(f"  = {[tokenizer.decode([t]) for t in top5.indices.tolist()]}")
    next_token2 = int(logits2[0, 0, :].argmax().item())
    print(f"Next token: {next_token2} = '{tokenizer.decode([next_token2])}'")

    # Compare prefill vs decode for same position
    print(f"\n=== Comparing prefill last position vs decode ===")
    print(f"Prefill position {S-1} predicted: {logits[0, -1, :].argmax().item()} = '{tokenizer.decode([logits[0, -1, :].argmax().item()])}'")
    print(f"Decode position {S} predicted: {logits2[0, 0, :].argmax().item()} = '{tokenizer.decode([logits2[0, 0, :].argmax().item()])}'")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
