# SPDX-License-Identifier: Apache-2.0
"""Compare prefill vs decode outputs for the same position."""

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
  else:
    input_ids = None
    S = 0

  seq_len_t = torch.tensor([S] if rank == 0 else [0], dtype=torch.int64, device=device)
  dist.broadcast(seq_len_t, src=0)
  S = seq_len_t.item()

  if rank != 0:
    input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
  dist.broadcast(input_ids, src=0)

  # === TEST 1: Prefill all S+1 tokens (6 tokens including next) ===
  if rank == 0:
    print(f"\n=== TEST 1: Prefill {S+1} tokens at once ===")

  # Add a "next" token to prefill
  next_token = 223  # space
  input_ids_full = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
  S_full = S + 1

  num_blocks = 4
  kv_caches_1 = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches_1 = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  positions_1 = torch.arange(S_full, device=device).unsqueeze(0)
  out_loc_1 = torch.arange(S_full, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens_1 = torch.tensor([S_full], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits_prefill = model(
      input_ids_full, positions_1,
      kv_caches=kv_caches_1, idx_k_caches=idx_k_caches_1,
      block_table=block_table, cache_seqlens=cache_seqlens_1,
      cache_seqlens_cpu=[S_full], out_loc=out_loc_1,
    )

  if rank == 0:
    # Position S (the next token after original prompt)
    top5_prefill = logits_prefill[0, S, :].topk(5)
    print(f"Prefill position {S} top5: {top5_prefill.indices.tolist()}")
    print(f"  = {[tokenizer.decode([t]) for t in top5_prefill.indices.tolist()]}")

  # === TEST 2: Prefill S tokens, then decode 1 token ===
  if rank == 0:
    print(f"\n=== TEST 2: Prefill {S} tokens, then decode 1 token ===")

  kv_caches_2 = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches_2 = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]

  # Prefill S tokens
  positions_2a = torch.arange(S, device=device).unsqueeze(0)
  out_loc_2a = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens_2a = torch.tensor([S], dtype=torch.int32, device=device)

  with torch.no_grad():
    _ = model(
      input_ids, positions_2a,
      kv_caches=kv_caches_2, idx_k_caches=idx_k_caches_2,
      block_table=block_table, cache_seqlens=cache_seqlens_2a,
      cache_seqlens_cpu=[S], out_loc=out_loc_2a,
    )

  # Decode 1 token at position S
  inp_decode = torch.tensor([[next_token]], device=device)
  pos_decode = torch.tensor([[S]], dtype=torch.int64, device=device)
  out_loc_decode = torch.tensor([[S]], dtype=torch.int32, device=device)
  cache_seqlens_decode = torch.tensor([S + 1], dtype=torch.int32, device=device)

  with torch.no_grad():
    logits_decode = model(
      inp_decode, pos_decode,
      kv_caches=kv_caches_2, idx_k_caches=idx_k_caches_2,
      block_table=block_table, cache_seqlens=cache_seqlens_decode,
      cache_seqlens_cpu=[S + 1], out_loc=out_loc_decode,
    )

  if rank == 0:
    top5_decode = logits_decode[0, 0, :].topk(5)
    print(f"Decode position {S} top5: {top5_decode.indices.tolist()}")
    print(f"  = {[tokenizer.decode([t]) for t in top5_decode.indices.tolist()]}")

    # Compare
    print(f"\n=== COMPARISON ===")
    print(f"Prefill position {S} logit for top-1: {logits_prefill[0, S, top5_prefill.indices[0]]:.4f}")
    print(f"Decode position {S} logit for top-1:  {logits_decode[0, 0, top5_decode.indices[0]]:.4f}")

    # Check if they're selecting from similar distribution
    cos_sim = torch.nn.functional.cosine_similarity(
      logits_prefill[0, S, :].unsqueeze(0),
      logits_decode[0, 0, :].unsqueeze(0)
    ).item()
    print(f"Cosine similarity between logits: {cos_sim:.4f}")

    if cos_sim < 0.9:
      print("⚠️  WARNING: Prefill and decode produce very different outputs!")
      print("This suggests the decode path has a bug.")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
