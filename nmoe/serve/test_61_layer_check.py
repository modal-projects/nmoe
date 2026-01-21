# SPDX-License-Identifier: Apache-2.0
"""Test full 61-layer model with token analysis."""

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
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  torch.cuda.set_device(local_rank)
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{local_rank}")

  if rank == 0:
    print("=" * 60)
    print("61-Layer Model Analysis")
    print(f"World size: {world_size}")
    print("=" * 60)

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

  # Test 1: Simple completion prompt
  if rank == 0:
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
  else:
    input_ids = None

  # Broadcast input
  seq_len = torch.tensor([input_ids.size(1) if input_ids is not None else 0], dtype=torch.int64, device=device)
  dist.broadcast(seq_len, src=0)
  S = seq_len.item()

  if rank != 0:
    input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
  dist.broadcast(input_ids, src=0)

  positions = torch.arange(S, device=device).unsqueeze(0)

  num_blocks = 2
  kv_caches = [
    torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    for _ in range(cfg.num_layers)
  ]
  idx_k_caches = [
    torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    for _ in range(cfg.num_layers)
  ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  with torch.no_grad():
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  # Check consistency across ranks
  local_argmax = logits[0, -1, :].argmax().to(torch.int64)
  argmaxes = [torch.empty_like(local_argmax) for _ in range(world_size)]
  dist.all_gather(argmaxes, local_argmax.contiguous())

  if rank == 0:
    print(f"\nPrompt: '{prompt}'")
    print(f"Input tokens: {input_ids[0].tolist()}")
    print(f"Logits shape: {logits.shape}")
    print(f"\nArgmax across ranks: {[int(a) for a in argmaxes]}")
    match = all(int(a) == int(argmaxes[0]) for a in argmaxes)
    print(f"All ranks agree: {match}")

    top10 = logits[0, -1, :].topk(10)
    print(f"\nTop 10 predictions:")
    for idx, val in zip(top10.indices.tolist(), top10.values.tolist()):
      token_str = tokenizer.decode([idx])
      print(f"  {idx}: '{token_str}' (logit={val:.2f})")

    # Check for expected token
    paris_tokens = tokenizer.encode(" Paris", add_special_tokens=False)
    print(f"\n'Paris' tokens: {paris_tokens}")
    for pt in paris_tokens:
      logit_val = logits[0, -1, pt].item()
      print(f"  Token {pt} ('{tokenizer.decode([pt])}'): logit={logit_val:.2f}")

  # Test 2: Compare 4-layer vs 61-layer
  if rank == 0:
    print("\n" + "=" * 60)
    print("Comparing vs 4-layer model outputs")
    print("=" * 60)

  # Also compare logits statistics
  logits_mean = logits[0, -1, :].float().mean()
  logits_std = logits[0, -1, :].float().std()
  logits_max = logits[0, -1, :].float().max()
  logits_min = logits[0, -1, :].float().min()

  stats = torch.tensor([logits_mean, logits_std, logits_max, logits_min], device=device)
  all_stats = [torch.empty_like(stats) for _ in range(world_size)]
  dist.all_gather(all_stats, stats.contiguous())

  if rank == 0:
    print("\nLogits statistics:")
    print(f"  Mean: {logits_mean:.4f}")
    print(f"  Std:  {logits_std:.4f}")
    print(f"  Max:  {logits_max:.4f}")
    print(f"  Min:  {logits_min:.4f}")
    print(f"  Stats consistent: {all(torch.allclose(s, all_stats[0]) for s in all_stats)}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
