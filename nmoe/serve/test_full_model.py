# SPDX-License-Identifier: Apache-2.0
"""Test full 61-layer model with tensor parallelism.

Run with: torchrun --nproc_per_node=8 -m nmoe.serve.test_full_model
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


def main():
  # Initialize distributed
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  torch.cuda.set_device(local_rank)
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{local_rank}")

  if rank == 0:
    print("=" * 60)
    print("Full Model Test (61 layers, 8-way TP)")
    print("=" * 60)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Full model config: 61 transformer layers (0-60), layer 61 in ckpt is MTP (unused)
  cfg = ModelConfig(num_layers=61, num_dense_layers=3)

  if rank == 0:
    print(f"Creating model: {cfg.num_layers} layers, world_size={world_size}")

  # DeepEP buffer for MoE communication
  # Use Buffer API to calculate required buffer sizes based on hidden dim and world size
  hidden_bytes = cfg.hidden_size * 2  # bf16 = 2 bytes
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)

  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  if rank == 0:
    print(f"DeepEP buffer: {num_nvl_bytes / 1e6:.1f} MB NVL")

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)

  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  if rank == 0:
    print(f"GPU {rank} memory after model: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

  # Load checkpoint with TP sharding
  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3.2-Speciale")
  if rank == 0:
    print(f"Loading checkpoint from {ckpt_path}...")

  missing, unexpected = load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)

  if rank == 0:
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    print(f"GPU {rank} memory after load: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")

  dist.barrier()

  # Test forward pass
  if rank == 0:
    print("\nRunning forward pass...")

  B, S = 1, 8
  input_ids = torch.tensor([[1, 100, 1000, 10000, 50000, 100000, 1234, 5678]], device=device)
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
      input_ids,
      positions,
      kv_caches=kv_caches,
      idx_k_caches=idx_k_caches,
      block_table=block_table,
      cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S],
      out_loc=out_loc,
    )

  if rank == 0:
    print(f"Logits: shape={logits.shape}, amax={logits.abs().max().item():.4f}")
    print(f"Has NaN: {torch.isnan(logits).any().item()}")

  has_nan = torch.isnan(logits).any()
  dist.barrier()

  if has_nan:
    if rank == 0:
      print("FAILED - logits contain NaN")
    dist.destroy_process_group()
    return

  # Generation test - all ranks must participate in forward passes
  if rank == 0:
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    prompt = "The capital of France is"

    # DeepSeek-V3.2-Speciale ships an explicit encoder implementation in
    # `encoding/encoding_dsv32.py`, which (for non-thinking mode) terminates the
    # user message with `</think>` after the assistant tag.
    formatted = (
      "<｜begin▁of▁sentence｜>"
      "You are a helpful Assistant."
      "\n\n"
      "<｜User｜>" + prompt + "<｜Assistant｜></think>"
    )
    input_ids = tokenizer.encode(formatted, return_tensors="pt", add_special_tokens=False).to(device)
  else:
    tokenizer = None
    input_ids = torch.zeros((1, 5), dtype=torch.int64, device=device)  # placeholder

  # Broadcast input_ids shape and content from rank 0
  seq_len = torch.tensor([input_ids.size(1)], dtype=torch.int64, device=device)
  dist.broadcast(seq_len, src=0)
  S = seq_len.item()

  if rank != 0:
    input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
  dist.broadcast(input_ids, src=0)

  # Reset caches for generation
  for kv in kv_caches:
    kv.zero_()
  for idx in idx_k_caches:
    idx.zero_()

  positions = torch.arange(S, device=device).unsqueeze(0)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

  generated = input_ids[0].tolist() if rank == 0 else []

  # Prefill - all ranks
  with torch.no_grad():
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  # Cross-rank sanity: after vocab all-gather in model forward, every rank must
  # agree on the global argmax at the last prompt position.
  local_argmax = logits[0, -1, :].argmax().to(torch.int64)
  gathered = [torch.empty_like(local_argmax) for _ in range(world_size)]
  dist.all_gather(gathered, local_argmax.contiguous())
  if rank == 0:
    for r, v in enumerate(gathered):
      if int(v) != int(gathered[0]):
        raise RuntimeError(f"TP inconsistency: rank {r} argmax={int(v)} != rank0 argmax={int(gathered[0])}")

  # TP correctness: sample on rank 0 and broadcast token to all ranks.
  if rank == 0:
    next_token = int(logits[0, -1, :].argmax().item())
  else:
    next_token = 0
  next_token_t = torch.tensor([next_token], dtype=torch.int64, device=device)
  dist.broadcast(next_token_t, src=0)
  next_token = int(next_token_t.item())
  if rank == 0:
    generated.append(next_token)
    print(f"  Prefill: token={next_token}, top5_logits={logits[0, -1, :].topk(5)}")

  # Decode loop - all ranks must participate
  eos_id = tokenizer.eos_token_id if rank == 0 else 0
  for i in range(20):
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

    if rank == 0:
      next_token = int(logits[0, 0, :].argmax().item())
    else:
      next_token = 0
    next_token_t = torch.tensor([next_token], dtype=torch.int64, device=device)
    dist.broadcast(next_token_t, src=0)
    next_token = int(next_token_t.item())
    if rank == 0:
      generated.append(next_token)
      if i < 5:  # Only print first 5 steps
        top5 = logits[0, 0, :].topk(5)
        print(f"  Step {i}: token={next_token}, top5_ids={top5.indices.tolist()}, top5_vals={[f'{v:.2f}' for v in top5.values.tolist()]}")
      if next_token == eos_id:
        break

    # Broadcast stop signal from rank 0
    stop = torch.tensor([1 if (rank == 0 and next_token == eos_id) else 0], device=device)
    dist.broadcast(stop, src=0)
    if stop.item():
      break

  if rank == 0:
    print(f"\nGeneration test:")
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(generated)}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
