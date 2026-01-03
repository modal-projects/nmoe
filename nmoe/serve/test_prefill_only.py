# SPDX-License-Identifier: Apache-2.0
"""Test prefill-only prediction."""

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

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    tokenizer = AutoTokenizer.from_pretrained("/data/models/DeepSeek-V3.2-Speciale", trust_remote_code=True)

  # Test prompts
  prompts = ["1+1=", "The capital of France is", "Hello"]

  for prompt in prompts:
    if rank == 0:
      input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
      print(f"\n{'='*60}")
      print(f"Prompt: '{prompt}'")
      print(f"Tokens: {input_ids.tolist()}")
    else:
      input_ids = None

    seq_len = torch.tensor([input_ids.size(1) if input_ids is not None else 0], dtype=torch.int64, device=device)
    dist.broadcast(seq_len, src=0)
    S = seq_len.item()

    if rank != 0:
      input_ids = torch.zeros((1, S), dtype=torch.int64, device=device)
    dist.broadcast(input_ids, src=0)

    num_blocks = 4
    kv_caches = [torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
    idx_k_caches = [torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
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
      last_logits = logits[0, -1, :]
      top10 = last_logits.topk(10)
      print(f"Top 10 predictions for next token:")
      for i, (idx, score) in enumerate(zip(top10.indices.tolist(), top10.values.tolist())):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx:6d} ({score:8.3f}): '{token}'")
      argmax_id = last_logits.argmax().item()
      print(f"Argmax: {argmax_id} = '{tokenizer.decode([argmax_id])}'")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
