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


def _greedy_argmax(logits: torch.Tensor) -> int:
  """Greedy argmax over full vocab logits (TP=1 mode)."""
  return int(torch.argmax(logits, dim=-1).item())


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

  # Allocate KV caches based on attention type
  max_seq = S + max_new_tokens + 64
  num_blocks = (max_seq + 63) // 64
  num_layers = cfg.num_layers

  if cfg.attention_type == "dsa":
    # DSA: compressed KV + indexer caches
    kv_caches = [
      torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
      for _ in range(num_layers)
    ]
    idx_k_caches = [
      torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
      for _ in range(num_layers)
    ]
    kv_caches_latent = None
    kv_caches_rope = None
  else:
    # MLA: latent + rope caches
    kv_caches = None
    idx_k_caches = None
    page_size = 64
    kv_caches_latent = [
      torch.zeros(num_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
      for _ in range(num_layers)
    ]
    kv_caches_rope = [
      torch.zeros(num_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
      for _ in range(num_layers)
    ]
  block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)

  generated_ids = input_ids.tolist()[0]

  # Prefill
  positions = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

  with torch.inference_mode():
    # Prefill: use "dense" mode for MLA (no cached prefix)
    prefill_mode = "dense" if cfg.attention_type == "mla" else None
    logits = model(
      input_ids,
      positions,
      kv_caches=kv_caches,
      idx_k_caches=idx_k_caches,
      kv_caches_latent=kv_caches_latent,
      kv_caches_rope=kv_caches_rope,
      block_table=block_table,
      cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S],
      out_loc=out_loc,
      prefill_mode=prefill_mode,
    )

    # Greedy argmax (TP=1 means full vocab on each rank)
    next_token = _greedy_argmax(logits[0, -1, :])
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

      # Decode: prefill_mode=None for single-token generation
      logits = model(
        input_ids,
        positions,
        kv_caches=kv_caches,
        idx_k_caches=idx_k_caches,
        kv_caches_latent=kv_caches_latent,
        kv_caches_rope=kv_caches_rope,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[cur_pos],
        out_loc=out_loc,
        prefill_mode=None,
      )

      next_token = _greedy_argmax(logits[0, 0, :])
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

  from nmoe.serve.model import DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint, load_sharded_checkpoint, load_model_config
  from deep_ep import Buffer

  # TP=1 for EP-only mode (dynamic disagg) - attention weights are replicated
  init_distributed(rank, world_size, tp_size=1)

  # Load model config from checkpoint (auto-detects MLA vs DSA)
  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324")
  cfg = load_model_config(ckpt_path)

  if rank == 0:
    print("=" * 60)
    print("Multi-GPU Generation Test")
    print("=" * 60)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Attention type: {cfg.attention_type}")
    print(f"Layers: {cfg.num_layers} ({cfg.num_dense_layers} dense)")

  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0, explicitly_destroy=True)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  # Auto-detect checkpoint format: mp8 (pre-sharded) vs HF (original)
  sharded_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(sharded_path):
    if rank == 0:
      print("Using pre-sharded checkpoint (load_sharded_checkpoint)")
    load_sharded_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)
  else:
    if rank == 0:
      print("Using HF checkpoint (load_checkpoint)")
    load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("Model loaded. Loading tokenizer...")

  # All ranks load tokenizer (needed for encoding prompts)
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
  buffer.destroy()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
