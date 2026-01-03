# SPDX-License-Identifier: Apache-2.0
"""Test text generation to validate model produces coherent output."""

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


def generate(
  model,
  tokenizer,
  prompt: str,
  max_new_tokens: int = 64,
  temperature: float = 0.7,
  top_k: int = 50,
  device: torch.device = None,
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

    # Sample from last position
    next_logits = logits[0, -1, :] / temperature
    if top_k > 0:
      v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
      next_logits[next_logits < v[-1]] = float("-inf")
    probs = torch.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    generated_ids.append(next_token)

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

      next_logits = logits[0, 0, :] / temperature
      if top_k > 0:
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[-1]] = float("-inf")
      probs = torch.softmax(next_logits, dim=-1)
      next_token = torch.multinomial(probs, num_samples=1).item()
      generated_ids.append(next_token)

      # Check for EOS
      if next_token == tokenizer.eos_token_id:
        break

  return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
  print("=" * 60)
  print("nmoe.serve Generation Test")
  print("=" * 60)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  # Initialize distributed
  if not dist.is_initialized():
    from nmoe.serve.test_utils import init_nccl_process_group
    init_nccl_process_group(rank=0, world_size=1)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(0, 1)

  # Model config - use more layers for better quality
  num_layers = 8  # Use 8 layers for reasonable quality
  num_dense = 3
  cfg = ModelConfig(num_layers=num_layers, num_dense_layers=num_dense)

  print(f"\nLoading model with {num_layers} layers...")
  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  # Load checkpoint
  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3.2-Speciale")
  print(f"Loading checkpoint from {ckpt_path}...")
  missing, unexpected = load_checkpoint(model, ckpt_path, rank=0, world_size=1, cfg=cfg)
  print(f"Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

  # Load tokenizer
  print("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

  # Test prompts
  prompts = [
    "The capital of France is",
    "def fibonacci(n):",
    "In machine learning, a neural network",
    "The quick brown fox",
  ]

  print("\n" + "=" * 60)
  print("Generation Tests")
  print("=" * 60)

  for prompt in prompts:
    print(f"\n--- Prompt: {repr(prompt)} ---")
    try:
      output = generate(model, tokenizer, prompt, max_new_tokens=32, temperature=0.7, device=device)
      print(f"Output: {output}")

      # Basic sanity checks
      if len(output) <= len(prompt):
        print("WARNING: No new tokens generated!")
      elif output == prompt:
        print("WARNING: Output identical to prompt!")
      else:
        print("✓ Generation produced new text")
    except Exception as e:
      print(f"ERROR: {e}")
      import traceback
      traceback.print_exc()

  print("\n" + "=" * 60)
  print("Generation Test Complete")
  print("=" * 60)


if __name__ == "__main__":
  main()
