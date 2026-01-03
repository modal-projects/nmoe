# SPDX-License-Identifier: Apache-2.0
"""End-to-end generation and benchmark tests for nmoe.serve."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

from nmoe.serve.ckpt import load_checkpoint
from nmoe.serve.engine import Engine, EngineConfig
from nmoe.serve.config import PROFILES
from nmoe.serve.types import Batch, Request
from nmoe.serve.model import DeepSeekV3, ModelConfig, init_distributed
from nmoe.serve.types import SamplingParams


@dataclass
class BenchmarkResult:
  """Results from a benchmark run."""
  prompt: str
  output: str
  prompt_tokens: int
  output_tokens: int
  prefill_time_ms: float
  decode_time_ms: float
  total_time_ms: float
  prefill_tps: float  # tokens per second
  decode_tps: float
  overall_tps: float
  ttft_ms: float  # time to first token


def load_model_and_tokenizer(
  model_path: str,
  num_layers: Optional[int] = None,
  device: str = "cuda:0",
) -> Tuple[Engine, any, ModelConfig]:
  """Load model and tokenizer."""
  from transformers import AutoTokenizer

  print(f"Loading tokenizer from {model_path}...")
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

  # Model config
  if num_layers is not None:
    num_dense = min(3, num_layers)
    cfg = ModelConfig(num_layers=num_layers, num_dense_layers=num_dense)
  else:
    cfg = ModelConfig()

  print(f"Creating model with {cfg.num_layers} layers ({cfg.num_dense_layers} dense)...")

  # Initialize distributed (single GPU)
  init_distributed(rank=0, world_size=1)

  # Engine config
  engine_config = EngineConfig(
    num_pages=2048,
    page_size=64,
    num_layers=cfg.num_layers,
    kv_lora_rank=cfg.kv_lora_rank,
    qk_rope_head_dim=cfg.qk_rope_head_dim,
    idx_dim=cfg.dsa_idx_dim,
    max_batch_size=32,
    max_seq_len=8192,
  )

  # Create engine
  print("Creating engine...")
  engine = Engine(
    model_config=cfg,
    engine_config=engine_config,
    rank=0,
    world_size=1,
  )

  # Load checkpoint
  print(f"Loading checkpoint from {model_path}...")
  start = time.time()
  missing, unexpected = load_checkpoint(engine.model, model_path, rank=0, world_size=1, cfg=cfg)
  load_time = time.time() - start
  print(f"Checkpoint loaded in {load_time:.1f}s")

  # Filter unexpected to only our layers
  our_unexpected = [k for k in unexpected if any(f'layers.{i}.' in k for i in range(cfg.num_layers))]
  if missing:
    print(f"  Missing keys: {len(missing)}")
    for k in sorted(missing)[:5]:
      print(f"    {k}")
  if our_unexpected:
    print(f"  Unexpected keys (our layers): {len(our_unexpected)}")
    for k in sorted(our_unexpected)[:5]:
      print(f"    {k}")

  return engine, tokenizer, cfg


def generate(
  engine: Engine,
  tokenizer,
  prompt: str,
  max_tokens: int = 128,
  temperature: float = 0.0,
  top_p: float = 1.0,
) -> BenchmarkResult:
  """Run generation and collect metrics."""
  # Tokenize (must be on CPU for Request)
  input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)[0].cpu()
  prompt_tokens = len(input_ids)

  sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

  # Sanity: token ids must be in range of model vocab.
  if prompt_tokens > 0:
    max_id = int(input_ids.max().item())
    if max_id >= int(engine.model_config.vocab_size):
      raise ValueError(f"Token id {max_id} exceeds model vocab_size={engine.model_config.vocab_size}")

  # Allocate pages for this request (simple sequential allocation for testing).
  page_size = engine.engine_config.page_size
  num_pages = (prompt_tokens + max_tokens + page_size - 1) // page_size
  page_ids = list(range(num_pages))
  if page_ids and max(page_ids) >= int(engine.engine_config.num_pages):
    raise ValueError("page_ids exceed engine KV cache capacity")

  def _out_loc_for_len(total_len: int) -> torch.Tensor:
    pages_needed = (total_len + page_size - 1) // page_size
    if pages_needed > len(page_ids):
      raise ValueError(f"Need {pages_needed} pages but only have {len(page_ids)}")
    pos = torch.arange(total_len, dtype=torch.int64)
    page = torch.div(pos, page_size, rounding_mode="floor")
    off = pos % page_size
    pids = torch.tensor(page_ids, dtype=torch.int64)
    phys = pids.index_select(0, page) * page_size + off
    return phys.to(torch.int32)

  req = Request(
    uid=0,
    input_ids=input_ids,
    sampling_params=sampling_params,
    profile_name="production_generate",
    forward_spec=PROFILES["production_generate"].to_forward_spec(),
    cached_len=0,
    table_idx=0,
    page_ids=page_ids,
  )

  # Prefill
  torch.cuda.synchronize()
  prefill_start = time.time()

  batch = Batch(
    reqs=[req],
    phase="prefill",
  )
  # Model expects [B, S] shape for all inputs, out_loc must be int32
  batch.input_ids = input_ids.unsqueeze(0).cuda().to(torch.int64)
  batch.positions = torch.arange(prompt_tokens, device="cuda", dtype=torch.int64).unsqueeze(0)
  batch.out_loc = _out_loc_for_len(prompt_tokens).cuda().unsqueeze(0)

  output = engine.forward_batch(batch)
  torch.cuda.synchronize()
  prefill_end = time.time()
  ttft = prefill_end - prefill_start

  # Get sampled token from engine output
  req.cached_len += prompt_tokens
  next_token = int(output.next_tokens_cpu.tolist()[0])
  output_ids = [next_token]
  req.output_ids = output_ids.copy()

  # Decode loop
  torch.cuda.synchronize()
  decode_start = time.time()

  for _ in range(max_tokens - 1):
    # Check EOS (DeepSeek uses 151643; tokenizer-dependent secondary EOS=2)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and output_ids[-1] == int(eos_id):
      break

    # Decode step
    cur_pos = prompt_tokens + len(output_ids) - 1
    batch = Batch(
      reqs=[req],
      phase="decode",
    )
    # Model expects [B, S] shape for all inputs, out_loc must be int32
    batch.input_ids = torch.tensor([[output_ids[-1]]], device="cuda", dtype=torch.int64)
    batch.positions = torch.tensor([[cur_pos]], device="cuda", dtype=torch.int64)
    batch.out_loc = _out_loc_for_len(cur_pos + 1)[-1:].cuda().view(1, 1)

    output = engine.forward_batch(batch)

    # Get sampled token
    req.cached_len += 1
    next_token = int(output.next_tokens_cpu.tolist()[0])
    output_ids.append(next_token)
    req.output_ids = output_ids.copy()

  torch.cuda.synchronize()
  decode_end = time.time()

  # Decode output
  output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

  # Calculate metrics
  prefill_time_ms = (prefill_end - prefill_start) * 1000
  decode_time_ms = (decode_end - decode_start) * 1000
  total_time_ms = prefill_time_ms + decode_time_ms
  output_tokens = len(output_ids)

  prefill_tps = prompt_tokens / (prefill_time_ms / 1000) if prefill_time_ms > 0 else 0
  decode_tps = (output_tokens - 1) / (decode_time_ms / 1000) if decode_time_ms > 0 and output_tokens > 1 else 0
  overall_tps = (prompt_tokens + output_tokens) / (total_time_ms / 1000) if total_time_ms > 0 else 0

  return BenchmarkResult(
    prompt=prompt,
    output=output_text,
    prompt_tokens=prompt_tokens,
    output_tokens=output_tokens,
    prefill_time_ms=prefill_time_ms,
    decode_time_ms=decode_time_ms,
    total_time_ms=total_time_ms,
    prefill_tps=prefill_tps,
    decode_tps=decode_tps,
    overall_tps=overall_tps,
    ttft_ms=ttft * 1000,
  )


def run_generation_tests(engine: Engine, tokenizer) -> List[BenchmarkResult]:
  """Run generation tests with various prompts."""
  test_prompts = [
    "Hello, my name is",
    "The capital of France is",
    "def fibonacci(n):\n",
    "Explain quantum computing in simple terms:",
    "Write a haiku about programming:",
  ]

  results = []
  for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt[:50]}...")
    result = generate(engine, tokenizer, prompt, max_tokens=64, temperature=0.0)
    results.append(result)

    print(f"Output: {result.output[:100]}...")
    print(f"Tokens: {result.prompt_tokens} -> {result.output_tokens}")
    print(f"TTFT: {result.ttft_ms:.1f}ms")
    print(f"Prefill: {result.prefill_time_ms:.1f}ms ({result.prefill_tps:.0f} tok/s)")
    print(f"Decode: {result.decode_time_ms:.1f}ms ({result.decode_tps:.1f} tok/s)")

  return results


def run_benchmark(
  engine: Engine,
  tokenizer,
  prompt_lens: List[int] = [128, 256, 512, 1024],
  output_len: int = 128,
  num_runs: int = 3,
) -> dict:
  """Run performance benchmark."""
  print(f"\n{'='*60}")
  print("PERFORMANCE BENCHMARK")
  print(f"{'='*60}")

  results = {}
  for prompt_len in prompt_lens:
    # Generate prompt of target length
    base_prompt = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = tokenizer.encode(base_prompt)[:prompt_len]
    prompt = tokenizer.decode(tokens)

    print(f"\nPrompt length: {prompt_len} tokens, Output: {output_len} tokens")

    run_results = []
    for run in range(num_runs):
      # Clear cache
      gc.collect()
      torch.cuda.empty_cache()

      result = generate(engine, tokenizer, prompt, max_tokens=output_len, temperature=0.0)
      run_results.append(result)

      print(f"  Run {run+1}: TTFT={result.ttft_ms:.1f}ms, "
            f"Prefill={result.prefill_tps:.0f} tok/s, "
            f"Decode={result.decode_tps:.1f} tok/s")

    # Average results
    avg_ttft = sum(r.ttft_ms for r in run_results) / len(run_results)
    avg_prefill_tps = sum(r.prefill_tps for r in run_results) / len(run_results)
    avg_decode_tps = sum(r.decode_tps for r in run_results) / len(run_results)

    results[prompt_len] = {
      "ttft_ms": avg_ttft,
      "prefill_tps": avg_prefill_tps,
      "decode_tps": avg_decode_tps,
    }

    print(f"  Avg: TTFT={avg_ttft:.1f}ms, "
          f"Prefill={avg_prefill_tps:.0f} tok/s, "
          f"Decode={avg_decode_tps:.1f} tok/s")

  return results


def run_accuracy_test(engine: Engine, tokenizer) -> dict:
  """Run accuracy/quality tests."""
  print(f"\n{'='*60}")
  print("ACCURACY TESTS")
  print(f"{'='*60}")

  tests = [
    {
      "name": "math_simple",
      "prompt": "What is 2 + 2? Answer with just the number:",
      "expected": ["4"],
    },
    {
      "name": "capital_city",
      "prompt": "What is the capital of Japan? Answer with just the city name:",
      "expected": ["Tokyo"],
    },
    {
      "name": "code_completion",
      "prompt": "def add(a, b):\n    return",
      "expected": ["a + b", "a+b"],
    },
    {
      "name": "language",
      "prompt": "Translate 'hello' to Spanish:",
      "expected": ["hola", "Hola"],
    },
  ]

  results = {}
  passed = 0
  for test in tests:
    result = generate(engine, tokenizer, test["prompt"], max_tokens=32, temperature=0.0)
    output = result.output.strip().lower()

    # Check if any expected answer is in output
    match = any(exp.lower() in output for exp in test["expected"])

    results[test["name"]] = {
      "prompt": test["prompt"],
      "output": result.output,
      "expected": test["expected"],
      "passed": match,
    }

    status = "PASS" if match else "FAIL"
    if match:
      passed += 1
    print(f"\n[{status}] {test['name']}")
    print(f"  Prompt: {test['prompt'][:50]}...")
    print(f"  Output: {result.output[:50]}...")
    print(f"  Expected: {test['expected']}")

  print(f"\nAccuracy: {passed}/{len(tests)} tests passed")
  return results


def main():
  parser = argparse.ArgumentParser(description="nmoe.serve end-to-end tests")
  parser.add_argument("--model-path", type=str, default="/data/models/DeepSeek-V3.2-Speciale")
  parser.add_argument("--num-layers", type=int, default=None, help="Number of layers (default: all)")
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--test", type=str, choices=["all", "generate", "benchmark", "accuracy"], default="all")
  parser.add_argument("--output", type=str, help="Output JSON file for results")
  args = parser.parse_args()

  torch.set_default_device(args.device)
  torch.cuda.set_device(args.device)

  print(f"nmoe.serve End-to-End Tests")
  print(f"{'='*60}")
  print(f"Model: {args.model_path}")
  print(f"Layers: {args.num_layers or 'all'}")
  print(f"Device: {args.device}")
  print(f"CUDA: {torch.cuda.get_device_name()}")
  print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

  engine = None
  try:
    # Load model
    engine, tokenizer, cfg = load_model_and_tokenizer(
      args.model_path,
      num_layers=args.num_layers,
      device=args.device,
    )

    all_results = {}

    # Run tests
    if args.test in ["all", "generate"]:
      gen_results = run_generation_tests(engine, tokenizer)
      all_results["generation"] = [
        {
          "prompt": r.prompt,
          "output": r.output,
          "prompt_tokens": r.prompt_tokens,
          "output_tokens": r.output_tokens,
          "ttft_ms": r.ttft_ms,
          "prefill_tps": r.prefill_tps,
          "decode_tps": r.decode_tps,
        }
        for r in gen_results
      ]

    if args.test in ["all", "benchmark"]:
      bench_results = run_benchmark(engine, tokenizer)
      all_results["benchmark"] = bench_results

    if args.test in ["all", "accuracy"]:
      acc_results = run_accuracy_test(engine, tokenizer)
      all_results["accuracy"] = acc_results

    # Save results
    if args.output:
      with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
      print(f"\nResults saved to {args.output}")

    print(f"\n{'='*60}")
    print("Tests complete!")
  finally:
    if engine is not None:
      engine.shutdown()


if __name__ == "__main__":
  main()
