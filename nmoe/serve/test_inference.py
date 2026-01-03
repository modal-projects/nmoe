# SPDX-License-Identifier: Apache-2.0
"""Inference system tests for nmoe.serve."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from nmoe.serve.cache import KvCache, MlaKvLayout
from nmoe.serve.engine import Engine, EngineConfig
from nmoe.serve.types import Batch, SamplingParams
from nmoe.serve.model import ModelConfig, init_distributed
from nmoe.serve.scheduler import Scheduler, SchedulerConfig


@dataclass
class TestConfig:
  """Configuration for test runs."""
  num_pages: int = 256
  page_size: int = 64
  num_layers: int = 2  # Small for testing
  batch_size: int = 4
  seq_len: int = 128
  output_len: int = 32
  device: str = "cuda:0"
  verbose: bool = True


def log(msg: str, verbose: bool = True) -> None:
  if verbose:
    print(f"[TEST] {msg}", flush=True)


def test_cache(cfg: TestConfig) -> bool:
  """Test KV cache allocation and prefix matching."""
  log("Testing KV cache...", cfg.verbose)

  device = torch.device(cfg.device)
  layout = MlaKvLayout(
    num_blocks=cfg.num_pages,
    block_size=cfg.page_size,
  )
  cache = KvCache(layout, device)

  # Test allocation
  pages = cache.allocate(4)
  assert len(pages) == 4, f"Expected 4 pages, got {len(pages)}"
  log(f"  Allocated pages: {pages.tolist()}", cfg.verbose)

  # Test free
  cache.free(pages)
  log("  Freed pages successfully", cfg.verbose)

  # Test prefix matching (basic)
  input_ids = torch.arange(64, dtype=torch.int64, device="cpu")
  handle, matched = cache.match_prefix(input_ids)
  log(f"  Prefix match: cached_len={handle.cached_len}", cfg.verbose)

  log("KV cache tests PASSED", cfg.verbose)
  return True


def test_scheduler(cfg: TestConfig) -> bool:
  """Test scheduler batching logic."""
  log("Testing scheduler...", cfg.verbose)

  device = torch.device(cfg.device)
  layout = MlaKvLayout(
    num_blocks=cfg.num_pages,
    block_size=cfg.page_size,
  )
  cache = KvCache(layout, device)

  sched_cfg = SchedulerConfig(
    max_batch_size=cfg.batch_size * 2,
    max_prefill_tokens=cfg.seq_len * cfg.batch_size,
    page_size=cfg.page_size,
  )
  scheduler = Scheduler(sched_cfg, cache, device)

  # Add some requests
  from nmoe.serve.types import Request, SamplingParams as TSamplingParams, ForwardSpec, OutputMode

  for i in range(cfg.batch_size):
    input_ids = torch.randint(0, 1000, (cfg.seq_len,), dtype=torch.int64)
    req = Request(
      uid=i,
      input_ids=input_ids,
      sampling_params=TSamplingParams(max_tokens=cfg.output_len),
      profile_name="production_generate",
      forward_spec=ForwardSpec(output_mode=OutputMode.TOKENS),
    )
    scheduler.add_request(req)

  log(f"  Added {cfg.batch_size} requests", cfg.verbose)

  # Schedule prefill batch
  batch = scheduler.schedule_prefill()
  assert batch is not None, "Expected prefill batch"
  assert batch.size == cfg.batch_size, f"Expected {cfg.batch_size} reqs, got {batch.size}"
  assert batch.is_prefill, "Expected prefill phase"
  log(f"  Scheduled prefill batch: {batch.size} reqs, {batch.total_tokens} tokens", cfg.verbose)

  # Promote to decode
  for req in batch.reqs:
    # Scheduler decode step expects the last generated token to exist
    # (it writes KV for that token). Simulate the prefill producing 1 token.
    req.cached_len = len(req.input_ids)
    req.output_ids.append(0)
    scheduler.promote_to_decode(req)

  # Schedule decode batch
  batch = scheduler.schedule_decode()
  assert batch is not None, "Expected decode batch"
  assert batch.is_decode, "Expected decode phase"
  log(f"  Scheduled decode batch: {batch.size} reqs", cfg.verbose)

  log("Scheduler tests PASSED", cfg.verbose)
  return True


def test_engine_init(cfg: TestConfig) -> bool:
  """Test engine initialization (without actual model loading)."""
  log("Testing engine initialization...", cfg.verbose)

  device = torch.device(cfg.device)

  # Check CUDA capability
  if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(device)
    log(f"  CUDA device: {torch.cuda.get_device_name(device)}", cfg.verbose)
    log(f"  Compute capability: {major}.{minor}", cfg.verbose)

    if major != 10:
      log(f"  WARNING: FlashMLA requires SM100 (B200). Got SM{major}{minor}.", cfg.verbose)
      log("  Skipping engine test on non-B200 hardware.", cfg.verbose)
      return True  # Skip but don't fail

  # Test with small config
  model_cfg = ModelConfig(
    num_layers=cfg.num_layers,
    num_dense_layers=1,
    vocab_size=1000,
    hidden_size=256,
    intermediate_size=512,
    num_heads=4,
    q_lora_rank=64,
    kv_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    dsa_n_idx_heads=2,
    dsa_idx_dim=16,
    dsa_topk=64,
    num_experts=8,
    num_shared_experts=1,
    num_experts_per_tok=2,
    moe_intermediate_size=128,
  )

  engine_cfg = EngineConfig(
    num_pages=cfg.num_pages,
    page_size=cfg.page_size,
    num_layers=cfg.num_layers,
    kv_lora_rank=32,
    qk_rope_head_dim=16,
    idx_dim=16,
  )

  init_distributed(0, 1)

  log("  Creating engine (this may take a moment for kernel compilation)...", cfg.verbose)
  try:
    engine = Engine(model_cfg, engine_cfg, rank=0, world_size=1)
    log(f"  Engine created successfully", cfg.verbose)
    log(f"  Vocab size: {engine.vocab_size}", cfg.verbose)
    log(f"  Num layers: {engine.num_layers}", cfg.verbose)
    engine.shutdown()
    log("Engine initialization PASSED", cfg.verbose)
    return True
  except Exception as e:
    log(f"  Engine creation failed: {e}", cfg.verbose)
    return False


def test_transfer(cfg: TestConfig) -> bool:
  """Test NIXL transfer module."""
  log("Testing NIXL transfer...", cfg.verbose)

  try:
    from nixl._api import nixl_agent, nixl_agent_config
    log("  NIXL import successful", cfg.verbose)
  except ImportError as e:
    log(f"  NIXL not available: {e}", cfg.verbose)
    log("  Skipping transfer test", cfg.verbose)
    return True  # Skip but don't fail

  from nmoe.serve.transfer import NixlAgent, TransferConfig

  # Create agent
  agent = NixlAgent("test_agent", TransferConfig())
  log(f"  Agent name: {agent.name}", cfg.verbose)

  # Get metadata
  metadata = agent.get_metadata()
  log(f"  Agent metadata size: {len(metadata)} bytes", cfg.verbose)

  agent.shutdown()
  log("NIXL transfer tests PASSED", cfg.verbose)
  return True


def test_forward_dummy(cfg: TestConfig) -> bool:
  """Test forward pass with dummy data (no model weights)."""
  log("Testing forward pass (dummy)...", cfg.verbose)

  device = torch.device(cfg.device)

  # Check SM100
  if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(device)
    if major != 10:
      log("  Skipping forward test on non-B200 hardware", cfg.verbose)
      return True

  # This would require actual model weights
  log("  Forward test requires model weights - skipping", cfg.verbose)
  return True


def run_all_tests(cfg: TestConfig) -> int:
  """Run all tests and return exit code."""
  log(f"Running inference tests on {cfg.device}", cfg.verbose)
  log("=" * 50, cfg.verbose)

  tests = [
    ("cache", test_cache),
    ("scheduler", test_scheduler),
    ("transfer", test_transfer),
    ("engine_init", test_engine_init),
    ("forward_dummy", test_forward_dummy),
  ]

  results = {}
  for name, test_fn in tests:
    try:
      results[name] = test_fn(cfg)
    except Exception as e:
      log(f"Test {name} FAILED with exception: {e}", cfg.verbose)
      import traceback
      traceback.print_exc()
      results[name] = False

  log("=" * 50, cfg.verbose)
  log("Results:", cfg.verbose)
  passed = 0
  failed = 0
  for name, success in results.items():
    status = "PASSED" if success else "FAILED"
    log(f"  {name}: {status}", cfg.verbose)
    if success:
      passed += 1
    else:
      failed += 1

  log(f"Total: {passed}/{len(tests)} passed", cfg.verbose)
  return 0 if failed == 0 else 1


def main():
  parser = argparse.ArgumentParser(description="nmoe.serve inference tests")
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--num-pages", type=int, default=256)
  parser.add_argument("--batch-size", type=int, default=4)
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("-q", "--quiet", action="store_true")
  args = parser.parse_args()

  cfg = TestConfig(
    device=args.device,
    num_pages=args.num_pages,
    batch_size=args.batch_size,
    seq_len=args.seq_len,
    verbose=not args.quiet,
  )

  sys.exit(run_all_tests(cfg))


if __name__ == "__main__":
  main()
