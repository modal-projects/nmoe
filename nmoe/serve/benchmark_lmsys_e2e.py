# SPDX-License-Identifier: Apache-2.0
"""LMSYS-faithful end-to-end benchmark using the real serving stack.

Measures steady-state *engine* throughput through:
  Scheduler -> Engine -> Model

This intentionally avoids the HTTP server path and runs under `torchrun` so that
all TP ranks execute the exact same batches. The benchmark is still "real"
in the sense that it exercises:
  - request objects + scheduler batching
  - page-table / out_loc construction
  - KV-cache writes + paged attention reads
  - TP vocab-parallel sampling collectives (greedy)

Targets (LMSYS blog, per node):
  - Prefill: ~50–57K tok/s/node
  - Decode:  ~22K tok/s/node
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, torch.device]:
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  return rank, world_size, device


def _broadcast_seed(seed: int, device: torch.device) -> int:
  t = torch.tensor([seed], dtype=torch.int64, device=device)
  dist.broadcast(t, src=0)
  return int(t.item())


def _load_sharded_mp(model, ckpt_path: str, rank: int, world_size: int) -> None:
  from safetensors.torch import safe_open

  fpath = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  state_dict = {}
  with safe_open(fpath, framework="pt", device="cpu") as f:
    for key in f.keys():
      state_dict[key] = f.get_tensor(key)
  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if rank == 0 and (missing or unexpected):
    raise RuntimeError(f"mp load mismatch: missing={len(missing)} unexpected={len(unexpected)}")


def _run_until(orchestrator, pred, max_steps: int = 10_000_000) -> int:
  steps = 0
  while not pred():
    orchestrator.run_step()
    steps += 1
    if steps >= max_steps:
      raise RuntimeError("benchmark loop exceeded max_steps (stuck?)")
  return steps


def _bench_prefill(orchestrator, rank: int, *, bs: int, prompt_len: int) -> None:
  if rank == 0:
    print("\n" + "=" * 80)
    print("PREFILL (E2E)")
    print("=" * 80)
    print(f"Config: bs={bs} prompt_len={prompt_len}")

  reqs = []
  for _ in range(bs):
    # CPU input ids required by Request contract.
    input_ids = torch.randint(0, 10000, (prompt_len,), dtype=torch.int64)
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=0.0,
      max_tokens=1,  # stop after first generated token (prefill completes first)
    )
    reqs.append(req)
    orchestrator.add_request(req)

  torch.cuda.synchronize()
  dist.barrier()
  t0 = time.perf_counter()
  _run_until(orchestrator, lambda: all(r.is_finished for r in reqs))
  torch.cuda.synchronize()
  dist.barrier()
  dt = time.perf_counter() - t0

  tok = bs * prompt_len
  if rank == 0:
    print(f"Prefill tok/s: {tok / dt:,.0f} (tokens={tok:,} time={dt:.3f}s)")


def _bench_decode(orchestrator, rank: int, *, bs: int, ctx_len: int, decode_steps: int) -> None:
  if rank == 0:
    print("\n" + "=" * 80)
    print("DECODE (E2E)")
    print("=" * 80)
    print(f"Config: bs={bs} ctx_len={ctx_len} decode_steps={decode_steps}")

  reqs = []
  for _ in range(bs):
    input_ids = torch.randint(0, 10000, (ctx_len,), dtype=torch.int64)
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=0.0,
      max_tokens=1 + decode_steps,  # 1 token from prefill + N tokens from decode
    )
    reqs.append(req)
    orchestrator.add_request(req)

  # Run prefill until all requests have produced the first token and entered decode.
  _run_until(orchestrator, lambda: all(r.status.name == "DECODING" for r in reqs))

  torch.cuda.synchronize()
  dist.barrier()
  t0 = time.perf_counter()
  _run_until(orchestrator, lambda: all(len(r.output_ids) >= 1 + decode_steps for r in reqs))
  torch.cuda.synchronize()
  dist.barrier()
  dt = time.perf_counter() - t0

  tok = bs * decode_steps
  if rank == 0:
    print(f"Decode tok/s:  {tok / dt:,.0f} (tokens={tok:,} time={dt:.3f}s)")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", type=str, required=True, help="Path to *mp* shard dir, e.g. /data/models/DeepSeek-V3-0324-mp8")
  ap.add_argument("--attention-type", type=str, default="mla", choices=["mla", "dsa"])
  # NOTE: bs=256, ctx_len=2000 with page_size=64 needs ~8192 pages just to hold prompt KV.
  ap.add_argument("--num-pages", type=int, default=8192)
  ap.add_argument("--page-size", type=int, default=64)
  ap.add_argument("--max-seq-len", type=int, default=32768)
  ap.add_argument("--max-batch-size", type=int, default=256)
  ap.add_argument("--max-prefill-tokens", type=int, default=16384)
  ap.add_argument("--disable-prefix-cache", action="store_true")
  ap.add_argument("--disable-chunked-prefill", action="store_true")
  ap.add_argument("--chunk-size", type=int, default=2048)
  ap.add_argument("--seed", type=int, default=1234)
  args = ap.parse_args()

  rank, world_size, device = _init_dist()
  seed = _broadcast_seed(args.seed, device)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  init_distributed(rank, world_size)

  # Model config defaults to DeepSeek-shaped; we only flip attention backend here.
  model_cfg = ModelConfig(attention_type=args.attention_type)

  engine_cfg = EngineConfig(
    num_pages=args.num_pages,
    page_size=args.page_size,
    num_layers=model_cfg.num_layers,
    kv_lora_rank=model_cfg.kv_lora_rank,
    qk_rope_head_dim=model_cfg.qk_rope_head_dim,
    max_batch_size=args.max_batch_size,
    max_seq_len=args.max_seq_len,
    attention_type=args.attention_type,
    idx_dim=model_cfg.dsa_idx_dim,
    max_step_tokens=args.max_prefill_tokens,
  )

  orch_cfg = OrchestratorConfig(
    max_batch_size=args.max_batch_size,
    max_prefill_tokens=args.max_prefill_tokens,
    max_seq_len=args.max_seq_len,
    num_pages=args.num_pages,
    page_size=args.page_size,
    enable_overlap=False,
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=args.chunk_size,
    enable_prefix_cache=not args.disable_prefix_cache,
    enable_cuda_graph=False,
  )

  orch = Orchestrator(model_cfg, engine_cfg, orch_cfg, rank=rank, world_size=world_size)
  _load_sharded_mp(orch.engine.model, args.ckpt, rank, world_size)
  dist.barrier()

  if rank == 0:
    print("=" * 80)
    print("nmoe.serve LMSYS E2E Benchmark")
    print("=" * 80)
    print(f"GPUs: {world_size}")
    print(f"attention_type: {args.attention_type}")
    print(f"ckpt: {args.ckpt}")

  # Warmup: small prefill + 2 decode steps.
  _bench_decode(orch, rank, bs=2, ctx_len=128, decode_steps=2)

  # Prefill target-like config: per LMSYS, 4096 input length and BS=4.
  _bench_prefill(orch, rank, bs=4, prompt_len=4096)

  # Decode target-like config: 256 sequences, ~2000 ctx.
  _bench_decode(orch, rank, bs=256, ctx_len=2000, decode_steps=20)

  orch.shutdown()
  dist.barrier()
  dist.destroy_process_group()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
