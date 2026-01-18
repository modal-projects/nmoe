# SPDX-License-Identifier: Apache-2.0
"""RDEP IPC barrier A/B microbenchmark (phase vs tag, and graph replay).

This benchmark is for *single-node* IPC mode (world_size == local_world_size).

Why this exists:
  - phase barrier (k_ipc_barrier_phase_*) is optimized for eager mode but uses a
    monotonically increasing host-side phase, which is not graph-replay-safe.
  - tag barrier (barrier_block pattern) is graph-replay-safe, but uses remote
    atomics.

We measure:
  1) Eager per-barrier latency: phase vs tag.
  2) CUDA-graph replay: tag barrier captured + replayed (validating capturability).

Run:
  torchrun --nproc_per_node=8 -m nmoe.serve.benchmark_rdep_barrier_ab
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist


def _setup_dist() -> None:
  if dist.is_initialized():
    return
  dist.init_process_group(backend="nccl", init_method="env://")


def _time_barrier(fn, *, iters: int, device: torch.device) -> float:
  # Average ms per call (synchronized).
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(iters):
    fn()
  end.record()
  end.synchronize()
  return float(start.elapsed_time(end)) / float(iters)


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--warmup-iters", type=int, default=200)
  ap.add_argument("--graph-capture-barriers", type=int, default=32, help="Number of barrier calls inside the captured graph.")
  ap.add_argument("--graph-replay-iters", type=int, default=2000, help="Number of graph replays to time.")
  args = ap.parse_args(argv)

  _setup_dist()
  rank = int(dist.get_rank())
  world = int(dist.get_world_size())
  if world != 8:
    raise SystemExit(f"world_size must be 8 for this benchmark (got {world}).")

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  stream = torch.cuda.current_stream(device)

  # Keep extension/JIT directories shared across ranks to avoid compile skew.
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")

  # Initialize IPC pointers/signals through the existing RDEP bootstrap.
  from nmoe.rdep import Rdep  # uses nmoe.csrc.rdep under the hood
  from nmoe.csrc import rdep as _C

  # Minimal allocation; we only need the barrier signal arrays + IPC pointer tables.
  _ = Rdep(dim=7168, n_local=32, topk=8, profile="bf16", capacity=4096)

  if _C.get_mode() != 1:
    raise SystemExit("RDEP mode is not IPC. This microbench is for single-node IPC only.")

  # Warmup: ensure IPC pointers are live and caches are warm.
  for _ in range(args.warmup_iters):
    _C.ipc_barrier_phase_bf16(stream)
  stream.synchronize()
  dist.barrier()

  # Eager timing: phase barrier.
  phase_ms = _time_barrier(lambda: _C.ipc_barrier_phase_bf16(stream), iters=args.iters, device=device)
  stream.synchronize()
  dist.barrier()

  # IMPORTANT: phase barrier writes monotonically increasing phase values into
  # the shared barrier signal slots. The tag barrier assumes the slots start at
  # zero. Reset local signal slots on all ranks before measuring tag.
  _C.ipc_barrier_zero_bf16(stream)
  stream.synchronize()
  dist.barrier()

  # Warmup and eager timing: tag barrier.
  for _ in range(args.warmup_iters):
    _C.ipc_barrier_tag_bf16(stream)
  stream.synchronize()
  dist.barrier()

  tag_ms = _time_barrier(lambda: _C.ipc_barrier_tag_bf16(stream), iters=args.iters, device=device)
  stream.synchronize()
  dist.barrier()

  # Graph timing (tag barrier only; phase barrier is not replay-safe as implemented).
  g = torch.cuda.CUDAGraph()
  _C.ipc_barrier_zero_bf16(stream)
  stream.synchronize()
  dist.barrier()

  # Capture: barrier calls must be identical across ranks.
  # IMPORTANT: pass the current stream explicitly; launching on the legacy default
  # stream (nullptr) will not be captured and can hang capture.
  with torch.cuda.graph(g):
    for _ in range(args.graph_capture_barriers):
      _C.ipc_barrier_tag_bf16(torch.cuda.current_stream(device))

  stream.synchronize()
  dist.barrier()

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(args.graph_replay_iters):
    g.replay()
  end.record()
  end.synchronize()
  graph_ms_per_replay = float(start.elapsed_time(end)) / float(args.graph_replay_iters)
  graph_ms_per_barrier = graph_ms_per_replay / float(args.graph_capture_barriers)

  # Gather rank-local measurements.
  stats = torch.tensor([phase_ms, tag_ms, graph_ms_per_barrier], device=device, dtype=torch.float64)
  all_stats = [torch.zeros_like(stats) for _ in range(world)]
  dist.all_gather(all_stats, stats)

  if rank == 0:
    all_stats_cpu = [t.detach().cpu() for t in all_stats]
    phase = [float(x[0]) for x in all_stats_cpu]
    tag = [float(x[1]) for x in all_stats_cpu]
    graph = [float(x[2]) for x in all_stats_cpu]
    print("=== RDEP IPC barrier microbench ===", flush=True)
    print(f"world_size={world}  iters={args.iters}  graph_capture_barriers={args.graph_capture_barriers}  graph_replay_iters={args.graph_replay_iters}", flush=True)
    print(f"phase barrier (eager) ms: mean={statistics.mean(phase):.4f}  max={max(phase):.4f}", flush=True)
    print(f"tag   barrier (eager) ms: mean={statistics.mean(tag):.4f}  max={max(tag):.4f}", flush=True)
    print(f"tag barrier (graph)  ms: mean={statistics.mean(graph):.4f}  max={max(graph):.4f}", flush=True)


if __name__ == "__main__":
  main()
