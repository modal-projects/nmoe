# SPDX-License-Identifier: Apache-2.0
"""Inference-RDEP IPC transport microbenchmark (decode operating point).

This validates the new CUDA-IPC + tag-barrier transport intended to replace
DeepEP in nmoe.serve for single-node EP (TP=1, world=8).

Operating point (launch requirement):
  - global decode batch size == 256 (=> T_cap == 32 per rank at world=8)

What it measures (ms/step):
  - dispatch_fp8 (write activations + local routing metadata)
  - barrier_tag (system-scope ordering)
  - return_bf16 (write per-token bf16 contribution to owners)
  - barrier_tag (ordering)
  - reduce_bf16 (owner reduce across ranks for local tokens)

Run:
  torchrun --nproc_per_node=8 -m nmoe.serve.benchmark_infer_rdep_transport
"""

from __future__ import annotations

import argparse
import os
import statistics

import numpy as np
import torch
import torch.distributed as dist


def _setup_dist() -> None:
  if dist.is_initialized():
    return
  dist.init_process_group(backend="nccl", init_method="env://")


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iters", type=int, default=2000)
  ap.add_argument("--warmup-iters", type=int, default=200)
  ap.add_argument("--t-cap", type=int, default=32, help="Per-rank token cap (decode: 32 for BS=256 at world=8).")
  ap.add_argument("--t-local", type=int, default=32, help="Local tokens on this rank (<= t-cap).")
  ap.add_argument("--hidden", type=int, default=7168)
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--num-experts", type=int, default=256)
  ap.add_argument(
    "--expert-placement",
    choices=["contiguous", "striped"],
    default=os.environ.get("NMOE_EXPERT_PLACEMENT", "contiguous").strip().lower(),
  )
  args = ap.parse_args(argv)

  _setup_dist()
  rank = int(dist.get_rank())
  world = int(dist.get_world_size())
  if world != 8:
    raise SystemExit(f"world_size must be 8 for this benchmark (got {world}).")

  t_cap = int(args.t_cap)
  t_local = int(args.t_local)
  if t_local < 0 or t_local > t_cap:
    raise SystemExit(f"t_local must be in [0, t_cap] (got t_local={t_local} t_cap={t_cap}).")

  bs_global = world * t_cap
  if bs_global != 256:
    raise SystemExit(f"global batch size must be 256 (world={world} t_cap={t_cap} => {bs_global}).")

  hidden = int(args.hidden)
  topk = int(args.topk)
  num_experts = int(args.num_experts)
  if num_experts % world != 0:
    raise SystemExit(f"num_experts must be divisible by world (num_experts={num_experts} world={world}).")
  n_local = num_experts // world

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  stream = torch.cuda.current_stream(device)

  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")

  from nmoe.csrc import rdep as _C
  from nmoe.csrc import infer_ipc  # torch-linked cudaMalloc allocator
  # Reuse BF16 RDEP's proven IPC barrier signals for infer_barrier_tag.
  from nmoe.rdep import Rdep
  from nmoe.serve.kernels.fp8_quant import quantize_fp8_ue8m0

  # Minimal BF16 RDEP allocation to bootstrap IPC barrier signal tables.
  _ = Rdep(dim=hidden, n_local=n_local, topk=topk, profile="bf16", capacity=4096)

  # Allocate cudaMalloc-backed buffers (CUDA-IPC safe) and exchange raw mem handles.
  slab, barrier, recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_w, ret_y, slab_off = infer_ipc.alloc_infer_ipc_slab_fp8(
    world, t_cap, hidden, topk
  )

  def _all_gather_mem_handle(t: torch.Tensor) -> np.ndarray:
    h_local = torch.from_numpy(_C.ipc_get_mem_handle(int(t.data_ptr()))).to(device=device, dtype=torch.uint8)
    all_h = [torch.empty_like(h_local) for _ in range(world)]
    dist.all_gather(all_h, h_local)
    return np.concatenate([x.detach().cpu().numpy() for x in all_h], axis=0)

  slab_h = _all_gather_mem_handle(slab)
  expert_placement = 0 if args.expert_placement == "contiguous" else 1
  _C.infer_init_ipc_slab_fp8_local(
    0,
    rank,
    world,
    t_cap,
    hidden,
    topk,
    n_local,
    expert_placement,
    slab_h,
    slab_off.cpu().numpy(),
    slab.data_ptr(),
  )

  # Per-rank local inputs (decode token vectors).
  if t_local > 0:
    x_bf16 = torch.randn((t_local, hidden), device=device, dtype=torch.bfloat16)
    x_fp8, x_scale = quantize_fp8_ue8m0(x_bf16)
    topk_idx = torch.randint(0, num_experts, (t_local, topk), device=device, dtype=torch.int64)
    w = torch.rand((t_local, topk), device=device, dtype=torch.float32)
    topk_w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
  else:
    x_fp8 = torch.empty((0, hidden), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((0, hidden // 128), device=device, dtype=torch.float32)
    topk_idx = torch.empty((0, topk), device=device, dtype=torch.int64)
    topk_w = torch.empty((0, topk), device=device, dtype=torch.float32)

  # Dummy per-token contribution (one vector per global slot).
  # Makes the correctness check trivial: owner sees sum_r rank = 28.
  y_partial = torch.full((bs_global, hidden), float(rank), device=device, dtype=torch.bfloat16)
  y_out = torch.empty((t_cap, hidden), device=device, dtype=torch.bfloat16)
  send_mask = torch.empty((t_cap,), device=device, dtype=torch.uint8)

  def _dispatch() -> None:
    _C.infer_dispatch_fp8(
      0,
      x_fp8.data_ptr(),
      x_scale.data_ptr(),
      topk_idx.data_ptr(),
      topk_w.data_ptr(),
      send_mask.data_ptr(),
      t_local,
      stream,
    )

  def _barrier() -> None:
    _C.infer_barrier_tag(stream)

  def _ret() -> None:
    _C.infer_return_bf16(0, y_partial.data_ptr(), stream)

  def _reduce() -> None:
    _C.infer_reduce_bf16(0, y_out.data_ptr(), send_mask.data_ptr(), stream)

  # Warmup.
  for _ in range(args.warmup_iters):
    _dispatch()
    _barrier()
    _ret()
    _barrier()
    _reduce()
  stream.synchronize()
  dist.barrier()

  # Timed segments (CUDA-event timings).
  def _time(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iters):
      fn()
    end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) / float(iters)

  dispatch_ms = _time(lambda: _dispatch(), args.iters)
  barrier_ms = _time(lambda: _barrier(), args.iters)
  ret_ms = _time(lambda: _ret(), args.iters)
  barrier2_ms = _time(lambda: _barrier(), args.iters)
  reduce_ms = _time(lambda: _reduce(), args.iters)

  stream.synchronize()
  dist.barrier()

  # Correctness: after a full step, y_out[:t_local] == sum_r rank (28) for all elements.
  _dispatch()
  _barrier()
  _ret()
  _barrier()
  _reduce()
  stream.synchronize()

  if t_local > 0:
    expected = float(sum(range(world)))
    max_abs = float((y_out[:t_local].float() - expected).abs().max().item())
  else:
    max_abs = 0.0

  stats = torch.tensor(
    [dispatch_ms, barrier_ms, ret_ms, barrier2_ms, reduce_ms, max_abs],
    device=device,
    dtype=torch.float64,
  )
  all_stats = [torch.zeros_like(stats) for _ in range(world)]
  dist.all_gather(all_stats, stats)

  if rank == 0:
    rows = [t.detach().cpu().numpy() for t in all_stats]
    def _col(i: int) -> list[float]:
      return [float(r[i]) for r in rows]

    dispatch = _col(0)
    barrier = _col(1)
    ret = _col(2)
    barrier2 = _col(3)
    reduce = _col(4)
    errs = _col(5)

    step_ms = [dispatch[i] + barrier[i] + ret[i] + barrier2[i] + reduce[i] for i in range(world)]
    print("=== Inference-RDEP transport microbench ===", flush=True)
    print(f"world_size={world}  t_cap={t_cap}  t_local={t_local}  hidden={hidden}  topk={topk}", flush=True)
    print(f"dispatch ms: mean={statistics.mean(dispatch):.4f}  max={max(dispatch):.4f}", flush=True)
    print(f"barrier  ms: mean={statistics.mean(barrier):.4f}  max={max(barrier):.4f}", flush=True)
    print(f"return   ms: mean={statistics.mean(ret):.4f}  max={max(ret):.4f}", flush=True)
    print(f"barrier2 ms: mean={statistics.mean(barrier2):.4f}  max={max(barrier2):.4f}", flush=True)
    print(f"reduce   ms: mean={statistics.mean(reduce):.4f}  max={max(reduce):.4f}", flush=True)
    print(f"step     ms: mean={statistics.mean(step_ms):.4f}  max={max(step_ms):.4f}", flush=True)
    print(f"max_abs_err: max={max(errs):.6f}", flush=True)


if __name__ == "__main__":
  main()
