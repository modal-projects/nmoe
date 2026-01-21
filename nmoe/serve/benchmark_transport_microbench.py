# SPDX-License-Identifier: Apache-2.0
"""DeepEP transport microbenchmark (dispatch+combine) at the decode operating point.

This is a benchmark (not a unit test). It is designed to answer:
  - How many ms/step are we spending in DeepEP dispatch+combine at BS=256?
  - How much does get_dispatch_layout cost (normal mode)?

Hard launch invariant (management): global decode batch size must be 256.

Run (single node, 8 GPUs):
  torchrun --nproc_per_node=8 -m nmoe.serve.benchmark_transport_microbench --mode normal
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def _percentile_ms(samples_ms: list[float], p: float) -> float:
  if not samples_ms:
    return float("nan")
  s = sorted(samples_ms)
  i = int(round((p / 100.0) * (len(s) - 1)))
  return float(s[max(0, min(i, len(s) - 1))])


def _setup_dist() -> None:
  if dist.is_initialized():
    return
  dist.init_process_group(backend="nccl", init_method="env://")


def _make_routing(*, T: int, num_experts: int, topk: int, device: torch.device, seed: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  if mode == "uniform":
    # Deterministic spread across experts: each token hits topk distinct experts.
    base = torch.arange(T, device=device, dtype=torch.int64)[:, None]
    offs = torch.arange(topk, device=device, dtype=torch.int64)[None, :]
    topk_idx = (base + offs) % num_experts
    topk_weights = torch.full((T, topk), 1.0 / float(topk), device=device, dtype=torch.float32)
    return topk_idx, topk_weights
  if mode == "random":
    # Stable-ish random routing (seeded).
    topk_idx = torch.randint(0, num_experts, (T, topk), device=device, dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(T, topk, device=device, dtype=torch.float32), dim=-1)
    return topk_idx, topk_weights
  raise ValueError(f"unknown routing mode: {mode}")


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("normal",), default="normal", help="DeepEP mode (LL is benchmarked elsewhere).")
  ap.add_argument("--decode-batch-size", type=int, default=256, help="Global decode batch size (must be 256).")
  ap.add_argument("--hidden-size", type=int, default=7168)
  ap.add_argument("--num-experts", type=int, default=256)
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--iters", type=int, default=200)
  ap.add_argument("--warmup-iters", type=int, default=20)
  ap.add_argument("--routing", choices=("uniform", "random"), default="random")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)

  if args.decode_batch_size != 256:
    raise SystemExit(
      f"--decode-batch-size must be 256 (got {args.decode_batch_size}). "
      "This is a management launch requirement and a benchmark invariant."
    )

  _setup_dist()
  rank = int(dist.get_rank())
  world = int(dist.get_world_size())
  if world != 8:
    raise SystemExit(f"world_size must be 8 for this benchmark (got {world}).")

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  # Keep extension/JIT directories shared across ranks to avoid compile skew.
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")

  from deep_ep import Buffer  # type: ignore

  T_local = args.decode_batch_size // world
  if T_local * world != args.decode_batch_size:
    raise SystemExit("decode_batch_size must be divisible by world_size")

  # DeepEP buffer sizing (NVLink only for single node).
  num_nvl_bytes = max(
    Buffer.get_dispatch_config(world).get_nvl_buffer_size_hint(args.hidden_size * 2, world),
    Buffer.get_combine_config(world).get_nvl_buffer_size_hint(args.hidden_size * 2, world),
  )
  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=int(num_nvl_bytes), num_rdma_bytes=0, explicitly_destroy=True)

  # Inputs.
  x = torch.randn(T_local, args.hidden_size, device=device, dtype=torch.bfloat16)
  topk_idx, topk_weights = _make_routing(
    T=T_local,
    num_experts=args.num_experts,
    topk=args.topk,
    device=device,
    seed=args.seed + rank,
    mode=args.routing,
  )

  # Layout (normal mode).
  dist.barrier()
  t0 = time.perf_counter()
  num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = buffer.get_dispatch_layout(topk_idx, args.num_experts)
  torch.cuda.synchronize(device)
  dist.barrier()
  layout_ms = (time.perf_counter() - t0) * 1e3

  # Warmup.
  for _ in range(args.warmup_iters):
    (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
      x,
      num_tokens_per_rank=num_tokens_per_rank,
      is_token_in_rank=is_token_in_rank,
      num_tokens_per_expert=num_tokens_per_expert,
      topk_idx=topk_idx,
      topk_weights=topk_weights,
    )
    buffer.combine(recv_x, handle, recv_topk_weights)
  torch.cuda.synchronize(device)
  dist.barrier()

  # Timed.
  dispatch_ms: list[float] = []
  combine_ms: list[float] = []
  step_ms: list[float] = []

  start = torch.cuda.Event(enable_timing=True)
  mid = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)

  for _ in range(args.iters):
    start.record()
    (recv_x, _, recv_topk_weights, _, handle, _) = buffer.dispatch(
      x,
      num_tokens_per_rank=num_tokens_per_rank,
      is_token_in_rank=is_token_in_rank,
      num_tokens_per_expert=num_tokens_per_expert,
      topk_idx=topk_idx,
      topk_weights=topk_weights,
    )
    mid.record()
    buffer.combine(recv_x, handle, recv_topk_weights)
    end.record()
    end.synchronize()
    d_ms = float(start.elapsed_time(mid))
    c_ms = float(mid.elapsed_time(end))
    dispatch_ms.append(d_ms)
    combine_ms.append(c_ms)
    step_ms.append(d_ms + c_ms)

  buffer.destroy()
  dist.barrier()

  # Aggregate with all_gather on NCCL (CUDA tensor).
  ms_t = torch.tensor(
    [statistics.mean(dispatch_ms), statistics.mean(combine_ms), statistics.mean(step_ms), layout_ms],
    device=device,
    dtype=torch.float64,
  )
  all_ms = [torch.zeros_like(ms_t) for _ in range(world)]
  dist.all_gather(all_ms, ms_t)

  if rank == 0:
    all_ms_cpu = [t.detach().cpu() for t in all_ms]
    disp = [float(t[0]) for t in all_ms_cpu]
    comb = [float(t[1]) for t in all_ms_cpu]
    step = [float(t[2]) for t in all_ms_cpu]
    lay = [float(t[3]) for t in all_ms_cpu]
    # Use the slowest rank as the step wall proxy (straggler-gated).
    step_ms_rankmax = max(step)
    tok_s = args.decode_batch_size / (step_ms_rankmax / 1e3)
    print("=== DeepEP transport microbench (normal) ===", flush=True)
    print(f"world_size={world}  decode_bs=256  T_local={T_local}  H={args.hidden_size}  E={args.num_experts}  topk={args.topk}", flush=True)
    print(f"routing={args.routing}  iters={args.iters}  warmup={args.warmup_iters}", flush=True)
    print(f"layout_ms (rank0 wall): {lay[0]:.3f}", flush=True)
    print(f"dispatch_ms avg across ranks: mean={statistics.mean(disp):.3f}  max={max(disp):.3f}", flush=True)
    print(f"combine_ms  avg across ranks: mean={statistics.mean(comb):.3f}  max={max(comb):.3f}", flush=True)
    print(f"step_ms     avg across ranks: mean={statistics.mean(step):.3f}  max={step_ms_rankmax:.3f}", flush=True)
    print(f"throughput_tok_s_node (rankmax): {tok_s:,.0f}", flush=True)


if __name__ == "__main__":
  main()
