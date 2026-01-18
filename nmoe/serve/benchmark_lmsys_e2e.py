# SPDX-License-Identifier: Apache-2.0
"""LMSYS-aligned end-to-end benchmark using the real serving stack.

Measures steady-state *engine* throughput through:
  Scheduler -> Engine -> Model

This intentionally avoids the HTTP server path and runs under `torchrun` so that
all ranks execute the same phase order and DeepEP collectives stay aligned.
The benchmark is still "real" in the sense that it exercises:
  - request objects + scheduler batching
  - page-table / out_loc construction
  - KV-cache writes + paged attention reads
  - MoE DeepEP dispatch/combine
  - sampling

Benchmark contract (LMSYS blogs; per node):
  - Prefill throughput is reported in prompt tokens/sec, using a fixed
    tokens-per-device budget (default: 16,384) and prompt lengths
    (default: 1K/2K/4K).
  - Decode throughput is reported in output tokens/sec, using a fixed global
    decode batch size (default: 256 sequences) and input length (default: 2000).

Note: Our engine samples the first token at the end of prefill and then
generates remaining tokens in decode. The decode benchmark reports decode-only
tokens (output_len - 1) to match the "steady-state decode" intent.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def _parse_int_list(s: str) -> list[int]:
  parts = [p.strip() for p in s.split(",") if p.strip()]
  if not parts:
    raise ValueError("expected a non-empty comma-separated list")
  out: list[int] = []
  for p in parts:
    try:
      out.append(int(p))
    except ValueError as e:
      raise ValueError(f"invalid int in list: {p!r}") from e
  return out


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


def _load_checkpoint(model, ckpt_path: str, rank: int, world_size: int, *, cfg) -> None:
  """Load a checkpoint, preferring pre-sharded mp files when present."""
  from nmoe.serve.ckpt import load_checkpoint, load_sharded_checkpoint

  shard_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(shard_path):
    missing, unexpected = load_sharded_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)
  else:
    missing, unexpected = load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  if rank == 0 and (missing or unexpected):
    raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")


def _run_until(orchestrator, pred, max_steps: int = 10_000_000) -> int:
  steps = 0
  while not pred():
    orchestrator.run_step()
    steps += 1
    if steps >= max_steps:
      raise RuntimeError("benchmark loop exceeded max_steps (stuck?)")
  return steps


def _get_cpu_ctrl_group(orchestrator):
  # Prefer the orchestrator's lockstep group if present (expected: gloo on CPU).
  g = getattr(orchestrator, "_lockstep_group", None)
  if g is not None:
    return g
  world_size = dist.get_world_size()
  if world_size > 1:
    return dist.new_group(ranks=list(range(world_size)), backend="gloo")
  return None


def _all_ranks_true(local_ok: bool, *, group) -> bool:
  t = torch.tensor([1 if local_ok else 0], dtype=torch.int64, device="cpu")
  if group is None:
    return bool(int(t.item()) == 1)
  # Use async all-reduce with a small timeout so benchmarks don't hang silently
  # if one rank diverges/crashes before entering the collective.
  work = dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group, async_op=True)
  deadline = time.perf_counter() + 10.0
  while not work.is_completed():
    if time.perf_counter() > deadline:
      r = dist.get_rank() if dist.is_initialized() else -1
      raise RuntimeError(f"benchmark: _all_ranks_true all_reduce timed out (rank={r})")
    time.sleep(0.01)
  work.wait()
  return bool(int(t.item()) == 1)


def _run_until_lockstep(
  orchestrator,
  pred,
  *,
  max_steps: int = 10_000_000,
  timeout_s: float | None = None,
  debug_name: str = "",
  debug_reqs: list | None = None,
  debug_every_s: float = 1.0,
) -> int:
  """Drive the orchestrator via its lockstep step coordinator.

  This keeps DeepEP collective ordering aligned even when ranks have different
  local work (including T=0 participation).
  """
  steps = 0
  t_start = time.perf_counter()
  t_next_log = t_start + float(debug_every_s)
  rank = int(getattr(orchestrator, "rank", 0))
  while True:
    ok = bool(pred())
    if ok:
      break
    if timeout_s is not None and (time.perf_counter() - t_start) > float(timeout_s):
      if rank == 0:
        print("\n" + "!" * 80)
        print(f"[benchmark] lockstep timeout after {timeout_s:.1f}s: {debug_name}")
        try:
          sched = getattr(orchestrator, "scheduler", None)
          qsz = int(getattr(orchestrator, "queue_size", -1))
          has_prefill = bool(getattr(sched, "has_pending_prefill", False))
          has_decode = bool(getattr(sched, "has_pending_decode", False))
          print(f"  steps={steps} queue_size={qsz} has_prefill={has_prefill} has_decode={has_decode}")
        except Exception as e:  # noqa: BLE001
          print(f"  (failed to read scheduler state: {e})")
        if debug_reqs:
          try:
            counts: dict[str, int] = {}
            out_lens: list[int] = []
            cached_lens: list[int] = []
            extend_lens: list[int] = []
            for r in debug_reqs:
              s = getattr(getattr(r, "status", None), "name", "UNKNOWN")
              counts[s] = counts.get(s, 0) + 1
              out_lens.append(len(getattr(r, "output_ids", [])))
              cached_lens.append(int(getattr(r, "cached_len", -1)))
              extend_lens.append(int(getattr(r, "extend_len", -1)))
            def _mm(xs: list[int]) -> tuple[int, int]:
              return (min(xs), max(xs)) if xs else (0, 0)
            print(f"  req_statuses={counts}")
            print(f"  out_ids_len[min,max]={_mm(out_lens)} cached_len[min,max]={_mm(cached_lens)} extend_len[min,max]={_mm(extend_lens)}")
          except Exception as e:  # noqa: BLE001
            print(f"  (failed to summarize reqs: {e})")
        print("!" * 80 + "\n", flush=True)
      raise RuntimeError(f"benchmark lockstep stuck: {debug_name} (timeout {timeout_s:.1f}s)")

    orchestrator._recv_requests()
    any_decode, any_prefill, any_shutdown = orchestrator._lockstep_any_work()
    if any_shutdown:
      raise RuntimeError("benchmark observed shutdown (unexpected)")
    if any_decode:
      orchestrator._lockstep_run_phase("decode")
    if any_prefill:
      orchestrator._lockstep_run_phase("prefill")
    steps += 1
    if steps >= max_steps:
      raise RuntimeError("benchmark loop exceeded max_steps (stuck?)")
    t_now = time.perf_counter()
    if rank == 0 and t_now >= t_next_log:
      t_next_log = t_now + float(debug_every_s)
      try:
        sched = getattr(orchestrator, "scheduler", None)
        qsz = int(getattr(orchestrator, "queue_size", -1))
        local_prefill = bool(getattr(sched, "has_pending_prefill", False))
        local_decode = bool(getattr(sched, "has_pending_decode", False))
        print(
          f"[benchmark] lockstep '{debug_name}': steps={steps} queue={qsz} "
          f"local(prefill={int(local_prefill)},decode={int(local_decode)}) "
          f"any(prefill={int(any_prefill)},decode={int(any_decode)})",
          flush=True,
        )
        if debug_reqs:
          counts: dict[str, int] = {}
          for r in debug_reqs:
            s = getattr(getattr(r, "status", None), "name", "UNKNOWN")
            counts[s] = counts.get(s, 0) + 1
          print(f"[benchmark] req_statuses={counts}", flush=True)
      except Exception as e:  # noqa: BLE001
        print(f"[benchmark] lockstep debug failed: {e}", flush=True)
    time.sleep(0.0005)
  return steps


def _disable_eos_stop_for_benchmark(orchestrator) -> None:
  """Ensure fixed-length generation for throughput benchmarking.

  Serving should stop on EOS; throughput benchmarks should not, otherwise
  requests may terminate early and the benchmark becomes ill-defined.
  """
  from nmoe.serve.types import RequestStatus

  def _check_finished_no_eos(self, req, token: int) -> bool:  # noqa: ARG001
    # Overlap mode appends -1 placeholders for in-flight decode steps; ignore
    # those in stop conditions.
    n = len(req.output_ids)
    while n > 0 and int(req.output_ids[n - 1]) == -1:
      n -= 1
    if n >= req.sampling_params.max_tokens:
      req.finish_reason = "length"
      return True
    if req.cancel_flag:
      req.status = RequestStatus.CANCELLED
      req.finish_reason = "cancelled"
      return True
    return False

  orchestrator._check_finished = _check_finished_no_eos.__get__(  # type: ignore[attr-defined]
    orchestrator, type(orchestrator)
  )


def _bench_prefill(
  orchestrator,
  rank: int,
  world_size: int,
  *,
  tokens_per_device: int,
  prompt_len: int,
) -> None:
  if rank == 0:
    print("\n" + "=" * 80)
    print("PREFILL (E2E)")
    print("=" * 80)
    print(f"Config: tokens_per_device={tokens_per_device} prompt_len={prompt_len}")

  if tokens_per_device <= 0:
    raise ValueError(f"tokens_per_device must be > 0, got {tokens_per_device}")
  if prompt_len <= 0:
    raise ValueError(f"prompt_len must be > 0, got {prompt_len}")
  if tokens_per_device % prompt_len != 0:
    raise ValueError(
      f"tokens_per_device ({tokens_per_device}) must be divisible by prompt_len ({prompt_len})"
    )
  bs_local = tokens_per_device // prompt_len
  bs_global = bs_local * world_size

  reqs = []
  for _ in range(bs_local):
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

  ctrl_group = _get_cpu_ctrl_group(orchestrator)

  torch.cuda.synchronize()
  dist.barrier()
  t0 = time.perf_counter()
  if world_size > 1:
    _run_until_lockstep(
      orchestrator,
      lambda: _all_ranks_true(all(r.is_finished for r in reqs), group=ctrl_group),
    )
  else:
    _run_until(orchestrator, lambda: all(r.is_finished for r in reqs))
  torch.cuda.synchronize()
  dist.barrier()
  dt = time.perf_counter() - t0

  tok = bs_global * prompt_len
  if rank == 0:
    print(f"Prefill tok/s/node: {tok / dt:,.0f} (tokens={tok:,} time={dt:.3f}s)")


def _bench_decode(
  orchestrator,
  rank: int,
  world_size: int,
  *,
  batch_size: int,
  ctx_len: int,
  output_len: int,
  profile: bool = False,
  timeout_s: float | None = None,
) -> None:
  if rank == 0:
    print("\n" + "=" * 80)
    print("DECODE (E2E)")
    print("=" * 80)
    print(f"Config: batch_size={batch_size} ctx_len={ctx_len} output_len={output_len}")

  if batch_size <= 0:
    raise ValueError(f"batch_size must be > 0, got {batch_size}")
  if ctx_len <= 0:
    raise ValueError(f"ctx_len must be > 0, got {ctx_len}")
  if output_len <= 1:
    raise ValueError(f"output_len must be > 1, got {output_len}")

  q, r = divmod(int(batch_size), int(world_size))
  bs_local = q + (1 if int(rank) < int(r) else 0)
  decode_steps = output_len - 1

  reqs = []
  for _ in range(bs_local):
    # Avoid random token IDs in benchmarks: some adversarial/random sequences can
    # trigger non-finite activations early in the stack, which is not useful for
    # benchmarking (and can deadlock lockstep EP).
    #
    # Use a deterministic, "non-weird" token pattern that still has variety so
    # routing isn't trivially constant across positions.
    input_ids = (torch.arange(ctx_len, dtype=torch.int64) % 10000) + 100
    req = orchestrator.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=0.0,
      max_tokens=output_len,  # 1 token from prefill + (output_len - 1) tokens from decode
    )
    reqs.append(req)
    orchestrator.add_request(req)

  ctrl_group = _get_cpu_ctrl_group(orchestrator)

  # Run prefill until all requests have produced the first token and entered decode.
  if world_size > 1:
    _run_until_lockstep(
      orchestrator,
      lambda: _all_ranks_true(all(r.status.name == "DECODING" for r in reqs), group=ctrl_group),
      timeout_s=timeout_s,
      debug_name="decode.prefill_to_decoding",
      debug_reqs=reqs if rank == 0 else None,
    )
  else:
    _run_until(orchestrator, lambda: all(r.status.name == "DECODING" for r in reqs))

  torch.cuda.synchronize()
  dist.barrier()

  do_profile = bool(profile)
  prof = None
  if do_profile:
    if rank == 0:
      print(f"\n=== Profiling decode (low_latency_mode={getattr(orchestrator.engine.buffer, 'low_latency_mode', 'NOT SET')}) ===")
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    prof.__enter__()

  t0 = time.perf_counter()
  if world_size > 1:
    _run_until_lockstep(
      orchestrator,
      lambda: _all_ranks_true(all(len(r.output_ids) >= output_len for r in reqs), group=ctrl_group),
      timeout_s=timeout_s,
      debug_name="decode.decode_steps",
      debug_reqs=reqs if rank == 0 else None,
    )
  else:
    _run_until(orchestrator, lambda: all(len(r.output_ids) >= output_len for r in reqs))
  torch.cuda.synchronize()

  if do_profile and prof:
    prof.__exit__(None, None, None)

  dist.barrier()
  dt = time.perf_counter() - t0

  tok = batch_size * decode_steps
  if rank == 0:
    print(f"Decode tok/s/node:  {tok / dt:,.0f} (tokens={tok:,} time={dt:.3f}s)")

  # If overlap is enabled, ensure no in-flight async decode steps leak into the
  # next benchmark phase (otherwise table slots and RDEP t_cap can overflow).
  if getattr(getattr(orchestrator, "orch_config", None), "enable_overlap", False):
    deadline = time.perf_counter() + 30.0
    while getattr(orchestrator, "_async_inflight", 0) and time.perf_counter() < deadline:
      orchestrator._drain_async_ready_steps(block=True, max_items=1)  # type: ignore[attr-defined]
    if getattr(orchestrator, "_async_inflight", 0):
      raise RuntimeError("benchmark: overlap drain timed out")

  # Print profile results
  if do_profile and prof and rank == 0:
    # Find .item() call stacks
    print("\n--- .item() Call Stacks ---")
    for e in prof.key_averages(group_by_stack_n=5):
      if "item" in e.key.lower() and e.count > 0:
        print(f"\n{e.key}: {e.count} calls")
        if hasattr(e, 'stack') and e.stack:
          for frame in e.stack[:5]:
            print(f"  {frame}")

    print("\n--- Kernel Launches ---")
    for key in ["cudaLaunchKernel", "cuLaunchKernelEx"]:
      matches = [e for e in prof.key_averages() if key in e.key]
      if matches:
        print(f"{key}: {matches[0].count} calls, {matches[0].cpu_time_total/1000:.1f}ms")

    print("\n--- Type Conversions ---")
    for key in ["aten::to", "aten::_to_copy", "aten::copy_"]:
      matches = [e for e in prof.key_averages() if e.key == key]
      if matches:
        print(f"{key}: {matches[0].count} calls, {matches[0].cpu_time_total/1000:.1f}ms")

    print("\n--- MoE Indexing ---")
    for key in ["aten::index", "aten::index_select", "aten::arange", "aten::nonzero", "aten::sort", "aten::bincount"]:
      matches = [e for e in prof.key_averages() if e.key == key]
      if matches:
        print(f"{key}: {matches[0].count} calls, {matches[0].cpu_time_total/1000:.1f}ms")

    print("\n--- Sync Points ---")
    for e in prof.key_averages():
      if "item" in e.key.lower():
        print(f"{e.key}: {e.count} calls, {e.cpu_time_total/1000:.1f}ms")

    print("\n--- Top 15 CUDA Kernels ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))



def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", type=str, required=True, help="Path to *mp* shard dir, e.g. /data/models/DeepSeek-V3-0324-mp8")
  ap.add_argument("--attention-type", type=str, default="mla", choices=["mla", "dsa"])
  ap.add_argument("--mode", type=str, default="all", choices=["all", "prefill", "decode"])
  ap.add_argument("--prefill-prompt-lens", type=str, default="1024,2048,4096", help="Comma-separated prompt lengths")
  ap.add_argument("--prefill-tokens-per-device", type=int, default=16384, help="Prompt tokens per device (per GPU)")
  ap.add_argument("--decode-batch-size", type=int, default=256, help="Global decode batch size (sequences per node)")
  ap.add_argument("--decode-ctx-len", type=int, default=2000, help="Input length per sequence")
  ap.add_argument("--output-len", type=int, default=100, help="Total output tokens per sequence (includes first token)")
  ap.add_argument("--num-pages", type=int, default=8192)
  ap.add_argument("--page-size", type=int, default=64)
  ap.add_argument("--max-seq-len", type=int, default=32768)
  ap.add_argument("--max-batch-size", type=int, default=256)
  ap.add_argument("--max-prefill-tokens", type=int, default=16384)
  ap.add_argument(
    "--moe-expected-m",
    type=int,
    default=256,
    help="MoE expected_m capacity per local expert (decode fast path; must be multiple of 16)",
  )
  ap.add_argument(
    "--probe-moe-load",
    action="store_true",
    help="Probe MoE load/overflow during decode.main (records global max masked_m and overflow count).",
  )
  ap.add_argument("--disable-prefix-cache", action="store_true")
  ap.add_argument("--disable-chunked-prefill", action="store_true")
  ap.add_argument("--chunk-size", type=int, default=2048)
  ap.add_argument("--enable-overlap", action="store_true", help="Enable lockstep decode overlap (GPU-driven decode + async consumer).")
  ap.add_argument(
    "--enable-cuda-graph",
    action="store_true",
    help="Enable CUDA-graph replay for greedy decode TOKENS (MLA-only; requires NMOE_DEEPEP_LOW_LATENCY=1).",
  )
  ap.add_argument("--seed", type=int, default=1234)
  args = ap.parse_args()

  # Launch requirement: the canonical decode throughput point is global BS=256.
  # Keep this benchmark fail-fast to avoid accidentally measuring/optimizing the
  # wrong operating point. Use other benchmarks for sweeps.
  if args.mode in ("all", "decode") and int(args.decode_batch_size) != 256:
    raise SystemExit(
      f"benchmark_lmsys_e2e: decode batch size must be 256 for launch (got {args.decode_batch_size}). "
      "Use nmoe.serve.benchmark_e2e for non-256 sweeps."
    )
  if int(args.moe_expected_m) <= 0 or (int(args.moe_expected_m) % 16) != 0:
    raise SystemExit(f"--moe-expected-m must be > 0 and a multiple of 16 (got {args.moe_expected_m}).")

  rank, world_size, device = _init_dist()
  seed = _broadcast_seed(args.seed, device)
  # Deterministic but de-correlated across ranks.
  torch.manual_seed(seed + rank)
  torch.cuda.manual_seed(seed + rank)

  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  init_distributed(rank, world_size, tp_size=1)

  # Validate RDEP env: the decode path is gated on NMOE_MOE_FUSED_PACK=1.
  # Without it, MoE falls through to DeepEP and crashes on RDEP's SimpleNamespace.
  transport = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
  fused_pack = os.environ.get("NMOE_MOE_FUSED_PACK", "0").strip().lower()
  if transport == "rdep" and fused_pack not in ("1", "true"):
    raise SystemExit(
      "NMOE_EP_TRANSPORT=rdep requires NMOE_MOE_FUSED_PACK=1 to use the RDEP decode path. "
      f"Got NMOE_MOE_FUSED_PACK={fused_pack!r}"
    )

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
    moe_expected_m=args.moe_expected_m,
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
    enable_overlap=bool(args.enable_overlap),
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=args.chunk_size,
    enable_prefix_cache=not args.disable_prefix_cache,
    enable_cuda_graph=bool(args.enable_cuda_graph),
  )

  orch = Orchestrator(model_cfg, engine_cfg, orch_cfg, rank=rank, world_size=world_size)
  _load_checkpoint(orch.engine.model, args.ckpt, rank, world_size, cfg=model_cfg)
  dist.barrier()
  _disable_eos_stop_for_benchmark(orch)

  moe_probe_mmax = None
  if bool(args.probe_moe_load):
    moe_probe_mmax = torch.zeros((1,), dtype=torch.int32, device=device)
    setattr(orch.engine.buffer, "_nmoe_moe_probe_mmax_gpu", moe_probe_mmax)

  if rank == 0:
    print("=" * 80)
    print("nmoe.serve LMSYS E2E Benchmark")
    print("=" * 80)
    print(f"GPUs: {world_size}")
    print(f"attention_type: {args.attention_type}")
    print(f"ckpt: {args.ckpt}")
    print(f"mode: {args.mode}")

  # Warmup: small prefill + 2 decode steps (never profiled).
  if hasattr(torch.cuda, "nvtx"):
    torch.cuda.nvtx.range_push("nmoe.serve.bench.decode.warmup_small")
  _bench_decode(
    orch,
    rank,
    world_size,
    batch_size=2 * world_size,
    ctx_len=128,
    output_len=3,
    profile=False,
    timeout_s=120.0,
  )
  if hasattr(torch.cuda, "nvtx"):
    torch.cuda.nvtx.range_pop()

  # CUDA graph warmup: trigger graph capture for the *target* decode batch size
  # outside the timed window (otherwise capture overhead contaminates results).
  if args.enable_cuda_graph and args.mode in ("all", "decode"):
    if hasattr(torch.cuda, "nvtx"):
      torch.cuda.nvtx.range_push("nmoe.serve.bench.decode.warmup_graph_capture")
    _bench_decode(
      orch,
      rank,
      world_size,
      batch_size=args.decode_batch_size,
      ctx_len=min(256, args.decode_ctx_len),
      output_len=4,
      profile=False,
      timeout_s=120.0,
    )
    if hasattr(torch.cuda, "nvtx"):
      torch.cuda.nvtx.range_pop()

  # Prefill (LMSYS): prompt lengths at fixed tokens/device.
  if args.mode in ("all", "prefill"):
    prompt_lens = _parse_int_list(args.prefill_prompt_lens)
    for prompt_len in prompt_lens:
      _bench_prefill(
        orch,
        rank,
        world_size,
        tokens_per_device=args.prefill_tokens_per_device,
        prompt_len=prompt_len,
      )

  # Decode (LMSYS): 2K input length, fixed global batch size, output length.
  if args.mode in ("all", "decode"):
    do_profile = os.environ.get("NMOE_PROFILE_DECODE") == "1"
    if hasattr(torch.cuda, "nvtx"):
      torch.cuda.nvtx.range_push("nmoe.serve.bench.decode.main")
    if moe_probe_mmax is not None:
      # Reset probe counters so warmups/capture don't contaminate the measurement.
      moe_probe_mmax.zero_()
      overflow_gpu = getattr(orch.engine, "_moe_overflow_gpu", None)
      if overflow_gpu is not None:
        overflow_gpu.zero_()
    _bench_decode(
      orch,
      rank,
      world_size,
      batch_size=args.decode_batch_size,
      ctx_len=args.decode_ctx_len,
      output_len=args.output_len,
      profile=do_profile,
    )
    if moe_probe_mmax is not None:
      ctrl_group = _get_cpu_ctrl_group(orch)
      mmax_cpu = torch.tensor([int(moe_probe_mmax.detach().cpu().item())], dtype=torch.int64, device="cpu")
      overflow_gpu = getattr(orch.engine, "_moe_overflow_gpu", None)
      overflow_cpu = int(overflow_gpu.detach().cpu().item()) if overflow_gpu is not None else 0
      overflow_cpu_t = torch.tensor([overflow_cpu], dtype=torch.int64, device="cpu")
      if ctrl_group is not None:
        dist.all_reduce(mmax_cpu, op=dist.ReduceOp.MAX, group=ctrl_group)
        dist.all_reduce(overflow_cpu_t, op=dist.ReduceOp.SUM, group=ctrl_group)
      if rank == 0:
        print(f"MoE probe (decode.main): m_max={int(mmax_cpu.item())} overflow_layers={int(overflow_cpu_t.item())}")
    if hasattr(torch.cuda, "nvtx"):
      torch.cuda.nvtx.range_pop()

  orch.shutdown()
  dist.barrier()
  dist.destroy_process_group()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
