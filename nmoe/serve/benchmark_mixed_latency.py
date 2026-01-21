# SPDX-License-Identifier: Apache-2.0
"""Mixed-load latency benchmark (TTFT/ITL) for DP=8 ownership + EP=8 experts.

Run (single node, 8×B200):
  torchrun --nproc_per_node=8 -m nmoe.serve.benchmark_mixed_latency --ckpt <mp_dir>

This benchmark measures *streaming tail latency* under mixed load using the
production DP=8 ownership control plane (rank0 driver, per-request owners):

- Owners execute the normal Orchestrator/Scheduler/Engine path locally.
- All ranks still participate in every DeepEP step via lockstep mode (T=0 allowed).
- Token emission timestamps are measured on rank0 as tokens arrive over the gloo
  backchannel (this matches client-visible timing, excluding HTTP).

Outputs (rank0):
- TTFT p50/p99 for probe requests
- ITL p50/p99 for decode sessions (steady-state)
- Prefill/Decode tok/s over the measurement window

Notes:
- This is a benchmark, not a pass/fail test. It is used to track progress
  against `docs/serve-targets.md`.
"""

from __future__ import annotations

import argparse
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return


def _percentile(xs: list[float], p: float) -> float:
  if not xs:
    return float("nan")
  ys = sorted(xs)
  if p <= 0:
    return float(ys[0])
  if p >= 100:
    return float(ys[-1])
  k = int(round((p / 100.0) * (len(ys) - 1)))
  k = max(0, min(k, len(ys) - 1))
  return float(ys[k])


def _load_checkpoint(model, ckpt_path: str, rank: int, world_size: int, *, cfg) -> None:
  from nmoe.serve.ckpt import load_checkpoint, load_sharded_checkpoint

  shard_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(shard_path):
    missing, unexpected = load_sharded_checkpoint(model, ckpt_path, rank=rank, world_size=world_size)
  else:
    missing, unexpected = load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  if rank == 0 and (missing or unexpected):
    raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")


@dataclass
class _ReqState:
  owner: int
  sent_t: float
  first_t: float = 0.0
  last_t: float = 0.0
  done: bool = False
  finish_reason: str = ""
  n_tokens: int = 0


def main() -> int:
  _maybe_set_cutlass_path()

  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, type=str)
  ap.add_argument("--attention-type", choices=["mla", "dsa"], default="mla")

  ap.add_argument("--concurrency", type=int, default=256, help="Active decode sessions (C)")
  ap.add_argument("--prompt-len", type=int, default=2000, help="Prompt length for sessions")
  ap.add_argument("--output-len", type=int, default=100, help="Generated tokens per session (N)")

  ap.add_argument("--prefill-backlog", type=int, default=0, help="Keep this many prefill-only requests queued")
  ap.add_argument("--prefill-prompt-len", type=int, default=2000)

  ap.add_argument("--probe-rps", type=float, default=0.0, help="Probe rate (req/s) for TTFT sampling")
  ap.add_argument("--probe-prompt-len", type=int, default=2000)
  ap.add_argument("--duration-s", type=float, default=30.0)

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
  ap.add_argument("--disable-prefix-cache", action="store_true")
  ap.add_argument("--disable-chunked-prefill", action="store_true")
  ap.add_argument("--chunk-size", type=int, default=2048)
  args = ap.parse_args()
  if int(args.moe_expected_m) <= 0 or (int(args.moe_expected_m) % 16) != 0:
    raise SystemExit(f"--moe-expected-m must be > 0 and a multiple of 16 (got {args.moe_expected_m}).")

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size != 8:
    raise RuntimeError(f"benchmark_mixed_latency requires world_size=8 (got {world_size})")
  torch.cuda.set_device(rank)

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from nmoe.serve.control_plane import (
    ControlPlane,
    OUTPUT_MODE_ID_TOKENS,
    RequestInit,
    finish_reason_id_to_str,
  )
  from nmoe.serve.ckpt import load_model_config
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig
  from nmoe.serve.types import ForwardSpec, OutputMode

  # Explicit JIT warm-up: build CUDA extensions before any DeepEP collectives.
  #
  # This avoids a failure mode where one rank is still compiling while others
  # already entered DeepEP dispatch/combine, triggering timeouts.
  def _jit_warmup() -> None:
    from nmoe.serve.kernels import (
      quantize_fp8_ue8m0,
      silu_mul_fp8,
      weighted_scatter_add,
      weighted_scatter_add_indexed,
    )

    dev = torch.device(f"cuda:{rank}")
    hidden = 128  # must be divisible by 128 for quantize_fp8_ue8m0 scales
    n = 128

    x = torch.randn((n, hidden), device=dev, dtype=torch.bfloat16)
    q, s = quantize_fp8_ue8m0(x)
    gate = torch.randn((n, hidden), device=dev, dtype=torch.bfloat16)
    up = torch.randn((n, hidden), device=dev, dtype=torch.bfloat16)
    _ = silu_mul_fp8(gate, up)

    token_ids = (torch.arange(n, device=dev, dtype=torch.int64) % 8).contiguous()
    w = torch.ones((n,), device=dev, dtype=torch.float32)
    out = torch.zeros((8, hidden), device=dev, dtype=torch.bfloat16)
    weighted_scatter_add(x.contiguous(), token_ids, w, out)

    src = torch.randn((n, hidden), device=dev, dtype=torch.bfloat16)
    src_idx = torch.arange(n, device=dev, dtype=torch.int64).contiguous()
    out2 = torch.zeros((8, hidden), device=dev, dtype=torch.bfloat16)
    weighted_scatter_add_indexed(src, src_idx, token_ids, w, out2)

    # Ensure all kernels have finished compiling/executing before proceeding.
    torch.cuda.synchronize(dev)

  _jit_warmup()
  dist.barrier()

  init_distributed(rank, world_size, tp_size=1)
  ctrl_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  cp = ControlPlane(rank=rank, world_size=world_size, ctrl_group=ctrl_group)

  if os.path.exists(os.path.join(args.ckpt, "config.json")):
    model_cfg = load_model_config(args.ckpt)
  else:
    model_cfg = ModelConfig(attention_type=args.attention_type)
  if args.attention_type:
    model_cfg = ModelConfig(**{**model_cfg.__dict__, "attention_type": args.attention_type})
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
    tp_size=1,
  )
  orch_cfg = OrchestratorConfig(
    max_batch_size=args.max_batch_size,
    max_prefill_tokens=args.max_prefill_tokens,
    max_decode_tokens=args.max_batch_size,
    max_seq_len=args.max_seq_len,
    num_pages=args.num_pages,
    page_size=args.page_size,
    enable_overlap=False,
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=args.chunk_size,
    enable_prefix_cache=not args.disable_prefix_cache,
    enable_cuda_graph=False,
  )
  orch = Orchestrator(model_cfg, engine_cfg, orch_cfg, rank=rank, world_size=world_size, control_plane=cp)
  _load_checkpoint(orch.engine.model, args.ckpt, rank, world_size, cfg=model_cfg)
  dist.barrier()

  def _all_ranks_ok(flag: bool) -> bool:
    t = torch.tensor([1 if flag else 0], dtype=torch.int64, device="cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=ctrl_group)
    return bool(int(t.item()) == 1)

  def _drive_lockstep_until(local_pred, *, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        raise RuntimeError("unexpected shutdown during benchmark")
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")
      if _all_ranks_ok(bool(local_pred())):
        return
      time.sleep(0.0005)
    raise TimeoutError(f"lockstep loop timed out after {timeout_s}s")

  # Warm up the *real decode path* to avoid first-use stalls (e.g., shape-keyed
  # kernel compilation/autotune) polluting TTFT/ITL tails.
  #
  # Key requirement: exercise the expected per-owner local decode batch size.
  # In this benchmark, rank0 distributes sessions across owners 1..(world_size-1),
  # so the maximum local batch is ceil(C / (world_size-1)).
  denom = max(1, int(world_size) - 1)
  warmup_local_bs = max(1, (int(args.concurrency) + denom - 1) // denom)
  warmup_prompt_len = int(args.prompt_len)
  warmup_max_tokens = min(16, int(args.output_len))

  warmup_reqs = []
  if rank != 0:
    for _ in range(int(warmup_local_bs)):
      req = orch.create_request(
        input_ids=torch.full((warmup_prompt_len,), 100, dtype=torch.int32, device="cpu"),
        profile_name="production_generate",
        temperature=0.0,
        max_tokens=int(warmup_max_tokens),
      )
      accepted = orch.try_add_request(req, timeout=0.0)
      if not accepted:
        raise RuntimeError("warmup request rejected (queue full)")
      warmup_reqs.append(req)

  _drive_lockstep_until(lambda: all(r.is_finished for r in warmup_reqs), timeout_s=1800.0)
  torch.cuda.synchronize()
  dist.barrier(group=ctrl_group)

  # Owner ranks (including rank0) receive init/cancel.
  if rank != 0:

    def _on_request_init(init: RequestInit) -> None:
      if int(init.uid) % int(world_size) != int(rank):
        cp.enqueue_error(uid=int(init.uid), msg="wrong owner for uid")
        return
      ok, err = orch.validate_request_bounds(int(init.prompt_len), int(init.max_tokens))
      if not ok:
        cp.enqueue_error(uid=int(init.uid), msg=err)
        return
      if int(init.output_mode_id) != OUTPUT_MODE_ID_TOKENS:
        cp.enqueue_error(uid=int(init.uid), msg="unsupported output_mode")
        return
      fs = ForwardSpec(output_mode=OutputMode.TOKENS, topk=int(init.topk))
      seed = None if int(init.seed_or_minus1) < 0 else int(init.seed_or_minus1)
      req = orch.create_request(
        input_ids=init.input_ids,
        profile_name="production_generate",
        uid=int(init.uid),
        forward_spec=fs,
        max_tokens=int(init.max_tokens),
        temperature=float(init.temperature),
        top_p=float(init.top_p),
        top_k=int(init.top_k),
        seed=seed,
      )
      accepted = orch.try_add_request(req, timeout=0.0)
      if not accepted:
        cp.enqueue_error(uid=int(init.uid), msg="owner queue full")

    cp.start_worker(
      on_request_init=_on_request_init,
      on_cancel=lambda uid: orch.cancel(int(uid)),
      on_shutdown=lambda: orch.request_stop(),
    )

    # Main-thread lockstep loop. Stop is triggered by rank0 MSG_SHUTDOWN callback.
    while True:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        break
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")
      time.sleep(0.0005)

    # Coordinated shutdown: stop compute first, then destroy DeepEP buffers, then
    # tear down control-plane threads and process group.
    all_stopped = _all_ranks_ok(True)

    local_cuda_ok = True
    try:
      torch.cuda.synchronize()
    except Exception:
      local_cuda_ok = False
    all_cuda_ok = _all_ranks_ok(local_cuda_ok)

    # All ranks reach the destroy point before tearing down DeepEP/NVSHMEM state.
    dist.barrier(group=ctrl_group)
    try:
      orch.shutdown()
    except Exception:
      all_cuda_ok = False
    dist.barrier(group=ctrl_group)

    cp.close(timeout_s=30.0)
    dist.destroy_process_group()
    if not all_cuda_ok:
      return 3
    return 0

  # Rank0 driver: token timestamps as client-visible ITL/TTFT.

  lock = threading.Lock()
  states: Dict[int, _ReqState] = {}
  ttft_ms: list[float] = []
  itl_ms: list[float] = []
  prefill_tokens: int = 0
  decode_tokens: int = 0

  def _on_token_update(batch) -> None:
    nonlocal decode_tokens
    uids = batch.uids.tolist()
    tokens = batch.tokens.tolist()
    flags = batch.uflags.tolist()
    now = time.perf_counter()
    with lock:
      for uid, tok, fl in zip(uids, tokens, flags, strict=False):
        uid = int(uid)
        st = states.get(uid)
        if st is None:
          continue
        done = bool(int(fl) & 0x1)
        rid = int((int(fl) >> 1) & 0x7) if done else 0
        if st.n_tokens == 0:
          st.first_t = now
          ttft_ms.append((now - st.sent_t) * 1000.0)
        else:
          itl_ms.append((now - st.last_t) * 1000.0)
        st.last_t = now
        st.n_tokens += 1
        decode_tokens += 1
        if done:
          st.done = True
          st.finish_reason = finish_reason_id_to_str(rid)

  def _on_error(uid: int, msg: str) -> None:
    now = time.perf_counter()
    with lock:
      st = states.get(int(uid))
      if st is None:
        return
      st.done = True
      st.finish_reason = "error"
      if st.n_tokens == 0:
        st.first_t = now
        ttft_ms.append((now - st.sent_t) * 1000.0)

  cp.start_rank0(on_token_update=_on_token_update, on_error=_on_error)

  driver_done = threading.Event()

  def _rank0_driver() -> None:
    nonlocal prefill_tokens, decode_tokens
    # v0.1: rank0 is control plane only (no local ownership), so distribute
    # requests across owners 1..7. Enforce uid%world_size==owner so the owner
    # can validate deterministically.
    next_uid_by_owner: dict[int, int] = {o: int(o) for o in range(1, world_size)}

    def _alloc_uid(owner: int) -> int:
      if int(owner) <= 0 or int(owner) >= int(world_size):
        raise ValueError(f"invalid owner={owner}; expected 1..{world_size-1}")
      uid = int(next_uid_by_owner[int(owner)])
      next_uid_by_owner[int(owner)] = int(uid) + int(world_size)
      return int(uid)

    for i in range(int(args.concurrency)):
      owner = 1 + (i % 7)
      uid = _alloc_uid(int(owner))
      _submit(uid, owner=int(owner), prompt_len=int(args.prompt_len), max_tokens=int(args.output_len))

    # Background prefill backlog: prefill-only (max_tokens=1).
    for i in range(int(args.prefill_backlog)):
      owner = 1 + (i % 7)
      uid = _alloc_uid(int(owner))
      _submit(uid, owner=int(owner), prompt_len=int(args.prefill_prompt_len), max_tokens=1)

    start = time.perf_counter()
    deadline = start + float(args.duration_s)
    next_probe_t = start
    probe_period = (1.0 / float(args.probe_rps)) if float(args.probe_rps) > 0 else 0.0

    while time.perf_counter() < deadline:
      if probe_period > 0 and time.perf_counter() >= next_probe_t:
        # Probe is a short-lived request (max_tokens=1); TTFT sample comes from its first token.
        owner = 1 + (int(next_probe_t * 1000) % 7)
        uid = _alloc_uid(int(owner))
        _submit(uid, owner=int(owner), prompt_len=int(args.probe_prompt_len), max_tokens=1)
        next_probe_t += probe_period

      # Keep background backlog full by replacing finished prefill-only requests.
      if args.prefill_backlog > 0:
        with lock:
          done_prefill = [k for k, st in states.items() if st.done and st.n_tokens <= 1 and st.finish_reason]
        for k in done_prefill[: int(args.prefill_backlog)]:
          with lock:
            states.pop(k, None)
          owner = 1 + (int(k) % 7)
          uid = _alloc_uid(int(owner))
          _submit(uid, owner=int(owner), prompt_len=int(args.prefill_prompt_len), max_tokens=1)

      time.sleep(0.001)

    # Snapshot stats.
    with lock:
      done = [st for st in states.values() if st.done]
      total_done = len(done)

    wall_s = time.perf_counter() - start
    prefill_toks = float(prefill_tokens) / max(wall_s, 1e-9)
    decode_toks = float(decode_tokens) / max(wall_s, 1e-9)

    print("=" * 80, flush=True)
    print("nmoe.serve Mixed Latency Benchmark (DP=8 ownership, EP=8 experts)", flush=True)
    print("=" * 80, flush=True)
    print(f"world_size: {world_size}", flush=True)
    print(f"attention_type: {args.attention_type}", flush=True)
    print(f"C={args.concurrency} P={args.prompt_len} N={args.output_len}", flush=True)
    print(f"prefill_backlog={args.prefill_backlog} prefill_prompt_len={args.prefill_prompt_len}", flush=True)
    print(f"probe_rps={args.probe_rps} probe_prompt_len={args.probe_prompt_len}", flush=True)
    print(f"duration_s={args.duration_s}", flush=True)
    print("-" * 80, flush=True)
    print(f"done_requests: {total_done}", flush=True)
    print(f"prefill_tok_s_node: {prefill_toks:,.1f}", flush=True)
    print(f"decode_tok_s_node: {decode_toks:,.1f}", flush=True)
    print(f"ttft_ms_p50: {_percentile(ttft_ms, 50):.2f}", flush=True)
    print(f"ttft_ms_p99: {_percentile(ttft_ms, 99):.2f}", flush=True)
    print(f"itl_ms_p50:  {_percentile(itl_ms, 50):.2f}", flush=True)
    print(f"itl_ms_p99:  {_percentile(itl_ms, 99):.2f}", flush=True)
    if ttft_ms:
      print(f"ttft_ms_mean: {statistics.mean(ttft_ms):.2f}", flush=True)
    if itl_ms:
      print(f"itl_ms_mean:  {statistics.mean(itl_ms):.2f}", flush=True)

    # Initiate shutdown: stop owners and stop lockstep loop.
    cp.shutdown()
    orch.request_stop()
    driver_done.set()

  def _submit(uid: int, *, owner: int, prompt_len: int, max_tokens: int) -> None:
    nonlocal prefill_tokens
    # Use a stable token id to avoid adversarial/random sequences triggering
    # non-finite activations in the stack (not relevant to this benchmark).
    ids = torch.full((prompt_len,), 100, dtype=torch.int32, device="cpu")
    sent_t = time.perf_counter()
    with lock:
      states[int(uid)] = _ReqState(owner=int(owner), sent_t=sent_t)
      prefill_tokens += int(prompt_len)
    cp.send_request_init(
      owner=int(owner),
      uid=int(uid),
      output_mode_id=OUTPUT_MODE_ID_TOKENS,
      topk=0,
      max_tokens=int(max_tokens),
      top_k=0,
      seed_or_minus1=-1,
      temperature=0.0,
      top_p=1.0,
      input_ids=ids,
    )

  driver_t = threading.Thread(target=_rank0_driver, daemon=False)
  driver_t.start()

  # Main-thread lockstep loop. Stop is triggered by the driver via orch.request_stop().
  while True:
    orch._recv_requests()
    any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
    if any_shutdown:
      break
    if any_decode:
      orch._lockstep_run_phase("decode")
    if any_prefill:
      orch._lockstep_run_phase("prefill")
    time.sleep(0.0005)

  driver_t.join(timeout=float(args.duration_s) + 180.0)
  all_stopped = _all_ranks_ok(bool(driver_done.is_set()))
  if not all_stopped:
    cp.close(timeout_s=30.0)
    dist.destroy_process_group()
    return 2

  local_cuda_ok = True
  try:
    torch.cuda.synchronize()
  except Exception:
    local_cuda_ok = False
  all_cuda_ok = _all_ranks_ok(local_cuda_ok)

  # All ranks reach the destroy point before tearing down DeepEP/NVSHMEM state.
  dist.barrier(group=ctrl_group)
  try:
    orch.shutdown()
  except Exception:
    all_cuda_ok = False
  dist.barrier(group=ctrl_group)

  cp.close(timeout_s=30.0)
  dist.destroy_process_group()
  if not all_cuda_ok:
    return 3
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
