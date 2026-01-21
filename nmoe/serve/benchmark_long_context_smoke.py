# SPDX-License-Identifier: Apache-2.0
"""Long-context smoke benchmark for TP=1/DP=8/EP=8 (single node).

This is a *correctness smoke* for long context lengths (128k–161k):
- Prefill a long prompt via the production orchestrator path (lockstep, T=0 ok).
- Decode a small number of tokens (greedy) to validate stability.

It is intentionally not a per-commit test: it is expensive.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path
import math

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


def _broadcast_i32_1d_cpu(x: torch.Tensor, *, src: int, group) -> torch.Tensor:
  if x.dtype != torch.int32 or x.device.type != "cpu" or x.ndim != 1:
    raise ValueError("expected int32[?] CPU tensor")
  n = torch.tensor([int(x.numel())], dtype=torch.int64, device="cpu")
  if dist.get_rank() == src:
    dist.broadcast(n, src=src, group=group)
  else:
    dist.broadcast(n, src=src, group=group)
    x = torch.empty((int(n.item()),), dtype=torch.int32, device="cpu")
  dist.broadcast(x, src=src, group=group)
  return x


def main() -> int:
  _maybe_set_cutlass_path()

  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True, type=str, help="Path to mp shard dir (e.g. /data/models/DeepSeek-V3-0324-ep8-tp1)")
  ap.add_argument("--tokenizer-path", type=str, default=None, help="Tokenizer dir (default: ckpt)")
  ap.add_argument("--attention-type", choices=["mla", "dsa"], default="mla")
  ap.add_argument(
    "--base-prompt",
    type=str,
    default="Once upon a time, ",
    help="Base prompt text (rank0); tokenized and repeated/truncated to --prompt-len.",
  )
  ap.add_argument("--prompt-len", type=int, default=128000)
  ap.add_argument("--output-len", type=int, default=16)
  ap.add_argument("--num-pages", type=int, default=4096)
  ap.add_argument("--page-size", type=int, default=64)
  ap.add_argument("--max-seq-len", type=int, default=163840)
  ap.add_argument("--max-batch-size", type=int, default=1)
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
  ap.add_argument("--progress-every", type=int, default=100, help="Print progress every N lockstep iterations (rank0 only).")
  ap.add_argument("--stall-secs", type=float, default=60.0, help="Abort if no progress for this many seconds.")
  ap.add_argument("--print-output", action="store_true", help="Decode and print generated output tokens (rank0 only).")
  ap.add_argument("--max-print-chars", type=int, default=2000, help="Max chars of decoded output to print.")
  args = ap.parse_args()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size != 8:
    raise RuntimeError(f"benchmark_long_context_smoke requires world_size=8 (got {world_size})")
  torch.cuda.set_device(rank)

  # KV capacity must be consistent across scheduler + CuTeDSL kernel contracts.
  # - Scheduler page tables are sized from max_seq_len/page_size.
  # - The MLA kernel buffers are sized from num_pages*page_size.
  # If max_seq_len exceeds KV capacity, paged prefill will fail when copying the
  # block table into CuTeDSL buffers.
  kv_capacity_tokens = int(args.num_pages) * int(args.page_size)
  if int(args.max_seq_len) > kv_capacity_tokens:
    need_pages = int(math.ceil(int(args.max_seq_len) / int(args.page_size)))
    raise RuntimeError(
      f"invalid KV config: max_seq_len={int(args.max_seq_len)} exceeds KV capacity "
      f"num_pages*page_size={kv_capacity_tokens} (num_pages={int(args.num_pages)} page_size={int(args.page_size)}). "
      f"Increase --num-pages to >= {need_pages} or lower --max-seq-len."
    )
  if int(args.prompt_len) + int(args.output_len) > int(args.max_seq_len):
    raise RuntimeError(
      f"invalid request: prompt_len+output_len={int(args.prompt_len) + int(args.output_len)} exceeds max_seq_len={int(args.max_seq_len)}"
    )
  if int(args.moe_expected_m) <= 0 or (int(args.moe_expected_m) % 16) != 0:
    raise RuntimeError(f"--moe-expected-m must be > 0 and a multiple of 16 (got {args.moe_expected_m}).")

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  init_distributed(rank, world_size, tp_size=1)
  cfg = load_model_config(args.ckpt) if os.path.exists(os.path.join(args.ckpt, "config.json")) else ModelConfig(attention_type=args.attention_type)
  if args.attention_type:
    cfg = ModelConfig(**{**cfg.__dict__, "attention_type": args.attention_type})

  engine_cfg = EngineConfig(
    num_pages=int(args.num_pages),
    page_size=int(args.page_size),
    num_layers=int(cfg.num_layers),
    kv_lora_rank=int(cfg.kv_lora_rank),
    qk_rope_head_dim=int(cfg.qk_rope_head_dim),
    max_batch_size=int(args.max_batch_size),
    max_seq_len=int(args.max_seq_len),
    moe_expected_m=int(args.moe_expected_m),
    attention_type=str(cfg.attention_type),
    idx_dim=int(cfg.dsa_idx_dim),
    max_step_tokens=int(args.max_prefill_tokens),
    tp_size=1,
  )
  orch_cfg = OrchestratorConfig(
    max_batch_size=int(args.max_batch_size),
    max_prefill_tokens=int(args.max_prefill_tokens),
    max_decode_tokens=int(args.max_batch_size),
    max_seq_len=int(args.max_seq_len),
    num_pages=int(args.num_pages),
    page_size=int(args.page_size),
    enable_overlap=False,
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=int(args.chunk_size),
    enable_prefix_cache=not args.disable_prefix_cache,
    enable_fast_path=True,
    max_prompt_tokens=max(0, int(args.max_seq_len) - int(args.output_len)),
    max_output_tokens=int(args.output_len),
  )

  orch = Orchestrator(cfg, engine_cfg, orch_cfg, rank=rank, world_size=world_size)

  shard_path = os.path.join(args.ckpt, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(shard_path):
    missing, unexpected = load_sharded_checkpoint(orch.engine.model, args.ckpt, rank=rank, world_size=world_size)
  else:
    missing, unexpected = load_checkpoint(orch.engine.model, args.ckpt, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()
  if rank == 0 and (missing or unexpected):
    raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")

  ctrl_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")

  # Build a "normal" prompt on rank0 and broadcast as int32 tokens.
  tokenizer = None
  if rank == 0:
    from transformers import AutoTokenizer

    tok_path = args.tokenizer_path or args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    base = tokenizer.encode(
      str(args.base_prompt),
      return_tensors="pt",
      add_special_tokens=False,
    )[0].to(torch.int32).cpu()
    if base.numel() == 0:
      raise RuntimeError("tokenizer returned empty base prompt")
    reps = (int(args.prompt_len) + int(base.numel()) - 1) // int(base.numel())
    ids = base.repeat(int(reps))[: int(args.prompt_len)].contiguous()
  else:
    ids = torch.empty((0,), dtype=torch.int32, device="cpu")

  input_ids = _broadcast_i32_1d_cpu(ids, src=0, group=ctrl_group)

  if rank == 0:
    ok, err = orch.validate_request_bounds(int(input_ids.numel()), int(args.output_len))
    if not ok:
      raise RuntimeError(f"request rejected: {err}")
    req = orch.create_request(
      input_ids=input_ids,
      profile_name="production_generate",
      temperature=0.0,
      max_tokens=int(args.output_len),
    )
    orch.add_request(req)
    print(
      f"[rank0] request_enqueued prompt_len={int(input_ids.numel())} output_len={int(args.output_len)} "
      f"chunked_prefill={int(not args.disable_chunked_prefill)} chunk_size={int(args.chunk_size)} "
      f"max_prefill_tokens={int(args.max_prefill_tokens)}",
      flush=True,
    )
  else:
    req = None

  done_local = torch.zeros((1,), device="cpu", dtype=torch.int64)
  t0 = time.perf_counter()
  prefill_done_t = None
  prefill_done_out_len = None
  first_token_t = None
  prof = None
  prof_decode_remaining = 0
  nsys_decode_remaining = 0
  nsys_active = False
  last_progress_t = t0
  last_cached = -1
  last_out = -1
  iters = 0
  prefill_steps = 0
  decode_steps = 0
  try:
    while True:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        raise RuntimeError("unexpected shutdown during run")
      if any_decode:
        if (
          rank == 0
          and req is not None
          and prefill_done_t is not None
          and not nsys_active
          and os.environ.get("NMOE_CUDA_PROFILER", "0") == "1"
        ):
          nsys_decode_remaining = max(1, min(4, int(args.output_len)))
          print(
            f"[rank0] starting CUDA profiler range for {nsys_decode_remaining} decode steps "
            "(use with: nsys profile --capture-range=cudaProfilerApi)",
            flush=True,
          )
          torch.cuda.synchronize()
          torch.cuda.profiler.start()
          nsys_active = True
        if rank == 0 and req is not None and prefill_done_t is not None and prof is None and os.environ.get("NMOE_PROFILE_DECODE", "0") == "1":
          prof_decode_remaining = max(1, min(4, int(args.output_len)))
          prof_dir = os.environ.get("NMOE_PROFILE_DIR", "/tmp/torch_prof_decode_longctx")
          os.makedirs(prof_dir, exist_ok=True)
          print(
            f"[rank0] starting torch profiler for {prof_decode_remaining} decode steps "
            f"(dir={prof_dir})",
            flush=True,
          )
          schedule = torch.profiler.schedule(wait=0, warmup=0, active=int(prof_decode_remaining), repeat=1)
          on_trace_ready = torch.profiler.tensorboard_trace_handler(
            str(prof_dir),
            worker_name=f"rank0_p{int(args.prompt_len)}",
          )
          prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
          )
          prof.__enter__()
        if rank == 0 and req is not None and (decode_steps < 3 or (args.progress_every > 0 and (decode_steps % int(args.progress_every) == 0))):
          print(
            f"[rank0] decode_step_start step={decode_steps} cached_len={int(req.cached_len)}/{int(input_ids.numel())} "
            f"out_len={len(req.output_ids)}/{int(args.output_len)}",
            flush=True,
          )
        try:
          orch._lockstep_run_phase("decode")
        except Exception as e:
          print(f"[rank{rank}] ERROR in lockstep decode: {e}", flush=True)
          traceback.print_exc()
          raise
        decode_steps += 1
        if nsys_active:
          nsys_decode_remaining -= 1
          if nsys_decode_remaining <= 0:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            nsys_active = False
            print("[rank0] CUDA profiler range stopped", flush=True)
        if prof is not None:
          prof.step()
          prof_decode_remaining -= 1
          if prof_decode_remaining <= 0:
            try:
              prof.__exit__(None, None, None)
              print("[rank0] torch profiler stopped", flush=True)
            finally:
              prof = None
      if any_prefill:
        if rank == 0 and req is not None and (prefill_steps < 3 or (args.progress_every > 0 and (prefill_steps % int(args.progress_every) == 0))):
          print(
            f"[rank0] prefill_step_start step={prefill_steps} cached_len={int(req.cached_len)}/{int(input_ids.numel())} "
            f"out_len={len(req.output_ids)}/{int(args.output_len)}",
            flush=True,
          )
        try:
          orch._lockstep_run_phase("prefill")
        except Exception as e:
          print(f"[rank{rank}] ERROR in lockstep prefill: {e}", flush=True)
          traceback.print_exc()
          raise
        prefill_steps += 1

      iters += 1
      if rank == 0 and req is not None:
        cached = int(req.cached_len)
        out_n = int(len(req.output_ids))
        now = time.perf_counter()
        if first_token_t is None and out_n > 0:
          first_token_t = now
        if prefill_done_t is None and cached >= int(input_ids.numel()):
          prefill_done_t = now
          prefill_done_out_len = int(out_n)
        if cached != last_cached or out_n != last_out:
          last_cached = cached
          last_out = out_n
          last_progress_t = time.perf_counter()
        if args.progress_every > 0 and (iters % int(args.progress_every) == 0):
          age = time.perf_counter() - last_progress_t
          print(
            f"[rank0] iter={iters} prefill_steps={prefill_steps} decode_steps={decode_steps} "
            f"cached_len={cached}/{int(input_ids.numel())} out_len={out_n}/{int(args.output_len)} "
            f"any_prefill={int(any_prefill)} any_decode={int(any_decode)} stall_age_s={age:.1f}",
            flush=True,
          )
        if float(args.stall_secs) > 0 and (time.perf_counter() - last_progress_t) > float(args.stall_secs):
          raise RuntimeError(
            f"STALL: no progress for {float(args.stall_secs):.1f}s "
            f"(cached_len={cached}/{int(input_ids.numel())} out_len={out_n}/{int(args.output_len)} "
            f"any_prefill={int(any_prefill)} any_decode={int(any_decode)} "
            f"prefill_steps={prefill_steps} decode_steps={decode_steps} iters={iters})"
          )

      done_local.fill_(1 if (rank == 0 and req is not None and req.is_finished) else 0)
      group = getattr(orch, "_lockstep_group", None)
      if group is not None:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX, group=group)
      else:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX)
      if bool(int(done_local.item())):
        break
      time.sleep(0.0005)
  finally:
    orch.shutdown()
    dist.destroy_process_group()

  if rank == 0:
    dt = time.perf_counter() - t0
    assert req is not None
    if not req.output_ids:
      raise SystemExit("FAIL: produced no output tokens")
    print("=" * 70, flush=True)
    print("Long-context smoke PASS", flush=True)
    print("=" * 70, flush=True)
    prompt_len = int(input_ids.numel())
    out_len = int(len(req.output_ids))
    total_s = float(dt)
    print(f"prompt_len={prompt_len} output_len={out_len} time_s={total_s:.2f}", flush=True)
    if prefill_done_t is not None:
      prefill_s = float(prefill_done_t - t0)
      decode_s = float((t0 + total_s) - prefill_done_t)
      prefill_out = int(prefill_done_out_len or 0)
      decode_tokens = max(0, out_len - prefill_out)
      decode_s_per_tok = (decode_s / decode_tokens) if decode_tokens > 0 else float("nan")
      decode_itl_ms_avg = float("nan")
      if first_token_t is not None and out_len > 1:
        decode_itl_ms_avg = float(((t0 + total_s) - first_token_t) / float(out_len - 1)) * 1000.0
      print(
        f"prefill_time_s={prefill_s:.2f} decode_time_s={decode_s:.2f} "
        f"decode_itl_ms_avg={decode_itl_ms_avg:.2f} "
        f"(prefill_out_len={prefill_out} decode_tokens={decode_tokens} decode_s_per_tok={decode_s_per_tok:.4f})",
        flush=True,
      )
    if first_token_t is not None:
      print(f"ttft_s={float(first_token_t - t0):.2f}", flush=True)
    print(f"finish_reason={req.finish_reason!r}", flush=True)
    if bool(args.print_output):
      if tokenizer is None:
        print("WARNING: tokenizer unavailable; cannot decode output tokens", flush=True)
      else:
        txt = tokenizer.decode(req.output_ids, skip_special_tokens=False)
        if int(args.max_print_chars) > 0 and len(txt) > int(args.max_print_chars):
          txt = txt[: int(args.max_print_chars)] + "\n...[truncated]..."
        print("-" * 70, flush=True)
        print("decoded_output:", flush=True)
        print(txt, flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
