# SPDX-License-Identifier: Apache-2.0
"""Inference orchestrator - main event loop coordinating scheduler and engine.

Optimized for TP=1/DP=8/EP=8 dynamic disaggregation:
- Pre-allocated input buffers eliminate per-step tensor allocations
- Lockstep coordination across ranks to keep DeepEP collectives aligned
- Prefill and decode advance independently (all ranks participate in each)
- Ranks with no local work still participate with T=0
- Expert dispatch/combine handled by DeepEP
"""

from __future__ import annotations

import asyncio
from collections import deque
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.distributed as dist

from nmoe.serve.buffers import InputBuffers
from nmoe.serve.cache import KvCache, MlaKvLayout
from nmoe.serve.config import PROFILES, BatchingMode, Profile, ServeConfig
from nmoe.serve.control_plane import ControlPlane, finish_reason_str_to_id
from nmoe.serve.engine import Engine, EngineConfig
from nmoe.serve.model import ModelConfig
from nmoe.serve.scheduler import Scheduler, SchedulerConfig
from nmoe.serve.types import Batch, ForwardOutput, OutputMode, Request, RequestStatus


@dataclass
class OrchestratorConfig:
  """Configuration for the orchestrator."""
  max_batch_size: int = 256
  max_prefill_tokens: int = 8192
  max_decode_tokens: int = 4096
  # DeepSeek V3 family supports long context (128k–161k depending on version).
  # This is a serving-time bound; the KV page budget must still be provisioned
  # via `num_pages` for the desired workload.
  max_seq_len: int = 163840
  num_pages: int = 4096
  page_size: int = 64
  enable_overlap: bool = True
  enable_chunked_prefill: bool = True
  chunk_size: int = 2048
  enable_prefix_cache: bool = True
  enable_cuda_graph: bool = False  # Greedy decode TOKENS only (MLA), via Engine._DecodeGraph
  enable_fast_path: bool = True  # Use pre-allocated buffers for zero-alloc hot path
  # Request bounds (P1 hardening)
  # Default keeps headroom for max_output_tokens within max_seq_len.
  max_prompt_tokens: int = 159744  # Max input/prompt tokens per request
  max_output_tokens: int = 4096   # Max generation tokens per request (caps sampling_params.max_tokens)
  max_pending_requests: int = 1024  # Bounded queue size for backpressure


@dataclass(frozen=True)
class _AsyncStep:
  """One in-flight step whose D2H copies complete asynchronously."""

  step_id: int
  batch: Batch
  output: ForwardOutput
  # For each request in batch.reqs, the output_ids index reserved for this step.
  out_pos: list[int]


_PENDING_TOKEN_ID: int = -1


def _materialized_output_len(req: "Request") -> int:
  """Return the number of materialized output tokens (ignores trailing placeholders).

  In overlap mode we append `_PENDING_TOKEN_ID` placeholders to reserve KV/table
  slots for in-flight decode steps. Those placeholders must not count toward
  max_tokens stop conditions.
  """
  n = len(req.output_ids)
  while n > 0 and int(req.output_ids[n - 1]) == _PENDING_TOKEN_ID:
    n -= 1
  return n


def validate_request_bounds_cfg(cfg: OrchestratorConfig, prompt_tokens: int, max_tokens: int) -> tuple[bool, str]:
  """Validate a request against orchestrator bounds (pure function).

  Used by both the server and tests. This must be deterministic and must fail
  fast with actionable errors (no silent downshifts).
  """
  prompt_tokens = int(prompt_tokens)
  max_tokens = int(max_tokens)

  if prompt_tokens < 0:
    return False, f"Prompt tokens must be >= 0 (got {prompt_tokens})"
  if max_tokens < 0:
    return False, f"max_tokens must be >= 0 (got {max_tokens})"

  # Check prompt length
  if prompt_tokens > int(cfg.max_prompt_tokens):
    return False, f"Prompt too long: {prompt_tokens} tokens exceeds limit of {int(cfg.max_prompt_tokens)}"

  # Check max_tokens
  if max_tokens > int(cfg.max_output_tokens):
    return False, f"max_tokens too large: {max_tokens} exceeds limit of {int(cfg.max_output_tokens)}"

  # Check total sequence length
  total_seq_len = prompt_tokens + max_tokens
  if total_seq_len > int(cfg.max_seq_len):
    return (
      False,
      f"Total sequence length {total_seq_len} (prompt={prompt_tokens} + max_tokens={max_tokens}) "
      f"exceeds limit of {int(cfg.max_seq_len)}",
    )

  # Fail fast if the request cannot fit in the provisioned KV page budget.
  #
  # NOTE: This assumes the request is owned by a single rank (DP ownership). In
  # dynamic disagg mode, other ranks may participate with T=0, but KV pages are
  # only allocated on the owner rank.
  page_size = int(cfg.page_size)
  num_pages = int(cfg.num_pages)
  if page_size <= 0:
    return False, f"Invalid page_size={page_size} (must be > 0)"
  if num_pages <= 0:
    return False, f"Invalid num_pages={num_pages} (must be > 0)"

  pages_needed = (total_seq_len + page_size - 1) // page_size
  if pages_needed > num_pages:
    return (
      False,
      f"Insufficient KV pages: need {pages_needed} pages for seq_len={total_seq_len} "
      f"(page_size={page_size}), but num_pages={num_pages}. "
      f"Increase --num-pages or reduce prompt/max_tokens.",
    )

  return True, ""


class Orchestrator:
  """
  Main inference orchestrator.

  Coordinates:
  - Request queue (from API server)
  - Scheduler (prefill/decode batching)
  - Engine (forward pass)
  - Result dispatch (back to API)

  Event loop pattern from mini-sglang:
  1. Receive new requests
  2. Schedule next batch (prefill or decode)
  3. Forward pass
  4. Process results (sample tokens, update state)
  5. Repeat
  """

  def __init__(
    self,
    model_config: ModelConfig,
    engine_config: EngineConfig,
    orch_config: OrchestratorConfig,
    rank: int = 0,
    world_size: int = 1,
    *,
    control_plane: Optional[ControlPlane] = None,
  ) -> None:
    self.model_config = model_config
    self.engine_config = engine_config
    self.orch_config = orch_config
    self.rank = rank
    self.world_size = world_size
    self.control_plane = control_plane

    # Device
    self.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(self.device)

    # Engine (owns model and KV caches)
    self.engine = Engine(
      model_config=model_config,
      engine_config=engine_config,
      rank=rank,
      world_size=world_size,
    )
    self.engine.enable_cuda_graph = orch_config.enable_cuda_graph
    if self.engine.enable_cuda_graph:
      if self.engine.attention_type == "dsa":
        raise NotImplementedError("enable_cuda_graph=True requires attention_type=mla (DSA decode graphs not implemented).")
      transport = getattr(self.engine.buffer, "_nmoe_ep_transport", "deepep")
      if transport == "deepep" and not bool(getattr(self.engine.buffer, "low_latency_mode", False)):
        raise RuntimeError(
          "enable_cuda_graph=True requires DeepEP low-latency mode. "
          "Set NMOE_DEEPEP_LOW_LATENCY=1 (or enable multi-node auto-detect)."
        )

    # KV cache for scheduler (separate from engine's internal caches)
    kv_layout = MlaKvLayout(
      num_blocks=orch_config.num_pages,
      block_size=orch_config.page_size,
    )
    self.kv_cache = KvCache(kv_layout, self.device, enable_prefix_cache=orch_config.enable_prefix_cache)

    # Pre-allocated input buffers for zero-allocation hot path
    # For TP=1/DP=8: Each GPU has its own buffers, no sharing
    self.input_buffers: Optional[InputBuffers] = None
    if orch_config.enable_fast_path:
      # Use chunk_size for input_ids/positions/out_loc (per-step buffers)
      # but always use max_seq_len for block_table (full page table)
      step_seq = orch_config.chunk_size if orch_config.enable_chunked_prefill else orch_config.max_seq_len
      self.input_buffers = InputBuffers.create(
        device=self.device,
        max_batch_size=orch_config.max_batch_size,
        max_seq_len=step_seq,
        max_total_seq_len=orch_config.max_seq_len,  # For block_table sizing
        page_size=orch_config.page_size,
      )

    # Scheduler
    sched_config = SchedulerConfig(
      max_batch_size=orch_config.max_batch_size,
      max_prefill_tokens=orch_config.max_prefill_tokens,
      max_decode_tokens=orch_config.max_decode_tokens,
      max_seq_len=orch_config.max_seq_len,
      page_size=orch_config.page_size,
      enable_chunked_prefill=orch_config.enable_chunked_prefill,
      chunk_size=orch_config.chunk_size,
    )
    self.scheduler = Scheduler(sched_config, self.kv_cache, self.device, self.input_buffers)

    # Lockstep coordinator communicator.
    #
    # DeepEP uses dist.group.WORLD (NCCL). Lockstep coordination uses a separate
    # gloo group on CPU to avoid GPU stream synchronization overhead. This
    # eliminates ~36ms of waste per step that was caused by NCCL all-reduce on
    # tiny flag tensors forcing a GPU stream sync.
    self._lockstep_group = None
    if self.world_size > 1 and dist.is_initialized():
      self._lockstep_group = dist.new_group(ranks=list(range(self.world_size)), backend="gloo")

    # Request queues (bounded for backpressure)
    self._input_queue: queue.Queue[Request] = queue.Queue(maxsize=orch_config.max_pending_requests)
    self._finished_reqs: list[Request] = []
    # Best-effort tracking for DP ownership control-plane cancellation.
    self._uid_to_req: dict[int, Request] = {}
    self._cancelled_uids: set[int] = set()

    # State
    self._running = False
    self._ready = False  # Set True after first successful forward pass or explicit ready signal
    self._async_loop: Optional[asyncio.AbstractEventLoop] = None  # Set by AsyncOrchestrator
    self._uid_counter = 0
    self._last_moe_dispatch_mode: Optional[Literal["normal", "ll"]] = None
    self._shutdown_requested: bool = False

    # Overlap scheduling state
    self._prev_batch: Optional[Batch] = None
    self._prev_output: Optional[ForwardOutput] = None

    # Lockstep overlap (multi-rank): async step completion pipeline.
    # The orchestrator thread never blocks on CUDA events; a consumer thread
    # waits for copy completion and hands results back for single-threaded
    # scheduler/request-state updates.
    self._async_pending_steps: "queue.Queue[Optional[_AsyncStep]]" = queue.Queue()
    self._async_ready_steps: "queue.Queue[_AsyncStep]" = queue.Queue()
    self._async_consumer_thread: Optional[threading.Thread] = None
    self._async_inflight: int = 0
    self._async_step_id: int = 0
    self._async_last_drained_step_id: int = -1
    self._async_deferred_frees: "deque[tuple[int, Request]]" = deque()

  def add_request(self, req: Request) -> None:
    """Add request to input queue (thread-safe). Blocks if queue is full."""
    self._uid_to_req[int(req.uid)] = req
    self._input_queue.put(req)

  def try_add_request(self, req: Request, timeout: float = 0.0) -> bool:
    """Try to add request to input queue. Returns False if queue is full (backpressure).

    Args:
      req: The request to add
      timeout: How long to wait for space (0.0 = non-blocking)

    Returns:
      True if request was accepted, False if queue is full (reject with 503)
    """
    try:
      self._uid_to_req[int(req.uid)] = req
      self._input_queue.put(req, block=(timeout > 0), timeout=timeout if timeout > 0 else None)
      return True
    except queue.Full:
      self._uid_to_req.pop(int(req.uid), None)
      return False

  @property
  def queue_size(self) -> int:
    """Current number of requests in the input queue."""
    return self._input_queue.qsize()

  @property
  def queue_capacity(self) -> int:
    """Max capacity of the input queue."""
    return self.orch_config.max_pending_requests

  @property
  def is_ready(self) -> bool:
    """True if orchestrator is ready to accept requests."""
    return self._ready and self._running

  def mark_ready(self) -> None:
    """Mark orchestrator as ready (called after model load and warmup)."""
    self._ready = True

  def validate_request_bounds(self, prompt_tokens: int, max_tokens: int) -> tuple[bool, str]:
    """Validate request against server bounds.

    Args:
      prompt_tokens: Number of tokens in the prompt
      max_tokens: Requested max_tokens for generation

    Returns:
      (is_valid, error_message) - error_message is empty if valid
    """
    return validate_request_bounds_cfg(self.orch_config, prompt_tokens, max_tokens)

  def create_request(
    self,
    input_ids: torch.Tensor,
    profile_name: str = "production_generate",
    *,
    uid: Optional[int] = None,
    forward_spec: Optional["ForwardSpec"] = None,
    **sampling_kwargs,
  ) -> Request:
    """Create a new request with auto-assigned UID."""
    from nmoe.serve.types import ForwardSpec, SamplingParams

    profile = PROFILES.get(profile_name)
    if profile is None:
      raise ValueError(f"Unknown profile: {profile_name}")

    if uid is None:
      uid = self._uid_counter
      self._uid_counter += 1
    else:
      uid = int(uid)
      self._uid_counter = max(self._uid_counter, uid + 1)

    sampling_params = SamplingParams(**sampling_kwargs)
    forward_spec = forward_spec or profile.to_forward_spec()

    return Request(
      uid=uid,
      input_ids=input_ids,
      sampling_params=sampling_params,
      profile_name=profile_name,
      forward_spec=forward_spec,
    )

  def create_forced_request(
    self,
    input_ids: torch.Tensor,
    *,
    forced_output_ids: list[int],
    profile_name: str = "rl_sample",
    uid: Optional[int] = None,
    forward_spec: Optional["ForwardSpec"] = None,
    **sampling_kwargs,
  ) -> Request:
    """Create a teacher-forcing request.

    The orchestrator will append `forced_output_ids` in order instead of sampling.
    This is used by RL/distillation workflows that need logprobs/logits for known
    target tokens under the production serving execution path.
    """
    forced = [int(x) for x in forced_output_ids]
    if not forced:
      raise ValueError("forced_output_ids must be non-empty")
    # Ensure max_tokens matches the forced sequence length unless explicitly set.
    if "max_tokens" not in sampling_kwargs:
      sampling_kwargs["max_tokens"] = int(len(forced))
    req = self.create_request(
      input_ids=input_ids,
      profile_name=profile_name,
      uid=uid,
      forward_spec=forward_spec,
      **sampling_kwargs,
    )
    req.forced_output_ids = forced
    req.forced_output_pos = 0
    return req

  def cancel(self, uid: int) -> None:
    """Best-effort cancellation for a request owned by this rank."""
    uid = int(uid)
    req = self._uid_to_req.get(uid)
    if req is None:
      self._cancelled_uids.add(uid)
    else:
      req.cancel_flag = True

    # DP=8 ownership: cancellation must close out the rank0 proxy immediately
    # even if the request is still in local queues and has produced no tokens.
    if self.control_plane is not None and self.world_size > 1 and self.rank != 0:
      rid = int(finish_reason_str_to_id("cancelled"))
      flag = 1 | ((rid & 0x7) << 1)
      self.control_plane.enqueue_token_updates([uid], [-1], [flag])

  def run_step(self) -> None:
    """Run one iteration of the event loop."""
    # 1. Receive new requests
    self._recv_requests()

    # 2. Schedule next batch
    batch = self._schedule_batch()
    if batch is None:
      return

    # 3. Forward pass
    output = self.engine.forward_batch(batch)

    # 4. Process results
    self._process_results(batch, output)

  def run_overlap_step(self) -> None:
    """
    Run one iteration with CPU/GPU overlap.

    Pattern from mini-sglang:
    - While GPU runs forward on batch N
    - CPU processes results from batch N-1 and schedules batch N+1
    """
    # Process previous batch results (CPU work)
    if self._prev_batch is not None and self._prev_output is not None:
      self._process_results(self._prev_batch, self._prev_output)

    # Receive new requests (CPU work)
    self._recv_requests()

    # Schedule next batch (CPU work)
    batch = self._schedule_batch()

    # Forward pass (GPU work - starts async)
    output = None
    if batch is not None:
      output = self.engine.forward_batch(batch)

    # Save for next iteration
    self._prev_batch = batch
    self._prev_output = output

  def run(self) -> None:
    """Run the main event loop (blocking)."""
    self._running = True
    self._ready = True  # Mark ready when event loop starts
    print(f"[Orchestrator] run() started, enable_overlap={self.orch_config.enable_overlap}", flush=True)

    try:
      # Multi-rank dynamic disaggregation requires a single global step order:
      # all ranks must agree on (prefill vs decode) each step, and every rank
      # must participate even with T=0.
      if self.world_size > 1:
        if self.orch_config.enable_overlap:
          self._start_async_consumer()
        self._run_lockstep_loop()
        return
      if self.orch_config.enable_overlap:
        self._run_overlap_loop()
      else:
        self._run_simple_loop()
    except Exception as e:
      print(f"[Orchestrator] EXCEPTION in run loop: {e}", flush=True)
      import traceback
      traceback.print_exc()
      raise

  def _run_simple_loop(self) -> None:
    """Simple event loop without overlap."""
    while self._running:
      if self.scheduler.is_idle and self._input_queue.empty():
        time.sleep(0.001)
        continue
      self.run_step()

  def _run_overlap_loop(self) -> None:
    """Event loop with CPU/GPU overlap."""
    while self._running:
      if self.scheduler.is_idle and self._input_queue.empty():
        # Drain any pending work
        if self._prev_batch is not None and self._prev_output is not None:
          self._process_results(self._prev_batch, self._prev_output)
          self._prev_batch = None
          self._prev_output = None
        time.sleep(0.001)
        continue
      self.run_overlap_step()

  def _run_lockstep_loop(self) -> None:
    """Lockstep multi-rank event loop for dynamic disaggregation.

    DeepEP dispatch/combine require that all ranks call collectives in the same
    order. We enforce a global ordering for each collective stream:
    - A decode step is executed if any rank has decode work.
    - A prefill step is executed if any rank has prefill work.
    All ranks participate in each executed step; ranks without local work
    participate with an explicit T=0 MoE-only step (still executes DeepEP
    collectives for remote tokens).
    """
    if not dist.is_initialized():
      raise RuntimeError("Lockstep mode requires torch.distributed initialized.")
    while self._running:
      # Drain completed async steps first (keeps request state fresh).
      if self.orch_config.enable_overlap:
        self._drain_async_ready_steps()
      self._recv_requests()

      any_decode, any_prefill, any_shutdown = self._lockstep_any_work()
      if any_shutdown:
        break
      if not any_decode and not any_prefill:
        time.sleep(0.001)
        continue

      # Decode is latency-sensitive; run it first when present.
      if any_decode:
        self._lockstep_run_phase("decode")
      if any_prefill:
        self._lockstep_run_phase("prefill")

    # Final drain on exit.
    if self.orch_config.enable_overlap:
      self._drain_async_ready_steps(block=True)

  def _lockstep_any_work(self) -> tuple[bool, bool, bool]:
    """Return (any_decode, any_prefill, any_shutdown) across ranks via all-reduce.

    Uses CPU tensors with gloo backend to avoid GPU stream synchronization.
    """
    local_decode = int(self.scheduler.has_pending_decode)
    local_prefill = int(self.scheduler.has_pending_prefill)
    local_shutdown = int(bool(self._shutdown_requested))
    if self.world_size <= 1 or not dist.is_initialized() or self._lockstep_group is None:
      any_decode = bool(local_decode)
      any_prefill = bool(local_prefill)
      any_shutdown = bool(local_shutdown)
    else:
      # CPU tensor for gloo (avoids GPU sync overhead).
      flags = torch.tensor([local_decode, local_prefill, local_shutdown], device="cpu", dtype=torch.int64)
      dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self._lockstep_group)
      any_decode, any_prefill, any_shutdown = (bool(int(x)) for x in flags.tolist())
    if any_shutdown:
      self._running = False
    return any_decode, any_prefill, any_shutdown

  def _lockstep_run_phase(self, phase: Literal["prefill", "decode"]) -> None:
    """Run one lockstep step for a single phase on this rank."""
    # Some harnesses (benchmarks/tests) drive lockstep by calling _lockstep_run_phase
    # directly rather than via Orchestrator.run(). Ensure the async consumer is
    # started in that case.
    if self.orch_config.enable_overlap and self._async_consumer_thread is None:
      self._start_async_consumer()
    if self.orch_config.enable_overlap:
      # Keep request state and deferred frees moving even when a harness drives
      # lockstep manually.
      self._drain_async_ready_steps()

    # Ensure all scheduler GPU writes (H2D copies, metadata tensors) are enqueued
    # on the engine stream. The engine forward uses self.engine.stream, and we
    # rely on stream-ordering rather than global/device synchronization.
    with torch.cuda.stream(self.engine.stream):
      batch = self.scheduler.schedule_phase(phase)

    # Decide DeepEP dispatch mode for this step. In lockstep mode, *all ranks*
    # must call the same sequence of DeepEP collectives. When low-latency mode
    # is enabled, a step may still need to run the normal path (e.g., LOGPROBS /
    # LOGITS profiles that require determinism).
    use_ll = False
    transport = getattr(self.engine.buffer, "_nmoe_ep_transport", "deepep")
    if (
      phase == "decode"
      and transport == "deepep"
      and bool(getattr(self.engine.buffer, "low_latency_mode", False))
    ):
      local_want_ll = 0
      local_want_normal = 0
      if batch is not None:
        mode = batch.reqs[0].forward_spec.output_mode
        if mode == OutputMode.TOKENS:
          local_want_ll = 1
        else:
          local_want_normal = 1
      if self.world_size <= 1 or not dist.is_initialized() or self._lockstep_group is None:
        use_ll = bool(local_want_ll and not local_want_normal)
      else:
        # CPU tensor for gloo (avoids GPU sync overhead).
        flags = torch.tensor([local_want_ll, local_want_normal], device="cpu", dtype=torch.int64)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=self._lockstep_group)
        any_ll, any_normal = (bool(int(x)) for x in flags.tolist())
        use_ll = bool(any_ll and not any_normal)
      if use_ll:
        self._maybe_clean_low_latency_buffer()
      self._last_moe_dispatch_mode = "ll" if use_ll else "normal"
    else:
      self._last_moe_dispatch_mode = "normal"

    # Propagate step dispatch mode to MoE modules (DeepEP-only). For decode-only
    # transports (e.g. inference-RDEP), do not force DeepEP modes.
    if transport == "deepep":
      setattr(self.engine.buffer, "_nmoe_force_low_latency", bool(use_ll))
    else:
      if hasattr(self.engine.buffer, "_nmoe_force_low_latency"):
        delattr(self.engine.buffer, "_nmoe_force_low_latency")

    if batch is None:
      # In lockstep mode, all ranks must execute the same DeepEP collective stream.
      # T=0 ranks must follow the globally decided dispatch mode (LL vs normal).
      self._participate_moe_only(phase, low_latency=bool(use_ll))
      return

    # Async overlap path: decode TOKENS only (first increment). Other output modes
    # keep the synchronous path for correctness until extended.
    if (
      self.orch_config.enable_overlap
      and phase == "decode"
      and batch.reqs[0].forward_spec.output_mode == OutputMode.TOKENS
    ):
      # Bounded in-flight steps to keep pinned ring buffers safe and avoid
      # unbounded speculative enqueue.
      max_inflight = int(getattr(self.engine, "_cpu_ring_size", 2))
      while self._async_inflight >= max_inflight and self._running:
        # Block until at least one step completes.
        self._drain_async_ready_steps(block=True, max_items=1)

      step_id = int(self._async_step_id)
      copy_slot = int(step_id % max_inflight)
      output = self.engine.forward_batch(batch, copy_slot=copy_slot)

      # Update GPU last-token table in stream order (enables next-step scheduling).
      if batch.table_idx is None or output.next_tokens_gpu is None:
        raise RuntimeError("Async decode requires batch.table_idx and output.next_tokens_gpu.")
      with torch.cuda.stream(self.engine.stream):
        self.scheduler.update_last_tokens(batch.table_idx, output.next_tokens_gpu.to(torch.int64))

      # Reserve output slots and advance cached_len without waiting on D2H.
      out_pos: list[int] = []
      for req in batch.reqs:
        req.cached_len += 1
        req.output_ids.append(_PENDING_TOKEN_ID)
        out_pos.append(len(req.output_ids) - 1)

      self._async_pending_steps.put(_AsyncStep(step_id=step_id, batch=batch, output=output, out_pos=out_pos))
      self._async_inflight += 1
      self._async_step_id += 1
      return

    # Synchronous fallback.
    output = self.engine.forward_batch(batch)
    self._process_results(batch, output)

  def _maybe_clean_low_latency_buffer(self) -> None:
    """Prepare DeepEP low-latency buffer when switching normal -> low-latency."""
    buf = getattr(self.engine, "buffer", None)
    if buf is None:
      return
    if not bool(getattr(buf, "low_latency_mode", False)):
      return
    if self._last_moe_dispatch_mode == "ll":
      return
    clean = getattr(buf, "clean_low_latency_buffer", None)
    if clean is None:
      return
    num_max_dispatch_tokens_per_rank = int(
      getattr(self.engine, "ll_max_dispatch_tokens_per_rank", self.orch_config.max_batch_size)
    )
    clean(
      num_max_dispatch_tokens_per_rank,
      int(self.model_config.hidden_size),
      int(self.model_config.num_experts),
    )

  def _participate_moe_only(self, phase: str, *, low_latency: bool) -> None:
    """Participate in DeepEP collectives for a step with local T=0.

    We execute MoE layers only (no attention) with an empty token batch. This is
    sufficient for DeepEP because remote tokens are received and processed
    inside MoE.dispatch/compute/combine.
    """
    num_dense_layers = int(self.model_config.num_dense_layers)
    hidden = int(self.model_config.hidden_size)
    # For decode-only transports (e.g. inference-RDEP), empty ranks must enter
    # the MoE layers with a decode-shaped tensor (S==1) so the model can select
    # the correct dispatch/combine implementation without relying on global
    # "low-latency" mode flags.
    if phase == "decode":
      x = torch.empty((0, 1, hidden), device=self.device, dtype=torch.bfloat16)
    else:
      x = torch.empty((0, hidden), device=self.device, dtype=torch.bfloat16)
    # Match the engine stream ordering used by forward_batch().
    with torch.cuda.stream(self.engine.stream):
      for layer in self.engine.model.layers[num_dense_layers:]:
        # Only MoE layers have DeepEP collectives. Dense layers are skipped.
        if getattr(layer, "is_moe", False):
          layer.ffn(x, low_latency=low_latency)

    # Critical: ensure this rank finishes the step before starting the next one.
    # Ranks with local work synchronize via Engine.forward_batch() + output copy
    # events; T=0 ranks must explicitly synchronize the engine stream so DeepEP
    # collectives do not overlap across steps/phases.
    evt = torch.cuda.Event()
    evt.record(self.engine.stream)
    evt.synchronize()

  def stop(self) -> None:
    """Stop the event loop."""
    self._running = False
    self._ready = False

  def request_stop(self) -> None:
    """Request a coordinated stop in lockstep multi-rank mode.

    In world_size>1 lockstep mode, ranks must exit together to avoid hanging
    collectives. This sets a local flag that is OR-reduced in the lockstep
    coordinator so all ranks observe it and stop together.
    """
    self._shutdown_requested = True

  def shutdown(self) -> None:
    """Clean shutdown."""
    self.request_stop()
    self.stop()
    if self.orch_config.enable_overlap:
      # Best-effort drain so we don't destroy DeepEP buffers while async decode
      # steps are still in-flight.
      self._drain_async_ready_steps()
      deadline = time.perf_counter() + 5.0
      while self._async_inflight and time.perf_counter() < deadline:
        self._drain_async_ready_steps(block=True, max_items=1)
      if not self._async_inflight:
        self._release_async_deferred_frees()
    if self.control_plane is not None:
      self.control_plane.shutdown()
    self._stop_async_consumer()
    self.engine.shutdown()

  def _start_async_consumer(self) -> None:
    if self._async_consumer_thread is not None:
      return
    t = threading.Thread(target=self._async_consumer_main, daemon=True)
    self._async_consumer_thread = t
    t.start()

  def _stop_async_consumer(self) -> None:
    t = self._async_consumer_thread
    if t is None:
      return
    # Sentinel + join best-effort (shutdown path).
    try:
      self._async_pending_steps.put_nowait(None)
    except Exception:
      pass
    t.join(timeout=5.0)
    self._async_consumer_thread = None

  def _async_consumer_main(self) -> None:
    while True:
      item = self._async_pending_steps.get()
      if item is None:
        return
      if item.output.copy_event is not None:
        item.output.copy_event.synchronize()
      self._async_ready_steps.put(item)

  def _drain_async_ready_steps(self, *, block: bool = False, max_items: int = 0) -> None:
    """Apply completed async decode steps (single-threaded scheduler updates)."""
    n = 0
    while True:
      if max_items and n >= int(max_items):
        return
      try:
        if block and n == 0:
          item = self._async_ready_steps.get(timeout=1.0)
        else:
          item = self._async_ready_steps.get_nowait()
      except queue.Empty:
        return
      expected = self._async_last_drained_step_id + 1
      if int(item.step_id) != int(expected):
        raise RuntimeError(
          f"Async decode completion out of order: got step_id={int(item.step_id)}, expected={int(expected)}"
        )
      self._process_async_decode_tokens(item)
      self._async_last_drained_step_id = int(item.step_id)
      self._async_inflight = max(0, self._async_inflight - 1)
      self._release_async_deferred_frees()
      n += 1

  def _release_async_deferred_frees(self) -> None:
    """Release finished requests whose GPU lifetime is complete."""
    while self._async_deferred_frees:
      release_after, req = self._async_deferred_frees[0]
      if int(release_after) > int(self._async_last_drained_step_id):
        return
      self._async_deferred_frees.popleft()
      self.scheduler.release_finished_request(req)

  def _process_async_decode_tokens(self, item: _AsyncStep) -> None:
    """Process a completed decode TOKENS step whose D2H copies are ready."""
    batch = item.batch
    output = item.output
    if output.next_tokens_cpu is None:
      raise RuntimeError("Async decode expected next_tokens_cpu.")
    tokens = output.next_tokens_cpu
    if output.moe_overflow_cpu is not None:
      dropped = int(output.moe_overflow_cpu.item())
      if dropped != 0:
        expected_m = int(getattr(self.engine.buffer, "_nmoe_masked_gemm_expected_m", 0))
        print(
          f"[MoE][OVERFLOW][rank{self.rank}] expected_m={expected_m} dropped_pairs={dropped}",
          flush=True,
        )

    send_updates = (
      self.control_plane is not None and self.world_size > 1 and self.rank != 0
    )
    upd_uids: list[int] = []
    upd_tokens: list[int] = []
    upd_flags: list[int] = []

    for i, req in enumerate(batch.reqs):
      if req.is_finished:
        continue
      pos = int(item.out_pos[i])
      tok = int(tokens[i])
      if pos < 0 or pos >= len(req.output_ids):
        continue
      if req.output_ids[pos] != _PENDING_TOKEN_ID:
        # Already filled (can happen if a request was trimmed/finished).
        continue
      req.output_ids[pos] = tok

      finished = self._check_finished(req, tok)
      self._notify_request_done(req)
      if finished:
        # Trim any speculative placeholders/tokens beyond the finishing token.
        del req.output_ids[pos + 1 :]
        # In overlap mode, later in-flight decode steps may still touch this
        # request's KV/table slot. Detach now, then release once all in-flight
        # steps (scheduled before we observed DONE) have drained.
        release_after = max(int(item.step_id), int(self._async_step_id) - 1)
        defer_free = bool(int(release_after) > int(item.step_id))
        self.scheduler.finish_request(
          req,
          success=(req.status != RequestStatus.CANCELLED),
          defer_free=defer_free,
        )
        if defer_free:
          self._async_deferred_frees.append((int(release_after), req))
        self._finished_reqs.append(req)
        if send_updates:
          rid = int(finish_reason_str_to_id(req.finish_reason))
          upd_uids.append(int(req.uid))
          upd_tokens.append(int(tok))
          upd_flags.append(1 | ((rid & 0x7) << 1))
        self._uid_to_req.pop(int(req.uid), None)
      else:
        if send_updates:
          upd_uids.append(int(req.uid))
          upd_tokens.append(int(tok))
          upd_flags.append(0)

    if send_updates and upd_uids:
      self.control_plane.enqueue_token_updates(upd_uids, upd_tokens, upd_flags)

  def get_health_status(self) -> dict:
    """Get detailed health status for health endpoint.

    Returns dict with:
      - status: "healthy", "degraded", or "unhealthy"
      - ready: bool - can accept requests
      - running: bool - event loop is running
      - queue_size: int - current queue depth
      - queue_capacity: int - max queue size
      - details: dict - additional diagnostics
    """
    status = "unhealthy"
    if self._ready and self._running:
      status = "healthy"
    elif self._running and not self._ready:
      status = "degraded"  # Running but not yet ready (warmup)

    return {
      "status": status,
      "ready": self._ready,
      "running": self._running,
      "queue_size": self.queue_size,
      "queue_capacity": self.queue_capacity,
      "details": {
        "world_size": self.world_size,
        "rank": self.rank,
        "device": str(self.device),
      },
    }

  def _recv_requests(self) -> None:
    """Receive pending requests from input queue."""
    while True:
      try:
        req = self._input_queue.get_nowait()
      except queue.Empty:
        break

      if req.cancel_flag or int(req.uid) in self._cancelled_uids:
        req.cancel_flag = True
        req.status = RequestStatus.CANCELLED
        req.finish_reason = "cancelled"
        self._finished_reqs.append(req)
        self._notify_request_done(req)
        self._uid_to_req.pop(int(req.uid), None)
        continue

      self.scheduler.add_request(req)

  def _schedule_batch(self) -> Optional[Batch]:
    """Schedule next batch based on pending work."""
    # Prefill has priority (following vLLM v1 pattern)
    if self.scheduler.has_pending_prefill:
      # Fixed-batching profiles (eval_exact/offline_distill) do not enter the
      # prefill queue; schedule_phase("prefill") is the single correct entrypoint.
      batch = self.scheduler.schedule_phase("prefill")
      if batch is not None:
        return batch

    # Then decode
    if self.scheduler.has_pending_decode:
      return self.scheduler.schedule_phase("decode")

    return None

  def _process_results(self, batch: Batch, output: ForwardOutput) -> None:
    """Process forward output: sample tokens, update request state."""
    # Wait for async copy to complete
    if output.copy_event is not None:
      output.copy_event.synchronize()
    if batch.is_decode and output.moe_overflow_cpu is not None:
      dropped = int(output.moe_overflow_cpu.item())
      if dropped != 0:
        expected_m = int(getattr(self.engine.buffer, "_nmoe_masked_gemm_expected_m", 0))
        print(
          f"[MoE][OVERFLOW][rank{self.rank}] expected_m={expected_m} dropped_pairs={dropped}",
          flush=True,
        )

    output_mode = batch.reqs[0].forward_spec.output_mode
    for r in batch.reqs[1:]:
      if r.forward_spec.output_mode != output_mode:
        raise ValueError("Mixed output_mode in one batch is not supported.")

    any_forced = any(
      (r.forced_output_ids is not None and r.forced_output_pos < len(r.forced_output_ids))
      for r in batch.reqs
    )

    # Get sampled tokens
    if output.next_tokens_cpu is not None:
      tokens = output.next_tokens_cpu
    else:
      # Fallback: greedy from logits (should not be hit in production)
      logits = output.logits
      sample_logits = logits[:, -1, :] if logits.dim() == 3 else logits
      tokens = torch.argmax(sample_logits, dim=-1).cpu()

    logits_cpu = output.logits_cpu

    # LOGITS mode contract: for temperature==0 (greedy), the returned tokens must
    # match argmax(logits). Ensure the CPU-visible token stream is derived from
    # the same logits tensor we store on the Request.
    #
    # NOTE: Teacher-forcing (forced_output_ids) is allowed to violate greedy
    # argmax by design; in that case we keep the forced token.
    if (
      output_mode == OutputMode.LOGITS
      and logits_cpu is not None
      and float(batch.reqs[0].sampling_params.temperature) <= 0.0
      and not any_forced
    ):
      tokens = torch.argmax(logits_cpu, dim=-1).to(torch.int64)

    sampled_tokens_list = [int(t) for t in tokens.tolist()]
    logprobs_list: Optional[list[float]] = None
    topk_ids_list: Optional[list[list[int]]] = None
    topk_logprobs_list: Optional[list[list[float]]] = None

    # Fast path: rely on engine-computed aux outputs unless we need teacher-forcing.
    if output_mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS) and not any_forced:
      if output.next_logprobs_cpu is None:
        raise RuntimeError("Engine did not return next_logprobs for LOGPROBS mode.")
      logprobs_list = [float(x) for x in output.next_logprobs_cpu.tolist()]

    if output_mode == OutputMode.TOPK_LOGPROBS and not any_forced:
      if output.next_topk_ids_cpu is None or output.next_topk_logprobs_cpu is None:
        raise RuntimeError("Engine did not return next_topk for TOPK_LOGPROBS mode.")
      topk_ids_list = [[int(x) for x in row] for row in output.next_topk_ids_cpu.tolist()]
      topk_logprobs_list = [[float(x) for x in row] for row in output.next_topk_logprobs_cpu.tolist()]

    if output_mode == OutputMode.LOGITS and logits_cpu is None:
      raise RuntimeError("Engine did not return logits_cpu for LOGITS mode.")
    S = batch.seqlen_q

    # Teacher-forcing needs logprobs/topk for forced tokens under the same policy.
    sample_args = None
    topk = int(batch.reqs[0].forward_spec.topk) if output_mode == OutputMode.TOPK_LOGPROBS else 0
    if any_forced and output_mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS):
      sample_args = self.engine.sampler.prepare(batch)

    def _slice_args(args, i: int):
      # Avoid importing BatchSamplingArgs at module scope.
      from nmoe.serve.engine import BatchSamplingArgs

      temps = args.temperatures[i : i + 1] if args.temperatures is not None else None
      top_k = [int(args.top_k[i])] if args.top_k is not None else None
      top_p = [float(args.top_p[i])] if args.top_p is not None else None
      seeds = [args.seeds[i]] if args.seeds is not None else None
      return BatchSamplingArgs(temps, top_k, top_p, seeds)

    def _score_forced(i: int, token: int) -> tuple[Optional[float], Optional[list[tuple[int, float]]]]:
      if sample_args is None:
        return None, None
      args_i = _slice_args(sample_args, i)
      logits_i = output.logits[i : i + 1]
      tok_i = torch.tensor([int(token)], device=logits_i.device, dtype=torch.int64)
      # Keep teacher-forcing scoring on the engine's stream to avoid introducing
      # additional stream concurrency that can amplify nondeterminism in later
      # MoE atomics.
      with torch.cuda.stream(self.engine.stream):
        lp, topk_ids, topk_lp = self.engine.sampler.logprobs_for_tokens(
          logits_i, args_i, tok_i, return_topk=topk
        )
        lp_cpu = lp.to("cpu", non_blocking=True)
        topk_ids_cpu = topk_ids.to("cpu", non_blocking=True) if topk_ids is not None else None
        topk_lp_cpu = topk_lp.to("cpu", non_blocking=True) if topk_lp is not None else None
        ev = torch.cuda.Event()
        ev.record()
      ev.synchronize()

      lp_f = float(lp_cpu.item())
      if topk_ids_cpu is None or topk_lp_cpu is None:
        return lp_f, None
      ids = [int(x) for x in topk_ids_cpu[0].tolist()]
      lps = [float(x) for x in topk_lp_cpu[0].tolist()]
      return lp_f, list(zip(ids, lps))

    def _pack_flags(done: bool, finish_reason: Optional[str]) -> int:
      if not done:
        return 0
      rid = int(finish_reason_str_to_id(finish_reason))
      return 1 | ((rid & 0x7) << 1)

    send_updates = (
      self.control_plane is not None and self.world_size > 1 and self.rank != 0
    )
    upd_uids: list[int] = []
    upd_tokens: list[int] = []
    upd_flags: list[int] = []

    # Update each request
    for i, req in enumerate(batch.reqs):
      if batch.is_prefill:
        # Advance cached_len by the number of prompt tokens we just processed.
        req.cached_len += S
        if req.cached_len < len(req.input_ids):
          # Chunked prefill: keep prefilling until the full prompt is cached.
          self.scheduler.requeue_prefill(req)
          continue

        # Prefill completed: sample the first generated token from the last
        # prompt position's logits.
        forced = (
          req.forced_output_ids is not None and req.forced_output_pos < len(req.forced_output_ids)
        )
        token = (
          int(req.forced_output_ids[req.forced_output_pos]) if forced else sampled_tokens_list[i]
        )
        if forced:
          req.forced_output_pos += 1
        req.output_ids.append(int(token))
        # Seed decode with the newly generated token (GPU-driven decode reads this).
        with torch.cuda.stream(self.engine.stream):
          self.scheduler.set_last_token_for_req(req, int(token))
        if output_mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS):
          if forced:
            lp, topk_row = _score_forced(i, token)
            if lp is None:
              raise RuntimeError("teacher-forcing requested but could not score token logprob")
            req.output_logprobs.append(float(lp))
            if output_mode == OutputMode.TOPK_LOGPROBS and topk_row is not None:
              req.output_topk.append(list(topk_row))
          else:
            if logprobs_list is None:
              raise RuntimeError("Engine did not return next_logprobs for LOGPROBS mode.")
            req.output_logprobs.append(float(logprobs_list[i]))
            if output_mode == OutputMode.TOPK_LOGPROBS:
              if topk_ids_list is None or topk_logprobs_list is None:
                raise RuntimeError("Engine did not return next_topk for TOPK_LOGPROBS mode.")
              req.output_topk.append(list(zip(topk_ids_list[i], topk_logprobs_list[i])))
        if output_mode == OutputMode.LOGITS and logits_cpu is not None:
          req._logits_chunks.append(logits_cpu[i].contiguous())
        req.mark_prefill_end()
        req.mark_first_token()

        finished = self._check_finished(req, token)
        self._notify_request_done(req)
        if finished:
          if output_mode == OutputMode.LOGITS:
            req.output_logits = torch.stack(req._logits_chunks, dim=0) if req._logits_chunks else None
            req._logits_chunks.clear()
          self.scheduler.finish_request(req, success=(req.status != RequestStatus.CANCELLED))
          self._finished_reqs.append(req)
          if send_updates:
            upd_uids.append(int(req.uid))
            upd_tokens.append(int(token))
            upd_flags.append(_pack_flags(True, req.finish_reason))
          self._uid_to_req.pop(int(req.uid), None)
          continue

        # Move to decode phase. The newly appended token is not yet cached and
        # will be processed on the next decode step.
        self.scheduler.promote_to_decode(req)
        if send_updates:
          upd_uids.append(int(req.uid))
          upd_tokens.append(int(token))
          upd_flags.append(0)
        continue

      # Decode: the batch token is the most recent generated token; KV is now
      # written for it.
      req.cached_len += 1
      forced = (
        req.forced_output_ids is not None and req.forced_output_pos < len(req.forced_output_ids)
      )
      token = (
        int(req.forced_output_ids[req.forced_output_pos]) if forced else sampled_tokens_list[i]
      )
      if forced:
        req.forced_output_pos += 1
      req.output_ids.append(int(token))
      # Advance decode token stream (GPU-driven decode reads this).
      with torch.cuda.stream(self.engine.stream):
        self.scheduler.set_last_token_for_req(req, int(token))
      if output_mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS):
        if forced:
          lp, topk_row = _score_forced(i, token)
          if lp is None:
            raise RuntimeError("teacher-forcing requested but could not score token logprob")
          req.output_logprobs.append(float(lp))
          if output_mode == OutputMode.TOPK_LOGPROBS and topk_row is not None:
            req.output_topk.append(list(topk_row))
        else:
          if logprobs_list is None:
            raise RuntimeError("Engine did not return next_logprobs for LOGPROBS mode.")
          req.output_logprobs.append(float(logprobs_list[i]))
          if output_mode == OutputMode.TOPK_LOGPROBS:
            if topk_ids_list is None or topk_logprobs_list is None:
              raise RuntimeError("Engine did not return next_topk for TOPK_LOGPROBS mode.")
            req.output_topk.append(list(zip(topk_ids_list[i], topk_logprobs_list[i])))
      if output_mode == OutputMode.LOGITS and logits_cpu is not None:
        req._logits_chunks.append(logits_cpu[i].contiguous())

      finished = self._check_finished(req, token)
      self._notify_request_done(req)
      if finished:
        if output_mode == OutputMode.LOGITS:
          req.output_logits = torch.stack(req._logits_chunks, dim=0) if req._logits_chunks else None
          req._logits_chunks.clear()
        self.scheduler.finish_request(req, success=(req.status != RequestStatus.CANCELLED))
        self._finished_reqs.append(req)
        if send_updates:
          upd_uids.append(int(req.uid))
          upd_tokens.append(int(token))
          upd_flags.append(_pack_flags(True, req.finish_reason))
        self._uid_to_req.pop(int(req.uid), None)
      else:
        if send_updates:
          upd_uids.append(int(req.uid))
          upd_tokens.append(int(token))
          upd_flags.append(0)

    if send_updates and upd_uids:
      self.control_plane.enqueue_token_updates(upd_uids, upd_tokens, upd_flags)

  def _notify_request_done(self, req: Request) -> None:
    """Thread-safe notification that request made progress (token or done)."""
    if req.return_queue is None:
      return
    # Use call_soon_threadsafe for asyncio.Queue from background thread
    if self._async_loop is not None:
      try:
        self._async_loop.call_soon_threadsafe(req.return_queue.put_nowait, None)
      except RuntimeError:
        # Event loop closed
        pass
    else:
      # Fallback for non-async usage
      try:
        req.return_queue.put_nowait(None)
      except (asyncio.QueueFull, AttributeError):
        pass

  def _check_finished(self, req: Request, token: int) -> bool:
    """Check if request should finish. Sets req.finish_reason."""
    # Max tokens
    if _materialized_output_len(req) >= req.sampling_params.max_tokens:
      req.finish_reason = "length"
      return True

    # EOS token (vocab-specific - using 151643 for DeepSeek)
    if token in (151643, 1, 2):  # DeepSeek V3 EOS=1, common EOS=2, legacy=151643
      req.finish_reason = "stop"
      return True

    # Cancelled
    if req.cancel_flag:
      req.status = RequestStatus.CANCELLED
      req.finish_reason = "cancelled"
      return True

    return False

  @property
  def pending_requests(self) -> int:
    """Number of pending requests (prefill + decode)."""
    return (
      self._input_queue.qsize()
      + (1 if self.scheduler.has_pending_prefill else 0)
      + (1 if self.scheduler.has_pending_decode else 0)
    )

  @property
  def finished_requests(self) -> list[Request]:
    """Get and clear finished requests."""
    reqs = self._finished_reqs
    self._finished_reqs = []
    return reqs


class AsyncOrchestrator:
  """
  Async wrapper for orchestrator.

  Runs orchestrator in background thread, exposes async interface.
  """

  def __init__(self, orchestrator: Orchestrator) -> None:
    self.orchestrator = orchestrator
    self._thread: Optional[threading.Thread] = None
    self._loop: Optional[asyncio.AbstractEventLoop] = None

  async def start(self) -> None:
    """Start orchestrator in background thread."""
    self._loop = asyncio.get_event_loop()
    # Pass event loop to orchestrator for thread-safe queue operations
    self.orchestrator._async_loop = self._loop
    self._thread = threading.Thread(
      target=self.orchestrator.run,
      daemon=True,
    )
    self._thread.start()

  async def stop(self) -> None:
    """Stop orchestrator."""
    self.orchestrator.stop()
    if self._thread is not None:
      self._thread.join(timeout=5.0)

  async def add_request(self, req: Request) -> None:
    """Add request (async interface). Blocks if queue is full."""
    self.orchestrator.add_request(req)

  def try_add_request(self, req: Request) -> bool:
    """Try to add request. Returns False if queue full (503)."""
    return self.orchestrator.try_add_request(req, timeout=0.0)

  def validate_request_bounds(self, prompt_tokens: int, max_tokens: int) -> tuple[bool, str]:
    """Validate request bounds. Returns (is_valid, error_message)."""
    return self.orchestrator.validate_request_bounds(prompt_tokens, max_tokens)

  def get_health_status(self) -> dict:
    """Get health status from orchestrator."""
    return self.orchestrator.get_health_status()

  @property
  def is_ready(self) -> bool:
    """True if orchestrator is ready to accept requests."""
    return self.orchestrator.is_ready

  @property
  def limits(self) -> dict:
    """Get server limits for error messages."""
    cfg = self.orchestrator.orch_config
    return {
      "max_prompt_tokens": cfg.max_prompt_tokens,
      "max_output_tokens": cfg.max_output_tokens,
      "max_seq_len": cfg.max_seq_len,
      "max_pending_requests": cfg.max_pending_requests,
    }

  async def wait_for_request(self, req: Request, timeout: float = 60.0) -> Request:
    """Wait for request to complete (return_queue is a wakeup channel)."""
    if req.return_queue is None:
      req.return_queue = asyncio.Queue()

    deadline = time.time() + float(timeout)
    while not req.is_finished:
      remaining = deadline - time.time()
      if remaining <= 0:
        req.cancel_flag = True
        raise asyncio.TimeoutError()
      try:
        await asyncio.wait_for(req.return_queue.get(), timeout=remaining)
      except asyncio.TimeoutError:
        req.cancel_flag = True
        raise
    return req

  async def generate(
    self,
    input_ids: torch.Tensor,
    profile_name: str = "production_generate",
    timeout: float = 60.0,
    **sampling_kwargs,
  ) -> Request:
    """Create request, submit, and wait for completion."""
    req = self.orchestrator.create_request(
      input_ids=input_ids,
      profile_name=profile_name,
      **sampling_kwargs,
    )
    req.return_queue = asyncio.Queue()

    await self.add_request(req)
    return await self.wait_for_request(req, timeout=timeout)
