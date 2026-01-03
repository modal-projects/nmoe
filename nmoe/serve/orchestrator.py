# SPDX-License-Identifier: Apache-2.0
"""Inference orchestrator - main event loop coordinating scheduler and engine.

Optimized for TP=1/DP=8/EP=8:
- Pre-allocated input buffers eliminate per-step tensor allocations
- Each GPU runs its own orchestrator independently (no cross-GPU sync for attention)
- Expert dispatch/combine handled by DeepEP
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from nmoe.serve.buffers import InputBuffers
from nmoe.serve.cache import KvCache, MlaKvLayout
from nmoe.serve.config import PROFILES, BatchingMode, Profile, ServeConfig
from nmoe.serve.engine import Engine, EngineConfig
from nmoe.serve.model import ModelConfig
from nmoe.serve.scheduler import Scheduler, SchedulerConfig
from nmoe.serve.types import Batch, ForwardOutput, Request, RequestStatus


@dataclass
class OrchestratorConfig:
  """Configuration for the orchestrator."""
  max_batch_size: int = 256
  max_prefill_tokens: int = 8192
  max_decode_tokens: int = 4096
  max_seq_len: int = 32768
  num_pages: int = 4096
  page_size: int = 64
  enable_overlap: bool = True
  enable_chunked_prefill: bool = True
  chunk_size: int = 2048
  enable_prefix_cache: bool = True
  enable_cuda_graph: bool = False  # TODO: implement
  enable_fast_path: bool = True  # Use pre-allocated buffers for zero-alloc hot path


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
  ) -> None:
    self.model_config = model_config
    self.engine_config = engine_config
    self.orch_config = orch_config
    self.rank = rank
    self.world_size = world_size

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
      max_seq = orch_config.chunk_size if orch_config.enable_chunked_prefill else orch_config.max_seq_len
      self.input_buffers = InputBuffers.create(
        device=self.device,
        max_batch_size=orch_config.max_batch_size,
        max_seq_len=max_seq,
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

    # Request queues
    self._input_queue: queue.Queue[Request] = queue.Queue()
    self._finished_reqs: list[Request] = []

    # State
    self._running = False
    self._uid_counter = 0

    # Overlap scheduling state
    self._prev_batch: Optional[Batch] = None
    self._prev_output: Optional[ForwardOutput] = None

  def add_request(self, req: Request) -> None:
    """Add request to input queue (thread-safe)."""
    self._input_queue.put(req)

  def create_request(
    self,
    input_ids: torch.Tensor,
    profile_name: str = "production_generate",
    **sampling_kwargs,
  ) -> Request:
    """Create a new request with auto-assigned UID."""
    from nmoe.serve.types import ForwardSpec, OutputMode, SamplingParams

    profile = PROFILES.get(profile_name)
    if profile is None:
      raise ValueError(f"Unknown profile: {profile_name}")

    uid = self._uid_counter
    self._uid_counter += 1

    sampling_params = SamplingParams(**sampling_kwargs)
    forward_spec = profile.to_forward_spec()

    return Request(
      uid=uid,
      input_ids=input_ids,
      sampling_params=sampling_params,
      profile_name=profile_name,
      forward_spec=forward_spec,
    )

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

    if self.orch_config.enable_overlap:
      self._run_overlap_loop()
    else:
      self._run_simple_loop()

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

  def stop(self) -> None:
    """Stop the event loop."""
    self._running = False

  def shutdown(self) -> None:
    """Clean shutdown."""
    self.stop()
    self.engine.shutdown()

  def _recv_requests(self) -> None:
    """Receive pending requests from input queue."""
    while True:
      try:
        req = self._input_queue.get_nowait()
        self.scheduler.add_request(req)
      except queue.Empty:
        break

  def _schedule_batch(self) -> Optional[Batch]:
    """Schedule next batch based on pending work."""
    # Prefill has priority (following vLLM v1 pattern)
    if self.scheduler.has_pending_prefill:
      batch = self.scheduler.schedule_prefill()
      if batch is not None:
        return batch

    # Then decode
    if self.scheduler.has_pending_decode:
      return self.scheduler.schedule_decode()

    return None

  def _process_results(self, batch: Batch, output: ForwardOutput) -> None:
    """Process forward output: sample tokens, update request state."""
    # Wait for async copy to complete
    if output.copy_event is not None:
      output.copy_event.synchronize()

    # Get sampled tokens
    if output.next_tokens_cpu is not None:
      tokens = output.next_tokens_cpu
    else:
      tokens = torch.argmax(output.logits[:, -1, :], dim=-1).cpu()

    tokens_list = tokens.tolist()
    S = batch.seqlen_q

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
        token = int(tokens_list[i])
        req.output_ids.append(token)
        req.mark_prefill_end()
        req.mark_first_token()

        finished = self._check_finished(req, token)
        if finished:
          self.scheduler.finish_request(req)
          self._finished_reqs.append(req)
          if req.return_queue is not None:
            try:
              req.return_queue.put_nowait(req)
            except asyncio.QueueFull:
              pass
          continue

        # Move to decode phase. The newly appended token is not yet cached and
        # will be processed on the next decode step.
        self.scheduler.promote_to_decode(req)
        continue

      # Decode: the batch token is the most recent generated token; KV is now
      # written for it.
      req.cached_len += 1
      token = int(tokens_list[i])
      req.output_ids.append(token)

      finished = self._check_finished(req, token)
      if finished:
        self.scheduler.finish_request(req)
        self._finished_reqs.append(req)
        if req.return_queue is not None:
          try:
            req.return_queue.put_nowait(req)
          except asyncio.QueueFull:
            pass

  def _check_finished(self, req: Request, token: int) -> bool:
    """Check if request should finish."""
    # Max tokens
    if len(req.output_ids) >= req.sampling_params.max_tokens:
      return True

    # EOS token (vocab-specific - using 151643 for DeepSeek)
    if token in (151643, 2):  # DeepSeek EOS, common EOS
      return True

    # Cancelled
    if req.cancel_flag:
      req.status = RequestStatus.CANCELLED
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
    """Add request (async interface)."""
    self.orchestrator.add_request(req)

  async def wait_for_request(self, req: Request, timeout: float = 60.0) -> Request:
    """Wait for request to complete."""
    if req.return_queue is None:
      req.return_queue = asyncio.Queue()

    try:
      return await asyncio.wait_for(
        req.return_queue.get(),
        timeout=timeout,
      )
    except asyncio.TimeoutError:
      req.cancel_flag = True
      raise

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
