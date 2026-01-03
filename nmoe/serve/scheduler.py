# SPDX-License-Identifier: Apache-2.0
"""Token-budget scheduler with profile-aware batching.

Optimized for TP=1/DP=8/EP=8:
- Pre-allocated buffers eliminate per-step tensor allocations
- Each GPU schedules its own batch independently (no cross-GPU sync)
- Expert dispatch/combine handled by DeepEP
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from nmoe.serve.cache import CacheHandle, KvCache
from nmoe.serve.config import BatchingMode, Profile, PROFILES
from nmoe.serve.types import Batch, Request, RequestStatus

if TYPE_CHECKING:
  from nmoe.serve.buffers import InputBuffers


@dataclass
class SchedulerConfig:
  """Configuration for the scheduler."""
  max_batch_size: int = 256
  max_prefill_tokens: int = 8192
  max_decode_tokens: int = 4096
  max_seq_len: int = 32768
  page_size: int = 64
  enable_chunked_prefill: bool = True
  chunk_size: int = 2048  # Max tokens per prefill chunk


class PrefillQueue:
  """Queue for requests waiting for prefill."""

  def __init__(self) -> None:
    self._queue: deque[Request] = deque()
    self._total_tokens: int = 0

  def add(self, req: Request) -> None:
    """Add request to prefill queue."""
    self._queue.append(req)
    self._total_tokens += req.extend_len

  def pop(self) -> Optional[Request]:
    """Pop next request from queue."""
    if not self._queue:
      return None
    req = self._queue.popleft()
    self._total_tokens -= req.extend_len
    return req

  def peek(self) -> Optional[Request]:
    """Peek at next request without removing."""
    return self._queue[0] if self._queue else None

  @property
  def size(self) -> int:
    return len(self._queue)

  @property
  def total_tokens(self) -> int:
    return self._total_tokens

  @property
  def is_empty(self) -> bool:
    return len(self._queue) == 0


class DecodeQueue:
  """Queue for requests in decode phase."""

  def __init__(self) -> None:
    self._reqs: dict[int, Request] = {}

  def add(self, req: Request) -> None:
    """Add request to decode queue."""
    self._reqs[req.uid] = req

  def remove(self, req: Request) -> None:
    """Remove request from decode queue."""
    self._reqs.pop(req.uid, None)

  def get_all(self) -> list[Request]:
    """Get all decoding requests."""
    return list(self._reqs.values())

  @property
  def size(self) -> int:
    return len(self._reqs)

  @property
  def is_empty(self) -> bool:
    return len(self._reqs) == 0


@dataclass
class ChunkedPrefill:
  """Tracks a chunked prefill request."""
  req: Request
  current_offset: int = 0

  @property
  def remaining(self) -> int:
    return len(self.req.input_ids) - self.current_offset

  @property
  def is_complete(self) -> bool:
    return self.remaining <= 0


class Scheduler:
  """
  Token-budget scheduler with profile-aware batching.

  Supports:
  - Continuous batching (production_generate, online_distill, rl_sample)
  - Fixed batching (eval_exact, offline_distill)
  - Chunked prefill to prevent decode starvation
  - Profile-driven scheduling policies

  Optimized for TP=1/DP=8/EP=8:
  - Pre-allocated buffers eliminate per-step tensor allocations
  - Pinned CPU memory for fast H2D transfers
  """

  def __init__(
    self,
    config: SchedulerConfig,
    kv_cache: KvCache,
    device: torch.device,
    input_buffers: Optional["InputBuffers"] = None,
  ) -> None:
    self.config = config
    self.kv_cache = kv_cache
    self.device = device
    self._buffers = input_buffers

    # Queues
    self._prefill_queue = PrefillQueue()
    self._decode_queue = DecodeQueue()
    self._chunked_prefills: dict[int, ChunkedPrefill] = {}

    # Page table: maps request table_idx to page indices
    self._page_table = torch.zeros(
      config.max_batch_size,
      (config.max_seq_len + config.page_size - 1) // config.page_size,
      dtype=torch.int32,
      device=device,
    )
    self._table_slots: list[int] = list(range(config.max_batch_size))

    # Fixed batching state (for eval_exact, offline_distill)
    self._fixed_batch_reqs: list[Request] = []

    # Pre-allocated reusable tensors for batch construction
    # These avoid allocations in the hot path
    self._pos_range = torch.arange(
      config.chunk_size if config.enable_chunked_prefill else config.max_seq_len,
      device=device,
      dtype=torch.int64,
    )
    self._table_idx_buffer = torch.zeros(
      config.max_batch_size, dtype=torch.int64, device=device
    )
    self._starts_buffer = torch.zeros(
      config.max_batch_size, dtype=torch.int64, device=device
    )

  def add_request(self, req: Request) -> None:
    """Add a new request to the scheduler."""
    profile = PROFILES.get(req.profile_name)
    if profile is None:
      raise ValueError(f"Unknown profile: {req.profile_name}")
    if req.forward_spec.output_mode != profile.output_mode:
      raise ValueError(
        f"Request forward_spec.output_mode={req.forward_spec.output_mode} "
        f"does not match profile[{req.profile_name}].output_mode={profile.output_mode}"
      )

    # Assign table slot
    if not self._table_slots:
      raise RuntimeError("No table slots available")
    req.table_idx = self._table_slots.pop()

    # Check prefix cache
    handle, cached_pages = self.kv_cache.match_prefix(req.input_ids)
    req.cached_len = handle.cached_len
    req.cache_handle = handle
    self.kv_cache.lock(handle)

    # Copy cached pages to page table
    if len(cached_pages) > 0:
      num_cached_pages = len(cached_pages)
      self._page_table[req.table_idx, :num_cached_pages] = cached_pages
      req.page_ids.extend([int(x) for x in cached_pages.tolist()])

    req.metrics.arrival_time = self._get_time()

    if profile.batching == BatchingMode.FIXED:
      self._fixed_batch_reqs.append(req)
    else:
      self._prefill_queue.add(req)

  def requeue_prefill(self, req: Request) -> None:
    """Re-enqueue a partially-prefilled request."""
    if req.extend_len <= 0:
      return
    self._prefill_queue.add(req)

  def schedule_prefill(self, token_budget: Optional[int] = None) -> Optional[Batch]:
    """Schedule next prefill batch within token budget."""
    if self._prefill_queue.is_empty:
      return None

    budget = token_budget or self.config.max_prefill_tokens
    req0 = self._prefill_queue.peek()
    if req0 is None:
      return None

    # MLA requires homogeneous prefill mode per batch:
    # - cache miss / first chunk: cached_len == 0  -> dense prefill
    # - cache hit or later chunks: cached_len > 0  -> paged prefill
    want_cached = req0.cached_len > 0

    # FlashMLA model contract is [B,S] without out_loc padding, so we only batch
    # requests that can contribute the same S tokens this step.
    step_len = min(req0.extend_len, self.config.chunk_size) if self.config.enable_chunked_prefill else req0.extend_len
    if step_len <= 0:
      self._prefill_queue.pop()
      return None

    reqs: list[Request] = []
    total_tokens = 0
    kept: deque[Request] = deque()
    while not self._prefill_queue.is_empty and len(reqs) < self.config.max_batch_size:
      req = self._prefill_queue.pop()
      if req is None:
        break

      if (req.cached_len > 0) != want_cached:
        kept.append(req)
        continue

      this_len = min(req.extend_len, self.config.chunk_size) if self.config.enable_chunked_prefill else req.extend_len
      if this_len != step_len:
        kept.append(req)
        continue
      if total_tokens + step_len > budget and reqs:
        kept.append(req)
        break

      # Allocate pages for tokens up to req.cached_len + step_len.
      need_total = req.cached_len + step_len
      need_pages = (need_total + self.config.page_size - 1) // self.config.page_size
      have_pages = len(req.page_ids)
      if need_pages > have_pages:
        pages_needed = need_pages - have_pages
        allocated = self.kv_cache.allocate(pages_needed)
        self._page_table[req.table_idx, have_pages:need_pages] = allocated
        req.page_ids.extend(allocated.tolist())

      reqs.append(req)
      total_tokens += step_len
      req.mark_prefill_start()

    # Put back skipped requests preserving order.
    while kept:
      self._prefill_queue._queue.appendleft(kept.pop())

    if not reqs:
      return None

    return self._prepare_batch(reqs, phase="prefill")

  def schedule_decode(self) -> Optional[Batch]:
    """Schedule decode batch (all active decode requests)."""
    if self._decode_queue.is_empty:
      return None

    reqs = self._decode_queue.get_all()
    if not reqs:
      return None

    # Each decode step processes 1 token per request
    # Allocate new pages if needed
    for req in reqs:
      current_len = req.seq_len
      current_pages = (current_len + self.config.page_size - 1) // self.config.page_size
      pages_allocated = len(req.page_ids)

      if current_pages > pages_allocated:
        pages_needed = current_pages - pages_allocated
        allocated = self.kv_cache.allocate(pages_needed)
        self._page_table[req.table_idx, pages_allocated:current_pages] = allocated
        req.page_ids.extend(allocated.tolist())

      req.mark_decode_start()

    return self._prepare_batch(reqs, phase="decode")

  def schedule_fixed_batch(self) -> Optional[Batch]:
    """Schedule a fixed batch (for eval_exact, offline_distill)."""
    if not self._fixed_batch_reqs:
      return None

    # For fixed batching, schedule all pending requests at once
    # respecting max_batch_size
    batch_size = min(len(self._fixed_batch_reqs), self.config.max_batch_size)
    reqs = self._fixed_batch_reqs[:batch_size]
    self._fixed_batch_reqs = self._fixed_batch_reqs[batch_size:]

    # Allocate pages
    for req in reqs:
      tokens_needed = req.extend_len
      pages_needed = (tokens_needed + self.config.page_size - 1) // self.config.page_size
      if pages_needed > 0:
        allocated = self.kv_cache.allocate(pages_needed)
        start_page = (req.cached_len + self.config.page_size - 1) // self.config.page_size
        self._page_table[req.table_idx, start_page:start_page + pages_needed] = allocated
        req.page_ids.extend(allocated.tolist())
      req.mark_prefill_start()

    return self._prepare_batch(reqs, phase="prefill")

  def schedule_next(self, profile: Profile) -> Optional[Batch]:
    """
    Schedule next batch based on profile.

    Continuous batching: Prefill has priority, but decode is always serviced.
    Fixed batching: Process full batches in order.
    """
    if profile.batching == BatchingMode.FIXED:
      return self.schedule_fixed_batch()

    # Continuous batching: prefer prefill, then decode
    # This matches vLLM v1's token-budget scheduler pattern
    batch = self.schedule_prefill()
    if batch is not None:
      return batch

    return self.schedule_decode()

  def finish_request(self, req: Request, success: bool = True) -> None:
    """Mark request as finished and free resources."""
    req.mark_finished()
    if not success:
      req.status = RequestStatus.CANCELLED

    # Remove from decode queue if present
    self._decode_queue.remove(req)

    # Free table slot
    self._table_slots.append(req.table_idx)

    # Return pages to allocator / prefix cache.
    if req.page_ids:
      pages = torch.tensor(req.page_ids, dtype=torch.int32, device="cpu")
      # Cache prompt-only (full pages) for future prefix hits. Never cache partial pages.
      handle = req.cache_handle or CacheHandle(0, None)
      prompt_cached = min(int(req.cached_len), int(req.input_ids.numel()))
      self.kv_cache.insert_and_free(handle, req.input_ids[:prompt_cached], pages)
    else:
      # No pages to return, but still release any prefix-cache lock.
      if req.cache_handle is not None:
        self.kv_cache.unlock(req.cache_handle)

    req.cache_handle = None

  def promote_to_decode(self, req: Request) -> None:
    """Move request from prefill to decode phase."""
    req.status = RequestStatus.DECODING
    self._decode_queue.add(req)

  def _prepare_batch(self, reqs: list[Request], phase: str) -> Batch:
    """Prepare a batch for forward pass.

    Uses pre-allocated buffers when available to eliminate per-step allocations.
    For TP=1/DP=8: Each GPU prepares its batch independently.
    """
    batch = Batch(reqs=reqs, phase=phase)
    B = len(reqs)

    # Use pre-allocated buffers if available
    if self._buffers is not None:
      return self._prepare_batch_fast(reqs, phase)

    # Fallback: original implementation (allocates tensors each call)
    if phase == "prefill":
      # For prefill, process a uniform chunk size S for all requests.
      # Tokens are the uncached prompt suffix: [cached_len, cached_len+S).
      S = min(r.extend_len for r in reqs)
      if self.config.enable_chunked_prefill:
        S = min(S, self.config.chunk_size)
      if S <= 0:
        raise RuntimeError("Prefill batch has no tokens to process.")

      # Use pre-allocated buffers for intermediate tensors
      for i, r in enumerate(reqs):
        self._starts_buffer[i] = r.cached_len
        self._table_idx_buffer[i] = r.table_idx
      starts = self._starts_buffer[:B]
      table_idx = self._table_idx_buffer[:B]

      # Positions: starts[:, None] + range[:S]
      token_pos = starts[:, None] + self._pos_range[:S]

      # CPU token ids to CUDA [B,S] - still allocates stack but reuses transfer
      ids_cpu = torch.stack([r.input_ids[r.cached_len : r.cached_len + S] for r in reqs], dim=0)
      batch.input_ids = ids_cpu.to(self.device, non_blocking=True).to(torch.int64)
      batch.positions = token_pos

      batch.block_table = self._page_table[table_idx]
      page_idx = torch.div(token_pos, self.config.page_size, rounding_mode="floor")
      off = token_pos % self.config.page_size
      page_ids = self._page_table[table_idx[:, None], page_idx]
      batch.out_loc = (page_ids * self.config.page_size + off).to(torch.int32)

      # Cache lengths after this prefill step
      cache_seqlens_cpu = [int(r.cached_len + S) for r in reqs]
      batch.cache_seqlens_cpu = cache_seqlens_cpu
      batch.cache_seqlens = torch.tensor(cache_seqlens_cpu, dtype=torch.int32, device=self.device)

    else:  # decode
      # For decode, one token per request: the most recently appended token.
      tokens = []
      positions = []
      for i, r in enumerate(reqs):
        if r.output_ids:
          tokens.append(int(r.output_ids[-1]))
          positions.append(int(r.seq_len - 1))
          self._table_idx_buffer[i] = r.table_idx
        else:
          raise RuntimeError("Decode request has no generated token.")

      token_t = torch.tensor(tokens, dtype=torch.int64, device=self.device)[:, None]
      pos_t = torch.tensor(positions, dtype=torch.int64, device=self.device)[:, None]
      batch.input_ids = token_t
      batch.positions = pos_t

      table_idx = self._table_idx_buffer[:B]
      batch.block_table = self._page_table[table_idx]
      page_idx = torch.div(pos_t, self.config.page_size, rounding_mode="floor")
      off = pos_t % self.config.page_size
      page_ids = self._page_table[table_idx[:, None], page_idx]
      batch.out_loc = (page_ids * self.config.page_size + off).to(torch.int32)

      cache_seqlens_cpu = [int(r.cached_len + 1) for r in reqs]
      batch.cache_seqlens_cpu = cache_seqlens_cpu
      batch.cache_seqlens = torch.tensor(cache_seqlens_cpu, dtype=torch.int32, device=self.device)

    return batch

  def _prepare_batch_fast(self, reqs: list[Request], phase: str) -> Batch:
    """Fast batch preparation using pre-allocated buffers.

    Zero-allocation hot path for TP=1/DP=8/EP=8.
    All tensors are slices into pre-allocated buffers.
    """
    batch = Batch(reqs=reqs, phase=phase)
    B = len(reqs)
    buf = self._buffers

    if phase == "prefill":
      S = min(r.extend_len for r in reqs)
      if self.config.enable_chunked_prefill:
        S = min(S, self.config.chunk_size)
      if S <= 0:
        raise RuntimeError("Prefill batch has no tokens to process.")

      # Fill CPU buffer then copy (pinned memory -> fast)
      for i, r in enumerate(reqs):
        # Input IDs
        chunk = r.input_ids[r.cached_len : r.cached_len + S]
        buf.input_ids_cpu[i, :S] = chunk
        # Metadata
        self._starts_buffer[i] = r.cached_len
        self._table_idx_buffer[i] = r.table_idx
        buf.cache_seqlens_cpu[i] = r.cached_len + S

      # Non-blocking copy to GPU
      buf.input_ids[:B, :S].copy_(buf.input_ids_cpu[:B, :S], non_blocking=True)

      # Positions using pre-allocated range
      starts = self._starts_buffer[:B]
      buf.positions[:B, :S] = starts[:, None] + self._pos_range[:S]

      # Block table and out_loc
      table_idx = self._table_idx_buffer[:B]
      buf.block_table[:B].copy_(self._page_table[table_idx])

      positions = buf.positions[:B, :S]
      page_idx = torch.div(positions, self.config.page_size, rounding_mode="floor")
      off = positions % self.config.page_size
      page_ids = self._page_table[table_idx[:, None], page_idx]
      buf.out_loc[:B, :S] = (page_ids * self.config.page_size + off).to(torch.int32)

      # Cache seqlens - direct copy from pinned CPU (no .to() allocation)
      buf.cache_seqlens[:B].copy_(buf.cache_seqlens_cpu[:B], non_blocking=True)

      # Set batch tensors - use contiguous() for slices that will be .view()'d
      # This is still faster than allocating fresh tensors each step
      batch.input_ids = buf.input_ids[:B, :S].contiguous()
      batch.positions = buf.positions[:B, :S].contiguous()
      batch.out_loc = buf.out_loc[:B, :S].contiguous()
      batch.block_table = buf.block_table[:B].contiguous()
      batch.cache_seqlens = buf.cache_seqlens[:B].contiguous()
      # Use slice of pinned CPU tensor directly (avoid .tolist() in hot path)
      batch.cache_seqlens_cpu = [int(buf.cache_seqlens_cpu[i]) for i in range(B)]

    else:  # decode (S=1)
      for i, r in enumerate(reqs):
        if not r.output_ids:
          raise RuntimeError("Decode request has no generated token.")
        buf.input_ids_cpu[i, 0] = r.output_ids[-1]
        pos = r.seq_len - 1
        buf.cache_seqlens_cpu[i] = r.cached_len + 1
        self._starts_buffer[i] = pos  # reuse as position buffer
        self._table_idx_buffer[i] = r.table_idx

      # Non-blocking copies from pinned memory
      buf.input_ids[:B, :1].copy_(buf.input_ids_cpu[:B, :1], non_blocking=True)

      # Positions
      buf.positions[:B, 0] = self._starts_buffer[:B]

      # Block table and out_loc
      table_idx = self._table_idx_buffer[:B]
      buf.block_table[:B].copy_(self._page_table[table_idx])

      positions = buf.positions[:B, :1]
      page_idx = torch.div(positions, self.config.page_size, rounding_mode="floor")
      off = positions % self.config.page_size
      page_ids = self._page_table[table_idx[:, None], page_idx]
      buf.out_loc[:B, :1] = (page_ids * self.config.page_size + off).to(torch.int32)

      # Cache seqlens - direct copy from pinned CPU
      buf.cache_seqlens[:B].copy_(buf.cache_seqlens_cpu[:B], non_blocking=True)

      # Set batch tensors - use contiguous() for slices that will be .view()'d
      batch.input_ids = buf.input_ids[:B, :1].contiguous()
      batch.positions = buf.positions[:B, :1].contiguous()
      batch.out_loc = buf.out_loc[:B, :1].contiguous()
      batch.block_table = buf.block_table[:B].contiguous()
      batch.cache_seqlens = buf.cache_seqlens[:B].contiguous()
      batch.cache_seqlens_cpu = [int(buf.cache_seqlens_cpu[i]) for i in range(B)]

    return batch

  def get_page_table(self, req: Request) -> torch.Tensor:
    """Get page table row for a request."""
    seq_len = req.seq_len
    num_pages = (seq_len + self.config.page_size - 1) // self.config.page_size
    return self._page_table[req.table_idx, :num_pages]

  def _get_time(self) -> float:
    """Get current time for metrics."""
    import time
    return time.perf_counter()

  @property
  def has_pending_prefill(self) -> bool:
    return not self._prefill_queue.is_empty or bool(self._fixed_batch_reqs)

  @property
  def has_pending_decode(self) -> bool:
    return not self._decode_queue.is_empty

  @property
  def is_idle(self) -> bool:
    return (
      self._prefill_queue.is_empty
      and self._decode_queue.is_empty
      and not self._fixed_batch_reqs
    )
