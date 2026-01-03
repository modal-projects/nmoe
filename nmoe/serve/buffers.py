# SPDX-License-Identifier: Apache-2.0
"""Pre-allocated input buffers for zero-copy batch construction.

Pattern from SGLang's GraphInputBuffers and vLLM's CpuGpuBuffer:
- Pre-allocate all tensors at init to max capacity
- Copy INTO buffers each step (no allocation in hot path)
- Pinned CPU memory for fast H2D transfer

Optimized for TP=1/DP=8/EP=8:
- No cross-GPU tensor sharing for attention
- Each GPU manages its own batch independently
- Expert dispatch/combine handled by DeepEP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class InputBuffers:
  """Pre-allocated GPU tensors for batch construction.

  All tensors sized to max capacity. Use slices [:B, :S] for actual batch.
  This eliminates per-step allocations in the scheduling hot path.
  """

  # Core input tensors [max_batch_size, max_seq_len]
  input_ids: torch.Tensor      # int64
  positions: torch.Tensor      # int64
  out_loc: torch.Tensor        # int32 (physical KV slot indices)

  # Page table [max_batch_size, max_pages_per_seq]
  block_table: torch.Tensor    # int32

  # Sequence metadata [max_batch_size]
  cache_seqlens: torch.Tensor  # int32

  # CPU-side pinned buffers for async copy
  input_ids_cpu: torch.Tensor   # Pinned CPU for H2D
  cache_seqlens_cpu: torch.Tensor  # True CPU tensor (never goes to GPU)

  # Output buffer for sampled tokens
  next_tokens: torch.Tensor    # int64 [max_batch_size]
  next_tokens_cpu: torch.Tensor  # Pinned CPU for D2H

  # Sizes for bounds checking
  max_batch_size: int
  max_seq_len: int
  max_pages_per_seq: int

  @classmethod
  def create(
    cls,
    *,
    device: torch.device,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
  ) -> "InputBuffers":
    """Create pre-allocated buffers.

    Args:
      device: GPU device for tensor allocation
      max_batch_size: Maximum requests per batch
      max_seq_len: Maximum sequence length (for prefill chunks)
      page_size: KV cache page size (typically 64)
    """
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size

    with torch.device(device):
      # GPU tensors - pre-allocated to max size
      input_ids = torch.zeros((max_batch_size, max_seq_len), dtype=torch.int64)
      positions = torch.zeros((max_batch_size, max_seq_len), dtype=torch.int64)
      out_loc = torch.zeros((max_batch_size, max_seq_len), dtype=torch.int32)
      block_table = torch.zeros((max_batch_size, max_pages_per_seq), dtype=torch.int32)
      cache_seqlens = torch.zeros((max_batch_size,), dtype=torch.int32)
      next_tokens = torch.zeros((max_batch_size,), dtype=torch.int64)

    # CPU tensors - pinned for fast transfer
    input_ids_cpu = torch.zeros(
      (max_batch_size, max_seq_len), dtype=torch.int64, pin_memory=True
    )
    cache_seqlens_cpu = torch.zeros((max_batch_size,), dtype=torch.int32, pin_memory=True)
    next_tokens_cpu = torch.zeros((max_batch_size,), dtype=torch.int64, pin_memory=True)

    return cls(
      input_ids=input_ids,
      positions=positions,
      out_loc=out_loc,
      block_table=block_table,
      cache_seqlens=cache_seqlens,
      input_ids_cpu=input_ids_cpu,
      cache_seqlens_cpu=cache_seqlens_cpu,
      next_tokens=next_tokens,
      next_tokens_cpu=next_tokens_cpu,
      max_batch_size=max_batch_size,
      max_seq_len=max_seq_len,
      max_pages_per_seq=max_pages_per_seq,
    )

  def get_input_slice(self, B: int, S: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get sliced views of input tensors for batch size B, seq len S."""
    return (
      self.input_ids[:B, :S],
      self.positions[:B, :S],
      self.out_loc[:B, :S],
    )

  def get_block_table_slice(self, B: int, num_pages: int) -> torch.Tensor:
    """Get sliced view of block table."""
    return self.block_table[:B, :num_pages]

  def get_cache_seqlens_slice(self, B: int) -> torch.Tensor:
    """Get sliced view of cache sequence lengths."""
    return self.cache_seqlens[:B]


@dataclass
class CpuGpuBuffer:
  """Paired CPU (pinned) and GPU buffers for efficient transfer.

  Pattern from vLLM: Pre-allocate both, copy between them with non_blocking=True.
  """

  cpu: torch.Tensor   # Pinned CPU memory
  gpu: torch.Tensor   # GPU memory

  @classmethod
  def create(
    cls,
    *size: int,
    dtype: torch.dtype,
    device: torch.device,
  ) -> "CpuGpuBuffer":
    """Create paired buffers."""
    cpu = torch.zeros(*size, dtype=dtype, pin_memory=True)
    gpu = torch.zeros(*size, dtype=dtype, device=device)
    return cls(cpu=cpu, gpu=gpu)

  def copy_to_gpu(self, n: Optional[int] = None) -> torch.Tensor:
    """Copy CPU data to GPU (non-blocking)."""
    if n is None:
      return self.gpu.copy_(self.cpu, non_blocking=True)
    # Handle multi-dimensional tensors
    if self.cpu.ndim == 1:
      return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)
    return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)

  def copy_to_cpu(self, n: Optional[int] = None) -> torch.Tensor:
    """Copy GPU data to CPU (non-blocking). Caller must sync!"""
    if n is None:
      return self.cpu.copy_(self.gpu, non_blocking=True)
    if self.cpu.ndim == 1:
      return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)
    return self.cpu[:n].copy_(self.gpu[:n], non_blocking=True)


class BatchBuilder:
  """Efficient batch construction using pre-allocated buffers.

  Replaces per-step tensor allocations with copies into pre-allocated buffers.
  For TP=1/DP=8: Each GPU builds batches independently, no cross-GPU sync.
  """

  def __init__(
    self,
    buffers: InputBuffers,
    page_table: torch.Tensor,  # Scheduler's [max_batch_size, max_pages] page table
    page_size: int,
    device: torch.device,
  ) -> None:
    self.buffers = buffers
    self.page_table = page_table
    self.page_size = page_size
    self.device = device

    # Reusable position range tensor
    self._pos_range = torch.arange(
      buffers.max_seq_len, device=device, dtype=torch.int64
    )

  def build_prefill_batch(
    self,
    table_indices: list[int],  # Request table slots
    cached_lens: list[int],    # Tokens already cached
    step_len: int,             # Tokens to process this step (uniform)
    input_ids_list: list[torch.Tensor],  # CPU input_ids per request
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build prefill batch tensors using pre-allocated buffers.

    Returns sliced views into pre-allocated buffers (no allocation).
    """
    B = len(table_indices)
    S = step_len

    # Populate CPU buffer then copy to GPU (pinned memory -> fast)
    for i, (tid, cached_len, ids) in enumerate(zip(table_indices, cached_lens, input_ids_list)):
      # Copy token IDs for this request's chunk
      chunk = ids[cached_len : cached_len + S]
      self.buffers.input_ids_cpu[i, :S] = chunk

    # Copy input_ids to GPU (non-blocking)
    self.buffers.input_ids[:B, :S].copy_(
      self.buffers.input_ids_cpu[:B, :S], non_blocking=True
    )

    # Build positions: [cached_len, cached_len + S) for each request
    # Use broadcasting: starts[:, None] + range[None, :]
    starts = torch.tensor(cached_lens, dtype=torch.int64, device=self.device)
    self.buffers.positions[:B, :S] = starts[:, None] + self._pos_range[:S]

    # Build out_loc from page table
    # page_idx = positions // page_size, offset = positions % page_size
    # out_loc = block_table[page_idx] * page_size + offset
    positions = self.buffers.positions[:B, :S]
    page_idx = torch.div(positions, self.page_size, rounding_mode="floor")
    offset = positions % self.page_size

    # Gather page IDs from scheduler's page table
    table_idx_t = torch.tensor(table_indices, dtype=torch.int64, device=self.device)
    page_ids = self.page_table[table_idx_t[:, None], page_idx]
    self.buffers.out_loc[:B, :S] = (page_ids * self.page_size + offset).to(torch.int32)

    # Copy block table rows for this batch
    self.buffers.block_table[:B].copy_(self.page_table[table_idx_t])

    # Cache seqlens after this step
    cache_seqlens_cpu = [cached_len + S for cached_len in cached_lens]
    for i, csl in enumerate(cache_seqlens_cpu):
      self.buffers.cache_seqlens_cpu[i] = csl
    self.buffers.cache_seqlens[:B].copy_(
      self.buffers.cache_seqlens_cpu[:B].to(self.device), non_blocking=True
    )

    return (
      self.buffers.input_ids[:B, :S],
      self.buffers.positions[:B, :S],
      self.buffers.out_loc[:B, :S],
      self.buffers.block_table[:B],
      self.buffers.cache_seqlens[:B],
      cache_seqlens_cpu[:B],
    )

  def build_decode_batch(
    self,
    table_indices: list[int],
    cached_lens: list[int],  # Current cached lengths (before this step)
    last_tokens: list[int],  # Most recent token per request
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build decode batch tensors (S=1).

    Returns sliced views into pre-allocated buffers (no allocation).
    """
    B = len(table_indices)
    S = 1

    # Token IDs: last generated token per request
    for i, tok in enumerate(last_tokens):
      self.buffers.input_ids_cpu[i, 0] = tok
    self.buffers.input_ids[:B, :S].copy_(
      self.buffers.input_ids_cpu[:B, :S], non_blocking=True
    )

    # Positions: cached_len (the position of the token we're decoding)
    positions_cpu = torch.tensor(cached_lens, dtype=torch.int64)
    self.buffers.positions[:B, 0].copy_(positions_cpu.to(self.device), non_blocking=True)

    # out_loc: single slot for this token
    positions = self.buffers.positions[:B, :S]
    page_idx = torch.div(positions, self.page_size, rounding_mode="floor")
    offset = positions % self.page_size

    table_idx_t = torch.tensor(table_indices, dtype=torch.int64, device=self.device)
    page_ids = self.page_table[table_idx_t[:, None], page_idx]
    self.buffers.out_loc[:B, :S] = (page_ids * self.page_size + offset).to(torch.int32)

    # Block table
    self.buffers.block_table[:B].copy_(self.page_table[table_idx_t])

    # Cache seqlens: cached_len + 1 (after writing this token's KV)
    cache_seqlens_cpu = [cl + 1 for cl in cached_lens]
    for i, csl in enumerate(cache_seqlens_cpu):
      self.buffers.cache_seqlens_cpu[i] = csl
    self.buffers.cache_seqlens[:B].copy_(
      self.buffers.cache_seqlens_cpu[:B].to(self.device), non_blocking=True
    )

    return (
      self.buffers.input_ids[:B, :S],
      self.buffers.positions[:B, :S],
      self.buffers.out_loc[:B, :S],
      self.buffers.block_table[:B],
      self.buffers.cache_seqlens[:B],
      cache_seqlens_cpu[:B],
    )
