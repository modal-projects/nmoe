# SPDX-License-Identifier: Apache-2.0
"""Core types for nmoe.serve."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Optional

import torch

if TYPE_CHECKING:
  from nmoe.serve.cache import CacheHandle
  from nmoe.serve.config import Profile


class OutputMode(Enum):
  """What the engine should compute and return."""
  TOKENS = auto()       # Just sampled tokens (streaming)
  LOGPROBS = auto()     # Log probabilities of sampled tokens
  TOPK_LOGPROBS = auto()  # Top-k log probabilities
  LOGITS = auto()       # Full vocabulary logits


class RequestStatus(Enum):
  """Lifecycle state of a request."""
  PENDING = auto()      # Waiting to be scheduled
  PREFILLING = auto()   # In prefill phase
  TRANSFERRING = auto() # KV being transferred (disagg)
  DECODING = auto()     # In decode phase
  FINISHED = auto()     # Complete (success or error)
  CANCELLED = auto()    # Cancelled by client


@dataclass
class SamplingParams:
  """Parameters controlling token sampling."""
  temperature: float = 1.0
  top_k: int = 0        # 0 = disabled
  top_p: float = 1.0    # 1.0 = disabled
  max_tokens: int = 256
  seed: Optional[int] = None  # For deterministic sampling (rl_sample, eval_exact)
  stop_sequences: list[str] = field(default_factory=list)


@dataclass
class ForwardSpec:
  """Specification for what engine should compute. Profile-agnostic."""
  output_mode: OutputMode
  return_hidden_states: bool = False
  topk: int = 10  # For TOPK_LOGPROBS mode


@dataclass
class RequestMetrics:
  """Timing metrics for a request."""
  arrival_time: float = 0.0
  prefill_start: float = 0.0
  prefill_end: float = 0.0
  transfer_start: float = 0.0
  transfer_end: float = 0.0
  decode_start: float = 0.0
  first_token_time: float = 0.0
  finish_time: float = 0.0

  @property
  def ttft(self) -> float:
    """Time to first token."""
    if self.first_token_time > 0 and self.arrival_time > 0:
      return self.first_token_time - self.arrival_time
    return 0.0

  @property
  def total_time(self) -> float:
    """Total request time."""
    if self.finish_time > 0 and self.arrival_time > 0:
      return self.finish_time - self.arrival_time
    return 0.0


@dataclass
class Request:
  """A single inference request."""
  uid: int
  input_ids: torch.Tensor  # [seq_len] on CPU
  sampling_params: SamplingParams
  profile_name: str
  forward_spec: ForwardSpec

  # Mutable state
  status: RequestStatus = RequestStatus.PENDING
  output_ids: list[int] = field(default_factory=list)
  output_logprobs: list[float] = field(default_factory=list)
  output_topk: list[list[tuple[int, float]]] = field(default_factory=list)
  output_logits: Optional[torch.Tensor] = None  # [num_tokens, vocab_size]
  # Internal accumulation for OutputMode.LOGITS (CPU tensors, stacked on finish).
  _logits_chunks: list[torch.Tensor] = field(default_factory=list, repr=False)

  # Optional teacher-forcing: when set, the orchestrator appends these tokens
  # instead of sampling from the model. This is used for RL/distillation
  # workflows that need logprobs of known target tokens under the serving path.
  forced_output_ids: Optional[list[int]] = None
  forced_output_pos: int = 0

  # KV cache tracking
  cached_len: int = 0       # Tokens already in KV cache (prefix hit)
  cache_handle: Optional["CacheHandle"] = None  # Prefix-cache handle (locked) for this request
  table_idx: int = -1       # Slot in page table
  page_ids: list[int] = field(default_factory=list)

  # Lifecycle
  return_queue: Optional[asyncio.Queue] = None  # For streaming responses
  cancel_flag: bool = False
  finish_reason: Optional[str] = None  # "stop" (EOS), "length" (max_tokens), "cancelled"
  metrics: RequestMetrics = field(default_factory=RequestMetrics)

  # Replica assignment (for disagg)
  prefill_replica: Optional[int] = None
  decode_replica: Optional[int] = None

  def __post_init__(self) -> None:
    if self.input_ids.device.type != "cpu":
      raise ValueError("Request.input_ids must be a CPU tensor.")
    if self.input_ids.ndim != 1:
      raise ValueError("Request.input_ids must be 1D: [seq_len].")
    if self.input_ids.dtype not in (torch.int32, torch.int64):
      raise ValueError("Request.input_ids must be int32 or int64.")

  @property
  def seq_len(self) -> int:
    """Current sequence length (input + generated)."""
    return len(self.input_ids) + len(self.output_ids)

  @property
  def extend_len(self) -> int:
    """Tokens to process this step (not yet in KV cache)."""
    return self.seq_len - self.cached_len

  @property
  def is_prefill(self) -> bool:
    """True if request hasn't generated any tokens yet."""
    return len(self.output_ids) == 0

  @property
  def is_finished(self) -> bool:
    """True if request is complete."""
    return self.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED)

  def mark_prefill_start(self) -> None:
    self.status = RequestStatus.PREFILLING
    self.metrics.prefill_start = time.perf_counter()

  def mark_prefill_end(self) -> None:
    self.metrics.prefill_end = time.perf_counter()

  def mark_transfer_start(self) -> None:
    self.status = RequestStatus.TRANSFERRING
    self.metrics.transfer_start = time.perf_counter()

  def mark_transfer_end(self) -> None:
    self.metrics.transfer_end = time.perf_counter()

  def mark_decode_start(self) -> None:
    self.status = RequestStatus.DECODING
    if self.metrics.decode_start == 0:
      self.metrics.decode_start = time.perf_counter()

  def mark_first_token(self) -> None:
    if self.metrics.first_token_time == 0:
      self.metrics.first_token_time = time.perf_counter()

  def mark_finished(self) -> None:
    self.status = RequestStatus.FINISHED
    self.metrics.finish_time = time.perf_counter()


@dataclass
class Batch:
  """A batch of requests for a single forward pass."""
  reqs: list[Request]
  phase: Literal["prefill", "decode"]

  # Prepared tensors (set by scheduler before forward)
  # NOTE: Our model contract is [B,S] (no padding in out_loc).
  input_ids: Optional[torch.Tensor] = None  # [B,S] int64
  positions: Optional[torch.Tensor] = None  # [B,S] int64 (absolute positions)
  out_loc: Optional[torch.Tensor] = None    # [B,S] int32 physical slot ids
  block_table: Optional[torch.Tensor] = None  # [B,max_pages] int32 page ids (paged cache)
  cache_seqlens: Optional[torch.Tensor] = None  # [B] int32 cache lengths after this step
  cache_seqlens_cpu: Optional[list[int]] = None  # DSA only: CPU lengths (no GPU scalar reads)
  # Table slot ids for each request (CUDA int64 [B]). This is a scheduling-time
  # artifact used by the lockstep overlap path; it must remain a view into
  # pre-allocated scheduler buffers (no per-step allocation).
  table_idx: Optional[torch.Tensor] = None  # [B] int64

  # Attention metadata (backend-specific)
  attn_metadata: Optional[object] = None

  @property
  def size(self) -> int:
    """Number of requests in batch."""
    return len(self.reqs)

  @property
  def seqlen_q(self) -> int:
    """Query sequence length S for this batch."""
    if self.input_ids is None:
      return 0
    return int(self.input_ids.size(1))

  @property
  def total_tokens(self) -> int:
    """Total tokens to process."""
    if self.input_ids is not None:
      return int(self.input_ids.numel())
    return sum(r.extend_len for r in self.reqs)

  @property
  def is_prefill(self) -> bool:
    return self.phase == "prefill"

  @property
  def is_decode(self) -> bool:
    return self.phase == "decode"


@dataclass
class ForwardOutput:
  """Output from engine forward pass."""
  logits: torch.Tensor  # [batch_size, vocab_size] or [total_tokens, vocab_size]
  logits_cpu: Optional[torch.Tensor] = None  # [batch_size, vocab_size] on CPU (LOGITS mode)
  hidden_states: Optional[torch.Tensor] = None

  # For async token copy to CPU
  next_tokens_gpu: Optional[torch.Tensor] = None
  next_tokens_cpu: Optional[torch.Tensor] = None
  moe_overflow_cpu: Optional[torch.Tensor] = None  # [1] int32 on CPU (dropped expert pairs)
  next_logprobs_gpu: Optional[torch.Tensor] = None  # [batch_size] float32
  next_logprobs_cpu: Optional[torch.Tensor] = None  # [batch_size] float32 on CPU
  next_topk_ids_gpu: Optional[torch.Tensor] = None  # [batch_size, K] int32
  next_topk_logprobs_gpu: Optional[torch.Tensor] = None  # [batch_size, K] float32
  next_topk_ids_cpu: Optional[torch.Tensor] = None  # [batch_size, K] int32 on CPU
  next_topk_logprobs_cpu: Optional[torch.Tensor] = None  # [batch_size, K] float32 on CPU
  copy_event: Optional[torch.cuda.Event] = None
