# SPDX-License-Identifier: Apache-2.0
"""CPU-only control plane for DP=8 request ownership (rank0 HTTP) + EP=8 experts.

Design goals:
- Rank 0 is the only HTTP/SSE frontend.
- Each request is owned by exactly one rank (v0.1: uid % world_size).
- Owner runs the normal Orchestrator/Scheduler/Engine path locally.
- Owners stream token updates back to rank 0 via gloo p2p (CPU tensors only).
- No new per-step collectives; DeepEP ordering remains unchanged.

Wire format is defined by management (hdr int64[8] + typed payload tensors).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.distributed as dist

PROTO_VER: int = 1

# msg_type enums (v0.1)
MSG_REQUEST_INIT: int = 1
MSG_TOKEN_UPDATE: int = 2
MSG_CANCEL: int = 3
MSG_ERROR: int = 4
MSG_SHUTDOWN: int = 5

# Stable IDs (do not depend on Python Enum values).
OUTPUT_MODE_ID_TOKENS: int = 0
OUTPUT_MODE_ID_LOGPROBS: int = 1
OUTPUT_MODE_ID_TOPK_LOGPROBS: int = 2
OUTPUT_MODE_ID_LOGITS: int = 3

FINISH_REASON_ID_NONE: int = 0
FINISH_REASON_ID_STOP: int = 1
FINISH_REASON_ID_LENGTH: int = 2
FINISH_REASON_ID_CANCELLED: int = 3
FINISH_REASON_ID_ERROR: int = 4


def _pack_uflag(done: bool, finish_reason_id: int) -> int:
  # bit0: DONE
  # bits1..3: finish_reason_id (valid iff DONE=1)
  if not done:
    return 0
  return 1 | ((int(finish_reason_id) & 0x7) << 1)


def _unpack_uflag(flag: int) -> tuple[bool, int]:
  done = bool(flag & 0x1)
  reason = int((flag >> 1) & 0x7) if done else FINISH_REASON_ID_NONE
  return done, reason


def finish_reason_id_to_str(reason_id: int) -> str:
  if int(reason_id) == FINISH_REASON_ID_STOP:
    return "stop"
  if int(reason_id) == FINISH_REASON_ID_LENGTH:
    return "length"
  if int(reason_id) == FINISH_REASON_ID_CANCELLED:
    return "cancelled"
  if int(reason_id) == FINISH_REASON_ID_ERROR:
    return "error"
  return "stop"


def finish_reason_str_to_id(reason: Optional[str]) -> int:
  if reason is None:
    return FINISH_REASON_ID_NONE
  if reason == "stop":
    return FINISH_REASON_ID_STOP
  if reason == "length":
    return FINISH_REASON_ID_LENGTH
  if reason == "cancelled":
    return FINISH_REASON_ID_CANCELLED
  if reason == "error":
    return FINISH_REASON_ID_ERROR
  return FINISH_REASON_ID_NONE


@dataclass(frozen=True)
class RequestInit:
  uid: int
  prompt_len: int
  output_mode_id: int
  topk: int
  max_tokens: int
  top_k: int
  seed_or_minus1: int
  temperature: float
  top_p: float
  input_ids: torch.Tensor  # int32[L] on CPU


@dataclass(frozen=True)
class TokenUpdateBatch:
  uids: torch.Tensor    # int64[U] on CPU
  tokens: torch.Tensor  # int32[U] on CPU
  uflags: torch.Tensor  # int32[U] on CPU


class ControlPlane:
  """gloo p2p control plane.

  Rank 0:
    - sends MSG_REQUEST_INIT / MSG_CANCEL to owners
    - receives MSG_TOKEN_UPDATE / MSG_ERROR from owners

  Owner ranks (including potentially rank0 for local ownership):
    - receives MSG_REQUEST_INIT / MSG_CANCEL from rank0
    - sends MSG_TOKEN_UPDATE / MSG_ERROR to rank0
  """

  def __init__(self, *, rank: int, world_size: int, ctrl_group) -> None:
    self.rank = int(rank)
    self.world_size = int(world_size)
    self.ctrl_group = ctrl_group

    self._running = False
    self._recv_threads: list[threading.Thread] = []
    self._send_thread: Optional[threading.Thread] = None
    self._shutdown_event = threading.Event()

    self._send_q: "queue.Queue[tuple[int, object]]" = queue.Queue()

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  def start_rank0(
    self,
    *,
    on_token_update: Callable[[TokenUpdateBatch], None],
    on_error: Callable[[int, str], None],
  ) -> None:
    if self.rank != 0:
      raise RuntimeError("start_rank0() called on non-zero rank.")
    if self._running:
      return
    self._running = True
    self._shutdown_event.clear()

    def _recv_loop(src_rank: int) -> None:
      hdr = torch.empty((8,), dtype=torch.int64, device="cpu")
      while self._running:
        dist.recv(hdr, src=src_rank, group=self.ctrl_group)
        if int(hdr[0].item()) != PROTO_VER:
          continue
        msg_type = int(hdr[1].item())
        if msg_type == MSG_TOKEN_UPDATE:
          U = int(hdr[3].item())
          uids = torch.empty((U,), dtype=torch.int64, device="cpu")
          tokens = torch.empty((U,), dtype=torch.int32, device="cpu")
          uflags = torch.empty((U,), dtype=torch.int32, device="cpu")
          if U > 0:
            dist.recv(uids, src=src_rank, group=self.ctrl_group)
            dist.recv(tokens, src=src_rank, group=self.ctrl_group)
            dist.recv(uflags, src=src_rank, group=self.ctrl_group)
          on_token_update(TokenUpdateBatch(uids=uids, tokens=tokens, uflags=uflags))
          continue
        if msg_type == MSG_ERROR:
          uid = int(hdr[2].item())
          n_bytes = int(hdr[3].item())
          buf = torch.empty((n_bytes,), dtype=torch.uint8, device="cpu")
          if n_bytes:
            dist.recv(buf, src=src_rank, group=self.ctrl_group)
          try:
            msg = bytes(buf.tolist()).decode("utf-8", errors="replace")
          except Exception:
            msg = "unknown error"
          on_error(uid, msg)
          continue
        if msg_type == MSG_SHUTDOWN:
          break

    # One receiver thread per non-zero rank (simple + avoids multiplexing).
    for src in range(1, self.world_size):
      t = threading.Thread(target=_recv_loop, args=(src,), daemon=False)
      t.start()
      self._recv_threads.append(t)

  def start_worker(
    self,
    *,
    on_request_init: Callable[[RequestInit], None],
    on_cancel: Callable[[int], None],
    on_shutdown: Optional[Callable[[], None]] = None,
  ) -> None:
    if self.rank == 0:
      raise RuntimeError("start_worker() called on rank0.")
    if self._running:
      return
    self._running = True
    self._shutdown_event.clear()

    def _recv_loop() -> None:
      hdr = torch.empty((8,), dtype=torch.int64, device="cpu")
      while self._running:
        dist.recv(hdr, src=0, group=self.ctrl_group)
        if int(hdr[0].item()) != PROTO_VER:
          continue
        msg_type = int(hdr[1].item())
        if msg_type == MSG_REQUEST_INIT:
          uid = int(hdr[2].item())
          L = int(hdr[3].item())
          output_mode_id = int(hdr[4].item())
          topk = int(hdr[5].item())
          sp_i64 = torch.empty((4,), dtype=torch.int64, device="cpu")
          sp_f32 = torch.empty((2,), dtype=torch.float32, device="cpu")
          input_ids = torch.empty((L,), dtype=torch.int32, device="cpu")
          dist.recv(sp_i64, src=0, group=self.ctrl_group)
          dist.recv(sp_f32, src=0, group=self.ctrl_group)
          if L:
            dist.recv(input_ids, src=0, group=self.ctrl_group)
          init = RequestInit(
            uid=uid,
            prompt_len=L,
            output_mode_id=output_mode_id,
            topk=topk,
            max_tokens=int(sp_i64[0].item()),
            top_k=int(sp_i64[1].item()),
            seed_or_minus1=int(sp_i64[2].item()),
            temperature=float(sp_f32[0].item()),
            top_p=float(sp_f32[1].item()),
            input_ids=input_ids,
          )
          on_request_init(init)
          continue
        if msg_type == MSG_CANCEL:
          uid = int(hdr[2].item())
          on_cancel(uid)
          continue
        if msg_type == MSG_SHUTDOWN:
          self._shutdown_event.set()
          if on_shutdown is not None:
            try:
              on_shutdown()
            except Exception:
              pass
          # Enqueue the ACK through the sender thread to preserve per-rank message
          # ordering (avoid racing with in-flight MSG_TOKEN_UPDATE sends).
          self._running = False
          try:
            self._send_q.put((MSG_SHUTDOWN, None))
          except Exception:
            pass
          break

    def _send_loop() -> None:
      while True:
        try:
          msg_type, payload = self._send_q.get(timeout=0.1)
        except queue.Empty:
          if not self._running:
            break
          continue
        if self._shutdown_event.is_set() and msg_type in (MSG_TOKEN_UPDATE, MSG_ERROR):
          # Best-effort drop on shutdown to avoid blocking if rank0 has stopped
          # receiving (shutdown ordering is handled by MSG_SHUTDOWN).
          continue
        if msg_type == MSG_TOKEN_UPDATE:
          batch: TokenUpdateBatch = payload  # type: ignore[assignment]
          U = int(batch.uids.numel())
          hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
          hdr[0] = PROTO_VER
          hdr[1] = MSG_TOKEN_UPDATE
          hdr[2] = 0
          hdr[3] = U
          hdr[4] = 0  # fields_mask
          hdr[5] = 0  # K
          dist.send(hdr, dst=0, group=self.ctrl_group)
          if U > 0:
            dist.send(batch.uids, dst=0, group=self.ctrl_group)
            dist.send(batch.tokens, dst=0, group=self.ctrl_group)
            dist.send(batch.uflags, dst=0, group=self.ctrl_group)
          continue
        if msg_type == MSG_ERROR:
          uid, msg = payload  # type: ignore[misc]
          b = msg.encode("utf-8", errors="replace")
          buf = torch.tensor(list(b), dtype=torch.uint8, device="cpu")
          hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
          hdr[0] = PROTO_VER
          hdr[1] = MSG_ERROR
          hdr[2] = int(uid)
          hdr[3] = int(buf.numel())
          hdr[4] = 0  # error_code
          dist.send(hdr, dst=0, group=self.ctrl_group)
          if buf.numel():
            dist.send(buf, dst=0, group=self.ctrl_group)
          continue
        if msg_type == MSG_SHUTDOWN:
          try:
            ack = torch.zeros((8,), dtype=torch.int64, device="cpu")
            ack[0] = PROTO_VER
            ack[1] = MSG_SHUTDOWN
            dist.send(ack, dst=0, group=self.ctrl_group)
          except Exception:
            pass
          break

    recv_t = threading.Thread(target=_recv_loop, daemon=False)
    recv_t.start()
    self._recv_threads.append(recv_t)
    self._send_thread = threading.Thread(target=_send_loop, daemon=False)
    self._send_thread.start()

  def shutdown(self) -> None:
    if not self._running:
      return
    self._running = False
    self._shutdown_event.set()
    if self.rank == 0:
      hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
      hdr[0] = PROTO_VER
      hdr[1] = MSG_SHUTDOWN
      for dst in range(1, self.world_size):
        try:
          dist.send(hdr, dst=dst, group=self.ctrl_group)
        except Exception:
          pass
    else:
      # Best-effort: unblock rank0 receiver threads even if rank0 initiated
      # shutdown locally (e.g., test teardown).
      try:
        hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
        hdr[0] = PROTO_VER
        hdr[1] = MSG_SHUTDOWN
        dist.send(hdr, dst=0, group=self.ctrl_group)
      except Exception:
        pass

  def join(self, *, timeout_s: Optional[float] = None) -> None:
    """Join control-plane threads after shutdown()."""
    # Python doesn't provide per-thread join timeout accumulation easily; do best-effort.
    for t in list(self._recv_threads):
      try:
        t.join(timeout=timeout_s)
      except Exception:
        pass
    if self._send_thread is not None:
      try:
        self._send_thread.join(timeout=timeout_s)
      except Exception:
        pass

  def close(self, *, timeout_s: Optional[float] = None) -> None:
    """Shutdown + join (best-effort)."""
    self.shutdown()
    self.join(timeout_s=timeout_s)

  def wait_for_shutdown(self, *, timeout_s: Optional[float] = None) -> bool:
    """Block the caller until MSG_SHUTDOWN is observed (or shutdown() called locally)."""
    return bool(self._shutdown_event.wait(timeout=timeout_s))

  # ---------------------------------------------------------------------------
  # Rank0 -> owner
  # ---------------------------------------------------------------------------

  def send_request_init(
    self,
    *,
    owner: int,
    uid: int,
    output_mode_id: int,
    topk: int,
    max_tokens: int,
    top_k: int,
    seed_or_minus1: int,
    temperature: float,
    top_p: float,
    input_ids: torch.Tensor,
  ) -> None:
    if self.rank != 0:
      raise RuntimeError("send_request_init only valid on rank0.")
    L = int(input_ids.numel())
    hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
    hdr[0] = PROTO_VER
    hdr[1] = MSG_REQUEST_INIT
    hdr[2] = int(uid)
    hdr[3] = L
    hdr[4] = int(output_mode_id)
    hdr[5] = int(topk)
    sp_i64 = torch.zeros((4,), dtype=torch.int64, device="cpu")
    sp_i64[0] = int(max_tokens)
    sp_i64[1] = int(top_k)
    sp_i64[2] = int(seed_or_minus1)
    sp_f32 = torch.zeros((2,), dtype=torch.float32, device="cpu")
    sp_f32[0] = float(temperature)
    sp_f32[1] = float(top_p)
    ids = input_ids.to(dtype=torch.int32, device="cpu", non_blocking=False).contiguous()
    dist.send(hdr, dst=int(owner), group=self.ctrl_group)
    dist.send(sp_i64, dst=int(owner), group=self.ctrl_group)
    dist.send(sp_f32, dst=int(owner), group=self.ctrl_group)
    if L:
      dist.send(ids, dst=int(owner), group=self.ctrl_group)

  def send_cancel(self, *, owner: int, uid: int) -> None:
    if self.rank != 0:
      raise RuntimeError("send_cancel only valid on rank0.")
    hdr = torch.zeros((8,), dtype=torch.int64, device="cpu")
    hdr[0] = PROTO_VER
    hdr[1] = MSG_CANCEL
    hdr[2] = int(uid)
    dist.send(hdr, dst=int(owner), group=self.ctrl_group)

  # ---------------------------------------------------------------------------
  # Owner -> rank0
  # ---------------------------------------------------------------------------

  def enqueue_token_updates(self, uids: list[int], tokens: list[int], uflags: list[int]) -> None:
    if self.rank == 0:
      return
    if self._shutdown_event.is_set():
      return
    U = int(len(uids))
    batch = TokenUpdateBatch(
      uids=torch.tensor(uids, dtype=torch.int64, device="cpu"),
      tokens=torch.tensor(tokens, dtype=torch.int32, device="cpu"),
      uflags=torch.tensor(uflags, dtype=torch.int32, device="cpu"),
    )
    self._send_q.put((MSG_TOKEN_UPDATE, batch))

  def enqueue_error(self, *, uid: int, msg: str) -> None:
    if self.rank == 0:
      return
    if self._shutdown_event.is_set():
      return
    self._send_q.put((MSG_ERROR, (int(uid), str(msg))))
