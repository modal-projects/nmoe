from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

from nmoe.distill.schema import (
  DistillArtifact,
  PositionTarget,
  TargetEntry,
  merge_entries_by_bytes,
  stable_example_id,
  utf8_char_to_byte_offsets,
  validate_artifact,
)
from nmoe.distill.span_alignment import distill_mask_for_byte_span, slice_tokenization_by_byte_span


def _sample_gumbel(shape: torch.Size, *, generator: torch.Generator, eps: float = 1e-20) -> torch.Tensor:
  # torch.rand defaults to CPU unless device is specified; a CUDA generator must
  # be paired with CUDA sampling to avoid a device mismatch error.
  if hasattr(generator, "device"):
    u = torch.rand(shape, dtype=torch.float64, device=generator.device, generator=generator)
  else:
    u = torch.rand(shape, dtype=torch.float64, generator=generator)
  return -torch.log(-torch.log(u.clamp(min=eps)) + eps)


def _gumbel_survival(x: torch.Tensor) -> torch.Tensor:
  # For G~Gumbel(0,1): P(G > x) = 1 - exp(-exp(-x)).
  return 1.0 - torch.exp(-torch.exp(-x))


def _decode_token_to_bytes(tokenizer, token_id: int) -> bytes:
  if hasattr(tokenizer, "decode_single_token_bytes"):
    try:
      b = tokenizer.decode_single_token_bytes(int(token_id))
      if isinstance(b, (bytes, bytearray)):
        return bytes(b)
    except Exception:
      pass

  s = None
  if hasattr(tokenizer, "decode"):
    s = tokenizer.decode([int(token_id)])
  elif hasattr(tokenizer, "id_to_token") and hasattr(tokenizer, "convert_tokens_to_string"):
    tok = tokenizer.id_to_token(int(token_id))
    s = tokenizer.convert_tokens_to_string([tok])
  if s is None:
    raise TypeError("teacher_tokenizer must support decode([id]) -> str")
  return s.encode("utf-8")


@dataclass(frozen=True)
class ProducerConfig:
  k: int
  n_samples: int = 1
  rng_seed: int = 0
  temperature: float = 1.0

  render_id: str = ""
  render_version: str = ""
  teacher_id: str = ""
  teacher_ckpt: str = ""
  teacher_dtype: str = ""
  teacher_vocab_hash: str = ""

  masked_token_ids: tuple[int, ...] = ()


def _masked_log_probs(logits: torch.Tensor, *, temperature: float, masked_token_ids: Sequence[int]) -> torch.Tensor:
  if temperature <= 0.0 or not math.isfinite(float(temperature)):
    raise ValueError(f"temperature must be finite and > 0 (got {temperature})")
  log_probs = F.log_softmax(logits.to(dtype=torch.float64) / float(temperature), dim=-1)
  if masked_token_ids:
    ids = torch.tensor(list(masked_token_ids), dtype=torch.int64, device=log_probs.device)
    log_probs = log_probs.index_fill(0, ids, float("-inf"))
  return log_probs


def _gumbel_topk_ht_entries(
  log_probs: torch.Tensor,
  *,
  k: int,
  n_samples: int,
  generator: torch.Generator,
  id_to_bytes: Callable[[int], bytes],
) -> tuple[TargetEntry, ...]:
  vocab_size = int(log_probs.numel())
  k = int(k)
  n_samples = int(n_samples)
  if k < 1:
    raise ValueError("k must be >= 1")
  if n_samples < 1:
    raise ValueError("n_samples must be >= 1")
  if k >= vocab_size:
    raise ValueError(f"k must be < vocab_size (k={k}, vocab_size={vocab_size})")

  # Aggregate across draws in weight-space for lower variance.
  weights_by_bytes: dict[bytes, float] = {}
  for _draw in range(n_samples):
    g = _sample_gumbel(log_probs.shape, generator=generator)
    noised = log_probs + g

    top_vals, top_idx = torch.topk(noised, k + 1)
    selected = top_idx[:k]
    tau = top_vals[k]  # (k+1)-th overall, excludes any selected token with prob~1 (continuous)

    sel_logp = log_probs[selected]
    q = _gumbel_survival(tau - sel_logp).clamp(min=1e-12, max=1.0 - 1e-12)
    w = torch.exp(sel_logp) / q  # P(v)/q_v

    for tid, wi in zip(selected.tolist(), w.tolist(), strict=True):
      b = id_to_bytes(int(tid))
      weights_by_bytes[b] = weights_by_bytes.get(b, 0.0) + float(wi) / float(n_samples)

  entries = [TargetEntry(cont_bytes=b, log_w=math.log(w)) for b, w in weights_by_bytes.items() if w > 0.0]
  return merge_entries_by_bytes(entries)


def produce_artifact_from_logits(
  *,
  x_bytes: bytes,
  input_ids: Sequence[int],
  offsets: Sequence[tuple[int | None, int | None]],
  teacher_logits: torch.Tensor,
  teacher_tokenizer,
  cfg: ProducerConfig,
  distill_mask: Sequence[bool] | None = None,
  example_id: str | None = None,
) -> DistillArtifact:
  """
  Build an offline distillation artifact from already-computed teacher logits.

  This module intentionally does NOT implement teacher inference. The supported
  production path is:
    serve/inference stack (Issue 04) -> logits + offsets -> distill producer.
  """
  x_text = x_bytes.decode("utf-8")
  char_to_byte = utf8_char_to_byte_offsets(x_text)

  ids = list(map(int, input_ids))
  if len(offsets) != len(ids):
    raise ValueError(f"offsets length mismatch: ids={len(ids)} offsets={len(offsets)}")

  # token t predicts token t+1, and uses byte_offset(start of token t+1 in bytes).
  tok_byte_starts: list[int | None] = []
  for (cs, ce) in offsets:
    if cs is None or ce is None or cs == ce:
      tok_byte_starts.append(None)
      continue
    if cs < 0 or cs >= len(char_to_byte):
      tok_byte_starts.append(None)
      continue
    tok_byte_starts.append(char_to_byte[int(cs)])

  logits = teacher_logits
  if logits.ndim == 3:
    if logits.shape[0] != 1:
      raise ValueError("teacher_logits batch dim must be 1 if present")
    logits = logits[0]
  if logits.ndim != 2:
    raise ValueError("teacher_logits must be (seq, vocab) or (1, seq, vocab)")
  if logits.shape[0] == len(ids):
    # Common HF convention: logits for each input position; ignore final position.
    logits = logits[:-1]
  if logits.shape[0] != len(ids) - 1:
    raise ValueError(f"teacher_logits length mismatch: got={logits.shape[0]} expected={len(ids)-1}")

  g = torch.Generator(device=logits.device).manual_seed(int(cfg.rng_seed))

  targets_by_offset: dict[int, list[TargetEntry]] = {}
  for t in range(len(ids) - 1):
    if distill_mask is not None and not bool(distill_mask[t]):
      continue

    byte_offset = tok_byte_starts[t + 1]
    if byte_offset is None:
      continue

    log_probs = _masked_log_probs(logits[t], temperature=cfg.temperature, masked_token_ids=cfg.masked_token_ids)
    if torch.isneginf(log_probs).all().item():
      raise ValueError("all tokens masked at a distillation position; check masked_token_ids")

    entries = _gumbel_topk_ht_entries(
      log_probs,
      k=cfg.k,
      n_samples=cfg.n_samples,
      generator=g,
      id_to_bytes=lambda tid: _decode_token_to_bytes(teacher_tokenizer, tid),
    )
    if not entries:
      raise ValueError("produced empty entries for a distillation position")
    targets_by_offset.setdefault(int(byte_offset), []).extend(entries)

  targets: list[PositionTarget] = []
  for off, entries in sorted(targets_by_offset.items(), key=lambda kv: kv[0]):
    merged = merge_entries_by_bytes(entries)
    if merged:
      targets.append(PositionTarget(byte_offset=int(off), entries=merged))

  artifact = DistillArtifact(
    example_id=example_id or stable_example_id(x_bytes=x_bytes),
    x_bytes=x_bytes,
    k=int(cfg.k),
    n_samples=int(cfg.n_samples),
    targets=tuple(targets),
    render_id=cfg.render_id,
    render_version=cfg.render_version,
    teacher_id=cfg.teacher_id,
    teacher_ckpt=cfg.teacher_ckpt,
    teacher_dtype=cfg.teacher_dtype,
    teacher_vocab_hash=cfg.teacher_vocab_hash,
    temperature=float(cfg.temperature),
    rng_seed=int(cfg.rng_seed),
    special_tokens_masked_ids=tuple(int(x) for x in cfg.masked_token_ids),
  )
  return validate_artifact(artifact)


def produce_span_aligned_artifact_from_logits(
  *,
  full_x_bytes: bytes,
  full_input_ids: Sequence[int],
  full_offsets: Sequence[tuple[int | None, int | None]],
  teacher_logits: torch.Tensor,
  teacher_tokenizer,
  cfg: ProducerConfig,
  student_byte_start: int,
  student_byte_end: int | None = None,
  distill_byte_start: int | None = None,
  distill_byte_end: int | None = None,
  example_id: str | None = None,
) -> DistillArtifact:
  """
  Slice a full teacher-forcing run into a student-visible span and distill only a
  subset of that span.

  The intended Phase 0 usage is:
    teacher sees:   [memory] + [query] + [answer]
    artifact stores:         [query] + [answer]
    targets align to:                  [answer] tokens only

  This function assumes `full_input_ids/full_offsets` correspond to
  `full_x_bytes.decode(\"utf-8\")` with `add_special_tokens=False`.
  """
  if student_byte_end is None:
    student_byte_end = len(full_x_bytes)
  if distill_byte_start is None:
    distill_byte_start = int(student_byte_start)
  if distill_byte_end is None:
    distill_byte_end = int(student_byte_end)

  if not (0 <= int(student_byte_start) <= int(distill_byte_start) <= int(distill_byte_end) <= int(student_byte_end)):
    raise ValueError(
      "invalid span ordering: "
      f"student=[{student_byte_start},{student_byte_end}) "
      f"distill=[{distill_byte_start},{distill_byte_end})"
    )
  if not (0 <= int(student_byte_start) <= int(student_byte_end) <= len(full_x_bytes)):
    raise ValueError(
      f"student span out of range: start={student_byte_start} end={student_byte_end} len={len(full_x_bytes)}"
    )

  sl = slice_tokenization_by_byte_span(
    full_x_bytes=full_x_bytes,
    full_input_ids=full_input_ids,
    full_offsets=full_offsets,
    span_byte_start=int(student_byte_start),
    span_byte_end=int(student_byte_end),
  )

  logits = teacher_logits
  if logits.ndim == 3:
    if logits.shape[0] != 1:
      raise ValueError("teacher_logits batch dim must be 1 if present")
    logits = logits[0]
  if logits.ndim != 2:
    raise ValueError("teacher_logits must be (seq, vocab) or (1, seq, vocab)")
  if logits.shape[0] == len(full_input_ids):
    # Common HF convention: logits for each input position; ignore final position.
    logits = logits[:-1]
  if logits.shape[0] != len(full_input_ids) - 1:
    raise ValueError(
      f"teacher_logits length mismatch: got={logits.shape[0]} expected={len(full_input_ids)-1}"
    )

  # Produce sub-sequence logits: token t predicts token t+1.
  logits_sub = logits[int(sl.token_start) : int(sl.token_end) - 1]
  if logits_sub.shape[0] != len(sl.input_ids) - 1:
    raise ValueError("span/logits mismatch; ensure spans align to token boundaries")

  distill_start_local = int(distill_byte_start) - int(student_byte_start)
  distill_end_local = int(distill_byte_end) - int(student_byte_start)
  mask = distill_mask_for_byte_span(
    x_bytes=sl.x_bytes,
    offsets=sl.offsets,
    distill_byte_start=distill_start_local,
    distill_byte_end=distill_end_local,
    require_start_aligned=True,
  )

  return produce_artifact_from_logits(
    x_bytes=sl.x_bytes,
    input_ids=sl.input_ids,
    offsets=sl.offsets,
    teacher_logits=logits_sub,
    teacher_tokenizer=teacher_tokenizer,
    cfg=cfg,
    distill_mask=mask,
    example_id=example_id,
  )


def produce_memory_query_answer_artifact_from_logits(
  *,
  memory_bytes: bytes,
  query_bytes: bytes,
  answer_bytes: bytes,
  full_input_ids: Sequence[int],
  full_offsets: Sequence[tuple[int | None, int | None]],
  teacher_logits: torch.Tensor,
  teacher_tokenizer,
  cfg: ProducerConfig,
  example_id: str | None = None,
  distill_query: bool = False,
) -> DistillArtifact:
  """
  Convenience wrapper for Phase 0 consolidation.

  Note: `full_input_ids/full_offsets/teacher_logits` must be from a teacher run on:
    full_x_bytes = memory_bytes + query_bytes + answer_bytes
  """
  student_byte_start = len(memory_bytes)
  student_byte_end = student_byte_start + len(query_bytes) + len(answer_bytes)
  distill_byte_start = student_byte_start if distill_query else student_byte_start + len(query_bytes)
  distill_byte_end = student_byte_end

  full_x_bytes = memory_bytes + query_bytes + answer_bytes
  return produce_span_aligned_artifact_from_logits(
    full_x_bytes=full_x_bytes,
    full_input_ids=full_input_ids,
    full_offsets=full_offsets,
    teacher_logits=teacher_logits,
    teacher_tokenizer=teacher_tokenizer,
    cfg=cfg,
    student_byte_start=int(student_byte_start),
    student_byte_end=int(student_byte_end),
    distill_byte_start=int(distill_byte_start),
    distill_byte_end=int(distill_byte_end),
    example_id=example_id,
  )


def masked_token_ids_from_tokenizer(tokenizer) -> tuple[int, ...]:
  ids: set[int] = set()
  if hasattr(tokenizer, "all_special_ids"):
    try:
      ids.update(int(x) for x in tokenizer.all_special_ids)
    except Exception:
      pass
  for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
    v = getattr(tokenizer, attr, None)
    if v is not None:
      ids.add(int(v))
  return tuple(sorted(ids))


def merge_decode_collisions(
  *,
  token_ids: Iterable[int],
  teacher_probs: Iterable[float],
  teacher_tokenizer,
) -> tuple[TargetEntry, ...]:
  entries = []
  for tid, p in zip(token_ids, teacher_probs, strict=True):
    if p <= 0.0:
      continue
    b = _decode_token_to_bytes(teacher_tokenizer, int(tid))
    entries.append(TargetEntry(cont_bytes=b, log_w=math.log(float(p))))
  return merge_entries_by_bytes(entries)
