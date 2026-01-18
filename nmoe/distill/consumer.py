from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from nmoe.distill.schema import DistillArtifact


def _tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int | None, int | None]]]:
  # tiktoken Encoding API.
  if hasattr(tokenizer, "encode_ordinary") and hasattr(tokenizer, "decode_with_offsets"):
    ids = list(map(int, tokenizer.encode_ordinary(text)))
    decoded, starts = tokenizer.decode_with_offsets(ids)
    if decoded != text:
      raise ValueError("tiktoken decode_with_offsets() did not roundtrip input text")

    spans: list[tuple[int | None, int | None]] = []
    for i, s in enumerate(starts):
      si = int(s)
      if si < 0:
        spans.append((None, None))
        continue
      if i + 1 < len(starts):
        ei = int(starts[i + 1])
        spans.append((si, ei) if ei >= 0 and ei > si else (None, None))
      else:
        spans.append((si, len(decoded)) if len(decoded) > si else (None, None))

    if len(spans) != len(ids):
      raise ValueError("tiktoken offset length mismatch")
    return ids, spans

  if callable(tokenizer):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc.get("input_ids")
    offsets = enc.get("offset_mapping")
    if input_ids is not None and offsets is not None:
      if isinstance(input_ids, (list, tuple)):
        return list(map(int, input_ids)), [(a, b) for a, b in offsets]
      if hasattr(input_ids, "tolist"):
        ids = input_ids.tolist()
        off = offsets.tolist() if hasattr(offsets, "tolist") else offsets
        if len(ids) != 1:
          raise ValueError("expected batch size 1 from tokenizer")
        return list(map(int, ids[0])), [(a, b) for a, b in off[0]]
  if hasattr(tokenizer, "encode_plus"):
    enc = tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
    return list(map(int, enc["input_ids"])), [(a, b) for a, b in enc["offset_mapping"]]
  raise TypeError("student_tokenizer must support offsets via __call__ or encode_plus")


def _encode_no_specials(tokenizer, text: str) -> list[int]:
  if hasattr(tokenizer, "encode_ordinary"):
    return list(map(int, tokenizer.encode_ordinary(text)))
  if hasattr(tokenizer, "encode"):
    try:
      return list(map(int, tokenizer.encode(text, add_special_tokens=False)))
    except TypeError:
      return list(map(int, tokenizer.encode(text)))
  if callable(tokenizer):
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc.get("input_ids")
    if ids is None:
      raise TypeError("tokenizer(text) did not return input_ids")
    if isinstance(ids, (list, tuple)):
      return list(map(int, ids))
    if hasattr(ids, "tolist"):
      ids = ids.tolist()
      if len(ids) != 1:
        raise ValueError("expected batch size 1 from tokenizer")
      return list(map(int, ids[0]))
  raise TypeError("tokenizer must support encode() or __call__")


@dataclass(frozen=True)
class ByteOffsetMapper:
  """
  Maps byte offsets (boundaries in x_bytes) to student token indices.

  For a boundary b, we train on the student logits at position p where the
  prefix ends exactly at b, i.e. token[p] ends at byte offset b.
  """

  end_to_token_index: dict[int, int]

  @classmethod
  def from_bytes(cls, *, x_bytes: bytes, student_tokenizer) -> "ByteOffsetMapper":
    x_text = x_bytes.decode("utf-8")
    _, offsets = _tokenize_with_offsets(student_tokenizer, x_text)

    # Convert char offsets to byte offsets.
    char_to_byte: list[int] = [0]
    b = 0
    for ch in x_text:
      b += len(ch.encode("utf-8"))
      char_to_byte.append(b)

    end_to_idx: dict[int, int] = {}
    for i, (cs, ce) in enumerate(offsets):
      if cs is None or ce is None or cs == ce:
        continue
      if ce < 0 or ce >= len(char_to_byte):
        continue
      end_b = int(char_to_byte[int(ce)])
      # In normal tokenizations ends are strictly increasing; keep last write for safety.
      end_to_idx[end_b] = int(i)
    return cls(end_to_token_index=end_to_idx)

  def map_boundaries(self, byte_offsets: Iterable[int]) -> list[int | None]:
    out: list[int | None] = []
    for b in byte_offsets:
      out.append(self.end_to_token_index.get(int(b)))
    return out


_TIKTOKEN_BYTES_TO_ID: dict[int, dict[bytes, int]] = {}


def _tiktoken_id_from_token_bytes(tokenizer, token_bytes: bytes) -> int | None:
  cache_key = id(tokenizer)
  m = _TIKTOKEN_BYTES_TO_ID.get(cache_key)
  if m is None:
    n_vocab = getattr(tokenizer, "n_vocab", None)
    if n_vocab is None:
      return None
    decode_single_token_bytes = getattr(tokenizer, "decode_single_token_bytes", None)
    if decode_single_token_bytes is None:
      return None
    m = {}
    for tid in range(int(n_vocab)):
      try:
        b = decode_single_token_bytes(int(tid))
      except Exception:
        continue
      if isinstance(b, (bytes, bytearray)):
        m[bytes(b)] = int(tid)
    _TIKTOKEN_BYTES_TO_ID[cache_key] = m
  return m.get(token_bytes)


def sparse_first_token_distill_loss(
  *,
  student_logits: torch.Tensor,  # (batch, seq, vocab)
  artifacts: Sequence[DistillArtifact],
  student_tokenizer,
  train_mask: torch.Tensor | None = None,  # (batch, seq) boolean mask over student positions
  renormalize_weights: bool = False,
) -> torch.Tensor:
  if student_logits.ndim != 3:
    raise ValueError("student_logits must be (batch, seq, vocab)")
  if len(artifacts) != int(student_logits.shape[0]):
    raise ValueError("artifacts batch size mismatch")
  if train_mask is not None and train_mask.shape[:2] != student_logits.shape[:2]:
    raise ValueError("train_mask must be (batch, seq)")

  total = student_logits.new_zeros(())
  denom = student_logits.new_zeros(())

  for b, art in enumerate(artifacts):
    mapper = ByteOffsetMapper.from_bytes(x_bytes=art.x_bytes, student_tokenizer=student_tokenizer)
    byte_offsets = [t.byte_offset for t in art.targets]
    positions = mapper.map_boundaries(byte_offsets)

    token_cache: dict[bytes, int | None] = {}
    log_probs = F.log_softmax(student_logits[b], dim=-1)

    for tgt, pos in zip(art.targets, positions, strict=True):
      if pos is None:
        continue
      if train_mask is not None and not bool(train_mask[b, int(pos)].item()):
        continue

      ws = []
      for ent in tgt.entries:
        w = math.exp(float(ent.log_w))
        if w > 0.0 and math.isfinite(w):
          ws.append(w)
      wsum = sum(ws)
      if renormalize_weights and wsum <= 0.0:
        continue

      for ent in tgt.entries:
        if ent.cont_bytes not in token_cache:
          tok_id = None
          if hasattr(student_tokenizer, "decode_single_token_bytes"):
            tok_id = _tiktoken_id_from_token_bytes(student_tokenizer, ent.cont_bytes)
          if tok_id is not None:
            token_cache[ent.cont_bytes] = int(tok_id)
          else:
            try:
              cont_text = ent.cont_bytes.decode("utf-8")
            except UnicodeDecodeError:
              token_cache[ent.cont_bytes] = None
            else:
              if not cont_text:
                token_cache[ent.cont_bytes] = None
              else:
                ids = _encode_no_specials(student_tokenizer, cont_text)
                token_cache[ent.cont_bytes] = int(ids[0]) if ids else None

        first = token_cache[ent.cont_bytes]
        if first is None:
          continue

        w = math.exp(float(ent.log_w))
        if not (w > 0.0 and math.isfinite(w)):
          continue
        if renormalize_weights:
          w = w / wsum

        total = total + (float(w) * (-log_probs[int(pos), int(first)]))
        denom = denom + float(w)

  return total / denom.clamp(min=1e-8)
