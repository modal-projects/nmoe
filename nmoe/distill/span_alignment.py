from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

from nmoe.distill.schema import utf8_char_to_byte_offsets


@dataclass(frozen=True)
class TokenSpanSlice:
  x_bytes: bytes
  input_ids: list[int]
  offsets: list[tuple[int | None, int | None]]
  token_start: int
  token_end: int


def _char_index_for_byte_offset(char_to_byte: Sequence[int], byte_offset: int) -> int:
  i = bisect.bisect_left(char_to_byte, int(byte_offset))
  if i >= len(char_to_byte) or int(char_to_byte[i]) != int(byte_offset):
    raise ValueError(f"byte offset is not on a UTF-8 character boundary: {byte_offset}")
  return int(i)


def slice_tokenization_by_byte_span(
  *,
  full_x_bytes: bytes,
  full_input_ids: Sequence[int],
  full_offsets: Sequence[tuple[int | None, int | None]],
  span_byte_start: int,
  span_byte_end: int,
) -> TokenSpanSlice:
  if len(full_offsets) != len(full_input_ids):
    raise ValueError(f"offsets length mismatch: ids={len(full_input_ids)} offsets={len(full_offsets)}")
  if not (0 <= int(span_byte_start) <= int(span_byte_end) <= len(full_x_bytes)):
    raise ValueError(
      f"byte span out of range: start={span_byte_start} end={span_byte_end} len={len(full_x_bytes)}"
    )

  full_text = full_x_bytes.decode("utf-8")
  char_to_byte = utf8_char_to_byte_offsets(full_text)

  c0 = _char_index_for_byte_offset(char_to_byte, int(span_byte_start))
  c1 = _char_index_for_byte_offset(char_to_byte, int(span_byte_end))

  # Token spans in byte space.
  tok_start_b: list[int | None] = []
  tok_end_b: list[int | None] = []
  for (cs, ce) in full_offsets:
    if cs is None or ce is None or cs == ce:
      tok_start_b.append(None)
      tok_end_b.append(None)
      continue
    if cs < 0 or ce < 0 or cs >= len(char_to_byte) or ce >= len(char_to_byte):
      tok_start_b.append(None)
      tok_end_b.append(None)
      continue
    tok_start_b.append(int(char_to_byte[int(cs)]))
    tok_end_b.append(int(char_to_byte[int(ce)]))

  start_idx = None
  for i, b in enumerate(tok_start_b):
    if b is not None and int(b) == int(span_byte_start):
      start_idx = int(i)
      break
  if start_idx is None:
    raise ValueError(f"span start is not on a token boundary: byte_offset={span_byte_start}")

  end_idx = None
  for i, b in enumerate(tok_end_b):
    if b is not None and int(b) == int(span_byte_end):
      end_idx = int(i) + 1
      break
  if end_idx is None:
    raise ValueError(f"span end is not on a token boundary: byte_offset={span_byte_end}")
  if end_idx <= start_idx:
    raise ValueError(f"empty or reversed token span: start={start_idx} end={end_idx}")

  sub_bytes = full_x_bytes[int(span_byte_start) : int(span_byte_end)]

  sub_ids = list(map(int, full_input_ids[start_idx:end_idx]))
  sub_offsets: list[tuple[int | None, int | None]] = []
  for (cs, ce) in full_offsets[start_idx:end_idx]:
    if cs is None or ce is None or cs == ce:
      sub_offsets.append((None, None))
      continue
    if cs < c0 or ce < c0 or cs > c1 or ce > c1:
      raise ValueError("token offsets out of sliced span; ensure spans align to token boundaries")
    sub_offsets.append((int(cs) - int(c0), int(ce) - int(c0)))

  return TokenSpanSlice(
    x_bytes=sub_bytes,
    input_ids=sub_ids,
    offsets=sub_offsets,
    token_start=int(start_idx),
    token_end=int(end_idx),
  )


def distill_mask_for_byte_span(
  *,
  x_bytes: bytes,
  offsets: Sequence[tuple[int | None, int | None]],
  distill_byte_start: int,
  distill_byte_end: int,
  require_start_aligned: bool = True,
) -> list[bool]:
  if not (0 <= int(distill_byte_start) <= int(distill_byte_end) <= len(x_bytes)):
    raise ValueError(
      f"distill span out of range: start={distill_byte_start} end={distill_byte_end} len={len(x_bytes)}"
    )

  x_text = x_bytes.decode("utf-8")
  char_to_byte = utf8_char_to_byte_offsets(x_text)

  tok_start_b: list[int | None] = []
  for (cs, ce) in offsets:
    if cs is None or ce is None or cs == ce:
      tok_start_b.append(None)
      continue
    if cs < 0 or cs >= len(char_to_byte):
      tok_start_b.append(None)
      continue
    tok_start_b.append(int(char_to_byte[int(cs)]))

  if require_start_aligned and distill_byte_start not in (0, len(x_bytes)):
    if int(distill_byte_start) not in {int(x) for x in tok_start_b if x is not None}:
      raise ValueError(f"distill start is not on a token boundary: byte_offset={distill_byte_start}")

  # token t predicts token t+1; the target boundary is the start byte of token t+1.
  out: list[bool] = []
  for t in range(max(0, len(tok_start_b) - 1)):
    b = tok_start_b[t + 1]
    ok = b is not None and int(distill_byte_start) <= int(b) < int(distill_byte_end)
    out.append(bool(ok))
  return out
