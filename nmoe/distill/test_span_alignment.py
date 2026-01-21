from __future__ import annotations

import pytest

from nmoe.distill.span_alignment import distill_mask_for_byte_span, slice_tokenization_by_byte_span


def test_slice_tokenization_by_byte_span_ascii_roundtrip() -> None:
  full = b"memQANS"
  ids = list(range(len(full)))
  offsets = [(i, i + 1) for i in range(len(full))]

  sl = slice_tokenization_by_byte_span(
    full_x_bytes=full,
    full_input_ids=ids,
    full_offsets=offsets,
    span_byte_start=3,
    span_byte_end=len(full),
  )

  assert sl.x_bytes == b"QANS"
  assert sl.token_start == 3
  assert sl.token_end == len(full)
  assert sl.offsets == [(0, 1), (1, 2), (2, 3), (3, 4)]
  assert sl.input_ids == [3, 4, 5, 6]


def test_slice_tokenization_by_byte_span_requires_token_boundary() -> None:
  full = b"memQANS"
  ids = [0, 1]
  offsets = [(0, 4), (4, 7)]  # "memQ", "ANS"

  with pytest.raises(ValueError, match="span start is not on a token boundary"):
    slice_tokenization_by_byte_span(
      full_x_bytes=full,
      full_input_ids=ids,
      full_offsets=offsets,
      span_byte_start=3,
      span_byte_end=len(full),
    )


def test_slice_tokenization_by_byte_span_utf8_boundary() -> None:
  full_text = "αβγ"
  full = full_text.encode("utf-8")
  # Tokenize per character.
  ids = [0, 1, 2]
  offsets = [(0, 1), (1, 2), (2, 3)]

  # Slice off the first character (2 bytes).
  sl = slice_tokenization_by_byte_span(
    full_x_bytes=full,
    full_input_ids=ids,
    full_offsets=offsets,
    span_byte_start=len("α".encode("utf-8")),
    span_byte_end=len(full),
  )

  assert sl.x_bytes.decode("utf-8") == "βγ"
  assert sl.offsets == [(0, 1), (1, 2)]

  with pytest.raises(ValueError, match="UTF-8 character boundary"):
    slice_tokenization_by_byte_span(
      full_x_bytes=full,
      full_input_ids=ids,
      full_offsets=offsets,
      span_byte_start=1,  # inside a multibyte character
      span_byte_end=len(full),
    )


def test_distill_mask_for_byte_span_answer_only() -> None:
  x_bytes = b"QANS"
  offsets = [(0, 1), (1, 2), (2, 3), (3, 4)]  # per-char

  # Distill only "ANS": start at byte 1, end at byte 4.
  mask = distill_mask_for_byte_span(
    x_bytes=x_bytes,
    offsets=offsets,
    distill_byte_start=1,
    distill_byte_end=len(x_bytes),
  )
  assert mask == [True, True, True]


def test_distill_mask_for_byte_span_requires_token_boundary() -> None:
  x_bytes = b"abcd"
  offsets = [(0, 4)]  # single token

  with pytest.raises(ValueError, match="distill start is not on a token boundary"):
    distill_mask_for_byte_span(
      x_bytes=x_bytes,
      offsets=offsets,
      distill_byte_start=1,
      distill_byte_end=4,
      require_start_aligned=True,
    )
