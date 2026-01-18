from __future__ import annotations

import pytest
import torch

from nmoe.distill.consumer import sparse_first_token_distill_loss
from nmoe.distill.schema import DistillArtifact, PositionTarget, TargetEntry, stable_example_id, utf8_char_to_byte_offsets


def _tiktoken():
  return pytest.importorskip("tiktoken")


def test_tiktoken_cont_bytes_loss_mapping() -> None:
  tiktoken = _tiktoken()
  enc = tiktoken.get_encoding("o200k_harmony")

  text = "hello world"
  ids = list(map(int, enc.encode_ordinary(text)))
  if len(ids) < 2:
    pytest.skip("need at least 2 tokens for a next-token target")

  decoded, starts = enc.decode_with_offsets(ids)
  assert decoded == text

  spans: list[tuple[int | None, int | None]] = []
  for i, s in enumerate(starts):
    si = int(s)
    if si < 0:
      spans.append((None, None))
      continue
    ei = int(starts[i + 1]) if i + 1 < len(starts) else len(decoded)
    spans.append((si, ei))

  cs1, _ = spans[1]
  assert cs1 is not None
  char_to_byte = utf8_char_to_byte_offsets(text)
  byte_offset = int(char_to_byte[int(cs1)])

  next_tid = int(ids[1])
  cont_bytes = enc.decode_single_token_bytes(next_tid)

  art = DistillArtifact(
    example_id=stable_example_id(x_bytes=text.encode("utf-8")),
    x_bytes=text.encode("utf-8"),
    k=1,
    n_samples=1,
    targets=(
      PositionTarget(
        byte_offset=byte_offset,
        entries=(TargetEntry(cont_bytes=bytes(cont_bytes), log_w=0.0),),
      ),
    ),
  )

  vocab = int(getattr(enc, "n_vocab"))
  logits = torch.zeros((1, len(ids), vocab), dtype=torch.float32)
  logits[0, 0, next_tid] = 30.0

  loss = sparse_first_token_distill_loss(student_logits=logits, artifacts=[art], student_tokenizer=enc)
  assert torch.isfinite(loss)
  assert float(loss.item()) < 1e-4

