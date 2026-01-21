from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import BinaryIO, Iterable, Sequence


_MAGIC = b"DSTL"
_VERSION_V1 = 1
_HEADER_STRUCT = struct.Struct("<4sHHIIHH44s")  # 64 bytes
_POS_HEADER_STRUCT = struct.Struct("<IH")  # byte_offset:u32, num_entries:u16
_ENTRY_HEADER_STRUCT = struct.Struct("<H")  # cont_len:u16
_ENTRY_TAIL_STRUCT = struct.Struct("<e")  # log_w: f16


@dataclass(frozen=True)
class TargetEntry:
  cont_bytes: bytes
  log_w: float  # log(weight)

  def weight(self) -> float:
    return math.exp(float(self.log_w))


@dataclass(frozen=True)
class PositionTarget:
  byte_offset: int
  entries: tuple[TargetEntry, ...]


@dataclass(frozen=True)
class ArtifactDiagnostics:
  mass_mean: float
  mass_std: float
  max_weight: float
  p99_weight: float
  decode_collisions: int
  special_tokens_masked: int


@dataclass(frozen=True)
class DistillArtifact:
  # Identity (index / provenance)
  example_id: str

  # Canonical bytes against which boundaries are defined (stored in blob).
  x_bytes: bytes

  # Sampling config encoded in blob header.
  k: int
  n_samples: int

  # Targets (v1: Δ=1 only).
  targets: tuple[PositionTarget, ...]

  # Provenance (stored in DuckDB index; not in blob).
  render_id: str = ""
  render_version: str = ""
  teacher_id: str = ""
  teacher_ckpt: str = ""
  teacher_dtype: str = ""
  teacher_vocab_hash: str = ""
  temperature: float = 1.0
  method: str = "gumbel_ht"
  rng_seed: int = 0
  max_delta: int = 1
  distill_mask: tuple[bool, ...] | None = None
  special_tokens_masked_ids: tuple[int, ...] = ()

  # Diagnostics (stored in DuckDB index; optional in-memory convenience).
  diag: ArtifactDiagnostics | None = None


@dataclass(frozen=True)
class BlobSlice:
  shard_id: str
  blob_offset: int
  blob_length: int


def stable_example_id(*, x_bytes: bytes) -> str:
  return hashlib.sha256(x_bytes).hexdigest()


def sha256_bytes(x: bytes) -> bytes:
  return hashlib.sha256(x).digest()


def utc_now() -> datetime:
  return datetime.now(timezone.utc)


def utf8_char_to_byte_offsets(text: str) -> list[int]:
  """Returns a table mapping UTF-8 character index → byte offset (len = len(text)+1)."""
  offsets: list[int] = [0]
  b = 0
  for ch in text:
    b += len(ch.encode("utf-8"))
    offsets.append(b)
  return offsets


def encode_blob(artifact: DistillArtifact) -> bytes:
  k = int(artifact.k)
  n_samples = int(artifact.n_samples)
  if k < 1:
    raise ValueError(f"k must be >= 1 (got {k})")
  if n_samples < 1:
    raise ValueError(f"n_samples must be >= 1 (got {n_samples})")

  x_bytes = artifact.x_bytes
  num_positions = int(len(artifact.targets))
  if num_positions >= 2**32:
    raise ValueError("too many positions for v1 blob")
  if len(x_bytes) >= 2**32:
    raise ValueError("x_bytes too large for v1 blob")

  header = _HEADER_STRUCT.pack(
    _MAGIC,
    _VERSION_V1,
    0,  # flags
    num_positions,
    len(x_bytes),
    k,
    n_samples,
    b"\x00" * 44,
  )

  out = bytearray()
  out += header
  out += x_bytes
  for pos in artifact.targets:
    if not (0 <= int(pos.byte_offset) < 2**32):
      raise ValueError(f"byte_offset out of range: {pos.byte_offset}")
    if len(pos.entries) >= 2**16:
      raise ValueError("too many entries for v1 blob")

    out += _POS_HEADER_STRUCT.pack(int(pos.byte_offset), int(len(pos.entries)))
    for ent in pos.entries:
      if len(ent.cont_bytes) >= 2**16:
        raise ValueError("cont_bytes too long for v1 blob")
      out += _ENTRY_HEADER_STRUCT.pack(len(ent.cont_bytes))
      out += ent.cont_bytes
      out += _ENTRY_TAIL_STRUCT.pack(float(ent.log_w))
  return bytes(out)


def decode_blob(buf: bytes) -> DistillArtifact:
  if len(buf) < _HEADER_STRUCT.size:
    raise ValueError("blob too small")

  magic, version, flags, num_positions, x_len, k, n_samples, _ = _HEADER_STRUCT.unpack_from(buf, 0)
  if magic != _MAGIC:
    raise ValueError(f"bad magic: expected={_MAGIC!r} got={magic!r}")
  if version != _VERSION_V1:
    raise ValueError(f"unsupported distill blob version: {version}")
  if flags != 0:
    raise ValueError(f"unsupported distill blob flags: {flags}")

  off = _HEADER_STRUCT.size
  x_end = off + int(x_len)
  if x_end > len(buf):
    raise ValueError("truncated x_bytes")
  x_bytes = buf[off:x_end]
  off = x_end

  targets: list[PositionTarget] = []
  for _i in range(int(num_positions)):
    if off + _POS_HEADER_STRUCT.size > len(buf):
      raise ValueError("truncated position header")
    byte_offset, num_entries = _POS_HEADER_STRUCT.unpack_from(buf, off)
    off += _POS_HEADER_STRUCT.size

    entries: list[TargetEntry] = []
    for _j in range(int(num_entries)):
      if off + _ENTRY_HEADER_STRUCT.size > len(buf):
        raise ValueError("truncated entry header")
      (cont_len,) = _ENTRY_HEADER_STRUCT.unpack_from(buf, off)
      off += _ENTRY_HEADER_STRUCT.size
      cont_end = off + int(cont_len)
      if cont_end > len(buf):
        raise ValueError("truncated cont_bytes")
      cont_bytes = buf[off:cont_end]
      off = cont_end
      if off + _ENTRY_TAIL_STRUCT.size > len(buf):
        raise ValueError("truncated log_w")
      (log_w,) = _ENTRY_TAIL_STRUCT.unpack_from(buf, off)
      off += _ENTRY_TAIL_STRUCT.size
      entries.append(TargetEntry(cont_bytes=cont_bytes, log_w=float(log_w)))

    targets.append(PositionTarget(byte_offset=int(byte_offset), entries=tuple(entries)))

  if off != len(buf):
    raise ValueError("trailing bytes in blob")

  return DistillArtifact(
    example_id=stable_example_id(x_bytes=x_bytes),
    x_bytes=x_bytes,
    k=int(k),
    n_samples=int(n_samples),
    targets=tuple(targets),
  )


def write_blob(fp: BinaryIO, artifact: DistillArtifact) -> BlobSlice:
  if not fp.seekable():
    raise ValueError("fp must be seekable")
  offset = fp.tell()
  blob = encode_blob(artifact)
  fp.write(blob)
  return BlobSlice(shard_id="", blob_offset=int(offset), blob_length=len(blob))


def read_blob(fp: BinaryIO, *, offset: int, length: int) -> DistillArtifact:
  if not fp.seekable():
    raise ValueError("fp must be seekable")
  fp.seek(int(offset))
  buf = fp.read(int(length))
  if len(buf) != int(length):
    raise ValueError("short read")
  return decode_blob(buf)


def logsumexp2(a: float, b: float) -> float:
  if a == -math.inf:
    return b
  if b == -math.inf:
    return a
  m = a if a >= b else b
  return m + math.log(math.exp(a - m) + math.exp(b - m))


def merge_entries_by_bytes(entries: Iterable[TargetEntry]) -> tuple[TargetEntry, ...]:
  merged: dict[bytes, float] = {}
  for e in entries:
    prev = merged.get(e.cont_bytes, -math.inf)
    merged[e.cont_bytes] = logsumexp2(prev, float(e.log_w))
  return tuple(TargetEntry(cont_bytes=b, log_w=lw) for b, lw in sorted(merged.items(), key=lambda kv: kv[0]))


def compute_diagnostics(artifact: DistillArtifact) -> ArtifactDiagnostics:
  masses: list[float] = []
  weights: list[float] = []
  for tgt in artifact.targets:
    m = 0.0
    for e in tgt.entries:
      w = math.exp(float(e.log_w))
      if math.isfinite(w) and w > 0.0:
        weights.append(w)
        m += w
    masses.append(m)

  if not masses:
    raise ValueError("artifact has no targets")

  mean = sum(masses) / len(masses)
  var = sum((x - mean) ** 2 for x in masses) / len(masses)
  std = math.sqrt(var)
  max_w = max(weights) if weights else 0.0
  if weights:
    ws = sorted(weights)
    p99_w = ws[min(len(ws) - 1, int(math.floor(0.99 * (len(ws) - 1))))]
  else:
    p99_w = 0.0

  return ArtifactDiagnostics(
    mass_mean=float(mean),
    mass_std=float(std),
    max_weight=float(max_w),
    p99_weight=float(p99_w),
    decode_collisions=0,
    special_tokens_masked=len(artifact.special_tokens_masked_ids),
  )


def validate_artifact(
  artifact: DistillArtifact,
  *,
  mass_tol: float = 0.25,
) -> DistillArtifact:
  if not artifact.x_bytes:
    raise ValueError("x_bytes must be non-empty")
  if artifact.k < 1:
    raise ValueError(f"k must be >= 1 (got {artifact.k})")
  if artifact.n_samples < 1:
    raise ValueError(f"n_samples must be >= 1 (got {artifact.n_samples})")

  for tgt in artifact.targets:
    if tgt.byte_offset < 0 or tgt.byte_offset > len(artifact.x_bytes):
      raise ValueError(f"byte_offset out of range: {tgt.byte_offset}")
    if not tgt.entries:
      raise ValueError(f"empty target at byte_offset={tgt.byte_offset}")
    for e in tgt.entries:
      if not math.isfinite(float(e.log_w)):
        raise ValueError(f"non-finite log_w at byte_offset={tgt.byte_offset}")
      if not e.cont_bytes:
        raise ValueError(f"empty cont_bytes at byte_offset={tgt.byte_offset}")

  diag = compute_diagnostics(artifact)
  if not (math.isfinite(diag.mass_mean) and math.isfinite(diag.mass_std)):
    raise ValueError("non-finite mass diagnostics")
  if abs(diag.mass_mean - 1.0) > float(mass_tol):
    raise ValueError(f"distill mass out of tolerance: mean={diag.mass_mean:.6f} tol={mass_tol}")

  return replace(artifact, diag=diag)


def approx_bytes_per_position(artifact: DistillArtifact) -> float:
  blob = encode_blob(artifact)
  header = _HEADER_STRUCT.size + len(artifact.x_bytes)
  if len(artifact.targets) == 0:
    return float("inf")
  return float((len(blob) - header) / len(artifact.targets))
