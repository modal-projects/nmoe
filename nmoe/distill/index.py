from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from nmoe.distill.schema import DistillArtifact, BlobSlice, sha256_bytes, utc_now


def _require_duckdb():
  try:
    import duckdb  # type: ignore
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
      "duckdb is required for nmoe.distill.index; install via the repo's container images."
    ) from e
  return duckdb


_DDL = """
CREATE TABLE IF NOT EXISTS artifacts (
  example_id           VARCHAR PRIMARY KEY,
  x_bytes_hash         BLOB NOT NULL,

  render_id            VARCHAR NOT NULL,
  render_version       VARCHAR NOT NULL,
  teacher_id           VARCHAR NOT NULL,
  teacher_ckpt         VARCHAR NOT NULL,
  teacher_vocab_hash   VARCHAR NOT NULL,
  temperature          REAL NOT NULL,

  method               VARCHAR NOT NULL,
  k                    INTEGER NOT NULL,
  n_samples            INTEGER NOT NULL,
  rng_seed             BIGINT NOT NULL,
  artifact_version     INTEGER NOT NULL,
  max_delta            INTEGER NOT NULL,

  shard_id             VARCHAR NOT NULL,
  blob_offset          BIGINT NOT NULL,
  blob_length          INTEGER NOT NULL,

  num_positions        INTEGER NOT NULL,
  seq_length_bytes     INTEGER NOT NULL,

  diag_mass_mean       REAL,
  diag_max_weight      REAL,

  dataset_id           VARCHAR,
  split                VARCHAR,
  length_bucket        INTEGER,

  created_at           TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_teacher ON artifacts(teacher_id);
CREATE INDEX IF NOT EXISTS idx_render  ON artifacts(render_id);
CREATE INDEX IF NOT EXISTS idx_shard   ON artifacts(shard_id);
"""


@dataclass(frozen=True)
class DistillIndexRow:
  example_id: str
  shard_id: str
  blob_offset: int
  blob_length: int


class DistillIndex:
  def __init__(self, *, path: str):
    duckdb = _require_duckdb()
    self._duckdb = duckdb
    self._con = duckdb.connect(database=path)
    self._con.execute(_DDL)

  def close(self) -> None:
    self._con.close()

  def add(
    self,
    *,
    artifact: DistillArtifact,
    blob: BlobSlice,
    dataset_id: str | None = None,
    split: str | None = None,
    length_bucket: int | None = None,
    artifact_version: int = 1,
  ) -> None:
    diag = artifact.diag
    self._con.execute(
      """
      INSERT OR REPLACE INTO artifacts VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
      )
      """,
      [
        artifact.example_id,
        sha256_bytes(artifact.x_bytes),
        artifact.render_id,
        artifact.render_version,
        artifact.teacher_id,
        artifact.teacher_ckpt,
        artifact.teacher_vocab_hash,
        float(artifact.temperature),
        artifact.method,
        int(artifact.k),
        int(artifact.n_samples),
        int(artifact.rng_seed),
        int(artifact_version),
        int(artifact.max_delta),
        str(blob.shard_id),
        int(blob.blob_offset),
        int(blob.blob_length),
        int(len(artifact.targets)),
        int(len(artifact.x_bytes)),
        None if diag is None else float(diag.mass_mean),
        None if diag is None else float(diag.max_weight),
        dataset_id,
        split,
        length_bucket,
        utc_now(),
      ],
    )

  def iter_locations(
    self,
    *,
    teacher_id: str | None = None,
    render_id: str | None = None,
    limit: int | None = None,
  ) -> Iterable[DistillIndexRow]:
    where = []
    args: list[Any] = []
    if teacher_id is not None:
      where.append("teacher_id = ?")
      args.append(teacher_id)
    if render_id is not None:
      where.append("render_id = ?")
      args.append(render_id)
    sql = "SELECT example_id, shard_id, blob_offset, blob_length FROM artifacts"
    if where:
      sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at"
    if limit is not None:
      sql += " LIMIT ?"
      args.append(int(limit))

    for row in self._con.execute(sql, args).fetchall():
      yield DistillIndexRow(
        example_id=str(row[0]),
        shard_id=str(row[1]),
        blob_offset=int(row[2]),
        blob_length=int(row[3]),
      )
