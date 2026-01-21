# SPDX-License-Identifier: Apache-2.0
"""HuggingFace Hub download helpers (container-first, /data-friendly).

This intentionally mirrors the robust patterns used by vLLM/sglang:
- file lock to prevent multi-rank download races
- optional hf_transfer enablement
- safetensors index-aware allow_patterns to avoid downloading unused shards

This module is only used by offline tooling (download/convert), not the hot
path inference server.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from typing import IO, Optional


_SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _enable_hf_transfer() -> None:
  """Enable hf_transfer if installed (best-effort)."""
  if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
    return
  try:
    import huggingface_hub.constants

    # If hf_transfer is installed, setting this constant enables it.
    import hf_transfer  # type: ignore # noqa: F401

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
  except Exception:
    # Best-effort only. Falls back to regular HTTP downloads.
    return


def _lock_path(key: str, cache_dir: str | None) -> str:
  lock_dir = cache_dir or tempfile.gettempdir()
  os.makedirs(lock_dir, exist_ok=True)
  hash_name = hashlib.sha256(key.encode()).hexdigest()
  return os.path.join(lock_dir, f"nmoe-hf-{hash_name}.lock")


@contextmanager
def _file_lock(key: str, cache_dir: str | None) -> Iterable[None]:
  # Import lazily to keep serve imports light.
  from filelock import FileLock

  lock = FileLock(_lock_path(key, cache_dir), mode=0o666)
  with lock:
    yield


@contextmanager
def _atomic_writer(path: str, mode: str = "w", encoding: str | None = None) -> Iterable[IO]:
  """Atomic write helper (same-directory temp file then os.replace)."""
  tmp_dir = os.path.dirname(path) or "."
  os.makedirs(tmp_dir, exist_ok=True)
  fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
  try:
    with os.fdopen(fd, mode=mode, encoding=encoding) as f:
      yield f
    os.replace(tmp_path, path)
  finally:
    if os.path.exists(tmp_path):
      try:
        os.remove(tmp_path)
      except OSError:
        pass


def _maybe_expand_safetensors_patterns_from_index(
  *,
  repo_id: str,
  allow_patterns: list[str],
  revision: str | None,
  cache_dir: str | None,
) -> list[str]:
  """If the repo has a safetensors index, download it and narrow file list."""
  if "*.safetensors" not in allow_patterns:
    return allow_patterns

  _enable_hf_transfer()
  from huggingface_hub import hf_hub_download

  try:
    index_path = hf_hub_download(
      repo_id=repo_id,
      filename=_SAFE_WEIGHTS_INDEX_NAME,
      cache_dir=cache_dir,
      revision=revision,
    )
  except Exception:
    return allow_patterns

  try:
    with open(index_path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception:
    return allow_patterns

  weight_map = data.get("weight_map")
  if not isinstance(weight_map, dict) or not weight_map:
    return allow_patterns

  # Treat the shard filenames as the allowlist; keep the index file too.
  shard_files = sorted(set(weight_map.values()))
  return shard_files + [_SAFE_WEIGHTS_INDEX_NAME]


def snapshot_download_to_dir(
  model_id_or_path: str,
  *,
  local_dir: str,
  revision: str | None = None,
  cache_dir: str | None = None,
  allow_patterns: Optional[list[str]] = None,
  ignore_patterns: str | list[str] | None = None,
  local_dir_use_symlinks: bool = False,
) -> str:
  """Download a HF repo snapshot to a deterministic local directory.

  If model_id_or_path is already a directory, returns it unchanged.
  """
  if os.path.isdir(model_id_or_path):
    return model_id_or_path

  _enable_hf_transfer()
  from huggingface_hub import snapshot_download

  allow_patterns = allow_patterns or ["*.json", "*.safetensors"]
  allow_patterns = _maybe_expand_safetensors_patterns_from_index(
    repo_id=model_id_or_path,
    allow_patterns=allow_patterns,
    revision=revision,
    cache_dir=cache_dir,
  )

  # Make multi-rank (TP) launches safe: only one process downloads/links files.
  lock_key = f"{model_id_or_path}@{revision or 'main'}->{local_dir}"
  with _file_lock(lock_key, cache_dir):
    os.makedirs(local_dir, exist_ok=True)
    meta_path = os.path.join(local_dir, "nmoe_hf_snapshot.json")
    if os.path.exists(meta_path):
      # Best-effort fast path: if metadata exists, assume snapshot already present.
      return local_dir

    snapshot_download(
      repo_id=model_id_or_path,
      revision=revision,
      cache_dir=cache_dir,
      local_dir=local_dir,
      local_dir_use_symlinks=local_dir_use_symlinks,
      allow_patterns=allow_patterns,
      ignore_patterns=ignore_patterns,
    )

    # Record provenance for reproducibility (no credentials).
    with _atomic_writer(meta_path, mode="w", encoding="utf-8") as f:
      json.dump(
        {
          "repo_id": model_id_or_path,
          "revision": revision,
          "allow_patterns": allow_patterns,
          "ignore_patterns": ignore_patterns,
          "local_dir_use_symlinks": local_dir_use_symlinks,
        },
        f,
        indent=2,
        sort_keys=True,
      )
      f.write("\n")

  return local_dir

