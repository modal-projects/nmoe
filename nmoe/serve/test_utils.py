# SPDX-License-Identifier: Apache-2.0
"""Test utilities for nmoe.serve.

Keep this module tiny and dependency-free. It exists to avoid flaky test
infrastructure (e.g., fixed-port NCCL init) and to centralize common setup.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional


def init_nccl_process_group(*, rank: int = 0, world_size: int = 1, init_method: Optional[str] = None) -> None:
  """Initialize a NCCL process group without relying on fixed TCP ports.

  - For `world_size==1`, defaults to a unique `file://...` init method to avoid
    port conflicts in shared debug containers.
  - For `world_size>1`, defaults to `env://` (torchrun sets the env vars).
  """
  import torch
  import torch.distributed as dist

  if dist.is_initialized():
    return
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to init a NCCL process group.")

  if init_method is None:
    if world_size == 1:
      tmp = tempfile.NamedTemporaryFile(prefix="nmoe_pg_", suffix=".tmp", delete=False)
      tmp.close()
      init_method = f"file://{tmp.name}"
    else:
      init_method = os.environ.get("INIT_METHOD", "env://")

  dist.init_process_group(
    backend="nccl",
    init_method=init_method,
    world_size=int(world_size),
    rank=int(rank),
  )


def init_torchrun_nccl() -> tuple[int, int, int, "torch.device"]:
  """Initialize NCCL under torchrun with correct local-rank device binding.

  Torchrun sets RANK/WORLD_SIZE/LOCAL_RANK. Setting the CUDA device before
  init_process_group avoids NCCL device guessing that can hang collectives.
  """
  import torch
  import torch.distributed as dist

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

  if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

  return dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}")
