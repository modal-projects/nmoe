"""Runtime initialization and cleanup for nmoe training.

Handles platform checks, distributed setup, and seeds.
Seamlessly supports single GPU, single-node multi-GPU, and multi-node training.
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def _maybe_add_repo_third_party_to_sys_path(*, repo_root: str | Path | None = None) -> None:
  """Make vendored deps under repo-root/third_party importable.

  The supported execution path is container-first; these vendored deps are part
  of the image checkout and should be importable without requiring users to
  manually extend PYTHONPATH.
  """
  root = None
  if repo_root is not None:
    root = Path(repo_root)
  else:
    try:
      root = Path(__file__).resolve().parent.parent
    except Exception:
      return

  third_party = root / "third_party"
  if not third_party.is_dir():
    return

  # Vendored deps are placed as self-contained import roots, e.g.:
  # - third_party/quack/quack/...
  # - third_party/flash_attn/flash_attn/...
  # Add the package roots (not just third_party/).
  candidates = [
    third_party / "quack",
    third_party / "flash_attn",
    third_party / "nvshmem" / "nvshmem4py",
  ]
  for path in candidates:
    if path.is_dir():
      p = str(path)
      if p not in sys.path:
        sys.path.insert(0, p)


def _require_b200():
  """Hard-target NVIDIA B200 (sm_100a). Off-target is not supported."""
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required (B200, sm_100a). Off-target is not supported.")
  major, minor = torch.cuda.get_device_capability()
  if (major, minor) != (10, 0):
    raise RuntimeError(
      f"This repo targets NVIDIA B200 (sm_100a). Detected compute capability {major}.{minor}. "
      "Off-target is not supported."
    )


def init(seed: int = 42, *, require_b200: bool = True) -> tuple[int, int]:
  """Initialize runtime for training. Returns (rank, world).

  Handles:
  - Platform check (B200 required unless explicitly relaxed)
  - Seeds and TF32
  - Device assignment (LOCAL_RANK env var)
  - Distributed init (automatic for multi-GPU)

  Works seamlessly for:
  - Single GPU: rank=0, world=1
  - Single-node multi-GPU: torchrun sets LOCAL_RANK, init NCCL
  - Multi-node: same as single-node, world > local_world
  """
  _maybe_add_repo_third_party_to_sys_path()
  if require_b200:
    _require_b200()
  elif not torch.cuda.is_available():
    raise RuntimeError("CUDA device required.")

  # Seeds and TF32
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

  # Device assignment (torchrun sets LOCAL_RANK; single-process defaults to 0)
  local_rank = int(os.environ.get('LOCAL_RANK', '0'))
  torch.cuda.set_device(local_rank)

  # Distributed init (only when launched under torchrun)
  world_env = int(os.environ.get('WORLD_SIZE', '1'))
  if world_env > 1 and not dist.is_initialized():
    dist.init_process_group("nccl")

  # Get rank and world (or default to single GPU)
  rank = dist.get_rank() if dist.is_initialized() else 0
  world = dist.get_world_size() if dist.is_initialized() else 1

  return rank, world


def finalize():
  """Cleanup distributed state."""
  if dist.is_initialized():
    dist.destroy_process_group()
