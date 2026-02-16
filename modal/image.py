"""Shared Modal image definition, volume, and common env for nmoe.

All Modal apps import from here. The image reproduces the environment from
bootstrap.sh using native Modal image builder methods.

Structure:
  nmoe_build_image  -- all build steps (layers 1-5), no source mount
  nmoe_image        -- nmoe_build_image + Python source mount (must be last)

The split exists because add_local_dir with copy=False (mount) must be the
final operation in the chain -- no build steps (pip_install, run_commands,
etc.) can follow it. Code that needs to extend the image with additional
build steps (e.g. debug.py adding pytest) should start from nmoe_build_image
and add its own mount at the end.
"""
import re
import tomllib
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).parent.parent

# -- Project deps from pyproject.toml (single source of truth) -------------
# bootstrap.sh uses `uv sync` to install from pyproject.toml. The archived
# Docker chain had a manually maintained dep list in Dockerfile.deps that
# drifted (missing datasets, typer, rich). We read pyproject.toml directly
# and pass to uv_pip_install.
#
# Why not uv_sync? It creates its own venv at /.uv/.venv, separate from the
# /workspace/nmoe/.venv we set up in layer 1. Our multi-layer build installs
# torch/triton (layer 2), builds kernels (layer 4), then adds runtime deps
# (layer 5). uv_pip_install installs into the first Python on PATH (our
# existing venv); uv_sync does not.

with open(REPO_ROOT / "pyproject.toml", "rb") as _f:
  _PROJECT_DEPS = tomllib.load(_f)["project"]["dependencies"]

def _pkg_name(spec: str) -> str:
  return re.split(r"[><=!~\[\s]", spec)[0]

# Build deps: needed by the kernel Makefile (pybind11 for include headers).
# Changing these invalidates the kernel cache. torch is also a build dep
# but installed separately in layer 2 (nightly, special index).
_BUILD_DEPS = {"pybind11"}

# Version overrides for our CUDA 12.9 image (pyproject.toml pins for
# host-path CUDA 12.8; bootstrap.sh installs torch separately too).
_VERSION_OVERRIDES = {
  "cuda-python": "cuda-python==12.9.1",
  "nvidia-cutlass-dsl": "nvidia-cutlass-dsl==4.3.4",
}

def _resolve_build_deps() -> list[str]:
  """Deps needed before kernel compilation (changes invalidate kernel cache)."""
  return [spec for spec in _PROJECT_DEPS if _pkg_name(spec) in _BUILD_DEPS]

def _resolve_runtime_deps() -> list[str]:
  """Remaining project deps + modal (changes do NOT invalidate kernel cache)."""
  deps = []
  for spec in _PROJECT_DEPS:
    pkg = _pkg_name(spec)
    if pkg in _BUILD_DEPS:
      continue
    deps.append(_VERSION_OVERRIDES.get(pkg, spec))
  deps.append("modal")
  return deps

# -- Volumes ---------------------------------------------------------------
# Data (training shards, eval bundles) lives on a v2 Volume for high file
# counts and concurrent reads.  Checkpoints go to a v1 Volume — v2's FUSE
# layer has severe write amplification with torch.save's small-write pickle
# protocol (~10 MB/s effective), while v1 sustains ~350 MB/s.

data_vol = modal.Volume.from_name("nmoe-data", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("nmoe-data-v1")

# -- Source mount ignore list (shared between nmoe_image and debug_image) ---

SOURCE_MOUNT_IGNORE = [
  "triton",           # built from source in layer 2 (pinned commit)
  "third_party",      # flashmla + CUTLASS submodule, flash_attn/cute, quack -- all from layer 2
  "nmoe/csrc",        # already added with copy=True in layer 4
  "modal",            # shadows the `modal` pip package; mounted separately as modal_src
  ".git",
  "__pycache__",
  "*.pyc",
  "docker",           # archived, not needed at runtime
  "k8s",              # archived, not needed at runtime
  ".venv",            # created inside the image
  "build",
  "*.egg-info",
  "nviz",             # separate image (Dockerfile.nviz), not needed in training image
  "node_modules",
]

# -- Build image (layers 1-4, all build steps, no source mount) -------------

nmoe_build_image = (
  # -- Layer 1: Base (rarely changes) ----------------------------------------

  modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04")
  .apt_install(
    "build-essential", "cmake", "git", "curl", "wget",
    "ninja-build", "zlib1g-dev", "libxml2-dev", "python3.12-dev",
  )
  .run_commands(
    "curl -LsSf https://astral.sh/uv/install.sh | sh",
    "mkdir -p /workspace/nmoe",
    "cd /workspace/nmoe && /root/.local/bin/uv venv --python python3.12",
  )
  .env({
    "CUDA_HOME": "/usr/local/cuda",
    "PATH": "/workspace/nmoe/.venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/sbin:/sbin",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
    "TORCH_CUDA_ARCH_LIST": "10.0a",
    "CUTLASS_NVCC_ARCHS": "100a",
    "UV_HTTP_TIMEOUT": "900",
    "UV_HTTP_RETRIES": "5",
    "UV_CONCURRENT_DOWNLOADS": "1",
    "UV_CONCURRENT_INSTALLS": "1",
  })

  # -- Layer 2: System (almost never changes) --------------------------------

  # PyTorch nightly cu129
  .uv_pip_install("torch", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu129")

  # Source builds: Triton, FlashAttention CuTe, FlashMLA, Quack
  .run_commands(
    # Triton (pinned commit)
    "git clone https://github.com/triton-lang/triton.git /workspace/nmoe/triton && "
    "cd /workspace/nmoe/triton && git checkout bad25767a03107bc23e066d94aca489dc86ca70c && "
    "cd /workspace/nmoe && uv pip install -e ./triton",

    # FlashAttention CuTe (vendored, sparse checkout, pinned)
    # Sparse checkout into third_party/flash_attn/ creates the nested
    # flash_attn/cute/ structure the import expects. The upstream __init__.py
    # imports flash_attn_2_cuda (compiled ext we don't build), so we stub it.
    # bootstrap.sh flattens the structure instead, but that breaks the import
    # `from flash_attn.cute.interface import ...` — the nesting is needed.
    "mkdir -p /workspace/nmoe/third_party/flash_attn && "
    "cd /workspace/nmoe/third_party/flash_attn && "
    "git init && git remote add origin https://github.com/Dao-AILab/flash-attention.git && "
    "git sparse-checkout set flash_attn/cute && "
    "git fetch --depth 1 origin 9b6dbaceb658f576ea81e2b0189f4b5707a39aae && "
    "git checkout --detach FETCH_HEAD && "
    "echo '' > flash_attn/__init__.py",

    # FlashMLA (pinned, with CUTLASS submodule for kernel headers)
    "git clone https://github.com/deepseek-ai/FlashMLA.git /workspace/nmoe/third_party/flashmla && "
    "cd /workspace/nmoe/third_party/flashmla && "
    "git checkout 1408756a88e52a25196b759eaf8db89d2b51b5a1 && "
    "git submodule update --init --recursive && "
    "rm -rf .git .gitmodules benchmark docs flash_mla tests setup.py && "
    "rm -rf csrc/sm90 csrc/smxx csrc/sm100/decode csrc/sm100/prefill/sparse csrc/pybind.cpp csrc/params.h",

    # Quack (vendored, pinned — full clone then extract package + license)
    "mkdir -p /workspace/nmoe/third_party/quack_upstream && "
    "cd /workspace/nmoe/third_party/quack_upstream && "
    "git init && "
    "git remote add origin https://github.com/Dao-AILab/quack.git && "
    "git fetch --depth 1 origin 32b51a5b9c4620724158be60ff6a228667bbd391 && "
    "git checkout --detach FETCH_HEAD && "
    "cd /workspace/nmoe && "
    "mkdir -p third_party/quack && "
    "cp -a third_party/quack_upstream/quack third_party/quack/quack && "
    "cp third_party/quack_upstream/LICENSE third_party/quack/LICENSE && "
    "cp third_party/quack_upstream/pyproject.toml third_party/quack/pyproject.toml && "
    "rm -rf third_party/quack_upstream",
  )

  # -- Layer 3: Build deps (changes here invalidate kernel cache) -----------
  # Only deps the kernel Makefile needs (pybind11 for include headers).

  .uv_pip_install(*_resolve_build_deps())

  # -- Layer 4: CUDA kernels (rarely changes, ~45 min build, cached) ---------
  # Placed before runtime deps so that dep changes don't invalidate the
  # expensive kernel cache. Only pybind11 (above) and torch (layer 2) are
  # needed at build time.

  # copy=True required: subsequent run_commands needs these files in the image layer
  .add_local_dir(str(REPO_ROOT / "nmoe" / "csrc"), "/workspace/nmoe/nmoe/csrc", copy=True)
  .run_commands("cd /workspace/nmoe && make -C nmoe/csrc NMOE_CUDA_ARCH=100a", gpu="B200")

  # -- Layer 5: Runtime deps from pyproject.toml (single source of truth) ----
  # Placed after kernels so dep changes don't invalidate the expensive build.
  # Build deps (pybind11) excluded — already installed in layer 3.

  .uv_pip_install(*_resolve_runtime_deps())

  # -- Python path setup (via .pth file, avoids clobbering PYTHONPATH) -------

  .run_commands(
    "SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))') && "
    "printf '%s\\n' "
    "/workspace/nmoe "
    "/workspace/nmoe/nmoe/csrc "
    "/workspace/nmoe/third_party/flash_attn "
    "/workspace/nmoe/third_party/quack "
    "/workspace/nmoe/triton/python "
    "/workspace/nmoe/modal_src "
    "> \"$SITE/nmoe.pth\""
  )

  # -- Runtime environment ---------------------------------------------------

  .env({
    "NCCL_DEBUG": "WARN",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_SHM_DISABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "HF_HUB_DISABLE_XET": "1",
    "PYTHONUNBUFFERED": "1",
  })
  .workdir("/workspace/nmoe")
)

# -- Training image (build + Python source mount, must be last) -------------

nmoe_image = (
  nmoe_build_image
  .add_local_dir(str(REPO_ROOT), "/workspace/nmoe", ignore=SOURCE_MOUNT_IGNORE)
  .add_local_dir(str(REPO_ROOT / "modal"), "/workspace/nmoe/modal_src")
)
