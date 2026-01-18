# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for nmoe.serve with actionable remedies.

Usage:
    python -m nmoe.serve.launch --config config.toml --doctor
    python -m nmoe.serve --config config.toml --doctor  # (under torchrun)
"""

from __future__ import annotations

import math
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nmoe.serve.config import ServeConfig


def check_world_size() -> tuple[bool, str]:
    """Check WORLD_SIZE is 8."""
    ws = int(os.environ.get("WORLD_SIZE", "0"))
    if ws != 8:
        return False, (
            f"WORLD_SIZE={ws}, need 8. "
            "Launch via `python -m nmoe.serve.launch` or "
            "`torchrun --nproc_per_node=8`"
        )
    return True, f"WORLD_SIZE=8 ✓"


def check_cuda_allocator() -> tuple[bool, str]:
    """Check CUDA allocator is configured for graph capture."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        return False, (
            "Missing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            "(required for CUDA graph capture). "
            "Use `python -m nmoe.serve.launch` which sets this automatically."
        )
    return True, "CUDA allocator configured ✓"


def check_network_interface() -> tuple[bool, str]:
    """Check NCCL/Gloo network interface is set."""
    nccl_if = os.environ.get("NCCL_SOCKET_IFNAME", "")
    gloo_if = os.environ.get("GLOO_SOCKET_IFNAME", "")

    if not nccl_if and not gloo_if:
        return False, (
            "Neither NCCL_SOCKET_IFNAME nor GLOO_SOCKET_IFNAME set. "
            "This may cause binding to loopback. "
            "Use `python -m nmoe.serve.launch` or set manually: "
            "export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0"
        )

    parts = []
    if nccl_if:
        parts.append(f"NCCL={nccl_if}")
    if gloo_if:
        parts.append(f"Gloo={gloo_if}")
    return True, f"Network interfaces: {', '.join(parts)} ✓"


def check_extensions_dir() -> tuple[bool, str]:
    """Check TORCH_EXTENSIONS_DIR is set (avoids conflicts)."""
    ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR", "")
    if not ext_dir:
        return True, "TORCH_EXTENSIONS_DIR not set (will use default, ok for single instance)"
    return True, f"TORCH_EXTENSIONS_DIR={ext_dir} ✓"


def check_gpu_count() -> tuple[bool, str]:
    """Check CUDA device count."""
    try:
        import torch
        count = torch.cuda.device_count()
        if count < 8:
            return False, (
                f"Only {count} CUDA devices visible, need 8. "
                "Check CUDA_VISIBLE_DEVICES or hardware."
            )
        return True, f"CUDA devices: {count} ✓"
    except Exception as e:
        return False, f"Cannot check CUDA devices: {e}"


def check_rdep_env() -> tuple[bool, str]:
    """Check RDEP transport env vars."""
    transport = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
    fused_pack = os.environ.get("NMOE_MOE_FUSED_PACK", "0").strip().lower()

    if transport == "rdep" and fused_pack not in ("1", "true"):
        return False, (
            f"NMOE_EP_TRANSPORT=rdep requires NMOE_MOE_FUSED_PACK=1. "
            f"Got NMOE_MOE_FUSED_PACK={fused_pack!r}. "
            "Use `python -m nmoe.serve.launch` which sets this automatically."
        )
    return True, f"RDEP transport: {transport}, fused_pack={fused_pack} ✓"


def check_kv_capacity(config: "ServeConfig") -> tuple[bool, str]:
    """Report KV capacity. Gate on worst-case (engine max_seq_len).

    - Model has absolute max (e.g., 128k for DSV3)
    - ServeConfig.max_seq_len is the engine ceiling (may be lower)
    - Actual capacity depends on request length distribution at runtime
    """
    page_size = config.kv_layout.page_size
    num_pages = config.num_pages
    engine_max = config.max_seq_len

    if num_pages == 0:
        return True, "KV capacity: auto (num_pages=0)"

    # Worst-case: all sequences at engine max
    pages_per_seq_max = math.ceil(engine_max / page_size)
    capacity_worst = num_pages // pages_per_seq_max

    # Must be able to fit at least 1 sequence at engine max
    if capacity_worst < 1:
        return False, (
            f"KV capacity insufficient: 0 seqs @ engine max_seq_len={engine_max}. "
            f"Increase num_pages or reduce max_seq_len."
        )

    # Report capacity at various lengths for visibility
    ctx_lengths = [2048, 8192, engine_max]
    caps = []
    for ctx in ctx_lengths:
        pages_per_seq = math.ceil(ctx / page_size)
        cap = num_pages // pages_per_seq
        caps.append(f"{cap}@{ctx//1024}k")

    return True, f"KV capacity (seqs): {', '.join(caps)}"


def check_model_path(config: "ServeConfig") -> tuple[bool, str]:
    """Check model path exists."""
    path = config.model_path

    # Handle env var substitution
    if path.startswith("${") and path.endswith("}"):
        var_name = path[2:-1]
        path = os.environ.get(var_name, "")
        if not path:
            return False, (
                f"model_path uses env var ${{{var_name}}} but it's not set. "
                f"Export {var_name}=/path/to/model"
            )

    if not path:
        return False, "model_path is empty"

    if not os.path.exists(path):
        return False, f"model_path does not exist: {path}"

    return True, f"model_path: {path} ✓"


def run_doctor(config: "ServeConfig") -> bool:
    """Run all preflight checks. Returns True if all pass."""
    checks = [
        ("world_size", check_world_size()),
        ("cuda_allocator", check_cuda_allocator()),
        ("network_interface", check_network_interface()),
        ("extensions_dir", check_extensions_dir()),
        ("gpu_count", check_gpu_count()),
        ("rdep_env", check_rdep_env()),
        ("kv_capacity", check_kv_capacity(config)),
        ("model_path", check_model_path(config)),
    ]

    all_passed = True
    print("=" * 70)
    print("nmoe.serve Doctor")
    print("=" * 70)

    for name, (passed, msg) in checks:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}: {msg}")
        all_passed = all_passed and passed

    print("=" * 70)

    if all_passed:
        print("All checks passed. Ready to serve.")
    else:
        print("Some checks failed. Fix the issues above before serving.")

    return all_passed
