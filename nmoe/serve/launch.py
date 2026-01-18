# SPDX-License-Identifier: Apache-2.0
"""One-command launcher for nmoe.serve.

Sets known-safe env defaults and runs torchrun if not already under torchrun.

Usage:
    python -m nmoe.serve.launch --config configs/serve/production.toml
    python -m nmoe.serve.launch --config configs/serve/production.toml --doctor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def detect_interface() -> str:
    """Auto-detect network interface for NCCL/Gloo.

    Uses `ip route get 1.1.1.1` to find the interface that would be used
    for external traffic.
    """
    try:
        result = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Parse: "1.1.1.1 via X.X.X.X dev eth0 src Y.Y.Y.Y ..."
            parts = result.stdout.split()
            if "dev" in parts:
                idx = parts.index("dev")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    except Exception:
        pass
    # Fallback to common interface names
    for iface in ["eth0", "eno1", "ens5", "enp0s31f6"]:
        if os.path.exists(f"/sys/class/net/{iface}"):
            return iface
    return "eth0"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="nmoe.serve launcher - sets safe env defaults and runs torchrun"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run preflight checks only (no server)",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of processes per node (default: 8)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for torchrun (default: 29500)",
    )
    args = parser.parse_args()

    # Check if already under torchrun
    if os.environ.get("RANK") is not None:
        print(
            "[launch.py] Already running under torchrun (RANK is set). "
            "Use `python -m nmoe.serve --config ...` directly.",
            file=sys.stderr,
        )
        return 1

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"[launch.py] Config file not found: {args.config}", file=sys.stderr)
        return 1

    # Auto-detect network interface
    interface = detect_interface()
    print(f"[launch.py] Detected network interface: {interface}")

    # Build environment with safe defaults
    env = os.environ.copy()

    # CUDA allocator: required for CUDA graph capture
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Torch extensions: unique dir to avoid conflicts
    ext_dir = env.get("TORCH_EXTENSIONS_DIR") or f"/tmp/torch_ext_{args.master_port}"
    env["TORCH_EXTENSIONS_DIR"] = ext_dir

    # Network interfaces for NCCL/Gloo
    env.setdefault("NCCL_SOCKET_IFNAME", interface)
    env.setdefault("GLOO_SOCKET_IFNAME", interface)

    # RDEP: fused pack required for RDEP decode path
    env.setdefault("NMOE_EP_TRANSPORT", "rdep")
    env.setdefault("NMOE_MOE_FUSED_PACK", "1")

    print(f"[launch.py] Environment:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF={env['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  TORCH_EXTENSIONS_DIR={env['TORCH_EXTENSIONS_DIR']}")
    print(f"  NCCL_SOCKET_IFNAME={env['NCCL_SOCKET_IFNAME']}")
    print(f"  GLOO_SOCKET_IFNAME={env['GLOO_SOCKET_IFNAME']}")
    print(f"  NMOE_EP_TRANSPORT={env['NMOE_EP_TRANSPORT']}")
    print(f"  NMOE_MOE_FUSED_PACK={env['NMOE_MOE_FUSED_PACK']}")

    # Build torchrun command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "-m", "nmoe.serve",
        "--config", args.config,
    ]

    if args.doctor:
        cmd.append("--doctor")

    print(f"[launch.py] Running: {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
