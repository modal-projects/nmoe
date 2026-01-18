# nmoe.serve Overhaul Plan

## Goal
Align nmoe.serve with AGENTS.md principles: elegant minimalism, one path per use-case, TOML config as source of truth, fail-fast with actionable errors.

## Current State (Problems)

| Issue | Current | Target |
|-------|---------|--------|
| Files | 114 in one directory | ~15 core + organized tests/benchmarks |
| Env vars | 14+ scattered runtime switches | 0 runtime switches, config-only |
| Code paths | DeepEP + RDEP + router variants | RDEP only, one router |
| Config | CLI + env vars + implicit defaults | Single TOML + ServeConfig dataclass |
| Tests | 72 ad-hoc files | ~5 canonical correctness tests |
| Launch | Manual torchrun + 12 env vars | `python -m nmoe.serve --config X.toml` |

---

## Phase 1: Delete Dead Code (Day 1)

### 1.1 Delete DeepEP Transport

**Files to modify:**
- `model.py`: Remove all `deepep` branches (~100 lines)
- `engine.py`: Remove DeepEP buffer init (~50 lines)
- `orchestrator.py`: Remove DeepEP low-latency mode logic

**Specific deletions in model.py:**
```python
# DELETE these env vars (lines 55-62):
_USE_MASKED_GEMM = os.environ.get("NMOE_MOE_MASKED_GEMM", "0") ...
_ROUTER_IMPL = os.environ.get("NMOE_ROUTER_IMPL", "auto") ...
_ROUTER_BF16_GEMM = os.environ.get("NMOE_ROUTER_BF16_GEMM", "0") ...

# DELETE router selector branches (lines 875-950):
if _ROUTER_IMPL == "fused": ...
if _ROUTER_IMPL in ("auto", "sglang"): ...

# DELETE DeepEP fallback path (lines 1706-1800):
else:  # DeepEP normal path
    ...get_dispatch_layout()...
```

**Specific deletions in engine.py:**
```python
# DELETE DeepEP init branch (around line 350):
if mode == "deepep":
    self._init_deep_ep()
elif mode == "rdep":
    ...

# KEEP only RDEP path, make it unconditional
```

### 1.2 Delete Debug/Probe Code from Production Path

**Move to separate debug module or delete:**
```python
# DELETE from model.py:
_RDEP_DEBUG_SYNC = os.environ.get("NMOE_RDEP_DEBUG_SYNC", "0") ...
_RDEP_LOAD_PROBE = os.environ.get("NMOE_RDEP_LOAD_PROBE", "0") ...
_ASSERT_FINITE = os.environ.get("NMOE_ASSERT_FINITE", "0") ...
_ASSERT_FUSED_PACK_NO_OVERFLOW = os.environ.get(...) ...

# DELETE all conditional blocks:
if _RDEP_DEBUG_SYNC: ...
if _RDEP_LOAD_PROBE: ...
if _ASSERT_FINITE: ...
```

### 1.3 Consolidate Router Implementation

**Keep only the best-performing router, delete selector:**
```python
# BEFORE (model.py):
_ROUTER_IMPL = os.environ.get("NMOE_ROUTER_IMPL", "auto")
if _ROUTER_IMPL == "fused": ...
elif _ROUTER_IMPL == "sglang": ...

# AFTER:
# Just one implementation, no selector
def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Best implementation only
```

---

## Phase 2: Reorganize Files (Day 1-2)

### 2.1 Create Directory Structure

```bash
mkdir -p nmoe/serve/tests
mkdir -p nmoe/serve/benchmarks
mkdir -p nmoe/serve/configs
```

### 2.2 Move Test Files

```bash
# Keep only canonical tests:
mv test_mla_correctness.py tests/
mv test_generation_multi.py tests/
mv test_cache_scheduler_invariants.py tests/

# Delete the rest (72 - 3 = 69 files):
rm test_*.py  # After moving keepers
```

**Tests to keep (5 total):**
1. `tests/test_mla_correctness.py` - MLA attention paths
2. `tests/test_generation_multi.py` - E2E generation
3. `tests/test_cache_scheduler.py` - KV cache + scheduler invariants
4. `tests/test_api.py` - OpenAI API compatibility
5. `tests/test_profiles.py` - All 5 serving profiles

### 2.3 Move Benchmark Files

```bash
# Keep only canonical benchmarks:
mv benchmark_lmsys_e2e.py benchmarks/
mv benchmark_e2e.py benchmarks/

# Delete the rest (16 - 2 = 14 files):
rm benchmark_*.py
```

**Benchmarks to keep (3 total):**
1. `benchmarks/lmsys.py` - LMSYS decode throughput (the target)
2. `benchmarks/prefill.py` - Prefill throughput
3. `benchmarks/latency.py` - Mixed latency scenarios

### 2.4 Delete Debug Files

```bash
rm debug_*.py  # All ~4 files
```

---

## Phase 3: Create Config System (Day 2)

### 3.1 Create `envs.py` (SGLang-style typed env vars)

```python
# nmoe/serve/envs.py
"""Typed environment variable accessors."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class EnvConfig:
    """Environment-based defaults (can be overridden by TOML)."""

    # Required for correctness
    cuda_alloc_expandable: bool = True

    # Network interfaces (auto-detect or explicit)
    gloo_socket_ifname: Optional[str] = None
    nccl_socket_ifname: Optional[str] = None

    # Paths
    model_path: Optional[str] = None
    torch_extensions_dir: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EnvConfig":
        return cls(
            cuda_alloc_expandable=os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""
            ).find("expandable_segments") >= 0,
            gloo_socket_ifname=os.environ.get("GLOO_SOCKET_IFNAME"),
            nccl_socket_ifname=os.environ.get("NCCL_SOCKET_IFNAME"),
            model_path=os.environ.get("NMOE_MODEL_PATH"),
            torch_extensions_dir=os.environ.get("TORCH_EXTENSIONS_DIR"),
        )
```

### 3.2 Create `config.py` (Single ServeConfig dataclass)

```python
# nmoe/serve/config.py
"""Unified configuration for nmoe.serve."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import tomllib

@dataclass
class ModelConfig:
    """Model-specific configuration."""
    path: str
    attention_type: Literal["mla", "dsa"] = "mla"
    num_layers: int = 61
    num_dense_layers: int = 3
    num_experts: int = 256
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    max_seq_len: int = 131072
    vocab_size: int = 129280

@dataclass
class EngineConfig:
    """Engine configuration."""
    num_pages: int = 4096
    page_size: int = 64
    max_batch_size: int = 256
    max_prefill_tokens: int = 16384
    moe_expected_m: int = 256
    enable_cuda_graph: bool = True
    enable_overlap: bool = False

@dataclass
class ServerConfig:
    """HTTP server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass
class ServeConfig:
    """Top-level serving configuration."""
    model: ModelConfig
    engine: EngineConfig = field(default_factory=EngineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # Computed at init
    world_size: int = 8
    tp_size: int = 1

    @classmethod
    def from_toml(cls, path: str | Path) -> "ServeConfig":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        model_cfg = ModelConfig(**data.get("model", {}))
        engine_cfg = EngineConfig(**data.get("engine", {}))
        server_cfg = ServerConfig(**data.get("server", {}))

        return cls(
            model=model_cfg,
            engine=engine_cfg,
            server=server_cfg,
        )

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []

        if self.world_size != 8:
            errors.append(f"world_size must be 8 (got {self.world_size})")
        if self.tp_size != 1:
            errors.append(f"tp_size must be 1 (got {self.tp_size})")
        if self.engine.moe_expected_m % 16 != 0:
            errors.append(f"moe_expected_m must be multiple of 16")
        if self.engine.num_pages < 1024:
            errors.append(f"num_pages should be >= 1024 for production")

        return errors
```

### 3.3 Create Example TOML Configs

```toml
# nmoe/serve/configs/production.toml
[model]
path = "/data/models/DeepSeek-V3-0324-nmoe-ep8"
attention_type = "mla"

[engine]
num_pages = 8192
max_batch_size = 256
moe_expected_m = 256
enable_cuda_graph = true

[server]
host = "0.0.0.0"
port = 8000
```

```toml
# nmoe/serve/configs/benchmark_lmsys.toml
[model]
path = "/data/models/DeepSeek-V3-0324-nmoe-ep8"
attention_type = "mla"

[engine]
num_pages = 4096
max_batch_size = 256
moe_expected_m = 256
enable_cuda_graph = true

[benchmark]
mode = "decode"
decode_batch_size = 256
decode_ctx_len = 2000
output_len = 100
```

---

## Phase 4: Create Launch System (Day 2-3)

### 4.1 Create `doctor.py` (Preflight Checks)

```python
# nmoe/serve/doctor.py
"""Preflight checks for nmoe.serve."""

import os
import subprocess
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class DoctorResult:
    passed: bool
    checks: list[tuple[str, bool, str]]  # (name, passed, message)

    def print_report(self) -> None:
        print("=" * 60)
        print("nmoe.serve Doctor Report")
        print("=" * 60)
        for name, passed, msg in self.checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg}")
        print("=" * 60)
        if not self.passed:
            print("FAILED: Fix the above issues before starting server.")
        else:
            print("PASSED: All checks passed.")

def run_doctor(config: "ServeConfig") -> DoctorResult:
    """Run all preflight checks."""
    checks = []

    # 1. World size
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    checks.append((
        "world_size",
        world_size == 8,
        f"world_size={world_size} (need 8)"
    ))

    # 2. CUDA allocator config
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    has_expandable = "expandable_segments" in alloc_conf
    checks.append((
        "cuda_allocator",
        has_expandable,
        "expandable_segments:True" if has_expandable else "MISSING: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    ))

    # 3. Network interfaces
    gloo_if = os.environ.get("GLOO_SOCKET_IFNAME", "")
    nccl_if = os.environ.get("NCCL_SOCKET_IFNAME", "")
    has_interfaces = bool(gloo_if) and bool(nccl_if)
    checks.append((
        "network_interfaces",
        has_interfaces,
        f"GLOO={gloo_if or 'NOT SET'}, NCCL={nccl_if or 'NOT SET'}"
    ))

    # 4. TORCH_EXTENSIONS_DIR
    ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR", "")
    checks.append((
        "extensions_dir",
        bool(ext_dir),
        ext_dir or "NOT SET (will use default, may cause JIT skew)"
    ))

    # 5. GPU count
    gpu_count = torch.cuda.device_count()
    checks.append((
        "gpu_count",
        gpu_count >= 8,
        f"{gpu_count} GPUs available (need 8)"
    ))

    # 6. KV capacity check
    kv_tokens = config.engine.num_pages * config.engine.page_size
    max_needed = config.engine.max_batch_size * config.model.max_seq_len
    kv_ok = kv_tokens >= config.engine.max_batch_size * 4096  # Minimum viable
    checks.append((
        "kv_capacity",
        kv_ok,
        f"{kv_tokens:,} tokens ({config.engine.num_pages} pages × {config.engine.page_size})"
    ))

    # 7. Model path exists
    model_exists = os.path.isdir(config.model.path)
    checks.append((
        "model_path",
        model_exists,
        config.model.path if model_exists else f"NOT FOUND: {config.model.path}"
    ))

    all_passed = all(passed for _, passed, _ in checks)
    return DoctorResult(passed=all_passed, checks=checks)
```

### 4.2 Create `launch.py` (torchrun Wrapper)

```python
# nmoe/serve/launch.py
"""Launch wrapper for nmoe.serve."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def get_default_interface() -> str:
    """Auto-detect network interface."""
    try:
        result = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            capture_output=True, text=True
        )
        # Parse: "1.1.1.1 via X.X.X.X dev eno1 src ..."
        parts = result.stdout.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception:
        pass
    return "eth0"

def main():
    parser = argparse.ArgumentParser(
        description="Launch nmoe.serve inference server"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to TOML config file"
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run preflight checks only"
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Port for torch distributed"
    )
    args = parser.parse_args()

    # Detect if already under torchrun
    if os.environ.get("RANK") is not None:
        # Already under torchrun, run directly
        from nmoe.serve.__main__ import main as serve_main
        return serve_main()

    # Set up environment
    interface = get_default_interface()
    env = os.environ.copy()
    env.setdefault("GLOO_SOCKET_IFNAME", interface)
    env.setdefault("NCCL_SOCKET_IFNAME", interface)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{args.master_port}")

    # Run torchrun
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=8",
        f"--master_port={args.master_port}",
        "-m", "nmoe.serve",
        "--config", args.config,
    ]

    if args.doctor:
        cmd.append("--doctor")

    print(f"Launching: {' '.join(cmd)}")
    print(f"Interface: {interface}")
    print(f"Extensions: {env['TORCH_EXTENSIONS_DIR']}")

    return subprocess.call(cmd, env=env)

if __name__ == "__main__":
    sys.exit(main())
```

### 4.3 Update `__main__.py`

```python
# nmoe/serve/__main__.py
"""Main entry point for nmoe.serve."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()

    # Load config
    from nmoe.serve.config import ServeConfig
    config = ServeConfig.from_toml(args.config)

    # Validate config
    errors = config.validate()
    if errors:
        for e in errors:
            print(f"Config error: {e}", file=sys.stderr)
        return 1

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Run doctor if requested
    if args.doctor:
        if rank == 0:
            from nmoe.serve.doctor import run_doctor
            result = run_doctor(config)
            result.print_report()
        dist.barrier()
        return 0 if result.passed else 1

    # Doctor check (non-verbose)
    from nmoe.serve.doctor import run_doctor
    result = run_doctor(config)
    if not result.passed:
        if rank == 0:
            result.print_report()
        return 1

    # Start server
    from nmoe.serve.server import run_server
    return run_server(config, rank, world_size)

if __name__ == "__main__":
    sys.exit(main())
```

---

## Phase 5: Simplify Core Code (Day 3-4)

### 5.1 Simplify `model.py`

**Before:** ~2300 lines with branches
**After:** ~1500 lines, one path

Key changes:
1. Remove all `_ROUTER_IMPL` branches
2. Remove all `_USE_MASKED_GEMM` branches
3. Remove all DeepEP code paths
4. Remove all debug sync/probe code
5. Make RDEP the only transport (unconditional)

### 5.2 Simplify `engine.py`

**Before:** ~1000 lines
**After:** ~700 lines

Key changes:
1. Remove DeepEP buffer init
2. Remove transport selector
3. Config passed in, not read from env
4. Remove debug probe logic

### 5.3 Simplify `orchestrator.py`

Key changes:
1. Remove low-latency mode selector
2. Remove force_low_latency attribute
3. Config determines behavior at init

---

## Phase 6: Documentation (Day 4)

### 6.1 Create README

```markdown
# nmoe.serve

DeepSeek-V3 inference serving on 8×B200 GPUs.

## Quick Start

```bash
# Start server
python -m nmoe.serve.launch --config configs/production.toml

# Run benchmark
python -m nmoe.serve.launch --config configs/benchmark_lmsys.toml
```

## Configuration

All configuration via TOML. See `configs/` for examples.

## Requirements

- 8× NVIDIA B200 GPUs
- CUDA 12.9+
- PyTorch 2.6+
```

---

## Summary: Files Changed

### Deleted (~95 files)
- 69 test files (keep 3)
- 14 benchmark files (keep 2)
- 4 debug files
- ~500 lines of DeepEP code
- ~200 lines of router selector code
- ~100 lines of debug sync/probe code

### Created (~8 files)
- `envs.py` - Typed env vars
- `config.py` - ServeConfig dataclass
- `doctor.py` - Preflight checks
- `launch.py` - torchrun wrapper
- `configs/production.toml`
- `configs/benchmark_lmsys.toml`
- `tests/__init__.py`
- `benchmarks/__init__.py`

### Modified (~5 files)
- `model.py` - Remove branches (-800 lines)
- `engine.py` - Remove transport selector (-300 lines)
- `orchestrator.py` - Simplify (-100 lines)
- `__main__.py` - Use config system
- `api.py` - Minor updates

### Final State
- **22 → 15 core files**
- **114 → 23 total files** (15 core + 5 tests + 3 benchmarks)
- **0 runtime env var switches**
- **1 config format (TOML)**
- **1 transport (RDEP)**
- **1 router implementation**
