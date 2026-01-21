# nmoe.serve Production Release Plan

## Executive Summary

Ship a production-grade DeepSeek-V3 inference server with vLLM/SGLang-level developer experience while maintaining AGENTS.md principles: elegant minimalism, one path per use-case, fail-fast with actionable errors.

**Target:** 22k tok/s decode on 8×B200 (currently at 4.6k tok/s)
**Current blockers:** DX friction, code duplication, untested paths, missing documentation

---

## Current State

| Metric | Current | Target |
|--------|---------|--------|
| Decode throughput | 4,589 tok/s | 22,000 tok/s |
| Launch command complexity | 12 env vars + torchrun | Single command |
| Code paths in model.py | Multiple (RDEP/DeepEP, router variants) | One (RDEP only) |
| Code shared with core nmoe/ | ~5% | ~40% |
| Env var switches in hot path | 14 | 0 (TOML-only at init) |
| Correctness test coverage | Ad-hoc | 5 canonical gate tests |

---

## Principles (from AGENTS.md)

1. **One clear path per use-case** - No DeepEP/RDEP selector, no router impl selector
2. **TOML config as source of truth** - Extend existing `ServeConfig`, don't create new
3. **Fail fast, fail loud** - Doctor preflight with actionable remedies
4. **Reuse battle-tested code** - Import from core `nmoe/` where possible
5. **Hot paths stay clean** - No `os.environ.get()` in forward pass
6. **Container-first reproducibility** - All paths validated in CI

---

## Phase 0: Foundation (Week 1)

### 0.1 Extend Existing Config System

**DO NOT** create new `config.py` or `envs.py`. Extend what exists.

```python
# nmoe/serve/config.py - EXTEND existing ServeConfig
@dataclass
class ServeConfig:
    # ... existing fields ...

    # Add launch/runtime fields
    master_port: int = 29500
    torch_extensions_dir: str = ""

    @classmethod
    def from_toml(cls, path: str) -> "ServeConfig":
        # Already exists - enhance validation
        ...

    def get_env_overrides(self) -> dict[str, str]:
        """Return env vars that must be set for this config."""
        return {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_EXTENSIONS_DIR": self.torch_extensions_dir or f"/tmp/torch_ext_{self.master_port}",
        }
```

### 0.2 Create Launch Wrapper

```python
# nmoe/serve/launch.py
"""One-command launcher for nmoe.serve."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()

    config = ServeConfig.from_toml(args.config)

    # Auto-detect network interface
    interface = detect_interface()  # ip route get 1.1.1.1

    # Set required env vars
    env = os.environ.copy()
    env.update(config.get_env_overrides())
    env.setdefault("GLOO_SOCKET_IFNAME", interface)
    env.setdefault("NCCL_SOCKET_IFNAME", interface)

    # Launch torchrun
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=8",
        f"--master_port={config.master_port}",
        "-m", "nmoe.serve",
        "--config", args.config,
    ]

    return subprocess.call(cmd, env=env)
```

**Usage:**
```bash
python -m nmoe.serve.launch --config configs/production.toml
```

### 0.3 Create Doctor Preflight

```python
# nmoe/serve/doctor.py
"""Preflight checks with actionable remedies."""

def check_world_size() -> tuple[bool, str]:
    ws = int(os.environ.get("WORLD_SIZE", "0"))
    if ws != 8:
        return False, f"WORLD_SIZE={ws}, need 8. Launch via nmoe.serve.launch or torchrun --nproc_per_node=8"
    return True, f"WORLD_SIZE=8 ✓"

def check_cuda_allocator() -> tuple[bool, str]:
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        return False, "Missing PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (required for CUDA graph capture)"
    return True, "CUDA allocator configured ✓"

def check_kv_capacity(config: ServeConfig) -> tuple[bool, str]:
    """Report KV capacity. Gate on worst-case (engine max_seq_len).

    - Model has absolute max (e.g., 128k for DSV3)
    - ServeConfig.max_seq_len is the engine ceiling (may be lower)
    - Actual capacity depends on request length distribution at runtime
    """
    page_size = config.kv_layout.page_size
    num_pages = config.num_pages
    engine_max = config.max_seq_len

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

def run_doctor(config: ServeConfig) -> bool:
    checks = [
        ("world_size", check_world_size()),
        ("cuda_allocator", check_cuda_allocator()),
        ("network_interface", check_network_interface()),
        ("extensions_dir", check_extensions_dir()),
        ("gpu_count", check_gpu_count()),
        ("kv_capacity", check_kv_capacity(config)),
        ("model_path", check_model_path(config)),
    ]

    all_passed = True
    print("=" * 60)
    print("nmoe.serve Doctor")
    print("=" * 60)
    for name, (passed, msg) in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {msg}")
        all_passed = all_passed and passed
    print("=" * 60)

    return all_passed
```

### 0.4 Canonical TOML Configs

**Important:** These configs match the existing `ServeConfig.from_toml()` schema (flat top-level keys + `[[replicas]]` array + `[kv_layout]` table).

```toml
# configs/serve/production.toml
# Top-level model config (use env substitution for paths)
model_path = "${NMOE_MODEL_PATH}"  # e.g., /data/models/DeepSeek-V3-0324-nmoe-ep8
model_family = "deepseek"

# Scheduling
max_batch_size = 256
max_prefill_tokens = 8192
max_seq_len = 32768

# Memory
gpu_memory_utilization = 0.9
num_pages = 8192  # 0 = auto-calculate

# KV cache layout (DeepSeek-V3 MLA defaults)
[kv_layout]
kv_lora_rank = 512
qk_rope_head_dim = 64
num_layers = 61
page_size = 64

# Replica configuration (8 GPUs, single "both" replica)
[[replicas]]
replica_id = 0
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
role = "both"

# API server
host = "0.0.0.0"
port = 8000
metrics_port = 9090
```

```toml
# configs/serve/benchmark_lmsys.toml
model_path = "${NMOE_MODEL_PATH}"
model_family = "deepseek"

max_batch_size = 256
max_seq_len = 32768
num_pages = 4096

[kv_layout]
page_size = 64

[[replicas]]
replica_id = 0
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
role = "both"

# Benchmark-specific (custom section, ignored by ServeConfig)
[benchmark]
mode = "decode"
batch_size = 256
ctx_len = 2000
output_len = 100
```

---

## Phase 1: Code Consolidation (Week 1-2)

### 1.1 Remove Dead Code Paths

**Delete from `model.py`:**

| Code Path | Lines | Reason |
|-----------|-------|--------|
| DeepEP transport | ~200 | Deprecated, RDEP only |
| `_ROUTER_IMPL` selector | ~80 | Keep one impl |
| `_USE_MASKED_GEMM` | ~50 | Not used in production |
| `_ROUTER_BF16_GEMM` | ~30 | Not used in production |
| Debug sync/probe in hot path | ~100 | Move to debug module |

**After:** `model.py` goes from 2239 → ~1700 lines

### 1.2 Converge serve onto core nmoe/

**Goal:** Serving uses the same battle-tested components as training. No reimplementations.

**Critical convergence targets:**

| Component | Core Location | serve/model.py Status | Action |
|-----------|---------------|----------------------|--------|
| **Quantization** | `nmoe/quant.py` | Reimplemented | Import `nmoe.quant.quantize_fp8`, `quantize_nvfp4` |
| **Block-scaled GEMM** | `nmoe/blockscaled/grouped.py` | Reimplemented | Import `expert_blockscaled`, `run_grouped_blockscaled_strided` |
| **RDEP transport** | `nmoe/rdep.py` | Partially shared | Use `nmoe.rdep.Rdep` class directly |
| **MoE compute** | `nmoe/moe.py` | Custom dispatch logic | Use `nmoe.moe.expert`, `_MoEBf16Fused`, `_MoEBlockscaledFused` |
| **RMSNorm** | `nmoe/norm.py` | Reimplemented (~20 lines) | Import `nmoe.norm.RMSNorm` |
| **Config** | `nmoe/config.py` | Separate `serve/model.py:ModelConfig` | Extend `nmoe.config.Config` for serve |

**Example migration:**

```python
# nmoe/serve/model.py

# BEFORE (reimplemented):
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6): ...

def custom_quantize_fp8(x): ...  # Custom quant logic

# AFTER (import from core):
from nmoe.norm import RMSNorm
from nmoe.quant import quantize_fp8, quantize_nvfp4
from nmoe.rdep import Rdep
from nmoe.moe import expert, _MoEBf16Fused, _MoEBlockscaledFused
```

**Config unification:**

```python
# nmoe/config.py already has Config dataclass with model arch fields:
#   dim, n_layers, n_heads, vocab_size, kv_lora_rank, qk_rope_head_dim,
#   n_routed_experts, n_activated_experts, etc.

# nmoe/serve/config.py - ServeConfig should reference or extend Config
@dataclass
class ServeConfig:
    # Could embed or extend nmoe.config.Config for model arch
    model_config_path: str  # Path to config.json / nmoe Config
    # serve-specific fields...
    max_batch_size: int = 256
    num_pages: int = 0
    # ...
```

### 1.3 Centralize Env Vars

**Create typed env module (not a new config system):**

```python
# nmoe/serve/environ.py
"""Typed environment variables for debug/platform detection only.

These are NOT config - they're for:
1. Debug flags (not in hot path)
2. Platform detection (auto-select backends)
3. CI/test overrides

Production behavior is determined by ServeConfig TOML.
"""

import os

class _EnvBool:
    def __init__(self, name: str, default: bool = False):
        self.name = name
        self.default = default

    def get(self) -> bool:
        val = os.environ.get(self.name, "").lower()
        if val in ("1", "true", "yes"):
            return True
        if val in ("0", "false", "no"):
            return False
        return self.default

# Debug flags (NOT in hot path)
DEBUG_SYNC = _EnvBool("NMOE_DEBUG_SYNC", False)
DEBUG_FINITE_CHECK = _EnvBool("NMOE_DEBUG_FINITE", False)

# Platform detection
CUDA_GRAPH_ENABLED = _EnvBool("NMOE_CUDA_GRAPH", True)
```

**Hot path stays clean:**
```python
# model.py forward() - NO env checks
def forward(self, x, ...):
    # Just computation, no os.environ.get()
    ...
```

---

## Phase 2: Test Consolidation (Week 2)

### 2.1 Define Must-Pass Suite

**5 canonical correctness tests:**

| Test | What it validates | Profile |
|------|-------------------|---------|
| `test_mla_correctness.py` | MLA attention paths (dense, chunked, decode) | MLA |
| `test_dsa_correctness.py` | DSA attention paths | DSA |
| `test_generation_e2e.py` | Full generation pipeline | Both |
| `test_kv_cache_invariants.py` | Cache consistency, page allocation | Both |
| `test_api_openai.py` | OpenAI API compatibility | Both |

**3 canonical benchmarks:**

| Benchmark | What it measures | Target |
|-----------|------------------|--------|
| `bench_lmsys_decode.py` | Decode throughput (BS=256) | 22k tok/s |
| `bench_prefill.py` | Prefill throughput | 40k tok/s |
| `bench_latency.py` | P50/P99 latency | TBD |

### 2.2 Directory Reorganization

```
nmoe/serve/
├── __init__.py
├── __main__.py          # Entry point
├── launch.py            # torchrun wrapper
├── doctor.py            # Preflight checks
├── config.py            # ServeConfig (extended)
├── environ.py           # Typed env vars (debug only)
├── engine.py            # Core engine
├── model.py             # Model (simplified)
├── scheduler.py
├── orchestrator.py
├── cache.py
├── api.py               # OpenAI endpoints
├── warmup.py
├── types.py
└── configs/
    ├── production.toml
    ├── benchmark_lmsys.toml
    └── eval.toml

tests/serve/             # MOVE tests here
├── test_mla_correctness.py
├── test_dsa_correctness.py
├── test_generation_e2e.py
├── test_kv_cache_invariants.py
└── test_api_openai.py

benchmarks/serve/        # MOVE benchmarks here
├── bench_lmsys_decode.py
├── bench_prefill.py
└── bench_latency.py
```

### 2.3 Delete Redundant Tests

**Process:**
1. Run each existing test, categorize as: duplicates must-pass | unique coverage | dead
2. Merge unique coverage into canonical tests
3. Delete the rest

**Expected reduction:** 72 → 5 test files

---

## Phase 3: Performance (Week 2-3)

### 3.1 Profile Decode Path

Current: 4,589 tok/s
Target: 22,000 tok/s (4.8× improvement needed)

**Profiling plan:**
1. NVTX trace of decode step
2. Identify top 5 time consumers
3. Compare to LMSYS reference architecture

**Likely bottlenecks:**
- MoE dispatch/combine overhead
- CUDA graph capture gaps
- Scheduler overhead

### 3.2 Optimization Priorities

| Optimization | Expected Gain | Effort |
|--------------|---------------|--------|
| Fix CUDA graph gaps | 1.5-2× | Medium |
| Optimize MoE dispatch | 1.2-1.5× | High |
| Reduce scheduler overhead | 1.1-1.2× | Low |
| Kernel fusion | 1.1-1.3× | High |

---

## Phase 4: Documentation (Week 3)

### 4.1 README

```markdown
# nmoe.serve

DeepSeek-V3 inference serving on 8×B200 GPUs.

## Quick Start

# Start server
python -m nmoe.serve.launch --config configs/serve/production.toml

# Run preflight checks
python -m nmoe.serve.launch --config configs/serve/production.toml --doctor

# Run LMSYS benchmark
python -m nmoe.serve.launch --config configs/serve/benchmark_lmsys.toml

## Requirements

- 8× NVIDIA B200 GPUs (sm_100a)
- CUDA 12.9+
- PyTorch 2.6+

## Configuration

All configuration via TOML. See `configs/serve/` for examples.

Environment variables are for debug/platform detection only:
- `NMOE_DEBUG_SYNC=1` - Enable debug synchronization (slow)
- `NMOE_DEBUG_FINITE=1` - Check for NaN/Inf (slow)
```

### 4.2 Runbook

```markdown
# nmoe.serve Runbook

## Common Issues

### "CUDA out of memory during graph capture"
**Cause:** Missing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
**Fix:** Use `nmoe.serve.launch` (sets this automatically) or export manually

### "Gloo binding to loopback"
**Cause:** `GLOO_SOCKET_IFNAME` not set
**Fix:** `export GLOO_SOCKET_IFNAME=eno1` (or your interface)

### "0 tok/s during warmup"
**Cause:** First-run JIT compilation (expected)
**Fix:** Wait for warmup to complete (~90s on first run)
```

---

## Milestones

| Milestone | Deliverable | Date |
|-----------|-------------|------|
| M0 | Launch wrapper + doctor working | Week 1 |
| M1 | Dead code removed, tests consolidated | Week 2 |
| M2 | 10k tok/s decode | Week 2-3 |
| M3 | Documentation complete | Week 3 |
| M4 | 22k tok/s decode | Week 4+ |

---

## Success Criteria

### DX
- [ ] `python -m nmoe.serve.launch --config X.toml` works out of box
- [ ] Doctor catches all common misconfigurations
- [ ] No manual env var setup required for production

### Code Quality (Primary Objectives)
- [ ] **One stack**: Single model.py implementation, no dead code paths
- [ ] **One config**: All runtime behavior from ServeConfig TOML, no env var switches in hot path
- [ ] **One transport**: RDEP only (DeepEP removed)
- [ ] **One model spec**: Shared ModelConfig between core nmoe/ and serve

### Code Quality (Secondary Metrics)
- [ ] ≥40% code shared with core nmoe/
- [ ] File count reduced (nice-to-have, not a gate)

### Correctness
- [ ] 5 canonical tests pass on every PR
- [ ] Reference comparison for MLA and DSA

### Performance
- [ ] ≥22k tok/s decode throughput (BS=256, ctx=2000)
- [ ] ≥40k tok/s prefill throughput

---

## Risks

| Risk | Mitigation |
|------|------------|
| Performance gap too large | Profile early, prioritize kernel work |
| Core nmoe/ changes break serve | Add serve tests to core CI |
| TOML config too rigid | Support env var substitution in TOML |
| Doctor misses edge cases | Expand based on user reports |

---

---

## Future TODOs (Post-Launch)

### F1: Native nmoe Checkpoint Serving

**Goal:** Serve models trained with nmoe directly, no HF conversion step.

| Task | Description | Priority |
|------|-------------|----------|
| Direct nmoe checkpoint loading | Load from `nmoe.checkpoint` format without conversion | P0 |
| Hot-reload weights | Update model weights without restart (LoRA, fine-tuned) | P1 |
| Training → Serving pipeline | `nmoe train` → `nmoe serve` with single checkpoint format | P1 |
| Checkpoint validation | Verify checkpoint compatibility before loading | P2 |

```toml
# Future config - extends existing ServeConfig schema
model_path = "/data/checkpoints/nmoe-run-42/step-100000"
model_format = "nmoe"  # vs "hf" or "nmoe-sharded" (new field)
```

### F2: Multi-Model Support

**Goal:** Serve models beyond DeepSeek-V3 family.

| Model | Architecture | Status | Notes |
|-------|--------------|--------|-------|
| DeepSeek-V3-0324 | MLA + MoE | ✅ Supported | Current target |
| DeepSeek-V3.2-Speciale | DSA + MoE | 🔄 In progress | Needs DSA attention path |
| GLM-4.7 | GQA + MoE | 📋 Planned | Different attention, similar MoE |
| MiniMax-2.1 | MLA variant + MoE | 📋 Planned | Check MoE compatibility |
| Qwen3 (all sizes) | GQA + MoE | 📋 Planned | Standard architecture |
| Qwen3-MoE | GQA + MoE | 📋 Planned | Similar to DeepSeek MoE |

**Implementation approach:**
1. Abstract attention backend (MLA, DSA, GQA, MQA)
2. Abstract MoE routing (top-k, expert choice, hash-based)
3. Model-specific adapters for weight loading
4. Unified config schema with model-specific overrides

```toml
# Future multi-model config - extends existing ServeConfig schema
model_path = "/data/models/Qwen3-72B-MoE"
model_family = "qwen3"     # existing field, add new families
model_variant = "72b-moe"  # new field for family-specific variants

# Model-specific tuning (new optional section)
[model_overrides]
moe_aux_loss_weight = 0.01
```

### F3: Large-Scale Production Configurations

**Goal:** Battle-tested configs for production deployments.

| Configuration | Scale | Use Case |
|---------------|-------|----------|
| Single-node (8×B200) | 1 node | Development, small deployments |
| Multi-node (8×8 B200) | 8 nodes | Medium production |
| Large cluster (64+ nodes) | 64+ nodes | High-throughput production |

**Infrastructure automation:**

```yaml
# k8s/serve/production-64node.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nmoe-serve
spec:
  replicas: 64
  template:
    spec:
      containers:
      - name: nmoe-serve
        image: nmoe/serve:latest
        command: ["python", "-m", "nmoe.serve.launch"]
        args: ["--config", "/config/production.toml"]
        resources:
          limits:
            nvidia.com/gpu: 8
```

**Load balancing & routing:**
- Request routing based on prompt length (short → low-latency pool, long → throughput pool)
- Automatic failover and health checks
- Graceful scaling (drain before shutdown)

### F4: Automation & CI/CD

**Goal:** Automated testing, deployment, and monitoring.

| Component | Description | Status |
|-----------|-------------|--------|
| **CI: Correctness** | Run 5 canonical tests on every PR | 📋 Planned |
| **CI: Performance** | Nightly LMSYS benchmark, alert on regression | 📋 Planned |
| **CD: Staging** | Auto-deploy to staging on main merge | 📋 Planned |
| **CD: Production** | Manual promotion with canary | 📋 Planned |
| **Monitoring** | Prometheus metrics, Grafana dashboards | 📋 Planned |

**Metrics to track:**
```python
# nmoe/serve/metrics.py
METRICS = {
    # Throughput
    "nmoe_decode_tokens_per_second": Gauge,
    "nmoe_prefill_tokens_per_second": Gauge,

    # Latency
    "nmoe_request_latency_seconds": Histogram,
    "nmoe_time_to_first_token_seconds": Histogram,

    # Utilization
    "nmoe_gpu_utilization_percent": Gauge,
    "nmoe_kv_cache_utilization_percent": Gauge,
    "nmoe_batch_size_current": Gauge,

    # Errors
    "nmoe_request_errors_total": Counter,
    "nmoe_moe_overflow_total": Counter,
}
```

**Alerting rules:**
```yaml
# alerts/serve.yaml
groups:
- name: nmoe-serve
  rules:
  - alert: DecodeTPSLow
    expr: nmoe_decode_tokens_per_second < 15000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Decode throughput below target"

  - alert: KVCacheNearFull
    expr: nmoe_kv_cache_utilization_percent > 90
    for: 1m
    labels:
      severity: critical
```

### F5: LoRA/DoRA Serving

**Goal:** Serve fine-tuned adapters individually and at scale.

#### F5.1: Single Adapter Serving

```toml
# configs/serve/lora_single.toml
# Base model config (existing ServeConfig fields)
model_path = "${NMOE_MODEL_PATH}"
model_family = "deepseek"

# Adapter config (new section)
[adapter]
type = "lora"  # or "dora"
path = "/data/adapters/customer-support-v2"
rank = 64
alpha = 128
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**Implementation:**
1. Load base model weights
2. Load adapter weights (A, B matrices for LoRA; + magnitude for DoRA)
3. Fuse or apply on-the-fly during forward pass
4. Support adapter hot-swap without base model reload

#### F5.2: Multi-Adapter Serving (Scale)

**Goal:** Serve 100s-1000s of adapters with dynamic loading.

| Feature | Description | Priority |
|---------|-------------|----------|
| **Adapter registry** | Track available adapters, metadata, versions | P0 |
| **Dynamic loading** | Load adapter on first request, LRU eviction | P0 |
| **Batched inference** | Mix requests for different adapters in same batch | P1 |
| **Adapter caching** | GPU memory pool for hot adapters | P1 |
| **S3/GCS backend** | Load adapters from cloud storage on-demand | P2 |

```toml
# configs/serve/lora_multi.toml
# Base model config (existing ServeConfig fields)
model_path = "${NMOE_MODEL_PATH}"
model_family = "deepseek"

# Multi-adapter config (new section)
[adapters]
registry = "/data/adapter_registry.json"  # or S3 URI
cache_size_gb = 10  # GPU memory for adapter cache
eviction_policy = "lru"
preload = ["customer-support-v2", "code-assist-v1"]  # Warm cache on startup

[adapters.defaults]
type = "lora"
rank = 64
alpha = 128
```

**API:**
```python
# Request with adapter selection
POST /v1/chat/completions
{
    "model": "deepseek-v3",
    "adapter": "customer-support-v2",  # or null for base model
    "messages": [...]
}
```

**Adapter registry format:**
```json
{
  "adapters": {
    "customer-support-v2": {
      "type": "lora",
      "path": "s3://adapters/customer-support-v2/",
      "rank": 64,
      "alpha": 128,
      "base_model": "deepseek-v3-0324",
      "created": "2025-01-15",
      "metrics": {"eval_loss": 0.42}
    },
    "code-assist-v1": {
      "type": "dora",
      "path": "/data/adapters/code-assist-v1/",
      "rank": 128,
      "alpha": 256
    }
  }
}
```

#### F5.3: LoRA/DoRA Training → Serving Pipeline

**Goal:** Seamless path from `nmoe train` with adapters to `nmoe serve`.

```bash
# Training (future)
nmoe train --config train_lora.toml --adapter-type lora --rank 64

# Checkpoint includes adapter weights
/data/checkpoints/run-42/
├── step-10000/
│   ├── model.safetensors      # Base (frozen)
│   ├── adapter.safetensors    # LoRA weights
│   └── adapter_config.json

# Serving (reads adapter automatically)
nmoe serve --config serve.toml --adapter /data/checkpoints/run-42/step-10000/
```

#### F5.4: Performance Considerations

| Scenario | Approach | Overhead |
|----------|----------|----------|
| Single adapter, fused | Merge LoRA into base weights at load | 0% runtime |
| Single adapter, on-the-fly | Apply LoRA in forward pass | ~5% runtime |
| Multi-adapter, same batch | Grouped LoRA GEMM (S-LoRA style) | ~10-15% runtime |
| Adapter swap | Load new adapter, evict old | ~100ms latency |

**Batching strategy for multi-adapter:**
```python
# Group requests by adapter for efficient batching
batch = {
    "base": [req1, req4, req7],           # No adapter
    "customer-support-v2": [req2, req5],  # Same adapter
    "code-assist-v1": [req3, req6],       # Different adapter
}
# Process each group with appropriate adapter applied
```

### F6: Advanced Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Speculative decoding** | Draft model for faster generation | P2 |
| **Prefix caching** | Share KV cache across requests with common prefix | P1 |
| **Continuous batching** | Dynamic batch assembly | P0 (exists) |
| **Disaggregated serving** | Separate prefill and decode pools | P2 |
| **Quantized serving** | INT8/INT4 weights for memory efficiency | P2 |

### F7: Benchmarking Suite

**Goal:** Comprehensive benchmarks beyond LMSYS decode.

| Benchmark | What it measures | Target |
|-----------|------------------|--------|
| LMSYS decode | Throughput at BS=256 | 22k tok/s |
| ShareGPT mixed | Real-world conversation patterns | TBD |
| Long context | 32k-128k context handling | TBD |
| Batch sweep | Throughput vs batch size curve | TBD |
| Latency percentiles | P50/P95/P99 at various loads | TBD |
| Cold start | Time from launch to first request | <60s |

```bash
# Future benchmark suite
python -m nmoe.serve.bench --suite full --output results/
# Generates: results/lmsys.json, results/sharegpt.json, results/longctx.json, ...
```

### F8: Model Zoo Integration

**Goal:** One-command download and serve for supported models.

```bash
# Future UX
nmoe serve deepseek-ai/DeepSeek-V3-0324  # Auto-download, convert, serve
nmoe serve Qwen/Qwen3-72B-MoE --port 8001
nmoe serve /local/path/to/checkpoint
```

**Implementation:**
1. Model registry with known-good configs
2. Automatic checkpoint format detection
3. On-the-fly conversion if needed
4. Config inference from model metadata

---

## Roadmap Summary

```
Q1 2026: Foundation
├── Week 1-2: DX improvements (launch, doctor, TOML)
├── Week 3-4: Code consolidation, 22k tok/s target
└── Deliverable: Production-ready DeepSeek-V3 serving

Q2 2026: Multi-Model
├── GLM-4.7 support
├── Qwen3 family support
├── MiniMax-2.1 support
└── Deliverable: Multi-model serving platform

Q3 2026: Scale
├── Multi-node configurations
├── Production automation (CI/CD, monitoring)
├── Advanced features (speculative, disaggregated)
└── Deliverable: Enterprise-ready serving platform

Q4 2026: Ecosystem
├── Native nmoe checkpoint serving
├── Model zoo integration
├── Community model support
└── Deliverable: Open model serving platform
```

---

## Appendix: Code Consolidation Targets

**Note:** File count is a secondary metric. The primary goal is convergence to one stack.

### Dead Code Paths in model.py (~500 lines)
```python
# Remove these selectors and keep only the production path:
# - DeepEP transport paths → Keep RDEP only
# - _ROUTER_IMPL selector → Keep one impl
# - _USE_MASKED_GEMM branches → Delete
# - _ROUTER_BF16_GEMM branches → Delete
# - Debug sync/probe in forward() → Move to debug module
```

### Test Organization
```
# Current: 72 test files scattered in nmoe/serve/
# Target: 5 gate tests in tests/serve/ + benchmarks/ separate

# Gate tests (must pass every PR):
tests/serve/test_mla_correctness.py   # MLA attention
tests/serve/test_dsa_correctness.py   # DSA attention
tests/serve/test_generation_e2e.py    # Full pipeline
tests/serve/test_kv_cache.py          # Cache invariants
tests/serve/test_api_openai.py        # API compatibility

# Benchmarks (nightly, not gates):
benchmarks/serve/bench_lmsys_decode.py
benchmarks/serve/bench_prefill.py
benchmarks/serve/bench_latency.py
```

### Reimplementations to Delete (after import from core)
```python
# After: from nmoe.quant import BlockScaledLinear
# Delete: nmoe/serve/model.py:class BlockScaledLinear (~300 lines)

# After: from nmoe.norm import RMSNorm
# Delete: nmoe/serve/model.py:class RMSNorm (~20 lines)

# After: from nmoe.moe import MoE
# Delete: custom dispatch logic in nmoe/serve/model.py (~200 lines)
```
