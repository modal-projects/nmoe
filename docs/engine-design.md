# nmoe Serve/Engine Design (Repo-Aligned): LMSYS-Grade Performance

This document is the **design source of truth for `nmoe/serve`**.

It has two goals:
1. **Describe what exists today** (accurate, file-by-file; no imaginary modules).
2. **Describe what we must build next** to match the LMSYS “large-scale EP” performance numbers (prefill + decode tok/s), while staying container-first and minimal.

## Targets (numbers)

We optimize for **system throughput**, not isolated kernel peak. The commonly cited targets are:
- Prefill: **50–57K tokens/sec/node**
- Decode: **~22K tokens/sec/node**

These are end-to-end engine numbers: scheduler + cache + comm + kernels + sampling.

## Hard constraints (repo contract)

- **B200 (sm_100a) is the performance target.** No silent downshift.
- **TOML is the config source of truth.** Avoid adding parallel config systems.
- **No duplicate stacks** for the same use-case (one engine path for serving).
- **No internal-only paths/hostnames/runbooks** in tracked files.

## Terminology

- **Prefill:** process prompt tokens, populate KV cache, produce the first next-token distribution.
- **Decode:** iterative 1-token steps using KV cache.
- **Dense vs MoE split:** layers `0..num_dense_layers-1` are dense; `num_dense_layers..num_layers-1` are MoE.
- **Disaggregation:** prefill and decode can run on different replicas; KV cache is transferred between them (NIXL).

---

# Current architecture (as implemented)

## Dataflow overview

```
Client → FastAPI → Orchestrator → Scheduler → Engine.forward_batch → Model.forward
                                                ↘ sampling / token streaming ↗
```

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI (api.py)                           │
│  /v1/chat/completions, /v1/completions, streaming               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AsyncOrchestrator (orchestrator.py)            │
│  Event loop: recv → schedule → forward → process_results        │
│  CPU/GPU overlap mode available                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌───────────────────────┐
│ Scheduler        │ │ KvCache      │ │ Engine                │
│ (scheduler.py)   │ │ (cache.py)   │ │ (engine.py)           │
│                  │ │              │ │                       │
│ • PrefillQueue   │ │ • RadixCache │ │ • DeepSeekV3 model    │
│ • DecodeQueue    │ │ • PageAlloc  │ │ • Sampler             │
│ • ChunkedPrefill │ │ • LRU evict  │ │ • FP8 KV caches       │
│ • Token budget   │ │              │ │                       │
└──────────────────┘ └──────────────┘ └───────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Model (model.py)                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ DsaFlashMla    │  │ MoE            │  │ Dense MLP        │  │
│  │ • Sparse attn  │  │ • MoEGate      │  │ • FP8Linear      │  │
│  │ • DSA indexer  │  │ • Local experts│  │ • First 3 layers │  │
│  │ • FP8 KV pack  │  │ • All-reduce   │  │                  │  │
│  │ • Paged cache  │  │ • 128-align    │  │                  │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component inventory

| Component | File | LOC | Status |
|-----------|------|-----|--------|
| Model (DeepSeek-V3) | model.py | ~930 | Production |
| FlashMLA + DSA | model.py | ~260 | Production |
| FP8 Quantization | model.py | ~150 | Production (UE8M0, SM100) |
| MoE (model-parallel) | model.py | ~200 | Production (all-reduce) |
| KV Cache (paged) | cache.py | ~330 | Production |
| Prefix Cache (radix) | cache.py | ~110 | Production |
| Scheduler | scheduler.py | ~400 | Production |
| Engine | engine.py | ~300 | Production |
| Orchestrator | orchestrator.py | ~300 | Production |
| API (OpenAI) | api.py | ~426 | Production |
| Config/Profiles | config.py | ~250 | Production |
| Checkpoint Loading | ckpt.py | ~400 | Production |
| NIXL Transfer | transfer.py | ~230 | Complete (not integrated) |
| Types | types.py | ~220 | Production |

## Implemented optimizations

| Optimization | Location | Notes |
|--------------|----------|-------|
| FlashMLA sparse attention | model.py | DSA indices, 2K sparse tokens |
| FP8 compute | model.py | DeepGEMM fp8_gemm_nt, grouped GEMM |
| Paged KV cache | cache.py | 64-token blocks, 656 bytes/token |
| Prefix caching | cache.py | Radix tree + LRU eviction |
| Chunked prefill | scheduler.py | Prevents decode starvation |
| Token-budget batching | scheduler.py | Prefill/decode fairness |
| Tensor parallelism | model.py | Column/row parallel + all-reduce |
| 128-alignment padding | model.py | DeepGEMM grouped GEMM requirement |
| CPU/GPU overlap | orchestrator.py | Batch prep while GPU runs |
| Per-request seeding | engine.py | Deterministic sampling |
| Streaming responses | api.py | SSE token streaming |

## Code map (`nmoe/serve` is the source of truth)

### API + entry
- `nmoe/serve/__main__.py`: CLI entry; initializes the orchestrator and HTTP server.
- `nmoe/serve/api.py`: OpenAI-ish endpoints; creates `Request`s and streams results.

### Orchestration + scheduling
- `nmoe/serve/orchestrator.py`
  - Main event loop coordinating scheduler and engine.
  - Has an overlap loop (CPU work overlaps GPU compute).
  - CUDA graph support is **declared but not implemented** (`enable_cuda_graph` is TODO).
- `nmoe/serve/scheduler.py`
  - Token-budget scheduler with chunked prefill and decode queue.
  - Maintains a **page table** mapping request slots → page ids.

### KV cache + prefix cache
- `nmoe/serve/cache.py`
  - GPU KV cache tensor plus CPU-owned metadata.
  - Radix prefix cache with eviction.
  - Current “page/block” unit is **64 tokens**.

### Engine (TP worker process)
- `nmoe/serve/engine.py`
  - Owns model + per-layer KV tensors.
  - Builds `block_table` and passes per-request `cache_seqlens` into the model.
  - Sampling is implemented here (greedy + per-request top-k/top-p).

### Model + kernels integration
- `nmoe/serve/model.py`
  - DeepSeek-shaped model: DSA + FlashMLA decode (FP8 KV cache) + MoE.
  - FP8 KV packing is done via a **pure PyTorch** helper (`nmoe/triton/flashmla_kv.py`) that implements UE8M0-style pow2 scaling.
  - FlashMLA sparse indices follow the standard contract: unused entries are **-1** and metadata is built with `cache_seqlens=topk`.
  - MoE currently has two semantic regimes in the code:
    - Dense layers (`< num_dense_layers`)
    - MoE layers (`>= num_dense_layers`) with shared experts + experts

### Checkpoint loading
- `nmoe/serve/ckpt.py`
  - Loads HuggingFace safetensors and shards for TP.
  - Stacks expert weights into grouped-GEMM-friendly layouts.
  - Enforces UE8M0 pow2 scale constraints where required by the FP8 GEMM path.
  - Canonical offline tooling (download + mp shard conversion):
    - Download HF snapshot to `/data/models/<name>`:
      - `python -m nmoe.serve.ckpt download --repo deepseek-ai/DeepSeek-V3-0324 --out /data/models/DeepSeek-V3-0324`
    - Convert HF → mp shards (one file per rank, includes UE8M0 fixes):
      - `python -m nmoe.serve.ckpt convert-hf-to-mp --hf /data/models/DeepSeek-V3-0324 --out /data/models/DeepSeek-V3-0324-mp8 --world-size 8 --attention-type mla`
    - One-shot download+convert:
      - `python -m nmoe.serve.ckpt download-and-convert --repo deepseek-ai/DeepSeek-V3-0324 --out-hf /data/models/DeepSeek-V3-0324 --out-mp /data/models/DeepSeek-V3-0324-mp8 --world-size 8 --attention-type mla`

### Disaggregation / KV transfer
- `nmoe/serve/config.py`
  - Profiles define output mode + batching + disaggregation mode.
  - `DisaggMode`: `NONE`, `FULL`, `DECODE_ONLY`.
- `nmoe/serve/transfer.py`
  - NIXL-based page transfer (VRAM↔VRAM) primitives and notification plumbing.
  - Intended to be wired into the scheduler/orchestrator for prefill↔decode splits.

### Bench tooling (today)
- `nmoe/serve/benchmark.py`: model-only throughput harness (upper bound; not end-to-end).
- `nmoe/serve/test_e2e.py` and other `test_*.py`: end-to-end and regression utilities used to validate behavior.

---

# Gap analysis vs LMSYS-grade performance

This section is intentionally concrete: it identifies the system-level mechanisms that must exist to hit the numbers and where the current code deviates.

## What's missing (visual)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🔴 CUDA Graph Runner (decode)                           │   │
│  │    • Capture graphs for BS=[1,2,4,8,16,32,64,128,256]   │   │
│  │    • Pre-allocated buffers for replay                   │   │
│  │    • Graph selection based on batch size                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MoE Layer                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🟡 DeepEP Dispatch/Combine (optional, for true EP)      │   │
│  │    • Normal dispatch for prefill                        │   │
│  │    • Low-latency dispatch for decode (CUDA graph safe)  │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🟡 Two-Batch Overlap (optional, prefill only)           │   │
│  │    • Split batch into micro-batches                     │   │
│  │    • Overlap dispatch/compute/combine                   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Not implemented (summary)

| Feature | Impact | Complexity |
|---------|--------|------------|
| **CUDA Graphs** | 10-50× decode | Medium |
| **DeepEP dispatch/combine** | 2-5× decode | High |
| **Two-Batch Overlap (TBO)** | ~30% prefill | High |
| **Disaggregated serving** | Cluster scale | Medium (transfer.py ready) |

## Expected impact of each fix

| Fix | Expected Speedup | Cumulative |
|-----|------------------|------------|
| Batching (BS=64) | 50-100× | 50-100× |
| CUDA Graphs | 10-20× | 500-2000× |
| DeepEP low-latency | 2-3× | 1000-6000× |
| Remove Python sync | 2-5× | 2000-30000× |
| TBO (prefill only) | 1.3× | — |

**Conclusion:** Batching + CUDA graphs alone should get us to ~1000-6000× improvement, close to LMSYS numbers.

## 1) Decode fast path needs CUDA graph replay

**Why:** decode is dominated by CPU launch overhead unless we graph replay stable shapes.

**Current:** `OrchestratorConfig.enable_cuda_graph` exists but is TODO; decode path allocates and builds tensors per step.

**Target:** add a decode execution mode that:
- Pads to a small set of capture sizes (e.g. `[1,2,4,8,16,32,64,128,256]`).
- Uses pre-allocated device buffers (no allocation during decode step).
- Replays captured CUDA graphs keyed by a batch descriptor (vLLM-style) or slot loop (JetStream-style).

## 2) MoE EP must be decode-safe (low-latency, fixed buffers)

**Why:** MoE dominates decode time; EP comm must be graph-compatible and avoid dynamic allocs.

**Current:** MoE code paths exist, and DeepEP `Buffer` is instantiated in engine code, but the end-to-end “decode-safe EP” execution discipline is not yet a clearly isolated fast path.

**Target:** implement/standardize:
- DeepEP low-latency dispatch/combine for decode (fixed buffers; graph-safe).
- A single “MoE execution contract” that is explicit about dense layers, shared experts, and routed experts.
- Grouped GEMM behavior validated for tiny per-expert M (decode regime).

## 3) Remove Python overhead from the hot path

Concrete current hotspots:
- `block_table` construction per batch involves Python loops and per-request tensor creation.
- Sampling has per-request loops for top-k/top-p.
- Some inputs are still threaded via Python lists (e.g., `cache_seqlens_cpu`) and shape-specialization is not enforced for decode.

**Target:** decode step should not do Python work that scales with batch size or tokens.

## 4) Disaggregation must be wired into the control flow (NIXL is already implemented)

**Why:** disaggregation (prefill pool + decode pool) is a primary lever for throughput at cluster scale; KV transfer must be non-blocking and overlapped.

**Current:** NIXL transfer primitives exist in `transfer.py`, profiles support disagg modes, but orchestration logic is not yet end-to-end in the main serving loop.

**Target:** minimal Dynamo-aligned behavior:
- Decode-side allocates KV pages and issues a remote prefill request.
- Prefill-side writes KV pages directly into decode-side allocation (NIXL WRITE) and sends completion notification.
- Decode engine continues serving other requests while awaiting notifications; upon completion, request moves into decode queue.

---

# Benchmarking (how we’ll measure progress)

We need a benchmark protocol that reflects the LMSYS targets:
- **TTFT** (p50/p99) under mixed workloads
- **ITL** (p50/p99) under sustained decode load
- **Prefill tok/s** and **Decode tok/s** at target concurrency
- GPU memory footprint + page cache behavior + prefix hit rate

Rules:
- Model-only benchmarks are allowed as upper bounds, but **pass/fail is engine-level**.
- Every performance change must be accompanied by a measured delta and a stable benchmark recipe.

## Metrics to track

```python
@dataclass
class BenchmarkMetrics:
    # Throughput
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float
    total_tokens_per_sec: float

    # Latency
    ttft_p50_ms: float  # Time to first token
    ttft_p99_ms: float
    tpot_p50_ms: float  # Time per output token (ITL)
    tpot_p99_ms: float

    # Utilization
    gpu_memory_gb: float
    gpu_compute_percent: float

    # Breakdown (decode)
    attention_ms: float
    moe_dispatch_ms: float
    moe_compute_ms: float
    moe_combine_ms: float
    sampling_ms: float
    kernel_launch_overhead_ms: float
```

## Benchmark configurations

| Config | Batch Size | Seq Len | Purpose |
|--------|------------|---------|---------|
| decode_latency | 1 | 1 | Single-token latency |
| decode_throughput | 64 | 1 | Batched decode |
| prefill_short | 32 | 128 | Short prompts |
| prefill_long | 8 | 2048 | Long prompts |
| mixed | 32 prefill + 64 decode | — | Realistic workload |

## Profiling commands

```bash
# CUDA profiling
nsys profile -o profile_decode torchrun --nproc_per_node=8 -m nmoe.serve.benchmark --mode decode

# Memory profiling
python -m torch.cuda.memory._record_memory_history -m nmoe.serve.benchmark

# Kernel timing (sync mode for accurate per-kernel timing)
CUDA_LAUNCH_BLOCKING=1 python -m nmoe.serve.benchmark --mode decode --batch_size 1
```

---

# Implementation plan (repo-aligned, minimal, measurable)

1. **Lock down baselines**: define the benchmark matrix and record current numbers (prefill tok/s, decode tok/s, TTFT/ITL).
2. **Decode CUDA graphs**: implement capture/replay and enforce stable decode shapes.
3. **Decode-safe MoE EP**: integrate low-latency dispatch/combine and validate grouped-GEMM behavior for tiny per-expert M.
4. **Cut Python hot path overhead**: block-table, sampling, and batch materialization.
5. **Wire disaggregation**: connect `transfer.py` into scheduler/orchestrator with non-blocking transfers and notifications.
6. **Compare against peers**: run the same benchmark matrix on sglang/vLLM and report deltas.

## Code sketches

### CUDA Graph Runner (engine.py addition)

```python
class CUDAGraphRunner:
    """CUDA graph capture and replay for decode batches."""

    def __init__(self, model: DeepSeekV3, max_batch_size: int = 256):
        self.model = model
        self.graph_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.graph_pool: Optional[torch.cuda.graph_pool_handle] = None

        # Pre-allocated buffers (largest size)
        self.input_ids = torch.empty(max_batch_size, 1, dtype=torch.long, device='cuda')
        self.positions = torch.empty(max_batch_size, 1, dtype=torch.long, device='cuda')
        self.out_loc = torch.empty(max_batch_size, 1, dtype=torch.int32, device='cuda')
        self.output_logits = torch.empty(max_batch_size, model.vocab_size, dtype=torch.bfloat16, device='cuda')

    def capture(self):
        """Capture CUDA graphs for all batch sizes (largest first)."""
        for bs in reversed(self.graph_batch_sizes):
            # Warmup
            self._run_decode(bs)
            torch.cuda.synchronize()

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.graph_pool):
                self.output_logits[:bs] = self._run_decode(bs)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph

    def run(self, batch: Batch) -> torch.Tensor:
        """Run decode using captured graph."""
        bs = len(batch.reqs)

        # Find smallest graph that fits
        graph_bs = next((s for s in self.graph_batch_sizes if s >= bs), None)
        if graph_bs is None:
            return self._run_decode_eager(batch)  # Fallback

        # Update pre-allocated buffers (no allocation!)
        self.input_ids[:bs].copy_(batch.input_ids)
        self.positions[:bs].copy_(batch.positions)
        self.out_loc[:bs].copy_(batch.out_loc)

        # Replay graph
        self.graphs[graph_bs].replay()
        return self.output_logits[:bs].clone()
```

### DeepEP Low-Latency MoE (model.py addition)

```python
def _forward_deepep_low_latency(self, x: torch.Tensor) -> torch.Tensor:
    """Low-latency dispatch for decode - CUDA graph compatible."""
    # Router
    router_logits = self.gate(x)
    topk_weights, topk_indices = self._route(router_logits)

    # DeepEP low-latency dispatch (fixed buffers, no dynamic alloc)
    recv_x, handle, recv_w, num_recv = self.buffer.low_latency_dispatch(
        x, topk_indices.to(torch.int32), topk_weights,
        num_experts=self.num_experts,
        config=self.low_latency_config,
    )

    # Compute local experts (with 128-alignment)
    if recv_x.numel() > 0:
        expert_out = self._compute_experts_padded(recv_x, num_recv)
    else:
        expert_out = recv_x

    # DeepEP combine
    output, _, _ = self.buffer.low_latency_combine(expert_out, handle, recv_w)

    # Add shared experts
    return output + self.shared(x)
```

## File changes summary

| File | Changes | Priority |
|------|---------|----------|
| `engine.py` | Add CUDAGraphRunner class | P0 |
| `orchestrator.py` | Wire CUDA graphs for decode | P0 |
| `benchmark.py` | Add batched benchmarks, profiling | P0 |
| `model.py` | Add DeepEP low-latency dispatch option | P1 |
| `config.py` | Add `cuda_graph`, `deepep_dispatch` flags | P1 |

## Success criteria

| Milestone | Prefill tok/s | Decode tok/s | Status |
|-----------|---------------|--------------|--------|
| Current baseline | TBD | TBD | Measure |
| After CUDA graphs | 5,000+ | 5,000+ | — |
| After DeepEP dispatch | 10,000+ | 10,000+ | — |
| Target (LMSYS) | 50,000+ | 22,000+ | Goal |

---

# Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CUDA graph capture fails for MoE | Medium | High | Fallback to eager, profile why |
| DeepEP low-latency mode incompatible | Low | Medium | Keep all-reduce path |
| Memory fragmentation with graphs | Medium | Medium | Tune graph pool, monitor |
| Benchmark methodology error | High | High | Validate against known good impl |

---

# Non-goals (for this doc)

- A second, parallel serving stack.
- Alternative config formats.
- Broad multi-arch support on the perf-critical path.
