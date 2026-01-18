# Inference-RDEP v0 (Single-Node EP Transport)

This document proposes an inference-specific EP transport (“inference-RDEP”) for `nmoe.serve` that targets **single-node (8×GPU) TP=1 / EP=8 / world_size=8**. The primary motivation is decode performance: after CUDA-graph replay removes CPU launch overhead, **EP transport becomes a dominant cost**.

## Motivation (Why this exists)

We can reproduce a graph-replay decode point of roughly:

- `BS=256, ctx=2000, output_len=32` → ~`2.9k tok/s/node` (~`86ms/step`).

At that point, profiling shows that DeepEP intranode kernels (`notify/dispatch/combine`) are ~**40%** of GPU kernel time in the replay window. Converting that to a step budget:

- `~0.40 × 86ms ≈ 34ms/step` is currently spent inside EP transport/protocol.

To approach an LMSYS-like `~11ms/step` throughput point on a single node, EP transport must plausibly be **<5ms/step** (order-of-magnitude improvement), and must be **graph-capturable**.

## Goals / Non-goals

**Goals**
- Single-node EP transport for `TP=1 / EP=8 / world_size=8` with **graph-capturable** forward path.
- Hard launch invariant: **global decode batch size is `BS=256` sequences per node**.
- **No host sync in the hot path**: no `cudaStreamSynchronize`, no D2H to compute `M_recv`, no CPU-side control decisions.
- Support dynamic-disagg semantics: per step, all ranks participate; ranks with `T=0` are valid.
- Provide an output layout that feeds the existing fast MoE decode path:
  - grouped expert layout + `masked_m` + deterministic scatter mapping (compatible with masked DeepGEMM and fused pack/scatter).

**Non-goals (v0)**
- Multi-node (NVSHMEM) transport. (We can reuse the hybrid bootstrap pattern later.)
- Backward pass / training. (This is inference-only.)
- A general-purpose collective replacement. This is an EP transport specialized for MoE dispatch/return.
- Prefill optimality. v0 may start decode-first if it accelerates the dominant online bottleneck; prefill can continue using the existing path until extended.

## Relationship to training RDEP

We should reuse training RDEP’s *infrastructure*:
- CUDA IPC shared-buffer allocation and handle exchange (`nmoe/rdep.py`, `nmoe/csrc/rdep.cu`).
- Device-resident pointer tables (`d_buffer_ptrs_*`, `d_barrier_signal_ptrs_*`).
- A **local-spin phase barrier** (`k_ipc_barrier_phase_*`) using sys-scope release/acquire to ensure cross-GPU visibility.

We should not reuse training RDEP’s inference-incompatible pieces:
- Any forward path that computes `M_recv` via `cudaStreamSynchronize` + D2H.
- Any per-dispatch CUB radix sort / prefix sums on dynamic sizes.

## Design overview (v0)

At a high level, inference-RDEP replaces DeepEP’s intranode protocol kernels with:

- **Direct cross-GPU writes** into per-rank IPC buffers (CUDA IPC pointers).
- **A small number of synchronization points** per MoE layer implemented as GPU kernels (graph-capturable), with local-memory polling only (no remote polling storms).
- Fixed-shape buffers per layer/step so CUDA graphs can replay without allocations and without shape-dependent control flow.

### Key definitions

- `world = 8`
- `BS_decode = 256` (global sequences per node; launch requirement)
- `T_cap_decode = BS_decode / world = 32` (fixed per-rank decode cap; `T_r ∈ [0, 32]`)
- `n_local = num_experts / world = 32` (DeepSeek-V3)
- `K = topk = 8`
- `T_r = local tokens on rank r` (decode often `~32`, may be `0`)
- `T_cap` = fixed per-rank token cap for a step (decode v0 uses `T_cap = T_cap_decode`)
- `expected_m` = fixed per-expert row cap for masked grouped GEMM (`expected_m` is per local expert group, not per rank)

## API proposals

We want a minimal surface that can be composed into the current MoE forward.

### Option A: transport-only API (mirrors DeepEP Buffer)

```python
handle = inference_rdep_dispatch(
  x: bf16[T_r, H],
  topk_idx: int32[T_r, K],      # global expert ids
  topk_w: fp32[T_r, K],
  *,
  T_cap: int,
)

# On each rank: compute local expert outputs for received tokens.
y_recv = inference_rdep_local_moe_compute(
  handle,
  *,
  expected_m: int,
  w13, w13_scale, w2, w2_scale,  # local expert weights
)

y = inference_rdep_combine(y_recv, handle)  # returns bf16[T_r, H] for local tokens
```

### Option B: fused MoE forward API (preferred long-term)

```python
y = inference_rdep_moe_forward(
  x: bf16[T_r, H],
  topk_idx: int32[T_r, K],
  topk_w: fp32[T_r, K],
  *,
  T_cap: int,
  expected_m: int,
  w13, w13_scale, w2, w2_scale,
)
```

This option allows the dispatch format to be chosen specifically to match the grouped+masked GEMM path and eliminates intermediate “recv_topk_idx/weights” materializations.

## Buffer layout (single-node IPC)

Inference-RDEP maintains a single shared IPC allocation (as training RDEP does), partitioned into per-rank regions. All pointers are made visible to device code via `cudaMemcpyToSymbol` once at init.

### Per-rank regions (conceptual)

For rank `d` (destination rank), we reserve:

1) **Recv tokens** (written by all source ranks, read by rank `d`)
- `recv_x[d]`: `bf16[world * T_cap, H]` (or `fp8+scale` in a later revision)

2) **Recv routing metadata** (written by all source ranks, read by rank `d`)
We need enough information for rank `d` to compute *all* local expert hits for each received token copy.

Two equivalent encodings:
- **Fixed-K encoding** (simple, fixed-shape):
  - `recv_local_eid[d]`: `int16[world * T_cap, K]` (local expert id 0..n_local-1, or -1)
  - `recv_w[d]`: `fp16/fp32[world * T_cap, K]` (weight, 0 for -1)
- **Bitmask+compact weights** (smaller, more complex):
  - `recv_mask[d]`: `uint32[world * T_cap]` (bitmask over 32 local experts)
  - `recv_w_compact[d]`: `fp16[world * T_cap, K]` + `recv_eid_compact[d]: uint8[...]`

v0 favors the fixed-K encoding because it is graph-friendly and avoids variable-length parsing.

3) **Return buffer** (written by rank `d`, read by each source rank)
We want rank `d` to return the aggregated per-token contribution back to the owning source rank (the rank that provided that token).

- `ret_y[d]`: `bf16[world * T_cap, H]`

4) **Sync signals**
- `phase_signal[d]`: `int32[world]` (one slot per source rank) used by `k_ipc_barrier_phase_*`.

### Deterministic slot mapping (no remote atomics)

For a token `t` on source rank `s` (0 ≤ t < T_cap), define:

- `slot = s * T_cap + t`  (0 ≤ slot < world*T_cap)

This mapping is:
- fixed-shape,
- graph-friendly,
- and eliminates the need for remote atomic allocation for decode.

## Step phases (per MoE layer)

For each MoE layer, each rank executes the same sequence (valid for `T=0`):

1) **Dispatch write (sender-side)**
Each source rank `s` writes:
- `recv_x[d][slot] = x_s[t]` for each destination rank `d` that has ≥1 local expert hit for that token.
- `recv_local_eid[d][slot, :]` and `recv_w[d][slot, :]` containing the subset of top-k experts that belong to `d` (local ids 0..31), padded with `-1/0`.

2) **Dispatch barrier**
Call `ipc_barrier_phase_*` so all ranks observe that all dispatch writes are globally visible.

3) **Local expert compute (receiver-side)**
Rank `d` reads its `recv_*[d]` arrays and produces:
- grouped packed activations per local expert (`[n_local, expected_m, H]`) and `masked_m`
- masked grouped GEMMs (gate/up + down)
- per-token aggregated contribution `y_recv[slot]` (one vector per received token copy)

4) **Return write**
Rank `d` writes `ret_y[d][slot] = y_recv[slot]` for every slot where it had ≥1 local expert hit.

5) **Return barrier**
Call `ipc_barrier_phase_*` again so owners can safely read `ret_y[*][slot]`.

6) **Owner combine**
Each source rank `s` reads, for each local token `t`, all `ret_y[d][slot]` across `d∈[0..world)` and sums only the ranks where it actually routed experts (mask-driven). This produces the final MoE MLP output for that layer.

### Barrier count

v0 uses **two barriers per MoE layer** (after dispatch, after return). This is intentionally simple and correct. Future revisions can pipeline to reduce barriers (e.g., fusing return barrier with next layer’s dispatch barrier) if it is proven safe and beneficial.

## Graph capture considerations

What must be true for capture/replay:
- All buffer pointers are fixed (IPC opened once; no resizing after).
- No host-visible counts (`M_recv`) are used in control flow.
- All kernels operate on fixed extents:
  - `T_cap`, `K`, `H`, `n_local`, `expected_m` are compile- or config-time constants.
- Any “phase” value used by barriers must be safe across replays.

### Barrier strategy for replay

The training RDEP codebase contains two families of barriers:
- an add/sub “tag returns to zero” barrier (`barrier_block`), and
- a monotonic “phase value” barrier (`k_ipc_barrier_phase_*`).

For graph replay, v0 prefers the phase barrier, but it requires a device-resident monotonic counter (incremented on-device) or a per-step reset kernel to avoid reusing the same phase values across replays.

## Failure modes / invariants

**Overflow**
- If `T_r > T_cap`: this is a scheduler violation. In v0 we fail fast (debug) and cap/queue upstream (production).
- If a token routes >K experts: model/router violation (fail fast).
- If `expected_m` is too small for a local expert in the compute phase: fail fast in debug; production must size `expected_m` for worst-case decode skew or implement a deterministic spill path.

**Timeout**
- Barrier kernels have a watchdog timeout. On timeout, we `trap()` to produce an actionable crash rather than silent deadlock.

**Correctness**
- T=0 ranks still participate in both dispatch and return barriers for every layer, maintaining lockstep and avoiding deadlocks.

## Implementation roadmap (v0)

1) **Bootstrap + buffer allocation**
- Reuse training RDEP’s IPC handle exchange and pointer table setup.
- Add an inference-only buffer layout for `recv_x`, `recv_local_eid`, `recv_w`, `ret_y`, and `phase_signal`.

2) **Dispatch kernel (fixed slot mapping)**
- Build per-destination local-eid/weight rows directly (no sort, no CUB).
- Write only to destination ranks that have ≥1 local hit for that token; keep unused destinations’ metadata as “all -1”.

3) **Receiver-side fused “route→pack”**
- A single kernel that reads `recv_local_eid/recv_w` and produces grouped tensors + `masked_m` (mirrors the existing fused pack idea but with a different input format).

4) **Masked grouped GEMM path**
- Reuse the current masked GEMM + fused SiLU + grouped scatter.

5) **Return + owner combine**
- Write fixed-slot returns and sum across ranks with a mask.

6) **Microbench + parity**
- A transport-only microbench that compares DeepEP vs inference-RDEP on decode (`T_cap=32`, `K=8`, `H=7168`) before integrating into the full model stack.
