# nmoe.serve Targets (LMSYS-aligned)

This document defines the first production targets for `nmoe.serve` and how we
claim “match LMSYS” in a way that is measurable and internally consistent.

It intentionally separates:

1. **Correctness requirements** (DeepEP / dynamic disaggregation invariants)
2. **Throughput targets** (LMSYS-style node throughput)
3. **Latency SLOs** (TTFT/ITL at a defined concurrency point)

## Terms and Metrics

### Tokens/sec (throughput)

- **Prefill tok/s**: prompt tokens processed during *prefill*.
- **Decode tok/s**: generated tokens processed during *decode* (steady-state).

These are *aggregate per node* metrics (sum over all GPUs on the node).

### TTFT (Time To First Token)

For a streaming request:

- **TTFT** is the wall time from request arrival to the first generated token
  becoming available for streaming.

TTFT depends heavily on prompt length and batching/scheduling.

### ITL (Inter-Token Latency)

For a streaming request in steady-state decode:

- **ITL** is the wall time between consecutive streamed tokens.

ITL depends heavily on decode batching and any interference from prefill.

### “Concurrency C” vs “Decode Batch Size”

These are not interchangeable:

- **Concurrency `C`**: number of simultaneously active streaming sequences.
- **Decode batch size**: number of sequences processed in one decode step.

In a real server, the decode batch is determined by the scheduler from the
concurrency and per-request readiness. A throughput benchmark typically fixes
the decode batch size directly to measure the engine’s capacity.

## What “Match LMSYS” Means (v1)

We adopt the LMSYS-style per-node throughput benchmark point:

- **Input length**: 2000 tokens
- **Output length**: 100 tokens
- **Decode batch size**: fixed global batch size per node

In this repo, the corresponding benchmark harness is:

- `nmoe/serve/benchmark_lmsys_e2e.py` (throughput)

## Targets (Single Node, 8×B200)

## Context Length Support (Correctness)

DeepSeek V3 family models support long context windows (128k–161k depending on
version). `nmoe.serve` must be able to run long prompts + short decode without
deadlock or non-finite activations (even if slow).

This is bounded by:
- `max_seq_len` (serving-time hard limit)
- KV page budget: `num_pages * page_size` (per-owner-rank capacity)

Use `nmoe/serve/benchmark_long_context_smoke.py` to validate 128k–161k on real
hardware.

### A) Throughput Targets (LMSYS benchmark point)

- **Prefill throughput**: ≥ **60,000 prompt tok/s per node**
- **Decode throughput**: ≥ **25,000 decode tok/s per node**

Reference points:

- LMSYS DeepSeek deployment (8×H100): **52.3k input tok/s/node**, **22.3k output tok/s/node**.
- LMSYS Kimi K2 deployment (8×H200): **56k prefill tok/s/node**, **24k decode tok/s/node**.

### B) Streaming Latency SLOs (Interactive anchor point)

Anchor point:

- **Concurrency**: `C = 256` active decode sequences
- **Prompt length**: `P = 2000` tokens
- **Output length**: `N = 100` tokens

Targets:

- **TTFT p99 ≤ 2.0s** at `(C=256, P=2000, N=100)`
- **ITL p99 ≤ 100ms** at `(C=256)` in steady-state decode

Guardrail (long prompts; initial bar, tighten later):

- **TTFT p99 ≤ 8.0s** at `(C=256, P=32k, N=100)`

### C) Correctness Requirements (Dynamic Disaggregation / DeepEP)

Dynamic disaggregation is a correctness contract, independent of latency goals:

- Every **prefill step** is globally synchronous across ranks.
- Every **decode step** is globally synchronous across ranks.
- Ranks with no local work still participate in the step with **T=0** so DeepEP
  dispatch/combine collective ordering stays aligned.

This is validated by:

- `nmoe/serve/test_zero_token_dispatch.py`
- `nmoe/serve/test_dynamic_disagg_advanced.py`

## How to Measure

See `docs/serve-testing.md` for canonical commands.
