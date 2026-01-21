# nmoe.serve Testing (Correctness + Performance)

This repo’s supported inference path is **container-first** and **torchrun-first**
for single-node EP serving. All tests below assume a single node with **8×B200**
(`world_size=8`, `tp=1`, `ep=8`, DP ownership across ranks).

Launch invariant (perf point): **global decode batch size is `BS=256` sequences per node**.

## TODO

- Add `fastapi` + `uvicorn` to `docker/Dockerfile.infer` so `nmoe.serve.test_api_streaming` passes in a fresh inference image (no manual `uv pip install`).

## Environment prelude (inside pod/container)

```bash
cd /workspace/nmoe
source .venv/bin/activate
export PYTHONPATH=/workspace/nmoe/third_party:$PYTHONPATH
export NMOE_MODEL_PATH=/data/models/DeepSeek-V3-0324-ep8-tp1
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Pick a unique port per run:

```bash
export MASTER_PORT=29530
```

## Correctness (must-pass)

DeepEP MoE parity vs local reference:

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_moe_deepep_vs_localmask
```

Dynamic disaggregation invariants (T=0 does not deadlock):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_zero_token_dispatch
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_dynamic_disagg_advanced
```

End-to-end generation sanity (TP=1, EP-only):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_generation_multi
```

Profiles contract harness (rank0-driven, T=0 participation on other ranks):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_profiles_correctness
NMOE_DEEPEP_LOW_LATENCY=1 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_profiles_correctness
```

DP=8 ownership control plane (rank0 assigns non-zero owner; streaming updates; cancel/error paths):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_control_plane_dp8
```

Teacher-forcing logprobs (RL/distill primitive; forced tokens + per-token logprobs):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_teacher_forcing_logprobs
```

Notes:
- `NMOE_TF_LOGPROB_TOL` controls the tolerated max-abs drift when comparing a sampled run vs a forced run (default `0.25`).
- `NMOE_RL_LOGPROB_TOL` controls the tolerated max-abs drift for `rl_sample` determinism inside `test_profiles_correctness` (default `0.2`).

## Performance (benchmarks; no pass/fail)

LMSYS-style throughput point:

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_lmsys_e2e \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla \
  --decode-batch-size 256 --decode-ctx-len 2000 --output-len 100 \
  --num-pages 4096 --page-size 64 --max-seq-len 32768 --max-batch-size 256 \
  --max-prefill-tokens 16384 --disable-prefix-cache --disable-chunked-prefill
```

Mixed-load latency tails (TTFT/ITL percentiles measured on rank0 token arrival):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_mixed_latency \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla \
  --concurrency 256 --prompt-len 2000 --output-len 100 \
  --prefill-backlog 0 --probe-rps 0.0 --duration-s 30
```

## Long-context smoke (128k–161k)

This is intentionally *not* a per-commit test: it is expensive. It is the
canonical way to validate that we can prefill a long prompt and decode a small
number of tokens without deadlock or non-finite outputs.

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_long_context_smoke \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla \
  --prompt-len 128000 --output-len 16 \
  --num-pages 4096 --page-size 64 --max-seq-len 163840 --max-batch-size 1 \
  --disable-prefix-cache --chunk-size 2048 --max-prefill-tokens 16384
```
