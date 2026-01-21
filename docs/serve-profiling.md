# nmoe.serve Profiling Runbook (Single Node, 8×B200)

This runbook captures the **canonical** commands we use to profile `nmoe.serve`
on the launch configuration:

- `world_size=8`, `tp=1`, `ep=8` (experts sharded), dense layers replicated
- Launch invariant: **global decode batch size is `BS=256` sequences per node**
- Dynamic disaggregation invariants (lockstep prefill/decode steps, `T=0` ok)

It is intentionally **container-first** and **torchrun-first**.

## Environment prelude (inside pod/container)

```bash
cd /workspace/nmoe
source .venv/bin/activate
export PYTHONPATH=/workspace/nmoe/third_party:$PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NMOE_MODEL_PATH=/data/models/DeepSeek-V3-0324-ep8-tp1
export MASTER_PORT=29530  # pick a unique port per run
```

Key toggles:

- `NMOE_DEEPEP_LOW_LATENCY=1` enables DeepEP low-latency dispatch/combine in decode.
- `NMOE_PROFILE_DECODE=1` enables a torch.profiler decode section in `benchmark_lmsys_e2e`.

## 1) Throughput benchmark (LMSYS-style point)

This is the primary “node tok/s” benchmark point (see `docs/serve-targets.md`).

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_lmsys_e2e \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla --mode all \
  --decode-batch-size 256 --decode-ctx-len 2000 --output-len 100 \
  --num-pages 4096 --page-size 64 --max-seq-len 32768 --max-batch-size 256 \
  --max-prefill-tokens 16384 --disable-prefix-cache --disable-chunked-prefill
```

Recommended matrix:

- DeepEP mode: `NMOE_DEEPEP_LOW_LATENCY=0` vs `NMOE_DEEPEP_LOW_LATENCY=1`
- Attention type: `--attention-type mla` (launch), then `dsa` as fast-follow

## 2) Torch profiler (decode-only breakdown)

Use this when you need counts and host-vs-device attribution:

- kernel launch count (`cudaLaunchKernel`, `cuLaunchKernelEx`)
- CPU dtype moves (`aten::to`, `aten::_to_copy`, `aten::copy_`)
- DeepEP kernels (`deep_ep::*`)

```bash
NMOE_PROFILE_DECODE=1 torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_lmsys_e2e \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla --mode decode \
  --decode-batch-size 256 --decode-ctx-len 2000 --output-len 32 \
  --num-pages 4096 --page-size 64 --max-seq-len 32768 --max-batch-size 256 \
  --max-prefill-tokens 16384 --disable-prefix-cache --disable-chunked-prefill
```

Notes:

- This prints a short table on rank0. It is not a JSON trace exporter; use Nsight
  for timeline-level diagnosis.
- Profiler overhead is non-trivial; use it for attribution, not headline tok/s.

## 2.1) Transport microbenches (DeepEP + RDEP barriers)

These are **microbenchmarks**, intended to isolate transport/barrier overheads
without running the full model.

DeepEP dispatch+combine transport (normal mode) at the launch decode point
(`BS=256` global, `T_local=32`):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_transport_microbench \
  --mode normal --decode-batch-size 256 --hidden-size 7168 --num-experts 256 --topk 8 \
  --routing random --warmup-iters 20 --iters 200
```

RDEP IPC barrier A/B (phase vs tag, plus graph replay on tag barrier):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_rdep_barrier_ab
```

Notes:
- The RDEP barrier microbench requires **single-node IPC mode** (`world_size == local_world_size`).
- The phase barrier is **not graph-replay-safe** as implemented (host-controlled monotonic phase).
  The tag barrier is graph-replay-safe and is the intended capture candidate.

## 3) Nsight Systems (timeline-level diagnosis)

Use Nsight Systems when you need to answer questions like:

- Are DeepEP kernels **waiting** (stream sync / protocol), or actually moving data?
- Where are the gaps between attention → MoE dispatch → GEMMs → combine?
- Is the step gated on an unexpected collective?

Example (run in a pod; adjust `<POD>`):

```bash
kubectl exec <POD> -- bash -lc '
set -euo pipefail
cd /workspace/nmoe
source .venv/bin/activate
export PYTHONPATH=/workspace/nmoe/third_party:$PYTHONPATH
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NMOE_DEEPEP_LOW_LATENCY=1
export MASTER_PORT=29530

nsys profile --force-overwrite=true --sample=none --trace=cuda,cublas,osrt \
  --duration=15 -o /tmp/nsys_decode \
  torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_lmsys_e2e \
    --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla --mode decode \
    --decode-batch-size 256 --decode-ctx-len 2000 --output-len 32 \
    --num-pages 4096 --disable-prefix-cache --disable-chunked-prefill
'
kubectl cp <POD>:/tmp/nsys_decode.nsys-rep .
```

What to look for first:

- `deep_ep::*dispatch*` and `deep_ep::*combine*` per-layer cost
- “mode flag” reductions / barriers that gate decode steps
- kernel launch fan-out (thousands of tiny kernels per step)

## 4) Long-context stability (128k–161k)

This is a **correctness smoke** (not a performance benchmark):

```bash
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.benchmark_long_context_smoke \
  --ckpt /data/models/DeepSeek-V3-0324-ep8-tp1 --attention-type mla \
  --prompt-len 128000 --output-len 16 \
  --num-pages 4096 --page-size 64 --max-seq-len 163840 --max-batch-size 1 \
  --disable-prefix-cache --chunk-size 2048 --max-prefill-tokens 16384
```
