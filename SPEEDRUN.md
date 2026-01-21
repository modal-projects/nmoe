# nmoe Speedrun Guide

## Quick Reference

| Model | Best Val Loss | Steps | Target | Notes |
|-------|---------------|-------|--------|-------|
| Dense SDPA | 3.4807 | 9536 | 3.28 | baseline |
| MoE-64 bf16 | 3.4319 | 9536 | 3.28 | -0.05 vs dense |
| **Ultra-256 bf16** | **3.2799** | 10752 | **3.28 ✓** | TARGET HIT |
| MoE-64 nvfp4 v2 (no dither) | 4.0351 | 9536 | 3.28 | +0.60 vs bf16 |
| **MoE-64 nvfp4 v3 (dither)** | **3.8272** | 9536 | 3.28 | +0.40 vs bf16 (35% better) |

---

## Pod Setup

### Active Pods (8x B200 each)

```bash
# List running pods
kubectl get pods | grep nmoe-infer-debug

# Current pods:
# - nmoe-infer-debug-9ltnp (primary)
# - nmoe-infer-debug-2-sxhvg (nvfp4 experiments)
# - nmoe-infer-debug-3-bjkd6 (available)
```

### Interactive Access

```bash
kubectl exec -it nmoe-infer-debug-9ltnp -- bash
```

---

## Syncing Code

**IMPORTANT**: Only sync specific changed files. NEVER delete `/workspace/nmoe` - full rebuild takes 45+ minutes.

```bash
# Python files (instant)
kubectl cp nmoe/opt.py nmoe-infer-debug-9ltnp:/workspace/nmoe/nmoe/opt.py
kubectl cp nmoe/train.py nmoe-infer-debug-9ltnp:/workspace/nmoe/nmoe/train.py
kubectl cp nmoe/metrics.py nmoe-infer-debug-9ltnp:/workspace/nmoe/nmoe/metrics.py

# CUDA files (requires rebuild after)
kubectl cp nmoe/csrc/adamw.cu nmoe-infer-debug-9ltnp:/workspace/nmoe/nmoe/csrc/adamw.cu

# Configs
kubectl cp configs/speedrun/small_moe.toml nmoe-infer-debug-9ltnp:/workspace/nmoe/configs/speedrun/small_moe.toml
```

### Rebuild CUDA Extensions (if csrc/ changed)

```bash
kubectl exec nmoe-infer-debug-9ltnp -- bash -c 'cd /workspace/nmoe && source .venv/bin/activate && make -C nmoe/csrc clean && make -C nmoe/csrc -j8'
```

---

## Running Training

```bash
# On pod - setup environment
cd /workspace/nmoe
source .venv/bin/activate
export PYTHONPATH=/workspace/nmoe/third_party:/workspace/nmoe/third_party/quack:$PYTHONPATH

# Kill existing runs
pkill -9 -f torchrun; pkill -9 -f nmoe.train

# Start training (background)
nohup torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe.toml > /tmp/train.log 2>&1 &

# Monitor
tail -f /tmp/train.log
```

### From kubectl (one-liner)

```bash
kubectl exec nmoe-infer-debug-9ltnp -- bash -c 'cd /workspace/nmoe && source .venv/bin/activate && export PYTHONPATH=/workspace/nmoe/third_party:/workspace/nmoe/third_party/quack:$PYTHONPATH && nohup torchrun --nproc_per_node=8 -m nmoe.train configs/speedrun/small_moe.toml > /tmp/train.log 2>&1 &'
```

---

## Monitoring

```bash
# Check if running
kubectl exec nmoe-infer-debug-9ltnp -- ps aux | grep -E 'torchrun|nmoe.train'

# View logs
kubectl exec nmoe-infer-debug-9ltnp -- tail -30 /tmp/train.log

# GPU utilization
kubectl exec nmoe-infer-debug-9ltnp -- nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

---

## Querying Metrics (DuckDB)

Metrics logged to `/data/metrics/run_*/rank_0.duckdb`. **DB is locked during training** - copy first:

```bash
# On pod
RUN_DIR=$(ls -td /data/metrics/run_* | head -1)
mkdir -p /tmp/m
cp $RUN_DIR/rank_0.duckdb /tmp/m/
cp $RUN_DIR/rank_0.duckdb.wal /tmp/m/

# Query
python -c "
import duckdb
con = duckdb.connect('/tmp/m/rank_0.duckdb', read_only=True)
# Flip rates
for row in con.execute(\"\"\"
    SELECT step, value FROM metrics
    WHERE tag LIKE '%flip_frac%'
    ORDER BY step DESC LIMIT 10
\"\"\").fetchall():
    print(row)
"
```

---

## Config Reference

### June 2024 Calibrated Settings

All speedrun configs use:
- `steps=9536`, `batch_size=256`, `seq_len=2048`
- `lr=1.8e-3`, `wd=0.1`
- `warmup=256`, `warmdown=2048`
- `attn="sdpa"` (MLA unstable at high LR)

### Key Files

| Config | Description |
|--------|-------------|
| `configs/speedrun/small_dense_sdpa.toml` | Dense baseline |
| `configs/speedrun/small_moe.toml` | MoE-64 (nvfp4 default) |
| `configs/speedrun/small_moe_ultra.toml` | Ultra-256 (bf16, 32800 steps) |

---

## NVFP4 Notes

### The Problem
NVFP4 without dither shows ~0.6 loss gap vs bf16 due to "code stickiness" - RTN quantization traps weights in bins.

### The Solution: Resonant Dither
Implemented in `nmoe/csrc/adamw.cu`. When quantized code would be unchanged (`new == old`), applies deterministic symmetric dither and re-quantizes.

**Results:** Gap reduced from 0.60 → 0.40 (35% improvement)

### Flip Rate Comparison
```
Step   v2 (no dither)   v3 (dither)
 800      2.2% ↓          7.9%
1800      1.0% ↓          7.2%
6000      0.3% ↓          6.0%
9536      0.2% ↓          5.5% stable ✓
```

### Tuning Constants (in adamw.cu)
```cpp
NVFP4_DITHER_AMPL = 0.25f      // Dither amplitude in normalized FP4 domain
NVFP4_DITHER_PROB_MASK = 0x1u  // ~50% of stuck codes get dithered
```
To close remaining gap, try increasing amplitude to 0.35 or mask to 0x0 (100%).

---

## Troubleshooting

### "pybind11/pybind11.h not found" during build
Activate venv first: `source .venv/bin/activate`

### DuckDB "Table metrics does not exist"
Copy the `.wal` file along with the `.duckdb` file.

### Training hangs at start
Check if previous run is still running: `ps aux | grep python`

### OOM during backward
Reduce `batch_size` in config (192 is safe, 512 is max on B200).
