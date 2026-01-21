# Architecture Ablations (Physics Harness)

> **Results**: See [PHYSICS_ABLATION_RESULTS.md](PHYSICS_ABLATION_RESULTS.md) for aggregated 3-seed results using PhysicsLM4-faithful evaluation.

This document defines a **staged ablation program** for the emerging "next architecture" we see converging across:
- DeepSeek (e.g. n‑gram memory / manifold‑constrained residual composition / local attention variants)
- Google/Gemma‑3n (AltUp, Per‑Layer Inputs, nested‑width MatFormer)
- Independent work (Canon layers, SWA/local mixers)

We frame everything through the continual learning lens:
1) **Memory substrate (read)**: where new information can live without rewriting the backbone.
2) **Consolidation/write rule**: how information migrates from read‑time memory into weights.
3) **Stability constraints**: how we prevent drift/forgetting.
4) **Measurement**: how we tell we’re making progress (and detect failure modes).

This phase is **physics + docs only**: micro‑scale, deterministic, mechanism‑isolating experiments.

---

## Scope and Non‑Goals

**In scope (now):**
- A **single runnable** physics harness that can train + evaluate a matrix of architecture variants.
- A matrix that covers the key axes: **width**, **residual/composition**, **memory**, **local mixing**.
- Metrics that specifically quantify **conditionality** (avoid always‑on brittle memory paths).

**Out of scope (later phases):**
- Production integration into `nmoe/model.py`.
- Full‑scale tokens/s benchmarking.
- The full “ICL → weights” (adapter consolidation) pipeline (tracked separately).

---

## Axes (“Map”) → Operational Knobs

We keep the surface small by mapping each conceptual axis to one knob in the physics runner.

### 1) Width (MatFormer‑style)
**Goal**: dynamically allocate *more* width only when needed.

Physics proxy:
- A nested‑width MLP with two widths: `hidden_small < hidden_large`.
- A per‑token gate `p_large(x)` that mixes the outputs:  
  `mlp(x) = (1-p)*mlp_small(x) + p*mlp_large(x)`.
- We track `mean(p_large)` as a compute‑allocation proxy.

### 2) Residual / Composition Law (AltUp / mHC)
**Goal**: improve information propagation and stability via better composition, not “extra hacks”.

Physics proxies:
- `residual=vanilla`: standard pre‑norm residual.
- `residual=altup`: multi‑stream predict/correct mixing (Gemma‑style) wrapped around each block.
- `residual=mhc`: multi‑stream hyper‑connections with Sinkhorn‑projected mixing (doubly‑stochastic).

### 3) Local Preconditioner (Canon)
**Goal**: add a cheap local “state preconditioner” that stacks cleanly with other mechanisms.

Physics proxy:
- `precond=canon`: add a short depthwise causal conv residual (Canon A/C‑style) as a local stabilizer.
- Back-compat shorthand: `residual=canon` means `residual=vanilla,precond=canon`.

### 4) Memory (Engram vs PLE+Ngrammer)
**Goal**: add an explicit read primitive that is **addressable** and **conditional**.

Physics proxies:
- `memory=none`
- `memory=engram`: hashed bigram embedding + hidden‑state gate (context‑aware conditional memory)
- `memory=ple_ngrammer`: hashed bigram embedding projected per layer (addressable but no hidden‑state gate)

### 5) Local Mixing (SWA‑style)
**Goal**: exploit chunk/locality primitives (local attention / conv) to improve representations.

Physics proxy:
- `attn=global` vs `attn=local(window=W)` sliding causal attention.
- `attn=converged(window=W)`: a learned per-token router between a **global** causal attention path and a **local** causal attention path with a *soft, learnable window* (window is per-head and differentiable).
  - Intended as a physics-harness proxy for a “converged NSA”: keep one local primitive + one global primitive, and learn how to mix them.
  - Diagnostics (slice-level, not per-token attribution):
    - `attn_local_gate_mean_by_layer`: mean local fraction per layer (0..1).
    - `attn_window_frac_mean_by_layer`: mean window fraction `w/T` per layer (0..1).

---

## Primary Metrics (CL‑Aligned)

All variants must report the same metrics schema.

### Task performance
- `train.loss`, `valid.loss`
- `train.answer_token_acc`, `valid.answer_token_acc`
- `train.answer_exact_match`, `valid.answer_exact_match` (where applicable)

### Conditionality (memory correctness under “should use memory” vs “should not”)
This is the **critical measurement** for memory designs.

For datasets that include a conditionality split (e.g. a mixed/noise `ngram` variant):
- Accuracy on “structured” vs “noise” regions/splits.
- Memory usage stats stratified by split:
  - Engram: `gate_mean` (should drop on noise)
  - PLE: `delta_rms` (should not stay high on noise unless it’s harmless)

We also support a cleaner prompt-level conditionality test:
- `ngram_polysemy`: memory is useful (disambiguation signal exists).
- `ngram_scrambled`: continuation is independent of the prefix, so hashed n-gram memory is useless.

If the memory system is behaving conditionally, Engram’s internal `gate_mean` should be lower on
`ngram_scrambled` than on `ngram_polysemy`.

### Compute allocation proxies
- MatFormer: `p_large_mean`, optionally histogram/quantiles.
- AltUp/mHC: coefficient norms / gain diagnostics (where available).

---

## Experiment Funnel (Staged)

We run a staged funnel to avoid combinatorial explosion.

### Stage 0: Baselines
- `BASE`: vanilla residual, no memory, global attention, fixed width.

### Stage 1: Single‑axis winners
- Width: `matformer` vs `fixed` (compute proxy matched via `p_large_mean` target or regularization later).
- Residual: `altup`, `mhc`, `canon` vs vanilla.
- Memory: `engram` vs `ple_ngrammer` vs none on:
  - a memory‑favoring task (e.g. `ngram`)
  - a conditionality task (mixed/noise `ngram` variant)
- Local mixing: `local(window=W)` vs global.

### Stage 2: Pairwise interactions (only winners)
- `memory × residual`
- `memory × width`
- `residual × width`

### Stage 3: Full candidate
Combine winners into one “candidate architecture” and re‑run the full physics suite.

---

## Minimal Matrix (what we actually schedule first)

Start with Stage 0 + Stage 1:

| ID | width | residual | precond | memory | attn |
|---:|:------|:---------|:-------|:-----|
| BASE | fixed | vanilla | none | none | global |
| W | matformer | vanilla | none | none | global |
| R_altup | fixed | altup | none | none | global |
| R_mhc | fixed | mhc | none | none | global |
| P_canon | fixed | vanilla | canon | none | global |
| M_engram | fixed | vanilla | none | engram | global |
| M_ple | fixed | vanilla | none | ple_ngrammer | global |
| A_local | fixed | vanilla | none | none | local(window=64) |

Then promote winners to Stage 2.

---

## How to Run (Physics)

Single matrix run (writes per‑variant logs + a summary JSON):
```bash
python -m nmoe.research.physics.arch_ablations --output /tmp/arch_ablations --steps 2000 --matrix stage1
```

### Attention compare (global vs local vs converged router)
```bash
python -m nmoe.research.physics.arch_ablations \
  --output /tmp/attn_compare --steps 2000 --matrix attn_compare \
  --slice-metrics --slice-metrics-n 512

python -m nmoe.research.physics.viz_slices --runs /tmp/attn_compare
```
This writes `analysis/viz/` SVGs, including slice curves and (when available) heatmaps for:
- memory gate means by layer (`mem_gate_heatmap_*.svg`)
- converged attention local-gate and window fraction by layer (`attn_converged_*_heatmap_*.svg`)
- per-slice LogitScope curves (per-layer CE) when `--layer-ce` is also enabled (`slices_layer_ce_*_*.svg`)

### Canon × attention redundancy test (`canon_attn_cross`)
This matrix is designed to answer one question:
**does local mixing inside attention make Canon-A redundant?**

It fixes `residual=mhc` and `memory=engram`, and varies:
- `canon_set ∈ {B, BC, BCD, ABCD}` (ABCD differs from BCD only by A)
- `attn ∈ {global, local:64, converged:64}`

Recommended run (use `ngram_polysemy` + `ngram_scrambled` so Engram’s conditionality is actually exercised):
```bash
python -m nmoe.research.physics.arch_ablations \
  --output /tmp/canon_attn_cross --steps 2000 --matrix canon_attn_cross \
  --slice-metrics --slice-metrics-n 512 \
  --tasks \
    ngram_polysemy:1.0:n_symbols=512,n_steps=128,table_seed=0 \
    ngram_scrambled:1.0:n_symbols=512,n_steps=128,table_seed=0 \
    depo:1.0:n_entities=6,max_hops=4 \
    mano:0.5:depth=2,ops=asm \
    mano:0.5:depth=6,ops=asm

python -m nmoe.research.physics.viz_slices --runs /tmp/canon_attn_cross
```
Interpretation:
- Within each attention kind: compare `canon_set=BCD` vs `canon_set=ABCD`.
  If they match, Canon-A isn’t doing useful work in that attention regime.

### LogitLens diagnostics (Engram-style evidence)

To reproduce the Engram-style “prediction convergence” evidence, run with `--logitlens` to compute
layer-wise `KL(p_final || p_layer)` on a subset of the validation set (per task):
```bash
python -m nmoe.research.physics.arch_ablations --output /tmp/arch_ablations --steps 2000 --matrix stage1 --logitlens --logitlens-n 256
```

This writes `analysis/logitlens_valid.json` in the output directory. Render publication-friendly SVG heatmaps:
```bash
python -m nmoe.research.physics.viz_logitlens --runs /tmp/arch_ablations
```

### CKA layer alignment (Engram-style heatmap evidence)

To reproduce Engram’s **layer alignment heatmap** evidence, run with `--cka` to compute **linear CKA**
between each variant’s layer representations and the baseline model’s layers (per task, on a validation subset):
```bash
python -m nmoe.research.physics.arch_ablations --output /tmp/arch_ablations --steps 2000 --matrix stage1 --cka --cka-n 256
```

This writes `analysis/cka_valid.json`. Render publication-friendly SVG heatmaps:
```bash
python -m nmoe.research.physics.viz_cka --runs /tmp/arch_ablations --mark-max
```

Single variant run:
```bash
python -m nmoe.research.physics.arch_ablations --output /tmp/arch_one --steps 2000 \
  --variant width=matformer,residual=altup,precond=canon,memory=engram,attn=local:64
```

Artifacts:
- `runs.json` (variant → paths)
- `runs/<variant>/train.jsonl` (step logs)
- `analysis/summary.json` (final metrics table)

---

## New Diagnostic Flags (Paper-Faithful Evaluation)

### Per-Layer CE to Ground Truth (`--layer-ce`)

The decisive metric for "are late layers doing useful work?":

```bash
python -m nmoe.research.physics.arch_ablations \
  --output /tmp/eval --matrix stage1 --layer-ce --layer-ce-n 256
```

### Lano-cfg DP KL (`--lano-cfg-kl`)

For `lano_cfg`, we can compute an **exact** next-token distribution from the CFG via DP
(PhysicsLM4-style) and report:
- `dp_kl`: mean KL(target_DP || model) across positions

```bash
python -m nmoe.research.physics.arch_ablations \
  --output /tmp/lano_eval \
  --steps 2000 \
  --tasks "lano_cfg:1.0:graph_seed=0,depth=6,num_sym=3,max_len=256,token_base=9400" \
  --variant "width=fixed,residual=vanilla,precond=canon,canon_set=ABCD,memory=none,attn=global" \
  --loss-mode answer_only \
  --lano-cfg-kl --lano-cfg-kl-n 16
```

---

## Prompt-Conditioned Stack Gate (Tagged Sanity)

We can test whether a model can learn an **optimal gating strategy per prompt** by adding a small
prompt-conditioned gate that controls:
- Canon placements: αA/αB/αC/αD

This is intended to answer: “can the model learn to enable the right mechanisms for the right prompt
distribution, consistent with our per-task winners?”

### Flags
- `--tag-task`: prepend a task-tag token after BOS (makes the partition trivially learnable).
- `--stack-gate`: enable the prompt-conditioned gate.
- `--gate-cond {embed_pool,layer1_pool}`: what signal the gate MLP sees.
  - `layer1_pool` is the default: mean-pool the hidden state after a minimal block0 probe.
- `--gate-budget B`: soft budget for Canon usage (Σ αA..αD). Only exceeding budget is penalized.
- `--gate-lambda λ`: weight on the budget-over penalty; sweep `λ ∈ {0, 0.01, 0.1}`.

### Minimal tagged run (single seed)
```bash
python -m nmoe.research.physics.arch_ablations \
  --output /tmp/stack_gate_tagged \
  --steps 2000 --seed 42 --init-seed 0 \
  --dim 256 --n-layers 24 --seq-len 256 --batch-size 32 \
  --loss-mode answer_only --slice-metrics --layer-ce --logitlens \
  --tasks \
    "ngram_polysemy:0.5:n_symbols=512,n_steps=128,table_seed=0" \
    "ngram_scrambled:0.5:n_symbols=512,n_steps=128,table_seed=0" \
    "depo_v2:1.0:n_words_max=12,max_hops=4,n_qa=4,mini_vocab=16,min_tlen=1,max_tlen=2,separator=true" \
    "mano:0.25:depth=2,ops=asm" \
    "mano:0.25:depth=4,ops=asm" \
    "mano:0.25:depth=6,ops=asm" \
    "mano:0.25:depth=8,ops=asm" \
  --variant "width=fixed,residual=mhc,precond=canon,canon_set=ABCD,memory=engram,attn=global" \
  --tag-task \
  --stack-gate --gate-cond layer1_pool --gate-budget 2.0 --gate-lambda 0.01
```

Expected evidence:
- `analysis/slices_valid.json` contains `stack_gate` summaries per task/slice.
- With a nontrivial budget (e.g. `--gate-budget 2.0`), the model must trade off among Canon placements (αA..αD) by task/slice.
- `python -m nmoe.research.physics.viz_slices --runs /tmp/stack_gate_tagged` will also emit a `stack_gate_heatmap_*.svg` for publication.

Output (`analysis/layer_ce_valid.json`):
- `ce_by_layer`: CE at each layer (lower = closer to correct answer)
- `layer_contributions`: CE drop per layer (higher = more useful layer)
- `frac_late`: Fraction of total CE reduction in last 1/3 of layers

**Interpretation:**
- `frac_late < 0.33` → Early/mid layers do most work (late layers underutilized)
- `frac_late ≈ 0.33` → Work is evenly distributed across depth
- `frac_late > 0.33` → Late layers contribute disproportionately

### Selectable CKA Baseline (`--cka-baseline`)

By default, CKA compares to vanilla baseline. For component ablations:

```bash
# Compare mHC to Canon (not to vanilla)
python -m nmoe.research.physics.arch_ablations \
  --variant "residual=mhc,precond=canon" \
  --cka \
  --cka-baseline "width=fixed,residual=vanilla,precond=canon,memory=none,attn=global"
```

### Separate Init Seed (`--init-seed`)

For cleaner CKA comparisons, use the same init seed across variants:

```bash
python -m nmoe.research.physics.arch_ablations \
  --matrix stage1 \
  --seed 42 \       # Data sampling seed
  --init-seed 0 \   # Model init seed (same for all variants)
  --cka
```

This eliminates noise from random initialization differences.

---

## Paper-Faithful Experiment Protocols

### Canon-D note (SwiGLU)

PhysicsLM4’s Canon-D is defined on the **concatenated SwiGLU streams** (gate_proj + up_proj).
To reproduce Canon-D semantics in this harness, use:
- `--mlp-type swiglu`

Without SwiGLU, Canon-D is only an approximation (conv on a single hidden stream).

### Protocol 1: "Does Engram accelerate prediction convergence?"

**Goal:** At matched final quality, does Engram make earlier layers more prediction-ready?

```bash
python -m nmoe.research.physics.arch_ablations \
  --output /data/exp1 \
  --steps 4000 \
  --variant "memory=none" \
  --cka --layer-ce --logitlens \
  --init-seed 0 \
  --tasks "ngram_polysemy:1.0:n_symbols=256,n_steps=128"

python -m nmoe.research.physics.arch_ablations \
  --output /data/exp1 \
  --steps 4000 \
  --variant "memory=engram" \
  --cka --layer-ce --logitlens \
  --init-seed 0 \
  --tasks "ngram_polysemy:1.0:n_symbols=256,n_steps=128"

python -m nmoe.research.physics.arch_ablations \
  --output /data/exp1 \
  --steps 4000 \
  --variant "memory=ple_ngrammer" \
  --cka --layer-ce --logitlens \
  --init-seed 0 \
  --tasks "ngram_polysemy:1.0:n_symbols=256,n_steps=128"
```

**Success criteria:**
- Engram has lower early-layer CE than PLE/baseline at matched loss
- Engram shows positive depth_shift in CKA

### Protocol 2: "Does mHC preserve Canon's gains?"

**Goal:** Does mHC redistribute useful computation to later layers?

```bash
python -m nmoe.research.physics.arch_ablations \
  --output /data/exp2 \
  --steps 4000 \
  --variant "precond=canon" \
  --variant "precond=canon,residual=mhc" \
  --variant "precond=canon,memory=engram" \
  --variant "precond=canon,memory=engram,residual=mhc" \
  --cka --layer-ce \
  --cka-baseline "width=fixed,residual=vanilla,precond=canon,memory=none,attn=global" \
  --init-seed 0
```

**Success criteria:**
- CAN+ENG+MHC has higher `frac_late` than CAN+ENG
- CAN+MHC maintains or improves accuracy vs CAN

---

## Visualization Tools

### CKA Heatmaps + Depth Shift Summary

```bash
python -m nmoe.research.physics.viz_cka \
  --runs "$OUTPUT_DIR" \
  --mark-max \    # Circle argmax per row
  --summary       # Print depth_shift table
```

Output:
```
=== CKA Depth Shift Summary ===
variant                                            task             shift    early      mid     late  max_cka
---------------------------------------------------------------------------------------------------------------
width=fixed,residual=vanilla,memory=engram,...     ngram             1.50     0.00     1.00     3.50    0.510
```

### Layer-CE Curves + frac_late Summary

```bash
python -m nmoe.research.physics.viz_layer_ce \
  --runs "$OUTPUT_DIR" \
  --summary       # Print frac_late table
```

Output:
```
=== Layer CE Summary ===
variant                                            task              CE@L0    CE@Lf  frac_late
-----------------------------------------------------------------------------------------------
width=fixed,residual=vanilla,memory=engram,...     ngram              8.50     2.10      0.280
```

### Gate Profile Visualization (Memory Depth Policy)

For Engram-based memory, visualize the per-layer gate activation pattern ("close-late" vs "flat-open"):

```bash
# At final checkpoint
python -m nmoe.research.physics.viz_gate \
  --runs "$OUTPUT_DIR" \
  --step -1

# At matched accuracy threshold (critical for fair comparison)
python -m nmoe.research.physics.viz_gate \
  --runs "$OUTPUT_DIR" \
  --match-acc 0.65

# At specific step
python -m nmoe.research.physics.viz_gate \
  --runs "$OUTPUT_DIR" \
  --step 1000
```

Output:
```
=== Gate Profile Summary ===
label                                              step     acc   early    late   ratio
----------------------------------------------------------------------------------------
canon_engram/...                                    600   68.0%   0.155   0.121    0.78
canon_engram_mhc/...                               1400   67.8%   0.236   0.212    0.90
```

**Interpretation:**
- `late/early < 0.8` → Strong "close-late" pattern (memory gates shut down in late layers)
- `late/early ≈ 1.0` → Flat/uniform gates (memory active at all depths)

This is the decisive visualization for showing that **mHC changes the depth-wise memory usage policy**.

---

## Reproduced Evidence (Paper-Style)

This section records **measured, mechanism-level evidence** for the claims we care about (Engram / mHC / Canon),
using the physics harness + paper-style diagnostics.

### PhysicsLM4-faithful structured suite (Canon / mHC)

Task mix (PhysicsLM4-style substrates + DP evaluation):
- `depo_v2` (multi-token words, multi-QA per sample, answer-only labels)
- `lano_cfg` with DP-computable next-token distribution (report `dp_kl`)
- `mano` (arithmetic)

3-seed aggregate (mean ± std):

| Variant | Loss | Token Acc | DP-KL |
|:--|--:|--:|--:|
| baseline | 0.789±0.043 | 0.625±0.002 | 0.196±0.021 |
| engram | 0.792±0.060 | 0.626±0.012 | 0.197±0.043 |
| mhc | 0.762±0.045 | 0.640±0.002 | 0.163±0.033 |
| canon (ABCD) | 0.518±0.031 | 0.737±0.005 | 0.040±0.007 |
| mhc + canon | 0.521±0.034 | 0.737±0.005 | 0.045±0.013 |
| mhc + canon + engram | 0.517±0.030 | 0.738±0.004 | 0.042±0.008 |

Interpretation:
- **Canon-ABCD is the dominant lever** on structured reasoning/grammar (DP-KL drops ~80%).
- **mHC helps modestly** on this suite, but is not additive with Canon at this scale.
- **Engram is orthogonal by design** here (memory activates only for n-gram symbol ranges).

### Engram repro suite (memory conditionality + depth policy)

Task mix:
- `ngram` (bigram memory helpful)
- `ngram_polysemy` (mode A/B; disambiguation stress)
- `ngram_scrambled` (no exploitable structure; memory should be ignored)

Setup (seeded, paper-style):
- `dim=256`, `n_layers=24`, `seq_len=256`, `steps=2000`, `batch_size=64`
- `n_symbols=512`, `n_steps=128`, `table_seed=0`
- metrics: `--slice-metrics --logitlens --cka`

3-seed aggregate (valid, mean ± std):

| Variant | Loss | Token Acc |
|:--|--:|--:|
| baseline | 3.615±0.076 | 0.096±0.004 |
| engram | 3.775±0.070 | 0.138±0.015 |
| ple_ngrammer | 4.081±0.073 | 0.087±0.008 |

Seed 0 per-task token accuracy (valid):

| Task/Slice | Baseline | Engram |
|:--|--:|--:|
| ngram/all | 0.15895 | 0.28140 |
| ngram_polysemy/mode=A | 0.16191 | 0.28795 |
| ngram_polysemy/mode=B | 0.09570 | 0.03855 |
| ngram_scrambled/all | 0.00183 | 0.00192 |

Depth-wise gate policy ("close-late") for Engram (seed 0; early=L0–7, late=L16–23):

| Task/Slice | late/early |
|:--|--:|
| ngram/all | 0.20 |
| ngram_polysemy/mode=A | 0.22 |
| ngram_polysemy/mode=B | 0.12 |
| ngram_scrambled/all | 0.07 |

Interpretation:
- Engram improves token accuracy where hashed bigram memory is useful (`ngram`, `polysemy/A`).
- Engram is neutral on `ngram_scrambled` (the **conditionality** claim).
- `polysemy/mode=B` is a real **polysemy/collision failure** for Engram at this scale: it improves `mode=A` but harms `mode=B`.

LogitScope diagnosis (per-slice `--layer-ce`):
- Engram is **worse than baseline at every layer** on `mode=B` (no “late overwrite” signature).
- Depth-wise gate policy differs by mode, but does not rescue `mode=B`:
  - `mode=A` late/early gate ratio ≈ 0.15
  - `mode=B` late/early gate ratio ≈ 0.08 (memory closes even harder late)

Takeaway: this is not “late layers didn’t train”; it’s “memory helps on average, but collision disambiguation is failing for `mode=B`.”

Figures (seed 0 run; copied into repo):
- `docs/figures/engram_repro_d24/logitlens_ngram_polysemy.svg`
- `docs/figures/engram_repro_d24/logitlens_ngram_polysemy_delta.svg`
- `docs/figures/engram_repro_d24/cka_ngram_polysemy__width_fixed_residual_vanilla_memory_engram_attn_global.svg`
- `docs/figures/engram_repro_d24/mem_gate_heatmap_width_fixed_residual_vanilla_memory_engram_attn_global.svg`
- `docs/figures/engram_repro_d24/slices_ngram_polysemy_mode_answer_exact_match.svg`
- `docs/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_none_attn_global.svg`
- `docs/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_engram_attn_global.svg`
- `docs/figures/engram_repro_d24/slices_layer_ce_ngram_polysemy_width_fixed_residual_vanilla_memory_ple_ngrammer_attn_global.svg`

## Output Schema

```
$OUTPUT_DIR/
├── runs/
│   └── <variant_key>/
│       ├── config.json
│       ├── variant.json
│       └── train.jsonl
├── runs.json
└── analysis/
    ├── summary.json
    ├── logitlens_valid.json      # KL(p_final || p_layer) curves
    ├── cka_valid.json            # NxN CKA matrices
    ├── cka_depth_shift.json      # Depth shift summaries (NEW)
    └── layer_ce_valid.json       # Per-layer CE to ground-truth (NEW)
```
