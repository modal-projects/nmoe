# Architecture Evaluation Specification

## Goal

Define a rigorous evaluation framework to:
1. **PROVE** any architectural change improves over baseline
2. **SELECT** between competing approaches (DeepSeek vs Google/Gemma)
3. **UNDERSTAND** why improvements occur (not just that they occur)

---

## Candidates Under Evaluation

### Memory Axis
| Approach | Source | Mechanism |
|----------|--------|-----------|
| **Engram** | DeepSeek | Hashed n-gram + context-aware gate: `σ(⟨q(x), k(mem)⟩/√d)` |
| **PLE** | Google/Gemma | Per-layer embedding injection: `h + gelu(W_gate·h) * side_input` |
| **N-Grammer** | Google | PQ → bigram → hash → embed (addressing only, no gate) |

### Residual/Stability Axis
| Approach | Source | Mechanism |
|----------|--------|-----------|
| **mHC** | DeepSeek | Multi-stream mixing with Sinkhorn-constrained doubly-stochastic H_res |
| **AltUp** | Google/Gemma | Multiple streams, process one, predict/correct rest |
| **Canon** | PhysicsLM4 | Causal depthwise conv on residual (local state preconditioner) |

### Local Mixing Axis
| Approach | Source | Mechanism |
|----------|--------|-----------|
| **SWA** | Various | Sliding window attention (bounded context per layer) |
| **Canon conv** | PhysicsLM4 | k=4 causal depthwise conv |
| **ShortConv** | DeepSeek/Engram | Lightweight depthwise conv within memory branch |

---

## Evaluation Probes

### 1. LogitLens (Prediction Convergence)

**What it measures**: How quickly representations become "prediction-ready"

**Protocol**:
```python
for layer_idx in range(n_layers):
    h_l = model.get_hidden_state(x, layer=layer_idx)
    logits_l = model.lm_head(model.final_norm(h_l))
    logits_final = model.forward(x)

    kl_divergence[layer_idx] = KL(softmax(logits_l) || softmax(logits_final))
```

**Success criterion**:
- Engram claims "systematically smaller KL divergence in early blocks"
- Plot KL vs layer depth; steeper descent = faster convergence
- Memory module should reduce early-layer KL vs baseline

**Comparison**:
- Engram vs PLE: Does gating affect convergence curve shape?
- Memory vs no-memory: How much does memory accelerate convergence?

### 2. CKA (Representation Alignment)

**What it measures**: At what depth do representations align?

**Protocol**:
```python
# CKA(layer_i of model_A, layer_j of model_B)
# Find j such that CKA is maximized for each i
```

**Success criterion**:
- Engram claims "layer 5 of Engram ≈ layer 12 of baseline"
- This would prove memory provides "effective depth" increase

### 3. Gradient Flow (Training Stability)

**What it measures**: Stability of optimization dynamics

**Protocol**:
```python
for step in training:
    grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()]))
    composite_gain = compute_H_res_product_spectral_norm()  # For mHC
    log(step, grad_norm, composite_gain)
```

**Success criterion**:
- mHC claims gradient norms "significantly better than HC"
- Composite gain bounded at ~1.6 vs HC's 3000x
- Plot grad_norm vs step; lower variance = more stable

**Comparison**:
- mHC vs AltUp: Which provides better gradient flow?
- mHC vs vanilla residual: Is multi-stream worth the complexity?

### 4. Sensitivity Ablation

**What it measures**: How much does each component contribute?

**Protocol**:
```python
# For each component C in {memory, gate, residual_law, local_mixing}:
perf_with_C = evaluate(model_with_C)
perf_without_C = evaluate(model_without_C)
sensitivity[C] = (perf_with_C - perf_without_C) / perf_with_C
```

**Success criterion**:
- Engram reports: factual tasks retain only 29-44% without Engram
- This proves memory is load-bearing, not just regularization

### 5. Task-Specific Breakdown

**What it measures**: Where does each approach help?

**Task categories**:
- **Local pattern** (n-gram prediction): Should favor memory
- **Multi-hop reasoning** (depo): Should favor depth/residual stability
- **Arithmetic** (mano): Should favor computation
- **Polysemy disambiguation**: Should favor content-aware gating

**Protocol**: Report accuracy per task, not just aggregate

---

## Evidence Required to PROVE Improvement

### Necessary Conditions (must pass ALL):

1. **Task performance**: Final accuracy > baseline on held-out tasks
2. **Seed stability**: Improvement holds across 3+ seeds (report mean±std)
3. **Scale consistency**: Improvement holds at multiple scales (not just toy)
4. **Not just optimization**: LogitLens/CKA shows representational improvement, not just faster convergence to same point

### Sufficient Conditions (at least ONE):

1. **Mechanistic evidence**: LogitLens shows accelerated convergence in early layers
2. **Effective depth**: CKA shows representation alignment shift
3. **Stability evidence**: Gradient norms demonstrably better
4. **Sensitivity evidence**: Large performance drop when component removed

---

## Decision Framework: DeepSeek vs Google

### Memory: Engram vs PLE

| Criterion | Test | Winner if... |
|-----------|------|--------------|
| Convergence speed | LogitLens KL curve | Steeper early descent |
| Polysemy handling | ngram_polysemy task | Higher accuracy on mode-ambiguous inputs |
| Collision robustness | High-collision hash table | Maintains performance under collisions |
| Simplicity | Param count, code complexity | Fewer params, simpler code for same perf |

**Key differentiator**: Does content-aware gating provide value?
- If Engram gate learns content selectivity → Engram wins
- If gate just learns per-layer schedule → PLE wins (simpler)

### Stability: mHC vs AltUp

| Criterion | Test | Winner if... |
|-----------|------|--------------|
| Training stability | Gradient norm variance | Lower variance across training |
| Scale | Performance at 27B+ | Maintains gains at scale |
| Overhead | Memory/compute cost | Lower overhead for same stability |

**Key differentiator**: Is Sinkhorn constraint necessary?
- If unconstrained HC explodes but mHC doesn't → mHC wins
- If AltUp provides similar stability cheaper → AltUp wins

### Local Mixing: Canon vs SWA

| Criterion | Test | Winner if... |
|-----------|------|--------------|
| Local pattern capture | n-gram task | Higher accuracy |
| Composability | Canon+memory vs SWA+memory | Better stacking |
| Efficiency | FLOPs per token | Lower cost for same gain |

---

## Implementation Plan

### Phase 1: Instrumentation
1. Add LogitLens probe to arch_ablations.py
2. Add gradient norm logging
3. Add per-task accuracy breakdown

### Phase 2: Baseline Establishment
1. Run baseline (no memory, vanilla residual) at multiple scales
2. Establish LogitLens/gradient curves as reference

### Phase 3: Comparative Evaluation
1. Run each candidate with same probe suite
2. Generate comparison plots
3. Apply decision framework

### Phase 4: Selection
1. Identify winning approach per axis
2. Test composition (winner_memory + winner_stability + winner_local)
3. Final validation at target scale

---

## Artifact Schema

Each evaluation run produces:
```json
{
  "variant": "memory=engram,residual=mhc,local=canon",
  "scale": {"dim": 2048, "layers": 6, "params": "400M"},
  "seed": 42,
  "tasks": {
    "ngram": {"acc": 0.85, "loss": 0.42},
    "depo": {"acc": 0.12, "loss": 2.1},
    "polysemy": {"acc": 0.56, "loss": 1.2}
  },
  "probes": {
    "logitlens": {"layer_0_kl": 2.1, "layer_1_kl": 1.8, ...},
    "gradient": {"mean_norm": 0.42, "max_norm": 1.2, "variance": 0.08},
    "sensitivity": {"memory_drop": 0.35, "gate_drop": 0.05}
  }
}
```

This schema allows automated comparison and decision-making.
