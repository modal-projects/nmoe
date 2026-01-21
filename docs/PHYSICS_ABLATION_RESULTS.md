# Physics Ablation Results

This document summarizes the architecture ablation experiments using PhysicsLM4-faithful evaluation methodology.

---

## Methodology

### PhysicsLM4 Alignment

We aligned our evaluation substrate with the PhysicsLM4 reference implementation:

1. **Depo v2**: Multi-token "words" (mini_vocab + end-of-word marker), multiple QA pairs per sample, answer-only loss masking
2. **Lano-cfg**: Context-free grammar with DP-computable ground-truth next-token distributions, KL divergence metric
3. **SwiGLU MLP**: Gated MLP for Canon-D semantics matching PhysicsLM4
4. **Canon placements**: A (pre-attention), B (QKV), C (pre-MLP), D (gated MLP streams)

### Metrics

- **Loss**: Cross-entropy on answer tokens only
- **Token Acc**: Per-token accuracy on answer positions
- **DP-KL**: KL divergence to ground-truth next-token distribution (Lano-cfg only)
- **Gate Mean**: Per-layer Engram gate activation (memory conditionality)

---

## Experiment 1: Structured Reasoning (Canon Validation)

**Setup**: 6L/256d, seq_len=2048, 2000 steps, SwiGLU MLP
**Tasks**: Depo v2 (40%), Lano-cfg (30%), Mano (30%)
**Seeds**: 0, 1, 2

### Results (3-seed aggregated)

| Variant | Loss | Token Acc | DP-KL |
|---------|------|-----------|-------|
| baseline | 0.789±0.043 | 0.625±0.002 | 0.196±0.021 |
| engram | 0.792±0.060 | 0.626±0.012 | 0.197±0.043 |
| mhc | 0.762±0.045 | 0.640±0.002 | 0.163±0.033 |
| **canon-ABCD** | **0.518±0.031** | **0.737±0.005** | **0.040±0.007** |
| mhc+canon | 0.521±0.034 | 0.737±0.005 | 0.045±0.013 |
| mhc+canon+engram | 0.517±0.030 | 0.738±0.004 | 0.042±0.008 |

### Key Findings

1. **Canon-ABCD is the dominant lever**: DP-KL drops 0.196 → 0.040 (−80%)
2. **mHC helps modestly alone**: 0.196 → 0.163 (−17%)
3. **Engram is orthogonal**: No effect on structured reasoning tasks (by design)
4. **Canon + mHC ≈ Canon alone**: No additive benefit at this scale

### Interpretation

Canon's local convolutions are the critical mechanism for structured reasoning. They help "stitch" token-level substructure into usable units before/inside attention/MLP. This matches PhysicsLM4's thesis.

---

## Experiment 2: Memory Conditionality (Engram Validation)

**Setup**: 24L/256d, seq_len=256, 2000 steps
**Tasks**: ngram (33%), ngram_polysemy (33%), ngram_scrambled (33%)
**Seeds**: 0, 1, 2

### Results (3-seed aggregated)

| Variant | Loss | Token Acc |
|---------|------|-----------|
| baseline | 3.615±0.076 | 0.096±0.004 |
| **engram** | 3.775±0.070 | **0.138±0.015** |
| ple_ngrammer | 4.081±0.073 | 0.087±0.008 |

### Per-Task Breakdown (seed 0)

| Task | Baseline | Engram | Delta |
|------|----------|--------|-------|
| ngram/all | 0.159 | **0.281** | **+77%** |
| ngram_polysemy/mode=A | 0.162 | **0.288** | **+78%** |
| ngram_polysemy/mode=B | 0.096 | 0.039 | −57% |
| ngram_scrambled/all | 0.002 | 0.002 | 0% |

### Gate "Close-Late" Pattern (seed 0)

| Task | Early (L0-7) | Late (L16-23) | Ratio (L/E) |
|------|--------------|---------------|-------------|
| ngram | 0.208 | 0.042 | 0.20 |
| ngram_polysemy/A | 0.204 | 0.044 | 0.22 |
| ngram_polysemy/B | 0.202 | 0.025 | 0.12 |
| **ngram_scrambled** | 0.208 | **0.016** | **0.07** |

### Key Findings

1. **Engram improves token accuracy +44%** on ngram tasks (0.096 → 0.138)
2. **PLE is worse than baseline** (0.087 vs 0.096)
3. **Conditionality verified**: Gate closes more in late layers for scrambled (0.07) than structured (0.20)
4. **Task-specific behavior**: Engram helps mode=A (+78%) but hurts mode=B (−57%), suggesting collision sensitivity

### Interpretation

Engram's bigram-based memory with hidden-state gating learns to be "conditional":
- High gate activation in early layers (memory is useful for local n-gram structure)
- Gate closes in late layers (final prediction doesn't need raw memory)
- Closes most on scrambled tasks where n-gram structure is absent

---

## Experiment 3: Attention Ratio (Global vs Local)

**Setup**: 30L/256d, seq_len=4096, 4000 steps, window=64
**Tasks**: Depo v2 + Lano-cfg + Mano
**Seeds**: 3 seeds

### Results

| Arm | Global % | Loss | Token Acc | Exact Match |
|-----|----------|------|-----------|-------------|
| G1L9 | 10% | **5.26±0.28** | **0.028±0.003** | 0.078±0.006 |
| G1L1 | 50% | 5.34±0.18 | 0.027±0.004 | 0.078±0.005 |
| G1L0 | 100% | 5.35±0.15 | 0.026±0.003 | 0.078±0.006 |
| G0L1 | 0% | 5.48±0.06 | 0.025±0.004 | 0.079±0.005 |

### Key Findings

1. **All ratios converge to ~7.8% EM** - attention ratio is not a first-order lever
2. **G1L9 (10% global) achieves best loss** with acceptable variance
3. **Pure local (G0L1) is mildly worse** on loss/token_acc but not EM
4. **Pragmatic default**: `attn=mixed:G1L9:64` (one global layer every 10, window=64)

### Interpretation

At convergence, the global:local ratio doesn't significantly affect final accuracy on these tasks. The task suite may not isolate the attention-span constraint cleanly enough. However, "some global somewhere" helps optimization slightly.

---

## Summary: Converged Architecture Stack

Based on PhysicsLM4-faithful evaluation:

```
Recommended stack:
- residual: mhc (modest gains, no downside)
- precond: canon (ABCD) (dominant lever for structured reasoning)
- memory: engram (conditional, helps on n-gram tasks)
- attn: mixed:G1L9:64 (10% global, window=64)
- mlp: swiglu (for Canon-D semantics)
```

### Mechanism Summary

| Component | Primary Benefit | Effect Size |
|-----------|-----------------|-------------|
| Canon-ABCD | Structured reasoning | −80% DP-KL |
| mHC | Residual composition | −17% DP-KL |
| Engram | N-gram memory | +44% token acc |
| Attention ratio | Optimization speed | Marginal |

---

## Next Steps

1. **ICL→L/DoRA consolidation**: Test if ICL behavior can be distilled into adapters without drift
2. **Scale verification**: Confirm findings hold at larger model sizes (1B+)
3. **Real-world eval**: Validate on downstream tasks (MMLU, etc.)
