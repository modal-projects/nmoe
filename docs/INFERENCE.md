# Geometry-Aware Inference

## The Problem

Standard sampling methods (greedy, top-p, top-k, beam search) are blind to model calibration. They sample from the output distribution without knowing whether that distribution is trustworthy.

A model may be:
- **Confident and correct** — low entropy, good predictions
- **Confident and wrong** — low entropy, poor predictions (entropy collapse)
- **Uncertain and correct** — high entropy, calibrated uncertainty
- **Uncertain and wrong** — high entropy, miscalibrated

Standard sampling treats all low-entropy states as "confident" and all high-entropy states as "uncertain." But our routing geometry research shows this is wrong — **entropy level alone doesn't tell you about calibration**.

The core problem: **you cannot sample your way out of a miscalibrated distribution**. If p(x) is wrong, all samples from p(x) are drawn from the wrong distribution.

---

## What Our Research Shows

### Measurement Framework

Our routing geometry research produces per-position measurements:

| Metric | What it captures |
|--------|------------------|
| H (Shannon entropy) | Output distribution spread |
| H2 (Rényi collision entropy) | Mass concentration (more sensitive than H) |
| V (Varentropy) | Variance in log-probability |
| KL from base | Drift from pretrained distribution |
| NLL | Actual predictive performance |
| calibration_gap = NLL - H | Confident but wrong (positive = bad) |
| R = H2 - (H - 0.5V) | Residual capturing structure (H,V) misses |

### Key Findings

1. **(H, V) and H2 are complementary** — each captures structure the other misses. R has 65% of H2's variance, meaning the (H,V) → H2 mapping loses significant information.

2. **Domain-dependent danger zones** — high-forgetting windows cluster at different (H, V, H2) locations:
   - fineweb: low H (~1.1), low V (~2.0) — collapsed, overconfident
   - code: high H (~2.1), high V (~3.4) — uncertain territory

   The same (H, V) coordinate means different things in different domains.

3. **Tuning shifts the geometry systematically** — dH < 0, dV < 0, dR < 0 consistently across checkpoints. The entire distribution compresses.

4. **High-KL regions have disproportionate forgetting** — F/S ≈ 2.5 at extreme KL. Not just more forgetting, but more forgetting *per visit*.

5. **Forgetting-prone windows are pre-existing in base** — the fragility is baked into the geometry before tuning exposes it.

6. **calibration_gap reveals the failure mode** — positive gap means confident but wrong. This is the signature of harmful entropy collapse vs. beneficial compression.

### The Core Insight

We can compute KL, calibration_gap, and drift **offline** when we have both base and instruct models. We can then train a predictor to estimate these quantities from signals we CAN compute at inference time.

**The base model's knowledge is distilled into the predictor. At inference, we query the predictor, not the base.**

---

## The Constraint

We cannot keep the base model loaded at inference time:
- 2x memory cost
- 2x compute cost
- Impractical for deployment

Therefore, any approach that requires "compare to base at inference" is not viable.

**Solution**: Distill base-instruct comparison into lightweight artifacts during offline analysis. Use artifacts at inference.

---

## Observable Signals at Inference

What we CAN compute (minimal additional cost):

**From output distribution (free — computed anyway):**
- H (Shannon entropy)
- H2 (Rényi collision entropy, α=2)
- V (Varentropy)
- R = H2 - (H - 0.5V)

**From router states (MoE, minimal cost):**
- Router entropy per layer
- Expert selection pattern
- Top-k expert utilization

**From forward pass (already computed):**
- Hidden state h_i at each position
- Attention patterns (optional)

**What we CANNOT compute at inference:**
- KL from base (need base model)
- True NLL (need ground truth next token)
- True calibration_gap (need ground truth)

---

## Architecture: MTP-Style Geometry Head

Following the DeepSeek-V3 Multi-Token Prediction architecture pattern, we design a lightweight prediction head that estimates calibration signals from observable features.

### Reference Implementation

See `nmoe/mtp.py` for the canonical MTP implementation. The geometry head follows the same pattern.

**MTPBlock (from mtp.py:27-68):**
```python
class MTPBlock(nn.Module):
    def __init__(self, dim, inter_dim, n_layers, depth, attn_cls, rms_eps):
        self.norm_h = RMSNorm(dim, rms_eps)
        self.norm_emb = RMSNorm(dim, rms_eps)
        self.proj = nn.Linear(dim * 2, dim, bias=False)
        self.attn_norm = RMSNorm(dim, rms_eps)
        self.ffn_norm = RMSNorm(dim, rms_eps)
        self.attn = attn_cls
        self.ffn = _MLP(dim, inter_dim)

    def forward(self, h_prev, token_emb, cos, sin):
        h_normed = self.norm_h(h_prev)
        emb_normed = self.norm_emb(token_emb)
        x = self.proj(torch.cat([h_normed, emb_normed], dim=-1))
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

Key design:
- RMSNorm both inputs separately before concatenation
- Projection reduces 2d → d
- Attention + FFN block with residual connections

### Geometry Head Adaptation

The geometry head reuses this pattern with modifications:

| MTP (token prediction) | Geometry Head (calibration prediction) |
|------------------------|---------------------------------------|
| `token_emb = embedding(t_{i+k})` | `geom_emb = GeomEmbed(H, V, H2, R, router_stats)` |
| Shared `lm_head` → next token logits | `PredictionHeads` → (KL, cal_gap, drift) |
| D depths for D future tokens | Single depth (one block sufficient) |
| `cross_entropy` loss | `mse` loss on regression targets |
| RoPE for positions | No RoPE (single position prediction) |

```
geometry_features = [H, V, H2, R, router_H_1, ..., router_H_L, ...]

geom_emb = Linear(geometry_features)              # geom_dim → d
h' = proj(cat(RMSNorm(h), RMSNorm(geom_emb)))    # 2d → d
h_out = h' + attn(norm(h')) + ffn(norm(h'))      # transformer block
predictions = PredictionHeads(RMSNorm(h_out))    # d → 3 scalars
```

### Implementation Note

The geometry head must be **model-specific** because it depends on:
- Model dimension `d`
- Attention implementation (MLA, MHA, GQA, etc.)
- Number of MoE layers for router stats
- Expert count for utilization features
- RMSNorm epsilon and other model-specific constants

Each model family (DeepSeek, Qwen, etc.) needs its own geometry head implementation following the `nmoe/mtp.py` pattern.

### Why This Architecture

1. **Hidden state h_i carries rich information** — the model's internal representation of "what's happening at this position"

2. **Geometry features provide explicit signals** — (H, V, H2, R, router stats) that we know correlate with calibration

3. **Concatenation allows learning interactions** — how do hidden state patterns combine with geometry features to predict calibration?

4. **Small transformer block adds expressivity** — can model complex relationships without massive parameter count

5. **MTP pattern is proven** — DeepSeek-V3 uses this successfully for multi-token prediction

---

## Training Procedure

### Requirements

- Base model (e.g., DeepSeek-V3.2 Base)
- Instruct model (e.g., Speciale)
- Evaluation data with text (e.g., fineweb, code parquets)

### Data Collection

```python
training_data = []

for window in eval_data:
    # Forward pass through both models
    base_out = base_model.forward_with_internals(window)
    inst_out = inst_model.forward_with_internals(window)

    # Compute geometry features from instruct model
    H = shannon_entropy(inst_out.logits)
    H2 = renyi_entropy(inst_out.logits, alpha=2)
    V = varentropy(inst_out.logits)
    R = H2 - (H - 0.5 * V)
    router_stats = summarize_router(inst_out.router_logits)

    geometry_features = concat(H, H2, V, R, router_stats)

    # Compute ground truth targets
    KL_true = kl_divergence(inst_out.logits, base_out.logits)
    NLL_true = nll(inst_out.logits, window.next_tokens)
    calibration_gap_true = NLL_true - H
    drift_true = router_divergence(inst_out.router_logits, base_out.router_logits)

    training_data.append({
        "hidden": inst_out.hidden_state,
        "geometry_features": geometry_features,
        "targets": {
            "KL": KL_true,
            "calibration_gap": calibration_gap_true,
            "drift": drift_true,
        }
    })
```

### Training

```python
geometry_head = GeometryAwareHead(config)
optimizer = AdamW(geometry_head.parameters(), lr=1e-4)

for batch in DataLoader(training_data):
    # Forward through geometry head
    predictions = geometry_head(batch["hidden"], batch["geometry_features"])

    # Multi-task loss
    loss = (
        λ_KL * mse_loss(predictions.KL, batch["targets"]["KL"]) +
        λ_cal * mse_loss(predictions.calibration_gap, batch["targets"]["calibration_gap"]) +
        λ_drift * mse_loss(predictions.drift, batch["targets"]["drift"])
    )

    loss.backward()
    optimizer.step()
```

### Training Modes

**Option A: Post-hoc training (recommended initially)**
- Freeze instruct model completely
- Train geometry head on collected data
- Simpler, no risk of degrading main model

**Option B: Joint training**
- Add geometry head during instruct fine-tuning
- Model learns representations predictive of calibration
- Potentially better but more complex

---

## Inference Procedure

### Initialization

```python
model = load_model("speciale")
geometry_head = load("geometry_head.pt")
config = load("geometry_config.json")
```

### Per-Position Sampling

```python
def geometry_aware_sample(model, geometry_head, config, context):
    # Forward pass (standard)
    logits, hidden, router_states = model.forward_with_internals(context)

    # Compute observable features
    H = shannon_entropy(logits[-1])
    H2 = renyi_entropy(logits[-1], alpha=2)
    V = varentropy(logits[-1])
    R = H2 - (H - 0.5 * V)
    router_stats = summarize_router(router_states)

    geometry_features = concat(H, H2, V, R, router_stats)

    # Query geometry head
    KL_pred, cal_gap_pred, drift_pred = geometry_head(hidden[-1], geometry_features)

    # Compute risk score
    risk = compute_risk(KL_pred, cal_gap_pred, drift_pred, H, V, H2, R, config)

    # Adaptive sampling
    if risk > config.high_risk_threshold:
        return cautious_sample(logits, config)
    elif risk > config.medium_risk_threshold:
        return exploratory_sample(logits, config)
    else:
        return standard_sample(logits, config)
```

### Risk Computation

```python
def compute_risk(KL_pred, cal_gap_pred, drift_pred, H, V, H2, R, config):
    risk = 0.0

    # Predicted signals
    if KL_pred > config.KL_threshold:
        risk += config.w_KL * (KL_pred - config.KL_threshold)

    if cal_gap_pred > config.cal_threshold:
        risk += config.w_cal * (cal_gap_pred - config.cal_threshold)

    if drift_pred > config.drift_threshold:
        risk += config.w_drift * (drift_pred - config.drift_threshold)

    # Direct geometry checks (precomputed danger zones)
    if in_danger_zone(H, V, H2, config.danger_zones):
        risk += config.danger_zone_penalty

    # R anomaly (deviation from expected)
    R_baseline = config.R_baseline[detected_domain]
    R_zscore = abs(R - R_baseline.mean) / R_baseline.std
    if R_zscore > config.R_anomaly_threshold:
        risk += config.w_R_anomaly * R_zscore

    return risk
```

### Sampling Strategies

```python
def standard_sample(logits, config):
    return sample_top_p(logits, p=config.standard_p, temp=1.0)

def exploratory_sample(logits, config):
    # Higher temperature, wider sampling
    return sample_top_p(logits, p=config.explore_p, temp=config.explore_temp)

def cautious_sample(logits, config):
    # Options based on deployment context:
    # - Flatten distribution (high temp)
    # - Return uncertainty signal
    # - Trigger human review
    # - Abstain from generation

    if config.cautious_mode == "flatten":
        return sample_top_p(logits, p=0.99, temp=2.0)
    elif config.cautious_mode == "signal":
        return UNCERTAINTY_TOKEN
    elif config.cautious_mode == "abstain":
        return ABSTAIN_TOKEN
```

---

## Artifacts

### Trained Components

```
geometry_artifacts/
├── geometry_head.pt              # Trained prediction head
├── geometry_head_config.json     # Architecture specification
└── training_metadata.json        # Data stats, hyperparameters
```

### Precomputed Statistics

```
geometry_artifacts/
├── danger_zones/
│   ├── fineweb.json              # High-risk (H,V,H2) regions
│   └── code.json
├── baselines/
│   ├── R_fineweb.json            # Expected R distribution
│   ├── R_code.json
│   └── ...
├── router_baselines/             # MoE only
│   └── healthy_router_stats.json
└── expert_trust.json             # MoE only: per-expert weights
```

### Inference Configuration

```json
{
  "KL_threshold": 0.15,
  "cal_threshold": 0.5,
  "drift_threshold": 0.3,
  "w_KL": 2.0,
  "w_cal": 1.5,
  "w_drift": 1.0,
  "danger_zone_penalty": 1.0,
  "R_anomaly_threshold": 2.0,
  "w_R_anomaly": 0.5,
  "high_risk_threshold": 3.0,
  "medium_risk_threshold": 1.5,
  "standard_p": 0.9,
  "explore_p": 0.95,
  "explore_temp": 1.2,
  "cautious_mode": "flatten"
}
```

---

## Integration with Research Pipeline

### From Measurement to Inference

| Research Phase | Output | Used For |
|----------------|--------|----------|
| testF_entropy_collapse | (H, V, H2, KL, NLL, cal_gap) | Training data |
| V1-V5 routing geometry | Router patterns, expert behavior | Drift detection |
| Domain analysis | Danger zone boundaries | Risk computation |
| LLMRI bridge | R baselines, (H,V) clustering | Anomaly detection |

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    OFFLINE (Research)                            │
├─────────────────────────────────────────────────────────────────┤
│  Base + Instruct + Eval Data                                     │
│       ↓                                                          │
│  Measurement scripts → (H, V, H2, KL, NLL, router stats)         │
│       ↓                                                          │
│  Analysis → Danger zones, baselines, expert trust                │
│       ↓                                                          │
│  Training → geometry_head.pt                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ONLINE (Inference)                            │
├─────────────────────────────────────────────────────────────────┤
│  Instruct Model + Geometry Head + Artifacts                      │
│       ↓                                                          │
│  Geometry-Aware Sampler                                          │
│       ↓                                                          │
│  Calibration-aware, domain-adaptive generation                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Future Extensions

### Expert Reweighting (MoE)

Modify routing based on expert trust:

```python
adjusted_logits = router_logits * expert_trust[layer]
output = route_with_adjusted_logits(adjusted_logits, experts)
```

This changes p(x) itself, not just sampling from it.

### Layer-Wise Intervention

If collapse detected, mix terminal with mid-band:

```python
if collapse_detected:
    output = α * terminal_output + (1-α) * midband_output
```

### Trajectory Monitoring

Track (H, V, H2) over recent positions:

```python
if trajectory_indicates_trouble(recent_positions):
    increase_risk_score()
```

### Online Calibration

Update thresholds based on deployment feedback:

```python
if feedback_available:
    calibrate_thresholds(prediction_errors)
```

---

## Open Questions

1. **Minimum architecture size** — How small can geometry head be?

2. **Cross-model transfer** — Do artifacts generalize across model families?

3. **Automatic domain detection** — How to select domain-specific thresholds?

4. **Threshold optimization** — Principled approach to setting risk thresholds?

5. **Latency budget** — Acceptable overhead for geometry head?

6. **Joint training effects** — Does training geometry head jointly improve results?

---

## References

- DeepSeek-V3 Technical Report (MTP architecture)
- RL's Razor (arxiv 2509.04259)
- From Entropy to Epiplexity (arxiv 2601.03220)
- Routing Geometry research (HANDOFF.md, routing_geometry_spec.tex)
