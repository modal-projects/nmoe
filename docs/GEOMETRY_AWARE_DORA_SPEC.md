# Geometry-Aware DoRA Specification

## Core Insight

If we have access to the base model during SFT/RL, we can actively prevent calibration collapse rather than just monitor it. The base model provides a reference distribution that lets us:

1. Detect collapse as it happens (per-batch, per-position)
2. Apply corrective signals during training
3. Preserve V2 set-boundary redundancy (both axes)
4. Maintain calibration gap ≈ 0

**Prerequisite**: Understand base model's V2 profile before starting. Different bases need different protection strategies.

---

## V2 Two-Axis Model

V2 set-boundary redundancy has two independent conditions:

| Condition | Formula | What it measures |
|-----------|---------|------------------|
| **C1** (ΔBI) | κ_hidden(boundary) - κ_hidden(interior) | Boundary tokens more similar than interior |
| **C2** (ΔBrand) | κ_hidden(boundary) - κ_hidden_rand(boundary) | Adjacent expert k+1 is *specifically* similar (not just any expert) |

**Cross-family findings:**

| Model | C1 (ΔBI) | C2 (ΔBrand) | Interpretation |
|-------|----------|-------------|----------------|
| DeepSeek V3 Base | **Strong** (-0.0017) | FAILS (~0) | Experts distinct, but adjacency not special |
| Qwen3 Base | Weak (-0.0008) | **Strong** (-0.0037) | Experts less distinct, but adjacency IS special |
| Qwen3 Thinking | FAILS (+0.0002) | **Strong** (-0.0033) | Tuning erodes C1, preserves C2 |
| DeepSeek R1 (code) | PASS | PASS | Tuning somehow gains both |

**Implication for fine-tunability:**

Qwen3's strong C2 (adjacency specificity) acts as a stability margin. When tuning erodes C1, routing still has "fallback coherence" because adjacent experts produce similar outputs. DeepSeek lacks this fallback — C2 ≈ 0 means no safety net when routing drifts.

**Training strategy by base type:**

| Base Profile | Strategy |
|--------------|----------|
| Strong C1, Weak C2 (DeepSeek-like) | Protect C1 aggressively; consider C2 auxiliary loss |
| Weak C1, Strong C2 (Qwen-like) | C2 is naturally stable; monitor C1 erosion |
| Both weak | Consider continued pretraining before SFT |

---

## Architecture: Geometry-Aware Training Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING STEP                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  batch → ┬→ tuned_model.forward() → tuned_logits, tuned_hidden  │
│          │                                                       │
│          └→ base_model.forward()  → base_logits, base_hidden    │
│                        (frozen, no grad)                         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Geometry Signals (per position):                            ││
│  │   KL(tuned || base)                                         ││
│  │   dH = H(tuned) - H(base)                                   ││
│  │   calibration_gap = NLL(tuned) - H(tuned)                   ││
│  │   router_divergence (MoE only)                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Loss = task_loss                                            ││
│  │      + λ_KL * KL_penalty(positions where KL > threshold)    ││
│  │      + λ_cal * calibration_penalty(positions where gap > 0) ││
│  │      + λ_H * entropy_preservation(positions where dH < -δ)  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### 1. Base-Guided Loss Components

```python
class GeometryAwareLoss(nn.Module):
    def __init__(self, config):
        self.λ_KL = config.lambda_kl           # KL penalty weight
        self.λ_cal = config.lambda_cal         # Calibration penalty weight
        self.λ_H = config.lambda_entropy       # Entropy preservation weight
        self.KL_threshold = config.kl_threshold
        self.dH_threshold = config.dh_threshold
        self.cal_threshold = config.cal_threshold

    def forward(self, tuned_logits, base_logits, labels):
        # Task loss (standard)
        task_loss = F.cross_entropy(tuned_logits, labels, reduction='none')

        # Per-position geometry signals
        H_tuned = entropy(tuned_logits, dim=-1)
        H_base = entropy(base_logits, dim=-1)
        dH = H_tuned - H_base

        KL = F.kl_div(
            F.log_softmax(tuned_logits, dim=-1),
            F.softmax(base_logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)

        NLL = task_loss  # NLL is just the per-position loss
        calibration_gap = NLL - H_tuned

        # Selective penalties (only where thresholds exceeded)
        KL_penalty = torch.where(
            KL > self.KL_threshold,
            KL - self.KL_threshold,
            torch.zeros_like(KL)
        )

        cal_penalty = torch.where(
            calibration_gap > self.cal_threshold,
            calibration_gap - self.cal_threshold,
            torch.zeros_like(calibration_gap)
        )

        entropy_penalty = torch.where(
            dH < -self.dH_threshold,
            (-dH - self.dH_threshold),
            torch.zeros_like(dH)
        )

        # Combined loss
        loss = (
            task_loss.mean()
            + self.λ_KL * KL_penalty.mean()
            + self.λ_cal * cal_penalty.mean()
            + self.λ_H * entropy_penalty.mean()
        )

        return loss, {
            "task_loss": task_loss.mean(),
            "KL_mean": KL.mean(),
            "dH_mean": dH.mean(),
            "cal_gap_mean": calibration_gap.mean(),
            "positions_penalized_KL": (KL > self.KL_threshold).float().mean(),
            "positions_penalized_cal": (calibration_gap > self.cal_threshold).float().mean(),
            "positions_penalized_H": (dH < -self.dH_threshold).float().mean(),
        }
```

### 2. Router Divergence Penalty (MoE)

```python
def router_divergence_loss(tuned_router_logits, base_router_logits, layer_weights):
    """Penalize routing drift, especially in mid-band layers."""
    total_div = 0.0

    for layer_idx, (tuned_r, base_r) in enumerate(zip(tuned_router_logits, base_router_logits)):
        # Compute top-k agreement
        tuned_topk = tuned_r.topk(k, dim=-1).indices
        base_topk = base_r.topk(k, dim=-1).indices

        # Jaccard distance on expert sets
        agreement = compute_set_overlap(tuned_topk, base_topk)
        divergence = 1.0 - agreement

        # Weight by layer importance (mid-band higher)
        total_div += layer_weights[layer_idx] * divergence.mean()

    return total_div
```

### 3. Adaptive Threshold Scheduling

```python
class AdaptiveGeometryScheduler:
    """Tighten thresholds as training progresses."""

    def __init__(self, initial_KL=0.15, final_KL=0.08, warmup_steps=1000):
        self.initial_KL = initial_KL
        self.final_KL = final_KL
        self.warmup_steps = warmup_steps

    def get_thresholds(self, step):
        if step < self.warmup_steps:
            # Linear warmup: start permissive, tighten
            ratio = step / self.warmup_steps
            KL_threshold = self.initial_KL - ratio * (self.initial_KL - self.final_KL)
        else:
            KL_threshold = self.final_KL

        return {
            "kl_threshold": KL_threshold,
            "dh_threshold": 0.1,  # Fixed
            "cal_threshold": 0.05,  # Fixed
        }
```

---

## Training Loop Integration

```python
def geometry_aware_training_step(
    tuned_model,
    base_model,
    batch,
    optimizer,
    loss_fn,
    scheduler
):
    # Forward through both models
    tuned_out = tuned_model(batch.input_ids, output_router_logits=True)
    with torch.no_grad():
        base_out = base_model(batch.input_ids, output_router_logits=True)

    # Compute geometry-aware loss
    loss, metrics = loss_fn(
        tuned_out.logits,
        base_out.logits,
        batch.labels
    )

    # Add router divergence for MoE
    if hasattr(tuned_out, 'router_logits'):
        router_loss = router_divergence_loss(
            tuned_out.router_logits,
            base_out.router_logits,
            layer_weights=get_midband_weights(tuned_model.config)
        )
        loss = loss + scheduler.λ_router * router_loss
        metrics["router_divergence"] = router_loss

    # Standard backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return metrics
```

---

## RL-Specific: Geometry-Aware PPO

For RL, the geometry signals can modulate the reward or KL penalty:

```python
def geometry_aware_ppo_step(
    policy_model,
    base_model,
    ref_model,      # PPO reference (can be same as base or separate)
    batch,
    reward_model
):
    # Get outputs
    policy_out = policy_model(batch.input_ids)
    with torch.no_grad():
        base_out = base_model(batch.input_ids)
        ref_out = ref_model(batch.input_ids)
        rewards = reward_model(batch.input_ids, batch.responses)

    # Standard PPO KL from reference
    kl_ref = kl_divergence(policy_out.logits, ref_out.logits)

    # Geometry signals from base
    kl_base = kl_divergence(policy_out.logits, base_out.logits)
    dH = entropy(policy_out.logits) - entropy(base_out.logits)
    cal_gap = compute_calibration_gap(policy_out.logits, batch.labels)

    # Geometry-modulated reward
    # Reduce reward for positions showing collapse
    geometry_penalty = torch.where(
        (kl_base > KL_THRESHOLD) | (dH < -DH_THRESHOLD) | (cal_gap > CAL_THRESHOLD),
        torch.tensor(GEOMETRY_PENALTY),
        torch.zeros_like(rewards)
    )

    adjusted_rewards = rewards - geometry_penalty

    # Standard PPO with adjusted rewards
    advantages = compute_gae(adjusted_rewards, values)
    ppo_loss = compute_ppo_loss(policy_out, ref_out, advantages, kl_ref)

    return ppo_loss
```

---

## Recommended Hyperparameters

Based on cross-family findings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| λ_KL | 0.1-0.5 | Start low, increase if KL explodes |
| λ_cal | 0.2-1.0 | Calibration preservation is important |
| λ_H | 0.1-0.3 | Prevent entropy collapse |
| KL_threshold | 0.08-0.15 | Qwen3 tolerates ~0.15, DeepSeek less |
| dH_threshold | 0.1 | Significant entropy drop |
| cal_threshold | 0.05 | Gap approaching problematic range |

For well-calibrated bases (Qwen3-like):
- Can use higher KL_threshold (0.15)
- Lower λ weights (0.1-0.2)

For poorly-calibrated bases (DeepSeek-like):
- Use lower KL_threshold (0.08)
- Higher λ weights (0.3-0.5)

---

## Memory Considerations

Keeping base model loaded doubles memory. Options:

1. **Full base in memory** (2x VRAM)
   - Simplest, most accurate
   - Use for smaller models or multi-GPU

2. **Base model sharding**
   - Keep base on CPU, move layers as needed
   - Slower but memory-efficient

3. **Cached base statistics**
   - Pre-compute base logits/entropy for training data
   - Only works for fixed datasets (SFT), not RL

4. **Periodic base queries**
   - Query base every N steps instead of every step
   - Trade accuracy for efficiency

---

## Validation: V2 Checkpoints

During training, periodically run V2 verification:

```python
def v2_checkpoint_validation(model, base_model, step, domains=["fineweb", "code"]):
    """Run V2 set-boundary redundancy check."""
    results = {}

    for domain in domains:
        v2_pass = run_v2_verification(model, domain)
        results[f"v2_{domain}"] = v2_pass

        if not v2_pass:
            log.warning(f"V2 FAIL on {domain} at step {step}")
            # Optionally: increase λ_router, reduce LR, or rollback

    return results
```

**Checkpointing rule**: Save checkpoint only if V2 passes on primary domain. If V2 fails, consider rolling back to last passing checkpoint.

---

## Summary

The key insight: **base model access during training enables active correction, not just passive monitoring**.

1. Compute geometry signals (KL, dH, cal_gap) per position
2. Apply selective penalties where thresholds exceeded
3. Preserve routing structure (MoE) via router divergence loss
4. Validate V2 periodically; checkpoint only on pass

This converts the observational findings (Qwen3 base is more calibrated → easier to RL) into an actionable intervention (force any base to stay calibrated during tuning).

---

## Continued Pretraining for V2 Improvement

If base fails V2 (especially C2), consider continued pretraining with V2-targeted auxiliary losses before SFT.

### C2 Auxiliary Loss (Adjacency Specificity)

```python
def c2_adjacency_loss(router_logits, expert_outputs, k=8):
    """Encourage adjacent expert k+1 to be specifically similar to expert k outputs."""

    # Get routing decisions
    scores = F.softmax(router_logits, dim=-1)
    topk_vals, topk_idx = scores.topk(k + 1, dim=-1)

    # Identify boundary tokens (small margin k vs k+1)
    margin = topk_vals[..., k-1] - topk_vals[..., k]
    boundary_mask = margin < BOUNDARY_THRESHOLD

    # For boundary tokens, compute:
    # - κ(expert_k, expert_k+1) for actual adjacent expert
    # - κ(expert_k, expert_rand) for random non-selected expert

    expert_k = gather_expert_output(expert_outputs, topk_idx[..., k-1])
    expert_k1 = gather_expert_output(expert_outputs, topk_idx[..., k])
    expert_rand = gather_expert_output(expert_outputs, sample_non_topk(topk_idx, n_experts))

    kappa_adjacent = normalized_diff(expert_k, expert_k1)  # Should be LOW
    kappa_random = normalized_diff(expert_k, expert_rand)   # Baseline

    # C2 loss: adjacent should be more similar than random
    # ΔBrand = kappa_adjacent - kappa_random should be NEGATIVE
    c2_loss = F.relu(kappa_adjacent - kappa_random + C2_MARGIN)

    return c2_loss[boundary_mask].mean()
```

### C1 Auxiliary Loss (Boundary-Interior Separation)

```python
def c1_boundary_interior_loss(router_logits, expert_outputs, k=8):
    """Encourage boundary tokens to have lower expert divergence than interior."""

    scores = F.softmax(router_logits, dim=-1)
    topk_vals, topk_idx = scores.topk(k + 1, dim=-1)

    margin = topk_vals[..., k-1] - topk_vals[..., k]
    boundary_mask = margin < BOUNDARY_THRESHOLD
    interior_mask = margin > INTERIOR_THRESHOLD

    # Compute κ_hidden for both strata
    expert_k = gather_expert_output(expert_outputs, topk_idx[..., k-1])
    expert_k1 = gather_expert_output(expert_outputs, topk_idx[..., k])
    kappa = normalized_diff(expert_k, expert_k1)

    kappa_boundary = kappa[boundary_mask].mean()
    kappa_interior = kappa[interior_mask].mean()

    # C1 loss: boundary should be lower than interior
    # ΔBI = kappa_boundary - kappa_interior should be NEGATIVE
    c1_loss = F.relu(kappa_boundary - kappa_interior + C1_MARGIN)

    return c1_loss
```

### Continued Pretraining Objective

```python
L_continued_pretrain = L_LM + λ_c1 * L_c1 + λ_c2 * L_c2
```

**Recommended schedule:**
1. Start with λ_c1 = λ_c2 = 0.01 (small)
2. Monitor V2 metrics every N steps
3. Increase λ if V2 not improving
4. Stop when V2 passes on target domains

**Expected outcome:**
- DeepSeek-like base (strong C1, weak C2): Focus on C2 loss → gain adjacency specificity
- Weak base (both fail): Apply both losses → build V2 from scratch

**Risk:** Excessive V2 pressure may hurt LM loss. Monitor perplexity and V2 jointly. There may be a Pareto front.

---

## Decision Tree (Full Pipeline)

```
1. Profile base model V2
   ├─ Both C1 and C2 pass → Proceed to SFT
   ├─ C1 pass, C2 fail (DeepSeek-like) → Option: continued pretraining with C2 loss
   ├─ C1 fail, C2 pass (Qwen-like after tuning) → Already tuned, just monitor
   └─ Both fail → Continued pretraining with both losses, or choose different base

2. During SFT/RL (with base model access)
   ├─ Compute per-position geometry signals
   ├─ Apply selective penalties (KL, dH, cal_gap)
   ├─ For MoE: add router divergence loss (weighted by layer)
   └─ Validate V2 at checkpoints

3. Checkpoint policy
   ├─ V2 passes → Save checkpoint
   ├─ V2 fails but improving → Continue training
   └─ V2 degrading → Reduce LR, increase λ, or rollback
```
