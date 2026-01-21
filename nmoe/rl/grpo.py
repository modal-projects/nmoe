"""GRPO (Group Relative Policy Optimization) with anti-reward-hacking measures.

Implements:
- Group-relative advantages (no critic needed)
- PPO-style clipping
- Reverse KL constraint to reference policy
- OPSM (Off-Policy Sequence Masking) - masks negative-advantage, high-KL sequences
- TIS (Truncated Importance Sampling) - clips IS ratios
- Dr.GRPO constant divisor normalization (arxiv 2503.20783)
- 3-tier importance ratio masking (token/sequence/geometric) from prime-rl

Reference:
- DeepSeek-R1 (https://arxiv.org/abs/2501.12948)
- THUDM/slime ppo_utils.py
- primeintellect/prime-rl loss.py
- Schulman KL approximations: http://joschu.net/blog/kl-approx.html
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch


@dataclass(frozen=True)
class GRPOMetrics:
    """Metrics from GRPO loss computation."""
    loss: float
    pg_loss: float
    kl_loss: float
    advantage_mean: float
    advantage_std: float
    kl_mean: float
    ratio_mean: float
    clip_frac: float
    opsm_frac: float = 0.0  # Fraction of sequences masked by OPSM
    # 3-tier importance masking stats
    token_mask_frac: float = 0.0  # Fraction of tokens masked
    seq_mask_frac: float = 0.0  # Fraction of sequences masked
    geo_mask_frac: float = 0.0  # Fraction masked by geometric ratio


@dataclass(frozen=True)
class ImportanceMaskConfig:
    """Configuration for 3-tier importance ratio masking.

    From primeintellect/prime-rl: masks tokens/sequences where the
    importance ratio (π/π_old) is too far from 1.0, indicating
    the policy has changed significantly since rollout.

    Three tiers:
    1. Token-level: mask individual tokens with extreme ratios
    2. Sequence-level: mask entire sequence if any token is extreme
    3. Geometric: mask if geometric mean ratio is extreme
    """
    # Token-level bounds (default from prime-rl)
    token_ratio_low: float = 0.125  # exp(-2.08)
    token_ratio_high: float = 8.0   # exp(2.08)

    # Sequence-level bounds
    seq_ratio_low: float = 0.0      # Disabled by default
    seq_ratio_high: float = 100.0   # Very permissive

    # Geometric mean bounds
    geo_ratio_low: float = 0.1      # exp(-2.3)
    geo_ratio_high: float = 10.0    # exp(2.3)

    # Whether to use each tier
    use_token_mask: bool = True
    use_seq_mask: bool = True
    use_geo_mask: bool = True


# =============================================================================
# KL Divergence Variants
# =============================================================================

def compute_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_type: Literal["k1", "k2", "k3"] = "k3",
) -> torch.Tensor:
    """Compute KL divergence approximation.

    Reference: Schulman blog http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of current policy.
        log_probs_base: Log probabilities of base (reference) policy.
        kl_type: Type of KL estimator:
            - k1: Simple log ratio (biased, can be negative)
            - k2: Squared log ratio / 2 (always positive, higher variance)
            - k3: exp(-log_ratio) - 1 + log_ratio (unbiased, lower variance, non-negative)

    Returns:
        Per-element KL divergence estimates.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_type == "k1":
        # Simple log ratio: E[log(π/π_ref)]
        return log_ratio
    elif kl_type == "k2":
        # Squared: (1/2) * E[(log(π/π_ref))^2]
        return 0.5 * log_ratio.pow(2)
    elif kl_type == "k3":
        # Reverse KL (DeepSeek-R1 formula):
        # D_KL(π||π_ref) = π_ref/π - log(π_ref/π) - 1
        #                = exp(-log_ratio) - (-log_ratio) - 1
        #                = exp(-log_ratio) + log_ratio - 1
        neg_log_ratio = -log_ratio
        return neg_log_ratio.exp() - neg_log_ratio - 1.0
    else:
        raise ValueError(f"Unknown kl_type: {kl_type}")


# =============================================================================
# Advantage Computation
# =============================================================================

def gdpo_decoupled_advantages(
    rewards_dict: dict[str, torch.Tensor],
    *,
    weights: dict[str, float] | None = None,
    normalize_mean: bool = True,
    normalize_std: bool = False,
    neg_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GDPO: Decoupled normalization for multi-reward settings.

    Reference: arxiv 2601.05242 "GDPO"

    The key insight is that normalizing summed rewards causes different reward
    combinations to collapse to identical advantages. GDPO normalizes each
    reward separately before aggregation, preserving training signal resolution.

    Args:
        rewards_dict: Dict of reward name -> [B, G] tensor
        weights: Dict of reward name -> weight (default 1.0 each)
        normalize_mean: Subtract group mean per reward
        normalize_std: Divide by group std per reward (for different reward scales)
        neg_scale: Scale factor for negative advantages
        eps: Numerical stability for std division

    Returns:
        advantages: [B, G] aggregated advantages
    """
    if not rewards_dict:
        raise ValueError("rewards_dict must be non-empty")

    weights = weights or {}
    first_key = next(iter(rewards_dict))
    shape = rewards_dict[first_key].shape
    device = rewards_dict[first_key].device
    dtype = rewards_dict[first_key].dtype

    if len(shape) != 2:
        raise ValueError(f"rewards must be rank-2 [B,G] (got shape={tuple(shape)})")

    # Normalize each reward separately (GDPO key insight)
    aggregated = torch.zeros(shape, device=device, dtype=dtype)
    for name, rewards in rewards_dict.items():
        if rewards.shape != shape:
            raise ValueError(f"Shape mismatch: {name} has {tuple(rewards.shape)}, expected {tuple(shape)}")

        w = weights.get(name, 1.0)
        mean = rewards.mean(dim=1, keepdim=True) if normalize_mean else 0.0
        centered = rewards - mean

        if normalize_std:
            # Per-reward variance normalization for different reward scales
            var = centered.pow(2).mean(dim=1, keepdim=True)
            std = torch.sqrt(var + eps)
            normalized = centered / std
        else:
            normalized = centered

        aggregated = aggregated + w * normalized

    # Asymmetric scaling for negative advantages
    if neg_scale != 1.0:
        aggregated = torch.where(aggregated < 0, aggregated * neg_scale, aggregated)

    return aggregated


def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    normalize_mean: bool = True,
    normalize_std: bool = False,
    use_rloo_baseline: bool = False,
    neg_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute group-relative advantages: A_i = r_i - baseline.

    This is the core GRPO insight: no critic needed, just use group statistics.

    NOTE on Dr. GRPO (arxiv 2503.20783):
    - normalize_std=True introduces "question-level difficulty bias"
    - Questions with low reward std get disproportionate weight
    - Dr. GRPO recommends normalize_std=False

    NOTE on RLOO vs mean baseline:
    - Mean baseline: A_i = r_i - mean(r)
    - RLOO baseline: A_i = r_i - mean_{j≠i}(r_j) = G/(G-1) * (r_i - mean(r))
    - For small G (e.g., G=2), RLOO gives 2x larger advantages
    - This scaling interacts with PPO clipping and KL penalties

    WARNING: Do not combine use_rloo_baseline=True with normalize_std=True.
    RLOO scaling after std normalization is not "RLOO baseline" - it's
    "std-normalized advantages scaled by G/(G-1)", a different estimator.

    NOTE on neg_scale:
    - Multiplying only negative advantages breaks zero-mean property
    - In PPO with ratios/clipping, this is a shaping choice, not just variance
    - E[A∇logπ] baseline cancellation argument is weaker here

    Args:
        rewards: [B, G] float tensor where B=prompts, G=samples per prompt
        normalize_mean: Subtract group mean (standard GRPO)
        normalize_std: Divide by group std (NOT recommended per Dr. GRPO)
        use_rloo_baseline: Use leave-one-out baseline (G/(G-1) scaling)
        neg_scale: Scale factor for negative advantages (shaping, not variance-only)
        eps: Numerical stability for std division

    Returns:
        advantages: [B, G] normalized advantages
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be rank-2 [B,G] (got shape={tuple(rewards.shape)})")

    if not normalize_mean and not normalize_std:
        adv = rewards
    else:
        mean = rewards.mean(dim=1, keepdim=True) if normalize_mean else 0.0

        if normalize_std:
            # WARNING: This introduces question-level difficulty bias per Dr. GRPO
            var = ((rewards - mean) if normalize_mean else rewards).pow(2).mean(dim=1, keepdim=True)
            std = torch.sqrt(var + eps)
            adv = (rewards - mean) / std
        else:
            adv = rewards - mean

        # RLOO scaling: A_i = r_i - mean_{j≠i}(r_j) = G/(G-1) * (r_i - mean(r))
        # This gives larger advantages for small G (e.g., G=2 gives 2x scaling)
        if use_rloo_baseline and normalize_mean:
            G = rewards.shape[1]
            if G > 1:
                rloo_scale = G / (G - 1)
                adv = adv * rloo_scale

    # Asymmetric scaling: penalize bad rollouts more heavily
    if neg_scale != 1.0:
        adv = torch.where(adv < 0, adv * neg_scale, adv)

    return adv


# =============================================================================
# OPSM (Off-Policy Sequence Masking)
# =============================================================================

def compute_opsm_mask(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    *,
    delta: float = 1e-4,
) -> Tuple[torch.Tensor, float]:
    """Compute Off-Policy Sequence Masking (OPSM).

    Masks sequences where:
    1. Advantage is negative (bad sample)
    2. KL divergence is high (off-policy)

    This prevents learning on sequences that are both bad AND far from
    the current policy, which reduces reward hacking and variance.

    NOTE: This is asymmetric by design - it masks when log_ratio < -delta
    (i.e., ratio < exp(-delta)), detecting "policy has moved away from old".
    It does NOT mask when ratio is large (policy moving toward old actions).
    This is intentional: we want to allow the policy to increase probability
    of actions the old policy took, but prevent it from learning on sequences
    where it has already diverged significantly AND those sequences are bad.

    IMPORTANT: In scaled-logprob PPO (with constant length normalization),
    delta operates in scaled space. The true ratio threshold is:
        mask if logp_old_scaled - logp_scaled > delta
        ⇔ (log π_old - log π) / C > delta
        ⇔ π / π_old < exp(-delta * C)

    Example: delta=1e-4 with C=512 masks when π/π_old < exp(-0.0512) ≈ 0.95.
    So OPSM is NOT tiny - it masks when true ratio drops below ~95% for
    negative-advantage samples. Adjust delta accordingly for your C.

    Reference: THUDM/slime ppo_utils.py

    Args:
        log_probs: [N] current policy log-probs (per sequence mean)
        log_probs_old: [N] old policy log-probs (per sequence mean)
        advantages: [N] advantage values
        delta: KL threshold for masking

    Returns:
        Tuple of (opsm_mask, opsm_frac) where:
        - opsm_mask: [N] mask (1=keep, 0=mask out)
        - opsm_frac: fraction of sequences masked
    """
    # Sequence-level KL: (old - new) approximates KL(old || new)
    # When seq_kl > delta: current policy has moved away from old policy
    seq_kl = log_probs_old - log_probs

    # Mask if advantage < 0 AND KL > delta
    should_mask = (advantages < 0) & (seq_kl > delta)
    opsm_mask = (~should_mask).float()
    opsm_frac = should_mask.float().mean().item()

    return opsm_mask, opsm_frac


# =============================================================================
# TIS (Truncated Importance Sampling)
# =============================================================================

def compute_tis_weights(
    log_probs_train: torch.Tensor,
    log_probs_rollout: torch.Tensor,
    *,
    clip_low: float = 0.1,
    clip_high: float = 2.0,
) -> Tuple[torch.Tensor, float]:
    """Compute Truncated Importance Sampling weights.

    When training policy differs from rollout policy, IS correction
    reduces variance from distribution shift.

    NOTE: This function is implemented but NOT currently integrated into
    the training loop. It's available for future use when:
    - Using a separate rollout policy from training policy
    - Doing off-policy correction in multi-epoch PPO
    - Combining samples from multiple policy checkpoints

    To integrate, multiply the PG loss by TIS weights before reduction.

    Args:
        log_probs_train: Log-probs from current training policy
        log_probs_rollout: Log-probs from rollout policy
        clip_low: Lower bound for IS ratio
        clip_high: Upper bound for IS ratio

    Returns:
        Tuple of (tis_weights, clip_frac)
    """
    is_ratio = torch.exp(log_probs_train - log_probs_rollout)
    tis_weights = is_ratio.clamp(min=clip_low, max=clip_high)
    clip_frac = (tis_weights != is_ratio).float().mean().item()
    return tis_weights, clip_frac


# =============================================================================
# 3-Tier Importance Ratio Masking (from prime-rl)
# =============================================================================

def compute_importance_mask(
    log_probs: torch.Tensor,
    log_probs_old: torch.Tensor,
    loss_mask: torch.Tensor,
    config: ImportanceMaskConfig | None = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Compute 3-tier importance ratio mask.

    From primeintellect/prime-rl loss.py: masks tokens/sequences where
    importance ratio is too extreme, indicating significant policy shift.

    Three tiers (applied independently, combined with OR):
    1. Token-level: mask if ratio < token_low or > token_high
    2. Sequence-level: mask entire seq if min/max ratio outside bounds
    3. Geometric: mask if exp(mean(log_ratio)) outside bounds

    Args:
        log_probs: [batch, seq] per-token log-probs from current policy
        log_probs_old: [batch, seq] per-token log-probs from rollout policy
        loss_mask: [batch, seq] boolean mask (True = include in loss)
        config: ImportanceMaskConfig (uses defaults if None)

    Returns:
        Tuple of:
        - keep_mask: [batch, seq] boolean mask (True = keep, False = mask out)
        - stats: dict with masking fractions
    """
    if config is None:
        config = ImportanceMaskConfig()

    # Compute per-token importance ratio
    log_ratio = log_probs - log_probs_old
    ratio = torch.exp(log_ratio)

    batch, seq = log_probs.shape
    device = log_probs.device

    # Initialize: start with loss_mask (only consider tokens in loss)
    is_masked = torch.zeros_like(loss_mask, dtype=torch.bool)

    stats = {}

    # --- Tier 1: Token-level masking ---
    if config.use_token_mask:
        token_mask_low = ratio < config.token_ratio_low
        token_mask_high = ratio > config.token_ratio_high
        token_masked = token_mask_low | token_mask_high
        is_masked = is_masked | token_masked

        # Stats: fraction of loss tokens masked
        loss_mask_bool = loss_mask.bool()
        stats["token_mask_low_frac"] = (token_mask_low & loss_mask_bool).float().sum() / loss_mask.float().sum().clamp(min=1)
        stats["token_mask_high_frac"] = (token_mask_high & loss_mask_bool).float().sum() / loss_mask.float().sum().clamp(min=1)

    # --- Tier 2: Sequence-level masking ---
    if config.use_seq_mask:
        # For each sequence, check min/max ratio among loss tokens
        # Use inf/-inf for tokens outside loss_mask
        loss_mask_bool = loss_mask.bool()
        ratio_for_min = torch.where(loss_mask_bool, ratio, torch.tensor(float('inf'), device=device))
        ratio_for_max = torch.where(loss_mask_bool, ratio, torch.tensor(float('-inf'), device=device))

        seq_min_ratio = ratio_for_min.min(dim=1, keepdim=True).values  # [batch, 1]
        seq_max_ratio = ratio_for_max.max(dim=1, keepdim=True).values  # [batch, 1]

        seq_mask_low = seq_min_ratio < config.seq_ratio_low
        seq_mask_high = seq_max_ratio > config.seq_ratio_high

        # Broadcast to all tokens in sequence
        seq_masked = (seq_mask_low | seq_mask_high).expand(-1, seq)
        is_masked = is_masked | seq_masked

        stats["seq_mask_low_frac"] = seq_mask_low.float().mean().item()
        stats["seq_mask_high_frac"] = seq_mask_high.float().mean().item()

    # --- Tier 3: Geometric mean masking ---
    if config.use_geo_mask:
        # Geometric mean = exp(mean(log_ratio)) over loss tokens
        # Compute mean log_ratio per sequence (only over loss tokens)
        loss_mask_bool = loss_mask.bool()
        log_ratio_masked = torch.where(loss_mask_bool, log_ratio, torch.zeros_like(log_ratio))
        token_counts = loss_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        mean_log_ratio = log_ratio_masked.sum(dim=1, keepdim=True) / token_counts
        geo_ratio = torch.exp(mean_log_ratio.clamp(max=10.0))  # Clamp for stability

        geo_mask_low = geo_ratio < config.geo_ratio_low
        geo_mask_high = geo_ratio > config.geo_ratio_high

        # Broadcast to all tokens in sequence
        geo_masked = (geo_mask_low | geo_mask_high).expand(-1, seq)
        is_masked = is_masked | geo_masked

        stats["geo_mask_low_frac"] = geo_mask_low.float().mean().item()
        stats["geo_mask_high_frac"] = geo_mask_high.float().mean().item()
        stats["geo_ratio_mean"] = geo_ratio.mean().item()

    # Final keep mask: in loss AND not masked
    keep_mask = loss_mask.bool() & ~is_masked

    # Overall stats
    total_loss_tokens = loss_mask.float().sum()
    total_kept = keep_mask.float().sum()
    stats["total_mask_frac"] = 1.0 - (total_kept / total_loss_tokens.clamp(min=1)).item()

    return keep_mask, stats


# =============================================================================
# Dr.GRPO Constant Divisor Normalization
# =============================================================================

def dr_grpo_normalize_loss(
    token_losses: torch.Tensor,
    loss_mask: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Dr.GRPO: Normalize loss by constant divisor instead of actual length.

    From arxiv 2503.20783: Standard per-token loss normalization divides by
    actual sequence length, which introduces "question-level difficulty bias".
    Questions with high reward variance dominate training because their
    gradients are larger per-token.

    Dr.GRPO fixes this by dividing by a constant (max_length) instead:
        loss = sum(token_losses) / max_length

    This ensures all questions contribute equally regardless of length variance.

    Args:
        token_losses: [batch, seq] per-token losses (already masked by loss_mask)
        loss_mask: [batch, seq] boolean mask for which tokens to include
        max_length: Constant divisor (e.g., max sequence length in batch)

    Returns:
        Scalar loss value
    """
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0 (got {max_length})")

    # Sum losses over sequence dimension
    seq_losses = (token_losses * loss_mask.float()).sum(dim=1)  # [batch]

    # Divide by constant instead of actual length
    normalized = seq_losses / max_length

    return normalized.mean()


def length_weighted_baseline(
    rewards: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute length-weighted baseline for advantage computation.

    From primeintellect/prime-rl: When sequences have different lengths,
    the baseline should account for length to avoid bias.

    baseline = sum(rewards * lengths) / sum(lengths)

    This gives longer sequences more weight in the baseline, which
    makes sense if longer sequences are inherently harder.

    Args:
        rewards: [B, G] reward tensor
        lengths: [B, G] sequence length tensor

    Returns:
        [B, 1] baseline per prompt group
    """
    if rewards.shape != lengths.shape:
        raise ValueError(f"Shape mismatch: rewards {rewards.shape} vs lengths {lengths.shape}")

    weighted_sum = (rewards * lengths).sum(dim=1, keepdim=True)
    length_sum = lengths.sum(dim=1, keepdim=True).clamp(min=1)

    return weighted_sum / length_sum


# =============================================================================
# REINFORCE++ Token-Level Returns (from slime)
# =============================================================================

def reinforce_pp_returns(
    rewards: torch.Tensor,
    kl: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    kl_coef: float = 0.01,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute REINFORCE++ token-level discounted returns.

    From arxiv 2501.03262: REINFORCE++ assigns token-level rewards:
    - Each token gets -kl_coef * KL(token) as immediate reward
    - Final token also gets the terminal reward

    Then computes discounted returns backwards:
        G_t = r_t + gamma * G_{t+1}

    This creates a credit assignment signal that propagates
    the final reward backwards through the sequence.

    Args:
        rewards: [batch] scalar rewards per sequence
        kl: [batch, seq] per-token KL divergence
        loss_mask: [batch, seq] mask for response tokens
        kl_coef: KL penalty coefficient
        gamma: Discount factor (1.0 = no discounting)

    Returns:
        [batch, seq] token-level returns G_t
    """
    batch, seq_len = kl.shape

    # Token-level rewards: -kl_coef * KL per token
    token_rewards = -kl_coef * kl  # [batch, seq]

    # Add terminal reward at last response token
    # Find last valid token per sequence
    last_indices = loss_mask.long().cumsum(dim=1).argmax(dim=1)  # [batch]

    # Add terminal reward at last position
    batch_indices = torch.arange(batch, device=kl.device)
    token_rewards[batch_indices, last_indices] += rewards

    # Compute discounted returns backwards
    returns = torch.zeros_like(token_rewards)
    running_return = torch.zeros(batch, device=kl.device, dtype=kl.dtype)

    for t in reversed(range(seq_len)):
        # G_t = r_t + gamma * G_{t+1}
        running_return = token_rewards[:, t] + gamma * running_return
        # Only accumulate for valid tokens
        running_return = torch.where(
            loss_mask[:, t].bool(),
            running_return,
            torch.zeros_like(running_return)
        )
        returns[:, t] = running_return

    return returns


def reinforce_pp_baseline_advantages(
    rewards: torch.Tensor,
    kl: torch.Tensor,
    *,
    kl_coef: float = 0.01,
) -> torch.Tensor:
    """Compute REINFORCE++ baseline advantages.

    Simpler variant that broadcasts scalar (reward - baseline)
    to all tokens, with KL penalty per token.

    A_t = (r - baseline) - kl_coef * KL_t

    This is faster than full returns computation but loses
    the temporal credit assignment signal.

    Args:
        rewards: [batch] scalar rewards (baseline already subtracted)
        kl: [batch, seq] per-token KL divergence
        kl_coef: KL penalty coefficient

    Returns:
        [batch, seq] token-level advantages
    """
    # Broadcast reward to all tokens
    advantages = rewards.unsqueeze(1).expand_as(kl) - kl_coef * kl
    return advantages


# =============================================================================
# GRPO Loss
# =============================================================================

def grpo_loss(
    logp_mean: torch.Tensor,
    logp_mean_old: torch.Tensor,
    logp_mean_ref: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float,
    clip_eps_high: float | None = None,
    dual_clip_c: float | None = None,
    kl_coef: float,
    kl_type: Literal["k1", "k2", "k3"] = "k3",
    use_opsm: bool = False,
    opsm_delta: float = 1e-4,
    opsm_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, GRPOMetrics]:
    """GRPO objective with PPO-style clipping and KL-to-reference penalty.

    Loss = -E[min(ratio * A, clip(ratio) * A)] + β * KL(π || π_ref)

    Supports dual-clip PPO for asymmetric handling of negative advantages:
    when advantage < 0 and dual_clip_c is set, applies extra clipping to
    prevent the policy from decreasing bad action probabilities too much.

    All inputs are per-sample scalars (sequence-mean logprobs).

    Args:
        logp_mean: [N] current policy mean logprob over completion tokens
        logp_mean_old: [N] rollout policy mean logprob (detached)
        logp_mean_ref: [N] reference policy mean logprob (detached)
        advantages: [N] group-normalized advantages (detached)
        clip_eps: PPO clipping epsilon lower bound (e.g., 0.2)
        clip_eps_high: PPO clipping epsilon upper bound (default: same as clip_eps)
        dual_clip_c: Dual-clip coefficient for negative advantages (e.g., 3.0).
                     Must be > 1.0. When set, negative advantages are clipped
                     more aggressively to prevent reward hacking.
        kl_coef: KL penalty coefficient (e.g., 0.001)
        kl_type: KL divergence type ("k1", "k2", "k3")
        use_opsm: Enable Off-Policy Sequence Masking
        opsm_delta: KL threshold for OPSM
        opsm_mask: Pre-computed OPSM mask (optional)

    Returns:
        Tuple of (loss, GRPOMetrics)
    """
    if clip_eps < 0.0:
        raise ValueError(f"clip_eps must be >= 0 (got {clip_eps})")
    if kl_coef < 0.0:
        raise ValueError(f"kl_coef must be >= 0 (got {kl_coef})")
    if dual_clip_c is not None and dual_clip_c <= 1.0:
        raise ValueError(f"dual_clip_c must be > 1.0 (got {dual_clip_c})")

    if clip_eps_high is None:
        clip_eps_high = clip_eps

    if not (logp_mean.shape == logp_mean_old.shape == logp_mean_ref.shape == advantages.shape):
        raise ValueError(
            "shape mismatch: "
            f"logp={tuple(logp_mean.shape)} old={tuple(logp_mean_old.shape)} "
            f"ref={tuple(logp_mean_ref.shape)} adv={tuple(advantages.shape)}"
        )

    # Policy ratio
    log_ratio = logp_mean - logp_mean_old
    ratio = torch.exp(log_ratio)

    # Clipped ratio (asymmetric: can have different lower/upper bounds)
    if clip_eps == 0.0 and clip_eps_high == 0.0:
        ratio_clipped = ratio
    else:
        ratio_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps_high)

    # Policy gradient objective (PPO-style)
    pg = ratio * advantages
    pg_clip = ratio_clipped * advantages
    pg_obj = torch.minimum(pg, pg_clip)

    # Dual-clip PPO: extra clipping for negative advantages
    # This prevents the policy from being too aggressive in reducing
    # probability of bad actions, which can lead to reward hacking.
    # For negative advantages, we limit the objective to be at least c * A
    # (which is negative), preventing the loss from growing unboundedly.
    if dual_clip_c is not None:
        # pg_obj for negative A is negative (we're penalizing bad actions)
        # We want to cap how negative it can get: max(pg_obj, c * A)
        # Since A < 0, c * A < 0, so this limits the penalty magnitude
        pg_floor = dual_clip_c * advantages  # negative for negative A
        pg_obj_dual = torch.maximum(pg_floor, pg_obj)
        # Only apply dual-clip when advantage < 0
        pg_obj = torch.where(advantages < 0, pg_obj_dual, pg_obj)

    # KL penalty to reference
    kl = compute_kl(logp_mean, logp_mean_ref, kl_type=kl_type)

    # OPSM: mask out bad off-policy sequences
    opsm_frac = 0.0
    if use_opsm:
        if opsm_mask is None:
            opsm_mask, opsm_frac = compute_opsm_mask(
                logp_mean.detach(), logp_mean_old, advantages,
                delta=opsm_delta,
            )
        pg_obj = pg_obj * opsm_mask

    # Final loss
    pg_loss = -pg_obj.mean()
    kl_loss = (kl_coef * kl).mean()
    loss = pg_loss + kl_loss

    # Metrics
    with torch.no_grad():
        if clip_eps > 0 or clip_eps_high > 0:
            clip_frac = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps_high))).float().mean()
        else:
            clip_frac = torch.zeros((), device=ratio.device)

        metrics = GRPOMetrics(
            loss=float(loss.detach().item()),
            pg_loss=float(pg_loss.detach().item()),
            kl_loss=float(kl_loss.detach().item()),
            advantage_mean=float(advantages.detach().mean().item()),
            advantage_std=float(advantages.detach().std(unbiased=False).item()),
            kl_mean=float(kl.detach().mean().item()),
            ratio_mean=float(ratio.detach().mean().item()),
            clip_frac=float(clip_frac.detach().item()),
            opsm_frac=opsm_frac,
        )

    return loss, metrics


# =============================================================================
# Dynamic Sampling Filter
# =============================================================================

def filter_zero_std_groups(
    rewards: torch.Tensor,
    *,
    min_std: float = 1e-6,
) -> torch.Tensor:
    """Filter out prompt groups with zero reward variance.

    Groups where all samples have the same reward provide no learning signal.
    This returns a mask indicating which groups to keep.

    Args:
        rewards: [B, G] reward tensor
        min_std: Minimum std to consider non-zero

    Returns:
        [B] boolean mask (True=keep, False=filter)
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be rank-2 [B,G] (got shape={tuple(rewards.shape)})")

    std = rewards.std(dim=1, unbiased=False)
    return std > min_std
