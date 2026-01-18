"""Comprehensive tests for nmoe.rl module."""
from __future__ import annotations

import sys
import torch
import traceback
from dataclasses import dataclass
from typing import Callable, List

# Test result tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    error: str | None = None

RESULTS: List[TestResult] = []

def test(name: str):
    """Decorator for test functions."""
    def decorator(fn: Callable[[], None]):
        def wrapper():
            try:
                fn()
                RESULTS.append(TestResult(name, True))
                print(f"  ✓ {name}")
            except Exception as e:
                RESULTS.append(TestResult(name, False, str(e)))
                print(f"  ✗ {name}: {e}")
                if "-v" in sys.argv:
                    traceback.print_exc()
        return wrapper
    return decorator


# =============================================================================
# Import Tests
# =============================================================================

@test("import: grpo module")
def test_import_grpo():
    from nmoe.rl.grpo import (
        GRPOMetrics,
        grpo_loss,
        group_relative_advantages,
        gdpo_decoupled_advantages,
        compute_kl,
        compute_opsm_mask,
        compute_tis_weights,
        filter_zero_std_groups,
    )

@test("import: rewards module")
def test_import_rewards():
    from nmoe.rl.rewards import (
        RewardResult,
        RewardFunc,
        Rubric,
        format_reward,
        gsm8k_accuracy_reward,
        python_tests_reward,
        condition_reward,
        conditioned_length_reward,
    )

@test("import: rollout module")
def test_import_rollout():
    from nmoe.rl.rollout import Trajectory, generate_one, completion_nll_mean

@test("import: top-level package")
def test_import_package():
    from nmoe.rl import (
        GRPOMetrics, grpo_loss, group_relative_advantages,
        gdpo_decoupled_advantages, RewardResult, Rubric,
    )


# =============================================================================
# KL Divergence Tests
# =============================================================================

@test("kl: k1 (log ratio)")
def test_kl_k1():
    from nmoe.rl.grpo import compute_kl
    log_p = torch.tensor([0.0, -0.5, -1.0])
    log_base = torch.tensor([0.0, 0.0, 0.0])
    kl = compute_kl(log_p, log_base, kl_type="k1")
    expected = log_p - log_base
    assert torch.allclose(kl, expected), f"k1 mismatch: {kl} vs {expected}"

@test("kl: k2 (squared/2)")
def test_kl_k2():
    from nmoe.rl.grpo import compute_kl
    log_p = torch.tensor([0.0, -0.5, -1.0])
    log_base = torch.tensor([0.0, 0.0, 0.0])
    kl = compute_kl(log_p, log_base, kl_type="k2")
    expected = 0.5 * (log_p - log_base).pow(2)
    assert torch.allclose(kl, expected), f"k2 mismatch: {kl} vs {expected}"

@test("kl: k3 (reverse KL, non-negative)")
def test_kl_k3():
    from nmoe.rl.grpo import compute_kl
    log_p = torch.tensor([0.0, -0.5, -1.0, 0.5, 1.0])
    log_base = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    kl = compute_kl(log_p, log_base, kl_type="k3")
    # k3 should always be non-negative
    assert (kl >= 0).all(), f"k3 should be non-negative: {kl}"
    # k3 = exp(-log_ratio) + log_ratio - 1
    log_ratio = log_p - log_base
    expected = torch.exp(-log_ratio) - (-log_ratio) - 1
    assert torch.allclose(kl, expected, atol=1e-5), f"k3 mismatch: {kl} vs {expected}"

@test("kl: invalid type raises")
def test_kl_invalid():
    from nmoe.rl.grpo import compute_kl
    try:
        compute_kl(torch.zeros(3), torch.zeros(3), kl_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown kl_type" in str(e)


# =============================================================================
# Advantage Computation Tests
# =============================================================================

@test("advantages: group mean subtraction")
def test_advantages_mean():
    from nmoe.rl.grpo import group_relative_advantages
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    adv = group_relative_advantages(rewards, normalize_mean=True, normalize_std=False)
    # Each row should have mean 0
    row_means = adv.mean(dim=1)
    assert row_means.abs().max() < 1e-5, f"Row means should be ~0: {row_means}"

@test("advantages: Dr.GRPO (no STD norm by default)")
def test_advantages_dr_grpo():
    from nmoe.rl.grpo import group_relative_advantages
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    # Default should be normalize_std=False (Dr.GRPO)
    adv = group_relative_advantages(rewards)
    # Without STD norm, values are just centered
    expected = rewards - rewards.mean(dim=1, keepdim=True)
    assert torch.allclose(adv, expected), f"Dr.GRPO default mismatch: {adv} vs {expected}"

@test("advantages: with STD norm (original GRPO)")
def test_advantages_std_norm():
    from nmoe.rl.grpo import group_relative_advantages
    rewards = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    adv = group_relative_advantages(rewards, normalize_std=True)
    # Each row should have std ~1
    row_stds = adv.std(dim=1, unbiased=False)
    assert (row_stds - 1.0).abs().max() < 0.1, f"Row stds should be ~1: {row_stds}"

@test("advantages: negative scaling")
def test_advantages_neg_scale():
    from nmoe.rl.grpo import group_relative_advantages
    rewards = torch.tensor([[1.0, 0.5, 0.0]])
    adv_sym = group_relative_advantages(rewards, neg_scale=1.0)
    adv_asym = group_relative_advantages(rewards, neg_scale=2.0)
    # Find negative values
    neg_mask = adv_sym < 0
    # Negatives should be 2x in asymmetric
    assert torch.allclose(adv_asym[neg_mask], adv_sym[neg_mask] * 2.0)

@test("advantages: invalid shape raises")
def test_advantages_invalid_shape():
    from nmoe.rl.grpo import group_relative_advantages
    try:
        group_relative_advantages(torch.zeros(10))  # 1D not allowed
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "rank-2" in str(e)

@test("advantages: RLOO baseline (G/(G-1) scaling)")
def test_advantages_rloo():
    from nmoe.rl.grpo import group_relative_advantages
    rewards = torch.tensor([[1.0, 3.0]])  # G=2

    # Mean baseline: A_i = r_i - mean(r)
    adv_mean = group_relative_advantages(rewards, use_rloo_baseline=False)

    # RLOO baseline: A_i = G/(G-1) * (r_i - mean(r)) = 2 * (r_i - mean)
    adv_rloo = group_relative_advantages(rewards, use_rloo_baseline=True)

    # With G=2, RLOO should be 2x the mean baseline
    expected_ratio = 2.0  # G/(G-1) = 2/1 = 2
    assert torch.allclose(adv_rloo, adv_mean * expected_ratio), \
        f"RLOO should be {expected_ratio}x mean baseline: {adv_rloo} vs {adv_mean * expected_ratio}"

@test("advantages: RLOO scaling factor varies with G")
def test_advantages_rloo_scaling():
    from nmoe.rl.grpo import group_relative_advantages

    # Test different group sizes
    for g in [2, 4, 8, 16]:
        rewards = torch.randn(2, g)
        adv_mean = group_relative_advantages(rewards, use_rloo_baseline=False)
        adv_rloo = group_relative_advantages(rewards, use_rloo_baseline=True)

        expected_scale = g / (g - 1)
        assert torch.allclose(adv_rloo, adv_mean * expected_scale, atol=1e-5), \
            f"G={g}: RLOO scale wrong. Expected {expected_scale}x"


# =============================================================================
# GDPO Decoupled Advantages Tests
# =============================================================================

@test("gdpo: decoupled normalization")
def test_gdpo_decoupled():
    from nmoe.rl.grpo import gdpo_decoupled_advantages
    # Two rewards: accuracy and format
    accuracy = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    format_r = torch.tensor([[1.0, 1.0], [0.0, 1.0]])

    adv = gdpo_decoupled_advantages(
        {"accuracy": accuracy, "format": format_r},
        weights={"accuracy": 1.0, "format": 0.5},
    )

    # Check shape
    assert adv.shape == (2, 2), f"Wrong shape: {adv.shape}"

    # Verify decoupled: each reward normalized separately
    acc_norm = accuracy - accuracy.mean(dim=1, keepdim=True)
    fmt_norm = format_r - format_r.mean(dim=1, keepdim=True)
    expected = 1.0 * acc_norm + 0.5 * fmt_norm
    assert torch.allclose(adv, expected), f"GDPO mismatch: {adv} vs {expected}"

@test("gdpo: prevents reward collapse")
def test_gdpo_no_collapse():
    from nmoe.rl.grpo import gdpo_decoupled_advantages, group_relative_advantages
    # Scenario where GRPO collapses but GDPO doesn't
    # Rewards: (0,1) and (0,2) should have different advantages
    accuracy = torch.tensor([[0.0, 0.0]])
    format_r = torch.tensor([[1.0, 2.0]])

    # GRPO on summed rewards
    summed = accuracy + format_r
    grpo_adv = group_relative_advantages(summed)

    # GDPO decoupled
    gdpo_adv = gdpo_decoupled_advantages(
        {"accuracy": accuracy, "format": format_r},
    )

    # GDPO should preserve distinction even when accuracy is same
    assert gdpo_adv[0, 0] != gdpo_adv[0, 1], f"GDPO should preserve distinction: {gdpo_adv}"

@test("gdpo: empty dict raises")
def test_gdpo_empty():
    from nmoe.rl.grpo import gdpo_decoupled_advantages
    try:
        gdpo_decoupled_advantages({})
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-empty" in str(e)

@test("gdpo: per-reward variance normalization")
def test_gdpo_variance_norm():
    from nmoe.rl.grpo import gdpo_decoupled_advantages
    # Two rewards with different scales
    accuracy = torch.tensor([[0.0, 1.0]])  # range [0,1]
    length = torch.tensor([[0.0, 100.0]])  # range [0,100], much larger scale

    # Without variance normalization, length dominates
    adv_no_norm = gdpo_decoupled_advantages(
        {"accuracy": accuracy, "length": length},
        normalize_std=False,
    )

    # With variance normalization, both contribute equally
    adv_with_norm = gdpo_decoupled_advantages(
        {"accuracy": accuracy, "length": length},
        normalize_std=True,
    )

    # After normalization, each row should have std ~1
    # The normalized values should be more balanced
    acc_centered = accuracy - accuracy.mean(dim=1, keepdim=True)
    len_centered = length - length.mean(dim=1, keepdim=True)

    # Without norm: length (50, -50) dominates accuracy (0.5, -0.5)
    # With norm: both are scaled to same magnitude
    assert abs(adv_no_norm[0, 1].item()) > abs(adv_with_norm[0, 1].item()) * 10, \
        "Without norm, large-scale rewards should dominate more"


# =============================================================================
# OPSM Tests
# =============================================================================

@test("opsm: masks negative-advantage high-KL sequences")
def test_opsm_basic():
    from nmoe.rl.grpo import compute_opsm_mask
    # Seq 0: neg adv, high KL -> mask
    # Seq 1: neg adv, low KL -> keep
    # Seq 2: pos adv, high KL -> keep
    log_probs = torch.tensor([-1.0, -0.5, -0.8])
    log_probs_old = torch.tensor([0.0, -0.4, 0.0])
    advantages = torch.tensor([-0.5, -0.2, 0.5])

    mask, frac = compute_opsm_mask(
        log_probs, log_probs_old, advantages, delta=0.5
    )

    expected = torch.tensor([0.0, 1.0, 1.0])
    assert torch.allclose(mask, expected), f"OPSM mask wrong: {mask} vs {expected}"
    assert abs(frac - 1/3) < 0.01, f"OPSM frac wrong: {frac}"

@test("opsm: all positive advantages -> no masking")
def test_opsm_all_positive():
    from nmoe.rl.grpo import compute_opsm_mask
    log_probs = torch.tensor([-1.0, -1.0, -1.0])
    log_probs_old = torch.zeros(3)
    advantages = torch.tensor([0.5, 0.3, 0.1])  # All positive

    mask, frac = compute_opsm_mask(log_probs, log_probs_old, advantages, delta=0.0)

    assert (mask == 1.0).all(), f"Should not mask positive advantages: {mask}"
    assert frac == 0.0, f"Frac should be 0: {frac}"


# =============================================================================
# TIS Tests
# =============================================================================

@test("tis: clips importance sampling ratios")
def test_tis_clipping():
    from nmoe.rl.grpo import compute_tis_weights
    # Create scenarios: ratio < low, within range, > high
    log_train = torch.tensor([0.0, 0.0, 0.0])
    log_rollout = torch.tensor([2.0, 0.0, -2.0])  # ratios: 0.14, 1.0, 7.4

    weights, clip_frac = compute_tis_weights(
        log_train, log_rollout, clip_low=0.2, clip_high=2.0
    )

    # Check clipping
    assert weights[0].item() >= 0.2, f"Should be clipped to >= 0.2: {weights[0]}"
    assert weights[2].item() <= 2.0, f"Should be clipped to <= 2.0: {weights[2]}"
    assert abs(weights[1].item() - 1.0) < 0.01, f"Unclipped ratio should be 1.0: {weights[1]}"


# =============================================================================
# Zero-STD Filter Tests
# =============================================================================

@test("filter: detects zero-std groups")
def test_filter_zero_std():
    from nmoe.rl.grpo import filter_zero_std_groups
    rewards = torch.tensor([
        [1.0, 2.0, 3.0],  # has variance
        [5.0, 5.0, 5.0],  # zero variance
        [7.0, 8.0, 9.0],  # has variance
    ])

    mask = filter_zero_std_groups(rewards)
    expected = torch.tensor([True, False, True])
    assert (mask == expected).all(), f"Filter mask wrong: {mask} vs {expected}"


# =============================================================================
# GRPO Loss Tests
# =============================================================================

@test("loss: basic computation")
def test_grpo_loss_basic():
    from nmoe.rl.grpo import grpo_loss
    n = 4
    logp = torch.randn(n, requires_grad=True)
    logp_old = logp.detach().clone()
    logp_ref = torch.randn(n)
    advantages = torch.randn(n)

    loss, metrics = grpo_loss(
        logp, logp_old, logp_ref, advantages,
        clip_eps=0.2, kl_coef=0.001,
    )

    assert loss.requires_grad, "Loss should require grad"
    assert isinstance(metrics.loss, float), "Metrics.loss should be float"
    assert isinstance(metrics.kl_mean, float), "Metrics.kl_mean should be float"

@test("loss: gradient flow")
def test_grpo_loss_gradient():
    from nmoe.rl.grpo import grpo_loss
    logp = torch.randn(4, requires_grad=True)
    logp_old = logp.detach().clone()
    logp_ref = torch.randn(4)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    loss, _ = grpo_loss(
        logp, logp_old, logp_ref, advantages,
        clip_eps=0.2, kl_coef=0.001,
    )

    loss.backward()
    assert logp.grad is not None, "Should have gradient"
    assert logp.grad.norm() > 0, "Gradient should be non-zero"

@test("loss: dual-clip PPO caps negative advantage penalty")
def test_grpo_loss_dual_clip():
    from nmoe.rl.grpo import grpo_loss
    # High ratio, negative advantage -> large penalty without dual-clip
    logp = torch.tensor([2.0], requires_grad=True)
    logp_old = torch.tensor([0.0])
    logp_ref = torch.tensor([0.0])
    advantages = torch.tensor([-2.0])

    loss1, _ = grpo_loss(
        logp, logp_old, logp_ref, advantages,
        clip_eps=0.2, kl_coef=0.0, dual_clip_c=None,
    )

    loss2, _ = grpo_loss(
        logp.detach().requires_grad_(), logp_old, logp_ref, advantages,
        clip_eps=0.2, kl_coef=0.0, dual_clip_c=3.0,
    )

    # Dual-clip should reduce the loss
    assert loss2.item() < loss1.item(), f"Dual-clip should reduce loss: {loss2} vs {loss1}"

@test("loss: OPSM integration")
def test_grpo_loss_opsm():
    from nmoe.rl.grpo import grpo_loss
    logp = torch.randn(4, requires_grad=True)
    logp_old = logp.detach().clone()
    logp_ref = torch.randn(4)
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

    _, metrics = grpo_loss(
        logp, logp_old, logp_ref, advantages,
        clip_eps=0.2, kl_coef=0.001,
        use_opsm=True, opsm_delta=1e-4,
    )

    assert hasattr(metrics, 'opsm_frac'), "Should have opsm_frac metric"

@test("loss: invalid params raise")
def test_grpo_loss_invalid():
    from nmoe.rl.grpo import grpo_loss
    logp = torch.randn(4)

    try:
        grpo_loss(logp, logp, logp, logp, clip_eps=-0.1, kl_coef=0.001)
        assert False, "Should raise for negative clip_eps"
    except ValueError:
        pass

    try:
        grpo_loss(logp, logp, logp, logp, clip_eps=0.2, kl_coef=-0.001)
        assert False, "Should raise for negative kl_coef"
    except ValueError:
        pass


# =============================================================================
# Reward Tests
# =============================================================================

@test("reward: format_reward valid")
def test_format_reward_valid():
    from nmoe.rl.rewards import format_reward
    text = "<think>Let me solve this.</think><answer>42</answer>"
    r = format_reward(text)
    assert r.value == 1.0, f"Valid format should get 1.0: {r.value}"
    assert r.category == "valid_format", f"Wrong category: {r.category}"

@test("reward: format_reward missing tags")
def test_format_reward_missing():
    from nmoe.rl.rewards import format_reward
    r = format_reward("Just the answer: 42")
    assert r.value == 0.0, f"Missing tags should get 0.0: {r.value}"
    assert r.category == "no_tags", f"Wrong category: {r.category}"

@test("reward: format_reward wrong order")
def test_format_reward_wrong_order():
    from nmoe.rl.rewards import format_reward
    text = "<answer>42</answer><think>Too late</think>"
    r = format_reward(text)
    assert r.value == 0.0, f"Wrong order should get 0.0: {r.value}"
    assert r.category == "wrong_tag_order", f"Wrong category: {r.category}"

@test("reward: format_reward empty answer")
def test_format_reward_empty():
    from nmoe.rl.rewards import format_reward
    text = "<think>thinking</think><answer></answer>"
    r = format_reward(text)
    assert r.value == 0.0, f"Empty answer should get 0.0: {r.value}"
    assert r.category == "empty_answer", f"Wrong category: {r.category}"

@test("reward: gsm8k_accuracy correct")
def test_gsm8k_correct():
    from nmoe.rl.rewards import gsm8k_accuracy_reward
    text = "<think>2+2=4</think><answer>\\boxed{4}</answer>"
    r = gsm8k_accuracy_reward(text, gold="4")
    assert r.value == 1.0, f"Correct answer should get 1.0: {r.value}"
    assert r.category == "correct", f"Wrong category: {r.category}"

@test("reward: gsm8k_accuracy wrong")
def test_gsm8k_wrong():
    from nmoe.rl.rewards import gsm8k_accuracy_reward
    text = "<think>2+2=5</think><answer>\\boxed{5}</answer>"
    r = gsm8k_accuracy_reward(text, gold="4")
    assert r.value == 0.0, f"Wrong answer should get 0.0: {r.value}"
    assert r.category == "wrong_answer", f"Wrong category: {r.category}"

@test("reward: gsm8k_accuracy no boxed")
def test_gsm8k_no_boxed():
    from nmoe.rl.rewards import gsm8k_accuracy_reward
    text = "<think>thinking</think><answer>The answer is 4</answer>"
    r = gsm8k_accuracy_reward(text, gold="4")
    assert r.value == 0.0, f"No boxed should get 0.0: {r.value}"
    assert r.category == "no_boxed", f"Wrong category: {r.category}"

@test("reward: gsm8k_gold_from_answer_field")
def test_gsm8k_gold():
    from nmoe.rl.rewards import gsm8k_gold_from_answer_field
    answer = "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\n#### 9"
    gold = gsm8k_gold_from_answer_field(answer)
    assert gold == "9", f"Gold extraction failed: {gold}"


# =============================================================================
# Rubric Tests
# =============================================================================

@test("rubric: composition with signature inspection")
def test_rubric_composition():
    from nmoe.rl.rewards import Rubric, format_reward, gsm8k_accuracy_reward

    rubric = Rubric()
    rubric.add(format_reward, weight=0.5, name="format")
    rubric.add(gsm8k_accuracy_reward, weight=1.0, name="accuracy")

    text = "<think>2+2=4</think><answer>\\boxed{4}</answer>"
    r = rubric.score(text, gold="4")

    # format=1.0*0.5 + accuracy=1.0*1.0 = 1.5
    assert abs(r.value - 1.5) < 0.01, f"Rubric score wrong: {r.value}"
    assert "format:valid_format" in r.category, f"Missing format category: {r.category}"
    assert "accuracy:correct" in r.category, f"Missing accuracy category: {r.category}"

@test("rubric: handles extra kwargs via signature inspection")
def test_rubric_kwargs():
    from nmoe.rl.rewards import Rubric, format_reward, gsm8k_accuracy_reward

    rubric = Rubric()
    rubric.add(format_reward, weight=0.5)  # Doesn't take 'gold'
    rubric.add(gsm8k_accuracy_reward, weight=1.0)  # Takes 'gold'

    text = "<think>ok</think><answer>\\boxed{4}</answer>"
    # This should work without error (format_reward ignores 'gold')
    r = rubric.score(text, gold="4")
    assert r.value > 0, f"Rubric should work: {r.value}"


# =============================================================================
# GDPO Conditional Reward Tests
# =============================================================================

@test("condition: gates secondary on primary")
def test_condition_reward():
    from nmoe.rl.rewards import condition_reward

    # Primary achieved -> secondary passes
    r1 = condition_reward(0.8, 1.0, threshold=1.0)
    assert r1 == 0.8, f"Should pass secondary: {r1}"

    # Primary not achieved -> secondary blocked
    r2 = condition_reward(0.8, 0.5, threshold=1.0)
    assert r2 == 0.0, f"Should block secondary: {r2}"

@test("condition: length reward gated on accuracy")
def test_conditioned_length():
    from nmoe.rl.rewards import conditioned_length_reward

    # Accurate -> length reward
    r1 = conditioned_length_reward("short", accuracy_reward=1.0, max_length=100)
    assert r1.value > 0, f"Should give length reward: {r1.value}"
    assert r1.category == "length_ok", f"Wrong category: {r1.category}"

    # Not accurate -> no length reward
    r2 = conditioned_length_reward("short", accuracy_reward=0.0, max_length=100)
    assert r2.value == 0.0, f"Should block length reward: {r2.value}"
    assert r2.category == "accuracy_gate_failed", f"Wrong category: {r2.category}"


# =============================================================================
# Code Execution Tests
# =============================================================================

@test("reward: python_tests pass")
def test_python_tests_pass():
    from nmoe.rl.rewards import python_tests_reward
    code = """<think>Write add function.</think><answer>
def add(a, b):
    return a + b
</answer>"""
    tests = "assert add(1, 2) == 3\nassert add(-1, 1) == 0"
    r = python_tests_reward(code, tests=tests)
    assert r.value == 1.0, f"Passing tests should get 1.0: {r.value}"
    assert r.category == "tests_passed", f"Wrong category: {r.category}"

@test("reward: python_tests fail")
def test_python_tests_fail():
    from nmoe.rl.rewards import python_tests_reward
    code = """<think>Wrong implementation.</think><answer>
def add(a, b):
    return a - b  # Wrong!
</answer>"""
    tests = "assert add(1, 2) == 3"
    r = python_tests_reward(code, tests=tests)
    assert r.value == 0.0, f"Failing tests should get 0.0: {r.value}"
    assert r.category == "tests_failed", f"Wrong category: {r.category}"

@test("reward: python_tests syntax error")
def test_python_tests_syntax():
    from nmoe.rl.rewards import python_tests_reward
    code = """<think>Syntax error.</think><answer>
def add(a, b)
    return a + b
</answer>"""
    tests = "assert add(1, 2) == 3"
    r = python_tests_reward(code, tests=tests)
    assert r.value == 0.0, f"Syntax error should get 0.0: {r.value}"

@test("reward: python_tests timeout")
def test_python_tests_timeout():
    from nmoe.rl.rewards import python_tests_reward
    code = """<think>Infinite loop.</think><answer>
while True:
    pass
</answer>"""
    tests = "pass"
    r = python_tests_reward(code, tests=tests, timeout_s=0.5)
    assert r.value == 0.0, f"Timeout should get 0.0: {r.value}"
    assert r.category == "timeout", f"Wrong category: {r.category}"


# =============================================================================
# Rollout NLL Tests
# =============================================================================

@test("rollout: nll with actual length normalization")
def test_nll_actual_length():
    from nmoe.rl.rollout import completion_nll_mean
    import torch.nn as nn

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 100)
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 100, device=x.device)

    model = MockModel().cuda()
    seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]]
    prompt_lens = [2, 2]
    completion_lens = [3, 5]

    nll = completion_nll_mean(
        model, seqs=seqs, prompt_lens=prompt_lens, completion_lens=completion_lens,
        pad_id=0, device=torch.device("cuda"),
        max_length=None,  # Actual length normalization
    )

    assert nll.shape == (2,), f"Wrong shape: {nll.shape}"
    assert (nll > 0).all(), f"NLL should be positive: {nll}"

@test("rollout: nll with constant length normalization (Dr.GRPO)")
def test_nll_constant_length():
    from nmoe.rl.rollout import completion_nll_mean
    import torch.nn as nn

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return torch.randn(x.shape[0], x.shape[1], 100, device=x.device)

    model = MockModel().cuda()
    seqs = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7]]
    prompt_lens = [2, 2]
    completion_lens = [3, 5]

    # With constant normalization
    nll_const = completion_nll_mean(
        model, seqs=seqs, prompt_lens=prompt_lens, completion_lens=completion_lens,
        pad_id=0, device=torch.device("cuda"),
        max_length=10,  # Constant normalization
    )

    # With actual length normalization
    nll_actual = completion_nll_mean(
        model, seqs=seqs, prompt_lens=prompt_lens, completion_lens=completion_lens,
        pad_id=0, device=torch.device("cuda"),
        max_length=None,
    )

    # Different normalization should give different values
    # (unless by coincidence)
    assert nll_const.shape == nll_actual.shape


# =============================================================================
# Config Tests
# =============================================================================

@test("config: rl options exist")
def test_config_rl_options():
    from nmoe.config import Config
    cfg = Config(
        dim=256, n_layers=2, n_heads=4, inter_dim=512,
        rl_enabled=True,
        rl_algorithm="grpo",
        rl_group_size=8,
        rl_kl_coef=0.001,
        rl_clip_eps=10.0,
        rl_kl_type="k3",
        rl_normalize_mean=True,
        rl_normalize_std=False,
        rl_length_norm_constant=True,
        rl_use_opsm=True,
        rl_neg_adv_scale=1.5,
        rl_dual_clip_c=3.0,
    )

    assert cfg.rl_enabled == True
    assert cfg.rl_group_size == 8
    assert cfg.rl_clip_eps == 10.0
    assert cfg.rl_normalize_std == False
    assert cfg.rl_dual_clip_c == 3.0


# =============================================================================
# Run Tests
# =============================================================================

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("NMOE RL Module Test Suite")
    print("="*60 + "\n")

    # Collect all test functions
    tests = [
        # Imports
        test_import_grpo,
        test_import_rewards,
        test_import_rollout,
        test_import_package,
        # KL
        test_kl_k1,
        test_kl_k2,
        test_kl_k3,
        test_kl_invalid,
        # Advantages
        test_advantages_mean,
        test_advantages_dr_grpo,
        test_advantages_std_norm,
        test_advantages_neg_scale,
        test_advantages_invalid_shape,
        test_advantages_rloo,
        test_advantages_rloo_scaling,
        # GDPO
        test_gdpo_decoupled,
        test_gdpo_no_collapse,
        test_gdpo_empty,
        test_gdpo_variance_norm,
        # OPSM
        test_opsm_basic,
        test_opsm_all_positive,
        # TIS
        test_tis_clipping,
        # Filter
        test_filter_zero_std,
        # Loss
        test_grpo_loss_basic,
        test_grpo_loss_gradient,
        test_grpo_loss_dual_clip,
        test_grpo_loss_opsm,
        test_grpo_loss_invalid,
        # Rewards
        test_format_reward_valid,
        test_format_reward_missing,
        test_format_reward_wrong_order,
        test_format_reward_empty,
        test_gsm8k_correct,
        test_gsm8k_wrong,
        test_gsm8k_no_boxed,
        test_gsm8k_gold,
        # Rubric
        test_rubric_composition,
        test_rubric_kwargs,
        # Conditional
        test_condition_reward,
        test_conditioned_length,
        # Code execution
        test_python_tests_pass,
        test_python_tests_fail,
        test_python_tests_syntax,
        test_python_tests_timeout,
        # Rollout (GPU required)
        test_nll_actual_length,
        test_nll_constant_length,
        # Config
        test_config_rl_options,
    ]

    # Run tests
    for test_fn in tests:
        test_fn()

    # Summary
    print("\n" + "="*60)
    passed = sum(1 for r in RESULTS if r.passed)
    failed = sum(1 for r in RESULTS if not r.passed)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed > 0:
        print("\nFailed tests:")
        for r in RESULTS:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
        return 1

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
