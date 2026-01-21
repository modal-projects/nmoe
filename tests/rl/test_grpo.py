"""Tests for nmoe.rl.grpo module."""
import pytest
import torch
import inspect
import math


class TestGrpoLossAPI:
    """Test grpo_loss function API contract."""

    def test_signature_has_required_params(self):
        """grpo_loss must accept the new API parameters."""
        from nmoe.rl.grpo import grpo_loss

        sig = inspect.signature(grpo_loss)
        params = set(sig.parameters.keys())

        required_positional = {"logp_mean", "logp_mean_old", "logp_mean_ref", "advantages"}
        required_kwargs = {"clip_eps", "kl_coef"}

        assert required_positional.issubset(params), f"Missing: {required_positional - params}"
        assert required_kwargs.issubset(params), f"Missing: {required_kwargs - params}"

    def test_rejects_old_api_kwargs(self):
        """grpo_loss must reject old API kwargs (policy_nll, ref_nll, group_size)."""
        from nmoe.rl.grpo import grpo_loss

        with pytest.raises(TypeError):
            grpo_loss(
                policy_nll=torch.zeros(4),
                ref_nll=torch.zeros(4),
                rewards=torch.zeros(4),
                kl_coef=0.01,
                group_size=2,
            )

    def test_accepts_new_api(self):
        """grpo_loss must accept new API and return (loss, metrics)."""
        from nmoe.rl.grpo import grpo_loss

        loss, metrics = grpo_loss(
            logp_mean=torch.randn(4),
            logp_mean_old=torch.randn(4),
            logp_mean_ref=torch.randn(4),
            advantages=torch.randn(4),
            clip_eps=0.2,
            kl_coef=0.01,
        )

        assert loss.shape == (), f"Loss should be scalar, got {loss.shape}"
        assert hasattr(metrics, "pg_loss"), "Metrics missing pg_loss"
        assert hasattr(metrics, "kl_loss"), "Metrics missing kl_loss"
        assert hasattr(metrics, "clip_frac"), "Metrics missing clip_frac"

    def test_loss_is_differentiable(self):
        """Loss must be differentiable for training."""
        from nmoe.rl.grpo import grpo_loss

        logp = torch.randn(4, requires_grad=True)
        loss, _ = grpo_loss(
            logp_mean=logp,
            logp_mean_old=logp.detach(),
            logp_mean_ref=torch.randn(4),
            advantages=torch.randn(4),
            clip_eps=0.2,
            kl_coef=0.01,
        )

        assert loss.requires_grad, "Loss should require grad"
        loss.backward()
        assert logp.grad is not None, "Gradient should flow to logp_mean"


class TestDrGrpoNormalization:
    """Test Dr.GRPO constant divisor normalization."""

    def test_differs_from_standard(self):
        """Dr.GRPO should produce different results than per-token normalization."""
        from nmoe.rl.grpo import dr_grpo_normalize_loss

        token_losses = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ])
        loss_mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ])

        standard = (token_losses * loss_mask).sum() / loss_mask.sum()
        dr_grpo = dr_grpo_normalize_loss(token_losses, loss_mask, max_length=4)

        assert abs(standard.item() - 1.0) < 1e-5
        assert abs(dr_grpo.item() - 0.625) < 1e-5

    def test_handles_empty_mask(self):
        """Should handle empty loss mask without NaN."""
        from nmoe.rl.grpo import dr_grpo_normalize_loss

        token_losses = torch.ones(2, 4)
        loss_mask = torch.zeros(2, 4)

        result = dr_grpo_normalize_loss(token_losses, loss_mask, max_length=4)
        assert not torch.isnan(result)


class TestImportanceMasking:
    """Test 3-tier importance ratio masking."""

    def test_identical_probs_no_masking(self):
        """With identical log-probs, ratio=1.0, nothing should be masked."""
        from nmoe.rl.grpo import compute_importance_mask, ImportanceMaskConfig

        log_probs = torch.zeros(4, 10)
        loss_mask = torch.ones(4, 10)

        config = ImportanceMaskConfig()
        mask, stats = compute_importance_mask(log_probs, log_probs, loss_mask, config)

        assert mask.dtype == torch.bool
        assert mask.all()

    def test_extreme_ratios_trigger_masking(self):
        """Extreme log-prob differences should trigger masking."""
        from nmoe.rl.grpo import compute_importance_mask, ImportanceMaskConfig

        log_probs = torch.zeros(4, 10) + 10.0
        log_probs_old = torch.zeros(4, 10)
        loss_mask = torch.ones(4, 10)

        config = ImportanceMaskConfig()
        mask, stats = compute_importance_mask(log_probs, log_probs_old, loss_mask, config)

        assert stats["total_mask_frac"] > 0.5


class TestReinforcePP:
    """Test REINFORCE++ token-level returns."""

    def test_returns_shape(self):
        """Returns should have same shape as KL tensor."""
        from nmoe.rl.grpo import reinforce_pp_returns

        batch, seq = 4, 16
        rewards = torch.randn(batch)
        kl = torch.randn(batch, seq).abs()
        loss_mask = torch.ones(batch, seq)

        returns = reinforce_pp_returns(rewards, kl, loss_mask, kl_coef=0.1, gamma=0.99)
        assert returns.shape == (batch, seq)

    def test_positive_reward_higher_returns(self):
        """Sequences with positive reward should have higher returns."""
        from nmoe.rl.grpo import reinforce_pp_returns

        rewards = torch.tensor([1.0, -1.0])
        kl = torch.ones(2, 8) * 0.1
        loss_mask = torch.ones(2, 8)

        returns = reinforce_pp_returns(rewards, kl, loss_mask, kl_coef=0.1)
        assert returns[0].sum() > returns[1].sum()


class TestLengthWeightedBaseline:
    """Test length-weighted baseline computation."""

    def test_equal_lengths_is_mean(self):
        """With equal lengths, baseline should be simple mean."""
        from nmoe.rl.grpo import length_weighted_baseline

        rewards = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
        lengths = torch.tensor([[10, 10], [20, 20]])

        baseline = length_weighted_baseline(rewards, lengths)
        expected = torch.tensor([[0.5], [0.5]])

        assert torch.allclose(baseline, expected)

    def test_weighted_by_length(self):
        """Longer sequences should have more weight."""
        from nmoe.rl.grpo import length_weighted_baseline

        rewards = torch.tensor([[1.0, 0.0]])
        lengths = torch.tensor([[10, 30]])

        baseline = length_weighted_baseline(rewards, lengths)
        expected = torch.tensor([[0.25]])

        assert torch.allclose(baseline, expected, atol=1e-5)


class TestKLEstimators:
    def test_k3_is_non_negative_and_matches_formula(self):
        from nmoe.rl.grpo import compute_kl

        logp = torch.tensor([0.0, -0.5, 0.5])
        logp_ref = torch.zeros_like(logp)
        kl = compute_kl(logp, logp_ref, kl_type="k3")
        assert (kl >= 0).all()

        log_ratio = logp - logp_ref
        expected = torch.exp(-log_ratio) + log_ratio - 1.0
        assert torch.allclose(kl, expected, atol=1e-6)


class TestDualClipPPO:
    def test_dual_clip_limits_negative_adv_penalty(self):
        from nmoe.rl.grpo import grpo_loss

        # Force ratio = 10.0
        logp_old = torch.tensor([0.0])
        logp = torch.tensor([math.log(10.0)], requires_grad=True)
        logp_ref = torch.tensor([0.0])
        adv = torch.tensor([-1.0])

        loss_no, _ = grpo_loss(
            logp_mean=logp,
            logp_mean_old=logp_old,
            logp_mean_ref=logp_ref,
            advantages=adv,
            clip_eps=0.0,
            kl_coef=0.0,
        )

        loss_dual, _ = grpo_loss(
            logp_mean=logp,
            logp_mean_old=logp_old,
            logp_mean_ref=logp_ref,
            advantages=adv,
            clip_eps=0.0,
            kl_coef=0.0,
            dual_clip_c=3.0,
        )

        assert loss_dual.item() < loss_no.item()
        assert abs(loss_dual.item() - 3.0) < 1e-5
