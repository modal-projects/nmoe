"""GRPO "actually improves" toy bandit.

This is a functional kill test: it must fail if GRPO stops pushing probability
mass onto higher-reward actions (i.e., if the policy-gradient signal is broken).
"""

from __future__ import annotations

import torch


class TestGRPOBandit:
    def test_probability_of_rewarded_action_increases(self):
        from nmoe.rl.grpo import grpo_loss, group_relative_advantages

        torch.manual_seed(0)

        n_actions = 4
        best_action = 2

        logits = torch.nn.Parameter(torch.zeros(n_actions, dtype=torch.float32))
        optimizer = torch.optim.Adam([logits], lr=0.2)

        with torch.no_grad():
            ref_logits = logits.detach().clone()
            p0 = torch.softmax(ref_logits, dim=0)[best_action].item()

        # GRPO shape: [B, G] rollouts per "prompt group".
        B, G = 256, 4

        for step in range(60):
            # Snapshot rollout policy ("old") and sample actions.
            with torch.no_grad():
                old_logits = logits.detach().clone()
                dist_old = torch.distributions.Categorical(logits=old_logits)
                actions = dist_old.sample((B, G))  # [B, G]

                rewards = (actions == best_action).to(torch.float32)  # [B, G] in {0,1}
                adv = group_relative_advantages(rewards, normalize_mean=True, normalize_std=False)  # [B, G]
                adv_flat = adv.reshape(-1).detach()

                a_flat = actions.reshape(-1)
                logp_old = torch.log_softmax(old_logits, dim=0)[a_flat].detach()
                logp_ref = torch.log_softmax(ref_logits, dim=0)[a_flat].detach()

            logp = torch.log_softmax(logits, dim=0)[a_flat]
            loss, _ = grpo_loss(
                logp_mean=logp,
                logp_mean_old=logp_old,
                logp_mean_ref=logp_ref,
                advantages=adv_flat,
                clip_eps=0.2,
                kl_coef=0.0,
                kl_type="k3",
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 0:
                # Sanity: we should have a real gradient signal on logits.
                assert logits.grad is not None
                assert torch.isfinite(logits.grad).all()

        p1 = torch.softmax(logits.detach(), dim=0)[best_action].item()

        assert p1 > p0 + 0.25, f"expected p(best) to increase by >=0.25 (p0={p0:.3f}, p1={p1:.3f})"
        assert int(torch.argmax(logits.detach()).item()) == best_action

