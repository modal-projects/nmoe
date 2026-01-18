## Scope

This repo’s north star is **state‑of‑the‑art MoE training on NVIDIA B200 with RDEP**.
The “RL / self‑play env” layer exists to produce **high-signal, low-noise** training data and reward signals (agentic code + math/proof loops) *without* silently corrupting logprobs/rewards.

This document is a *gap review* against commonly cited SOTA stacks:
- THUDM `slime`
- Volcengine `verl`
- Prime Intellect `prime-rl` + `verifiers`

It’s structured as: **capabilities → correctness → hardening → throughput**.

---

## Paper-to-system reconciliation

### Let’s Verify Step by Step (PRM)

Core system requirements implied by PRM training:
- Step-indexable solution representation (prefix through step `k`)
- A verifier that can be supervised **per-step** (binary or graded)
- A scoring API that is stable under multi-turn/tokenization (no retoken drift)

In practice, SOTA implementations converge on:
- **token-in/token-out** interfaces for multi-turn (store tokens, not strings)
- strict format contracts to avoid label ambiguity (single-token labels, or strict parsing)

### DeepSeek “Math‑V2 style” self-play (process)

Core system requirements implied by generator↔verifier↔meta-verifier loops:
- problem pool (no oracle) + acceptance filtering (meta-verifier as “soft oracle”)
- replay/debuggability: a sampled chain must be replayable offline (same prompt, same tool returns, same parsing)
- failure taxonomy: “parse error” vs “execution error” must be visible (no silent failures)

Key constraint we enforce: **Harmony-only** (no `\\boxed{}`, no custom tags).

---

## What SOTA frameworks emphasize (repo review)

### `slime`

Notable patterns:
- Decoupled training/rollout architecture (Megatron trainer + SGLang rollout).
- Multi-turn token correctness: rollout uses **token IDs returned by the server** (“avoid re-tokenization”) and can keep token-aligned logprobs.

Implication for nmoe:
- For frontier-scale RL, an inference server that returns `{token_ids, token_logprobs}` is table stakes.

### `verl`

Notable patterns:
- Explicit multi-turn configs with **tokenization sanity checks**.
- Multiple inference engines and placements; strong focus on throughput engineering and actor resharding.
- Sandbox/tool integration as a first-class extension point.

Implication for nmoe:
- Even if we don’t adopt their architecture, we need an equivalent “tokenization sanity check” gate and a clean seam to plug a rollout engine.

### `prime-rl` + `verifiers`

Notable patterns:
- Environments as a product surface: dataset + harness + rubric, packaged, versioned, and installable (hub).
- Trajectory-based tracking for token-in/token-out across turns (explicitly called out in `verifiers` changelog).
- Inference endpoint that supports **prompt tokens in request** and warns on retoken mismatch.

Implication for nmoe:
- A minimal internal “Environment SDK” (not a hub) is still a huge win: it becomes the single integration surface for self-play.

---

## nmoe current state (as of this worktree)

Correctness/hardening kernel (already in place):
- Tool execution **cwd inheritance** + per-task sandbox allowlist
- Kill tests preventing tool calls outside workspace
- Token-exact trajectory records + replay invariant
- Failure taxonomy + sampled replay bundles for debugging
- Mandatory provenance for code workspaces

Math capability additions (Harmony-first):
- PRM dataset pool + Harmony PRM tasks (step labels + whole-solution scores)
- Math‑V2 self-play loop now uses **Harmony verifier/meta-verifier tasks**
- A `ProblemPool` adapter that can source problems from:
  - DeepSeek-Math‑V2 repo inputs JSON
  - HuggingFace datasets (generic text field selection)

---

## Gaps (ordered by “capabilities → correctness → hardening → throughput”)

### 1) Capabilities gaps

- No single “Environment” abstraction that unifies:
  - dataset iteration/sharding
  - prompt construction (Harmony)
  - tool harness / workspace setup
  - verifier + meta-verifier loops
  - reward/rubric computation

- No first-class rollout engine seam (server that returns token IDs + logprobs).
  - This is the big blocker to scaling beyond “sanity trainer”.

### 2) Correctness gaps

- Behavior policy logprobs: for on-policy-ish PPO/GRPO at scale, SOTA stacks keep a clean separation of:
  - rollout logprobs (behavior)
  - trainer recomputed logprobs (current policy)
  - and sanity checks between them

- Multi-turn token contracts beyond tools:
  - we have tool replay; we still need “conversation replay” invariants once we add true multi-turn env state machines.

### 3) Hardening gaps

- Resource budgets per env (wallclock, disk, subprocess limits) should be standardized.
- Workspace lifecycle GC (when to delete, how to persist artifacts) needs a single policy.

### 4) Throughput gaps

- Asynchronous rollout (server + queue) and placement control (dedicated rollout GPUs) are required for frontier-scale runs.
- Partial rollouts / tail mitigation are likely necessary once tool loops get long-horizon.

---

## Recommended next PR-sized steps

1. **Introduce a minimal `Environment` interface + registry (TOML)**:
   - `env_id`, `dataset`, `harness`, `rubric`
   - strict Harmony prompt/output contracts
   - deterministic sharding + resume hooks

2. **Add a rollout-engine seam** (not full performance yet):
   - a “token-in/token-out” interface that can be backed by the in-flight serving work
   - strict retoken sanity checks

3. **Math‑V2 v1 pipeline**:
   - problem pool → generator → verifier → meta
   - acceptance filtering and emission of verifier training samples
   - replay bundles for any accepted sample

After these, we can credibly benchmark and start closing the gap to SOTA throughput.

