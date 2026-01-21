## Post-training story (nmoe)

### Thesis

We post-train via **RL-only** (no SFT), and we treat **verification** as the
primitive that makes RL scalable.

We also treat **format** as a learnable capability: Harmony structure and
channeling are rewarded via RL and scheduled via curriculum.

### Capability ordering (what the curriculum encodes)

Format → Reasoning → Tool use → Correctness → Efficiency

- You can’t score correctness reliably without a consistent format.
- You can’t optimize efficiency before you’re correct.

### System seam: Environment

An `Environment` binds together:
- a `TaskPool` (dataset-backed or self-play)
- an optional tool harness (`ToolConfig` / `AsyncToolExecutor`)
- an output format contract (Harmony)

Environment configs are TOML and can be validated/loaded via `nmoe.rl.environment`.

### Entry points

These are the supported “one clear path per use-case” commands:

1) **GRPO post-training (R1-Zero style)**

    `python -m nmoe.rl train <config.toml>`

2) **Verifier training (PRM-style)**

    `python -m nmoe.rl verifier <config.toml> --prm_source=prm800k --prm_split=train[:1024]`

3) **Distributed smoke tests**

    `torchrun --nproc_per_node=8 -m nmoe.rl tests-dist`

4) **Math‑V2 self-play data generation**

    `python -m nmoe.rl selfplay <config.toml> --problem_source=prm800k --output_dir=./mathv2_output --max_problems=1000 --replay_sample_rate=0.1`

### Ready vs blocked

Ready (runnable in-container today):
- `train`: GRPO post-training on any `Environment` + `TaskPool`
- `verifier`: PRM-style verifier training on PRM datasets
- `selfplay`: Math‑V2-style self-play data generation

Blocked (for frontier-scale runs):
- A trusted **pretrained nmoe checkpoint** from pretraining (the system can run without it, but meaningful learning requires it).

### TaskPool types

The `[task_pool]` table controls where verifiable problems come from.

1) `type="hf"`: single HuggingFace dataset split

2) `type="mixture"`: mix multiple sources by weight (curriculum / multi-domain)

Example:

```toml
[task_pool]
type = "mixture"
seed = 0

[[task_pool.sources]]
name = "gsm8k"
type = "hf"
dataset = "openai/gsm8k"
subset = "main"
split = "train"
task_type = "gsm8k"
problem_field = "question"
answer_field = "answer"
gold_extractor = "gsm8k_hash"
weight = 0.3
max_examples = 100000

[[task_pool.sources]]
name = "math"
type = "hf"
dataset = "hendrycks/competition_math"
split = "train"
task_type = "math"
problem_field = "problem"
answer_field = "solution"
gold_extractor = "boxed"
weight = 0.7
max_examples = 100000
```

### Current gaps (what’s next)

Even with the correctness/hardening kernel in place, frontier-scale RL needs:
- a rollout engine seam that returns **token IDs + logprobs** (no retoken drift)
- a unified “Environment SDK” used by all trainers (single-turn, tool multi-turn, self-play)

See `docs/rl/sota_gap_review.md` for the detailed gap analysis.
