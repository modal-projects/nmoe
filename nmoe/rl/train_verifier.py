r"""
nmoe.rl.train_verifier: GRPO sanity training for a text-verifier.

This trains the *verifier model* (not the prover) on PRM-labeled math solutions by
mapping stepwise PRM examples into Harmony-first verifier tasks.

Usage:
  python -m nmoe.rl.train_verifier <config.toml> --prm_source=prm800k --prm_split=train[:10000]
  torchrun --nproc_per_node=8 -m nmoe.rl.train_verifier <config.toml> --prm_source=math_shepherd --prm_split=train[:10000]

Notes:
  - This is a correctness/capability gate, not a throughput trainer.
  - Dataset access uses HuggingFace `datasets` (already in the container).
  - Output format is Harmony; no custom tags or \\boxed parsing.
"""

from __future__ import annotations

import random
import sys
import time
import tomllib

import torch

from nmoe.config import Config
from nmoe.model import Transformer
from nmoe.opt import build_optimizer, update_lr, step as opt_step
from nmoe import runtime

from nmoe.rl.grpo import group_relative_advantages, grpo_loss, filter_zero_std_groups
from nmoe.rl.rollout import generate_one, completion_nll_mean
from nmoe.rl.rewards_harmony import HARMONY_TOKENS, harmony_encode
from nmoe.rl.tasks.harmony_prm import HarmonyPRMStepLabelTask, HarmonyVerifierScoreTask
from nmoe.rl.tasks.prm_datasets import PRMTaskPool, to_verifier_task


def _parse_flag(name: str, default: str | None = None) -> str | None:
  pref = f"--{name}="
  for a in sys.argv[2:]:
    if a.startswith(pref):
      return a[len(pref):]
  return default


def _build_optimizers(model: torch.nn.Module, cfg: Config) -> tuple[torch.optim.Optimizer | None, torch.optim.Optimizer | None, list[dict]]:
  """Handle both 2-tuple and 3-tuple optimizer APIs (repo drift-safe)."""
  out = build_optimizer(model, cfg)
  if not isinstance(out, tuple):
    raise TypeError(f"build_optimizer must return a tuple, got {type(out)}")
  if len(out) == 2:
    opt, dense_groups = out
    return opt, None, dense_groups
  if len(out) == 3:
    opt, muon_opt, dense_groups = out
    return opt, muon_opt, dense_groups
  raise ValueError(f"unsupported build_optimizer return arity: {len(out)}")


def _require_harmony_tokens(enc) -> None:
  # Fail fast if tokenizer can't represent Harmony special tokens as single ids.
  for name in ("start", "end", "message", "channel"):
    ids = harmony_encode(enc, HARMONY_TOKENS[name])
    if not isinstance(ids, list) or len(ids) != 1:
      raise ValueError(
        "train_verifier requires a Harmony-capable tokenizer (single-token Harmony specials). "
        f"Token {HARMONY_TOKENS[name]!r} encoded as {ids}; use tokenizer=o200k_harmony."
      )


def train(
  cfg: Config,
  *,
  prm_source: str,
  prm_split: str,
  prm_max_examples: int,
  mode: str,
) -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("Verifier GRPO sanity training requires CUDA.")
  if not cfg.rl_enabled:
    raise ValueError("rl_enabled=false; refusing to run verifier trainer.")
  if cfg.rl_algorithm.lower() != "grpo":
    raise ValueError(f"unsupported rl_algorithm={cfg.rl_algorithm!r} (only 'grpo' supported)")
  if getattr(cfg, "perl_enabled", False):
    raise ValueError("perl_enabled=true not supported in verifier sanity trainer (keep surfaces small).")

  rank, world = runtime.init(cfg.seed)
  if cfg.batch_size % world != 0:
    raise ValueError(f"batch_size ({cfg.batch_size}) must be divisible by world_size ({world}).")
  local_batch_size = cfg.batch_size // world

  import tiktoken
  enc = tiktoken.get_encoding(cfg.tokenizer)
  _require_harmony_tokens(enc)

  pool = PRMTaskPool.from_hf(
    source=str(prm_source),
    split=str(prm_split),
    max_examples=int(prm_max_examples),
    seed=int(cfg.seed + rank),
    streaming=False,
  )
  if len(pool) == 0:
    raise RuntimeError("loaded 0 PRM tasks (check dataset access, split, and schema).")

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()

  model_ref = Transformer(cfg).cuda().eval()
  model_ref.load_state_dict(model.state_dict(), strict=False)
  for p in model_ref.parameters():
    p.requires_grad_(False)

  optimizer, muon_optimizer, dense_groups = _build_optimizers(model, cfg)
  zero2_state: dict = {}

  device = torch.device("cuda")
  max_new_tokens = int(getattr(cfg, "rl_max_new_tokens", 512))
  temperature = float(getattr(cfg, "rl_temperature", 0.7))
  top_p = float(getattr(cfg, "rl_top_p", 0.95))
  updates_per_batch = int(getattr(cfg, "rl_updates_per_batch", 2))
  normalize_mean = bool(getattr(cfg, "rl_normalize_mean", True))
  normalize_std = bool(getattr(cfg, "rl_normalize_std", False))
  length_norm_constant = bool(getattr(cfg, "rl_length_norm_constant", True))
  use_opsm = bool(getattr(cfg, "rl_use_opsm", False))
  opsm_delta = float(getattr(cfg, "rl_opsm_delta", 1e-4))
  filter_zero_std = bool(getattr(cfg, "rl_filter_zero_std", True))
  kl_type = str(getattr(cfg, "rl_kl_type", "k3"))
  neg_adv_scale = float(getattr(cfg, "rl_neg_adv_scale", 1.0))
  dual_clip_c = getattr(cfg, "rl_dual_clip_c", None)
  clip_eps_high = getattr(cfg, "rl_clip_eps_high", None)

  g = int(cfg.rl_group_size)
  if g <= 0:
    raise ValueError(f"rl_group_size must be > 0 (got {g})")

  for step_num in range(cfg.steps):
    rng = random.Random(cfg.seed + step_num * world + rank)
    b = int(local_batch_size)
    if b <= 0:
      raise ValueError(f"local_batch_size must be > 0 (got {b})")

    tasks = []
    if mode == "score":
      exs = pool.sample_examples(b, stratified=True)
      if len(exs) != b:
        raise RuntimeError(f"pool.sample_examples returned {len(exs)} examples, expected {b}")
      for ex in exs:
        vt = to_verifier_task(ex)
        tasks.append(HarmonyVerifierScoreTask(problem=vt.problem, solution=vt.proof, gold_score=vt.gold_score))
    elif mode == "step":
      steps = pool.sample_step_labels(b, stop_at_first_incorrect=True)
      if len(steps) != b:
        raise RuntimeError(f"pool.sample_step_labels returned {len(steps)} items, expected {b}")
      for it in steps:
        tasks.append(
          HarmonyPRMStepLabelTask(
            problem=it.prompt,
            steps=list(it.steps),
            step_idx=int(it.step_idx),
            gold_label=str(int(it.gold_label)),
          )
        )
    else:
      raise ValueError(f"unknown --mode={mode!r} (expected 'score' or 'step')")

    seqs: list[list[int]] = []
    prompt_lens: list[int] = []
    completion_lens: list[int] = []
    rewards_total: list[float] = []

    t0 = time.perf_counter()
    for task in tasks:
      prompt_ids = list(harmony_encode(enc, task.to_prompt()))
      for _ in range(g):
        traj = generate_one(
          model,
          enc=enc,
          prompt_ids=prompt_ids,
          max_new_tokens=max_new_tokens,
          eos_token_id=cfg.eos_token_id,
          temperature=temperature,
          top_p=top_p,
        )
        seqs.append(traj.tokens)
        prompt_lens.append(traj.prompt_len)
        completion_lens.append(traj.completion_len)
        ans = task.extract_answer(traj.completion_text)
        rewards_total.append(1.0 if task.verify(ans) else 0.0)

    rollout_ms = (time.perf_counter() - t0) * 1000.0

    rewards_t = torch.tensor(rewards_total, device=device, dtype=torch.float32).reshape(b, g)
    groups_filtered = 0
    if filter_zero_std:
      keep_mask = filter_zero_std_groups(rewards_t)
      groups_filtered = int((~keep_mask).sum().item())
      if groups_filtered > 0 and groups_filtered < b:
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        rewards_t = rewards_t[keep_idx]
        keep_seq_idx = []
        for idx in keep_idx.tolist():
          keep_seq_idx.extend(range(idx * g, (idx + 1) * g))
        seqs = [seqs[i] for i in keep_seq_idx]
        prompt_lens = [prompt_lens[i] for i in keep_seq_idx]
        completion_lens = [completion_lens[i] for i in keep_seq_idx]
        b = len(rewards_t)

    adv = group_relative_advantages(
      rewards_t,
      normalize_mean=normalize_mean,
      normalize_std=normalize_std,
      neg_scale=neg_adv_scale,
    ).reshape(-1).detach()

    nll_max_len = max_new_tokens if length_norm_constant else None
    with torch.inference_mode():
      nll_old = completion_nll_mean(
        model,
        seqs=seqs,
        prompt_lens=prompt_lens,
        completion_lens=completion_lens,
        pad_id=cfg.eos_token_id,
        device=device,
        max_length=nll_max_len,
      )
      nll_ref = completion_nll_mean(
        model_ref,
        seqs=seqs,
        prompt_lens=prompt_lens,
        completion_lens=completion_lens,
        pad_id=cfg.eos_token_id,
        device=device,
        max_length=nll_max_len,
      )
      logp_old = (-nll_old).detach()
      logp_ref = (-nll_ref).detach()

    try:
      lr = update_lr(optimizer, muon_optimizer, dense_groups, step_num, tokens_seen=0, cfg=cfg)
    except TypeError:
      # Back-compat for older optimizer API.
      lr = update_lr(optimizer, dense_groups, step_num, tokens_seen=0, cfg=cfg)  # type: ignore[misc]
    last_loss = 0.0
    for _k in range(updates_per_batch):
      nll = completion_nll_mean(
        model,
        seqs=seqs,
        prompt_lens=prompt_lens,
        completion_lens=completion_lens,
        pad_id=cfg.eos_token_id,
        device=device,
        max_length=nll_max_len,
      )
      logp = -nll
      loss, m = grpo_loss(
        logp_mean=logp,
        logp_mean_old=logp_old,
        logp_mean_ref=logp_ref,
        advantages=adv,
        clip_eps=float(cfg.rl_clip_eps),
        clip_eps_high=clip_eps_high,
        dual_clip_c=dual_clip_c,
        kl_coef=float(cfg.rl_kl_coef),
        kl_type=kl_type,
        use_opsm=use_opsm,
        opsm_delta=opsm_delta,
      )
      model.zero_grad(set_to_none=True)
      loss.backward()
      try:
        opt_step(model, optimizer, muon_optimizer, dense_groups, zero2_state, cfg, world)
      except TypeError:
        opt_step(model, optimizer, dense_groups, zero2_state, cfg, world)  # type: ignore[misc]
      last_loss = float(loss.detach().item())

    if rank == 0:
      reward_mean = float(rewards_t.mean().item()) if rewards_t.numel() else 0.0
      reward_std = float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() else 0.0
      print(
        f"[verifier] step={step_num+1} loss={last_loss:.4f} "
        f"reward_mean={reward_mean:.3f} reward_std={reward_std:.3f} "
        f"rollout_ms={rollout_ms:.1f} filtered={groups_filtered} "
        f"kl_mean={float(m.kl_mean):.4f} clip_frac={float(m.clip_frac):.3f} lr={lr:.3e}"
      )


def main() -> None:
  if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__)
    sys.exit(0)
  if len(sys.argv) < 2:
    print("Usage: python -m nmoe.rl.train_verifier <config.toml> --prm_source=... --prm_split=...", file=sys.stderr)
    sys.exit(1)

  prm_source = _parse_flag("prm_source")
  prm_split = _parse_flag("prm_split", "train")
  prm_max_examples_s = _parse_flag("prm_max_examples", "100000")
  mode = _parse_flag("mode", "score") or "score"
  if prm_source is None:
    raise ValueError("--prm_source is required (prm800k | math_shepherd)")
  try:
    prm_max_examples = int(prm_max_examples_s or "100000")
  except Exception as e:
    raise ValueError(f"invalid --prm_max_examples={prm_max_examples_s!r}") from e

  with open(sys.argv[1], "rb") as f:
    cfg_dict = tomllib.load(f)

  # Minimal overrides (same pattern as nmoe.rl.train), excluding PRM flags.
  reserved = {"prm_source", "prm_split", "prm_max_examples", "mode"}
  for arg in sys.argv[2:]:
    if not (arg.startswith("--") and "=" in arg):
      continue
    key, val = arg[2:].split("=", 1)
    if key in reserved:
      continue
    if val.lower() in ("true", "false"):
      val = val.lower() == "true"
    elif val.lstrip("-").isdigit():
      val = int(val)
    elif val.replace(".", "", 1).lstrip("-").isdigit():
      val = float(val)
    cfg_dict[key] = val
  cfg = Config(**cfg_dict)
  try:
    train(
      cfg,
      prm_source=prm_source,
      prm_split=prm_split or "train",
      prm_max_examples=prm_max_examples,
      mode=str(mode),
    )
  finally:
    runtime.finalize()


if __name__ == "__main__":
  main()
