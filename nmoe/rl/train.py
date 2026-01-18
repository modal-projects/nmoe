r"""
nmoe.rl.train: R1-Zero style GRPO (pure RL, no critic).

Supports multi-GPU via expert parallelism (Rdep) + ZeRO-2 for dense params.
Each GPU handles batch_size/world prompts with complete groups (no cross-GPU
coordination for advantages).

Usage:
  python -m nmoe.rl.train <config.toml> [--key=value ...]
  torchrun --nproc_per_node=8 -m nmoe.rl.train <config.toml>
"""

from __future__ import annotations

import os
import random
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from nmoe.config import Config, fingerprint
from nmoe.model import Transformer
from nmoe.opt import build_optimizer, update_lr, step as opt_step
from nmoe.checkpoint import Checkpointer, load_checkpoint, save_checkpoint
from nmoe.metrics import init_metrics, start_metrics, log_training_step, stop_metrics, register_model_timers
from nmoe.experiments import ExperimentTracker
from nmoe import runtime

from nmoe.rl.grpo import group_relative_advantages, grpo_loss, filter_zero_std_groups
from nmoe.rl.rollout import completion_nll_mean
from nmoe.rl.rollout_engine import LocalRolloutEngine, RolloutRequest, logp_mean_from_logprobs
from nmoe.rl.environment import Environment
from nmoe.rl.rewards_harmony import harmony_encode, parse_harmony_text
from nmoe.rl.tasks.code import iter_humaneval
from nmoe.rl.tasks.math import iter_gsm8k


def _load_tasks(tasks_file: str) -> dict[str, dict[str, Any]]:
  p = Path(tasks_file)
  if not p.exists():
    raise FileNotFoundError(f"eval_tasks_file not found: {tasks_file}")
  with p.open("rb") as f:
    obj = tomllib.load(f)
  tasks = obj.get("task", [])
  if not isinstance(tasks, list):
    raise ValueError(f"bad tasks.toml: expected [[task]] list (got {type(tasks)})")
  out: dict[str, dict[str, Any]] = {}
  for t in tasks:
    if not isinstance(t, dict):
      continue
    name = t.get("name")
    if isinstance(name, str) and name:
      out[name] = t
  return out


def _parse_hf_source(src: str) -> tuple[str, Optional[str], str]:
  # hf:<dataset>:<subset>:<split> or hf:<dataset>:<split>
  parts = src.split(":")
  if len(parts) < 3 or parts[0] != "hf":
    raise ValueError(f"unsupported source spec (expected hf:*): {src}")
  if len(parts) == 4:
    _, dataset, subset, split = parts
    return dataset, subset, split
  if len(parts) == 3:
    _, dataset, split = parts
    return dataset, None, split
  raise ValueError(f"bad source spec: {src}")


def _build_env_from_eval_tasks_file(tasks_file: str, *, seed: int) -> Environment:
  """Build an Environment from cfg.eval_tasks_file.

  For now, RL training expects the tasks file to define GSM8K + HumanEval.
  """
  tasks = _load_tasks(tasks_file)
  if "GSM8K" not in tasks or "HumanEval" not in tasks:
    raise RuntimeError("eval_tasks_file must define tasks 'GSM8K' and 'HumanEval' for RL training.")
  gsm8k_src = str(tasks["GSM8K"].get("source", ""))
  humaneval_src = str(tasks["HumanEval"].get("source", ""))
  if not gsm8k_src.startswith("hf:") or not humaneval_src.startswith("hf:"):
    raise RuntimeError("GSM8K/HumanEval sources must be HuggingFace ('hf:...') in tasks file.")

  gsm8k_tasks = list(iter_gsm8k(max_examples=10_000_000, source=gsm8k_src))
  dataset_name, _subset, _split = _parse_hf_source(humaneval_src)
  humaneval_tasks = list(iter_humaneval(source=dataset_name))
  if not gsm8k_tasks:
    raise RuntimeError("loaded 0 GSM8K tasks (check dataset access and schema).")
  if not humaneval_tasks:
    raise RuntimeError("loaded 0 HumanEval tasks (check dataset access and schema).")

  from nmoe.rl.tasks import TaskPool

  pool = TaskPool(
    tasks=[*gsm8k_tasks, *humaneval_tasks],
    weights={"gsm8k": 1.0, "humaneval": 1.0},
    seed=int(seed),
  )
  return Environment(env_id="train_grpo_core", task_pool=pool, format_type="harmony", tool_config=None)


def _think_len_tokens(text: str, *, enc) -> int:
  parsed = parse_harmony_text(text)
  think = parsed.analysis_content
  if not think.strip():
    return 0
  return len(harmony_encode(enc, think))

def _atomic_torch_save(obj: object, path: Path) -> None:
  tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
  torch.save(obj, str(tmp))
  os.replace(str(tmp), str(path))


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
  out: dict[str, torch.Tensor] = {}
  for k, v in model.state_dict().items():
    if not isinstance(v, torch.Tensor):
      raise TypeError(f"unexpected non-tensor state_dict entry: {k} -> {type(v)}")
    out[k] = v.detach().cpu()
  return out


def _ensure_ref_model(model_ref: torch.nn.Module, model: torch.nn.Module, *, path: Path, rank: int) -> None:
  is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
  if rank == 0 and not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(_cpu_state_dict(model), path)
  if is_dist:
    torch.distributed.barrier()
  sd = torch.load(str(path), map_location="cpu", weights_only=False)
  model_ref.load_state_dict(sd, strict=True)


def _is_pow2(n: int) -> bool:
  return n > 0 and (n & (n - 1) == 0)


def _should_check_irc(step_1_based: int, *, window: int) -> bool:
  """Log2 schedule inside the invariance window (spec-compatible, cheap)."""
  if step_1_based <= 0 or window <= 0:
    return False
  if step_1_based > window:
    return False
  return _is_pow2(step_1_based) or step_1_based == window


def train(cfg: Config) -> dict[str, float]:
  if not torch.cuda.is_available():
    raise RuntimeError("RL training requires CUDA (run in the container/GPU environment).")

  if not cfg.rl_enabled:
    raise ValueError("rl_enabled=false; refusing to run RL trainer.")
  if cfg.rl_algorithm.lower() != "grpo":
    raise ValueError(f"unsupported rl_algorithm={cfg.rl_algorithm!r} (only 'grpo' supported)")
  if cfg.rl_group_size <= 0:
    raise ValueError(f"rl_group_size must be > 0 (got {cfg.rl_group_size})")
  if cfg.rl_clip_eps < 0.0:
    raise ValueError(f"rl_clip_eps must be >= 0 (got {cfg.rl_clip_eps})")
  if cfg.rl_kl_coef < 0.0:
    raise ValueError(f"rl_kl_coef must be >= 0 (got {cfg.rl_kl_coef})")

  rank, world = runtime.init(cfg.seed)

  # Distribute prompts across GPUs: each rank handles batch_size/world prompts
  if cfg.batch_size % world != 0:
    raise ValueError(
      f"batch_size ({cfg.batch_size}) must be divisible by world_size ({world}) for RL training."
    )
  local_batch_size = cfg.batch_size // world

  # Tokenizer for prompt encoding/decoding
  import tiktoken
  enc = tiktoken.get_encoding(cfg.tokenizer)

  exp_tracker: ExperimentTracker | None = None
  run_id = os.getenv("NMOE_RUN", "")
  if rank == 0:
    exp_tracker = ExperimentTracker(cfg)
    run_id = exp_tracker.start_run(run_id=run_id or None)

  checkpointer = Checkpointer(
    base=cfg.checkpoint_dir,
    keep_last=getattr(cfg, "checkpoint_keep_last_n", 3),
    async_io=True,
    async_max_queue=1,
  )

  model = Transformer(cfg).cuda()
  model.init_weights()
  model.train()
  register_model_timers(model)

  perl_adapters = None
  perl_manifest = None
  if getattr(cfg, "perl_enabled", False):
    from nmoe.perl.apply import apply_ldora

    def _filter(name: str, _m: torch.nn.Linear) -> bool:
      # Embeddings are nn.Embedding. LM head is nn.Linear (exclude by default).
      if name == "lm_head" and not getattr(cfg, "perl_adapt_lm_head", False):
        return False
      return True

    perl_adapters, perl_manifest = apply_ldora(
      model,
      rank=int(getattr(cfg, "perl_rank_total", 32)),
      alpha=getattr(cfg, "perl_alpha", None),
      filter_fn=_filter,
      freeze_base=True,
    )

  optimizer, dense_groups = build_optimizer(model, cfg)
  if perl_adapters is not None:
    from nmoe.perl.policy import validate_optimizer_contract

    validate_optimizer_contract(dense_groups, perl_adapters)

  metrics_state = init_metrics(model, cfg.seq_len)
  metrics_ctx = start_metrics(run_id=run_id, metrics_dir=cfg.metrics_dir)

  start_step, tokens_seen, zero2_state = load_checkpoint(
    checkpointer, model, optimizer, loader=None, plan=None, cfg=cfg, rank=rank, print_fn=print
  )
  if perl_adapters is not None:
    for _name, m in perl_adapters.items():
      m._reset_g0_from_weight()
  config_fingerprint = fingerprint(cfg)
  checkpoint_every = getattr(cfg, "checkpoint_every", 100)

  model_ref = Transformer(cfg).cuda().eval()
  ref_path = Path(cfg.checkpoint_dir) / "rl_ref.pt"
  if perl_adapters is None:
    _ensure_ref_model(model_ref, model, path=ref_path, rank=rank)
  else:
    # PERL: reference must be the *base* model (no adapters) and fixed.
    #
    # We persist it for deterministic resume. This must be race-free under torchrun:
    # write to a temp file then atomically rename, and barrier before any rank loads.
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
    if rank == 0 and not ref_path.exists():
      model_ref.load_state_dict(model.state_dict(), strict=False)
      ref_path.parent.mkdir(parents=True, exist_ok=True)
      ref_keys = set(model_ref.state_dict().keys())
      base_sd = {k: v for k, v in model.state_dict().items() if k in ref_keys}
      missing = ref_keys.difference(base_sd.keys())
      if missing:
        raise RuntimeError(f"failed to build base reference state_dict: missing {len(missing)} keys")
      _atomic_torch_save({k: v.detach().cpu() for k, v in base_sd.items()}, ref_path)
    if is_dist:
      torch.distributed.barrier()
    try:
      sd = torch.load(str(ref_path), map_location="cpu", weights_only=False)
    except Exception as e:
      raise RuntimeError(
        f"failed to load reference checkpoint at {ref_path} "
        "(delete it to regenerate)"
      ) from e
    model_ref.load_state_dict(sd, strict=True)
  for p in model_ref.parameters():
    p.requires_grad_(False)

  # Rollout sampling settings from config
  max_new_tokens = getattr(cfg, "rl_max_new_tokens", 512)
  temperature = getattr(cfg, "rl_temperature", 0.7)
  top_p = getattr(cfg, "rl_top_p", 0.95)
  updates_per_batch = getattr(cfg, "rl_updates_per_batch", 2)
  normalize_mean = getattr(cfg, "rl_normalize_mean", True)
  normalize_std = getattr(cfg, "rl_normalize_std", False)  # Dr.GRPO default
  length_norm_constant = getattr(cfg, "rl_length_norm_constant", True)  # Dr.GRPO fix
  use_opsm = getattr(cfg, "rl_use_opsm", False)
  opsm_delta = getattr(cfg, "rl_opsm_delta", 1e-4)
  filter_zero_std = getattr(cfg, "rl_filter_zero_std", True)
  kl_type = getattr(cfg, "rl_kl_type", "k3")
  neg_adv_scale = getattr(cfg, "rl_neg_adv_scale", 1.0)
  dual_clip_c = getattr(cfg, "rl_dual_clip_c", None)
  clip_eps_high = getattr(cfg, "rl_clip_eps_high", None)
  perl_window = int(getattr(cfg, "perl_irc_window_steps", 256)) if perl_adapters is not None else 0
  if perl_adapters is not None and rank == 0:
    n = len(perl_adapters)
    r = int(getattr(cfg, "perl_rank_total", 32))
    print(f"[perl] enabled: adapters={n} rank_total={r} window={perl_window}")

  env = _build_env_from_eval_tasks_file(cfg.eval_tasks_file, seed=cfg.seed + rank)
  engine = LocalRolloutEngine(model=model, enc=enc, device="cuda")

  last_loss: float = 0.0

  try:
    for step_num in range(start_step, cfg.steps):
      # Per-step deterministic seeding: reproducible across restarts
      batch_seed = int(cfg.seed + step_num * world + rank)

      b = local_batch_size
      if b <= 0:
        raise ValueError(f"local_batch_size must be > 0 (got {b}, batch_size={cfg.batch_size}, world={world})")
      batch = env.sample(b, seed=batch_seed)

      # Rollout: for each prompt, sample G completions.
      g = int(cfg.rl_group_size)
      seqs: list[list[int]] = []
      prompt_lens: list[int] = []
      completion_lens: list[int] = []
      behavior_logp_means: list[float] = []
      rewards_total: list[float] = []
      think_lens: list[int] = []

      # Dr.GRPO: use constant length normalization to fix length bias
      logp_max_len = max_new_tokens if length_norm_constant else None

      t0 = time.perf_counter()
      torch.manual_seed(batch_seed)
      torch.cuda.manual_seed_all(batch_seed)
      for task in batch:
        prompt = task.to_prompt()
        prompt_ids = list(harmony_encode(enc, prompt))
        req = RolloutRequest(
          prompt_tokens=prompt_ids,
          n=g,
          max_new_tokens=max_new_tokens,
          eos_token_id=cfg.eos_token_id,
          temperature=temperature,
          top_p=top_p,
        )
        samples = engine.generate(req)
        if len(samples) != g:
          raise RuntimeError(f"RolloutEngine returned {len(samples)} samples for n={g}")
        for s in samples:
          seqs.append(s.tokens)
          prompt_lens.append(s.prompt_len)
          completion_lens.append(s.completion_len)
          behavior_logp_means.append(
            logp_mean_from_logprobs(s.logprobs, completion_len=s.completion_len, max_length=logp_max_len)
          )

          ans = task.extract_answer(s.completion_text) if hasattr(task, "extract_answer") else None
          ok = bool(task.verify(ans)) if hasattr(task, "verify") else False
          rewards_total.append(1.0 if ok else 0.0)
          think_lens.append(_think_len_tokens(s.completion_text, enc=enc))

      rollout_ms = (time.perf_counter() - t0) * 1000.0
      local_tokens = sum(len(s) for s in seqs)
      # Global token count: all_reduce sum across ranks (handles variable-length generation)
      if world > 1:
        tokens_tensor = torch.tensor([local_tokens], device="cuda", dtype=torch.int64)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        tokens_this_step = int(tokens_tensor.item())
      else:
        tokens_this_step = local_tokens
      tokens_seen += tokens_this_step

      device = torch.device("cuda")
      rewards_t = torch.tensor(rewards_total, device=device, dtype=torch.float32).reshape(b, g)

      # Filter zero-std groups if enabled (groups with all same reward provide no signal)
      groups_filtered = 0
      if filter_zero_std:
        keep_mask = filter_zero_std_groups(rewards_t)
        groups_filtered = int((~keep_mask).sum().item())
        if groups_filtered > 0 and groups_filtered < b:
          # Only keep groups with variance
          keep_idx = keep_mask.nonzero(as_tuple=True)[0]
          rewards_t = rewards_t[keep_idx]
          # Also filter sequences (each group has g sequences)
          keep_seq_idx = []
          for idx in keep_idx.tolist():
            keep_seq_idx.extend(range(idx * g, (idx + 1) * g))
          seqs = [seqs[i] for i in keep_seq_idx]
          prompt_lens = [prompt_lens[i] for i in keep_seq_idx]
          completion_lens = [completion_lens[i] for i in keep_seq_idx]
          behavior_logp_means = [behavior_logp_means[i] for i in keep_seq_idx]
          think_lens = [think_lens[i] for i in keep_seq_idx]  # Also filter think_lens
          b = len(rewards_t)  # Update batch size

      # Compute advantages with Dr.GRPO normalization + asymmetric scaling
      adv = group_relative_advantages(
          rewards_t,
          normalize_mean=normalize_mean,
          normalize_std=normalize_std,
          neg_scale=neg_adv_scale,
      ).reshape(-1).detach()

      with torch.inference_mode():
        nll_ref = completion_nll_mean(
          model_ref,
          seqs=seqs,
          prompt_lens=prompt_lens,
          completion_lens=completion_lens,
          pad_id=cfg.eos_token_id,
          device=device,
          max_length=logp_max_len,
        )
        logp_old = torch.tensor(behavior_logp_means, device=device, dtype=torch.float32)
        logp_ref = (-nll_ref).detach()

      # Optimize on this rollout batch (PPO-style multiple updates).
      lr = update_lr(optimizer, dense_groups, step_num, tokens_seen, cfg)
      for _k in range(updates_per_batch):
        nll = completion_nll_mean(
          model,
          seqs=seqs,
          prompt_lens=prompt_lens,
          completion_lens=completion_lens,
          pad_id=cfg.eos_token_id,
          device=device,
          max_length=logp_max_len,
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
        opt_step(model, optimizer, dense_groups, zero2_state, cfg, world)
        last_loss = float(loss.detach().item())

      # PERL: IRC guardrails (log2 schedule inside invariance window).
      if perl_adapters is not None:
        s = step_num + 1
        if _should_check_irc(s, window=perl_window):
          from nmoe.perl.irc import IrcThresholds, compute_irc_summary

          irc = compute_irc_summary(perl_adapters)
          thr = IrcThresholds()
          # Reduce max across ranks for safety.
          v = torch.tensor([irc.rho, irc.delta_frac, irc.radial_frac], device=device, dtype=torch.float32)
          if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(v, op=torch.distributed.ReduceOp.MAX)
          irc_rho, irc_delta, irc_radial = (float(v[0].item()), float(v[1].item()), float(v[2].item()))

          if rank == 0 and metrics_ctx is not None and metrics_ctx.writer is not None:
            metrics_ctx.writer.insert_many(
              s,
              [
                ("perl/irc_rho", irc_rho),
                ("perl/irc_delta_frac", irc_delta),
                ("perl/irc_radial_frac", irc_radial),
              ],
            )

          abort = (
            irc_rho > thr.rho_abort
            or irc_delta > thr.delta_frac_abort
            or irc_radial > thr.radial_frac_abort
          )
          warn = (
            irc_rho > thr.rho_warn
            or irc_delta > thr.delta_frac_warn
            or irc_radial > thr.radial_frac_warn
          )
          if rank == 0 and warn:
            print(
              f"[perl][warn] step={s} "
              f"irc_rho={irc_rho:.3f} irc_delta_frac={irc_delta:.3f} irc_radial_frac={irc_radial:.3f}"
            )
          if abort:
            raise RuntimeError(
              "[perl][abort] IRC exceeded abort thresholds inside invariance window: "
              f"step={s} irc_rho={irc_rho:.3f} (abort={thr.rho_abort:.3f}) "
              f"irc_delta_frac={irc_delta:.3f} (abort={thr.delta_frac_abort:.3f}) "
              f"irc_radial_frac={irc_radial:.3f} (abort={thr.radial_frac_abort:.3f})"
            )

      # Log
      s = step_num + 1
      reward_mean = float(rewards_t.mean().item())
      reward_std = float(rewards_t.std(unbiased=False).item())
      think_mean = float(sum(think_lens) / max(1, len(think_lens)))
      if rank == 0:
        print(
          f"[rl] step={s} loss={last_loss:.4f} "
          f"reward_mean={reward_mean:.3f} reward_std={reward_std:.3f} "
          f"think_toks_mean={think_mean:.1f} rollout_ms={rollout_ms:.1f} "
          f"filtered={groups_filtered} opsm={m.opsm_frac:.2f}"
        )

      log_training_step(
        s,
        model=model,
        loss=torch.tensor(last_loss, device=device),
        lr=lr,
        tokens_this_step=int(tokens_this_step),
        state=metrics_state,
        print_fn=lambda *_a, **_k: None,
        ctx=metrics_ctx,
        loader_wait_ms=rollout_ms,
      )
      # RL-specific metrics
      try:
        if metrics_ctx is not None and metrics_ctx.writer is not None:
          metrics_ctx.writer.insert_many(
            s,
            [
              ("rl/reward_mean", reward_mean),
              ("rl/reward_std", reward_std),
              ("rl/think_tokens_mean", think_mean),
              ("rl/kl_mean", float(m.kl_mean)),
              ("rl/ratio_mean", float(m.ratio_mean)),
              ("rl/clip_frac", float(m.clip_frac)),
              ("rl/opsm_frac", float(m.opsm_frac)),
              ("rl/groups_filtered", float(groups_filtered)),
              ("rl/pg_loss", float(m.pg_loss)),
              ("rl/kl_loss", float(m.kl_loss)),
              ("rl/advantage_mean", float(m.advantage_mean)),
              ("rl/advantage_std", float(m.advantage_std)),
            ],
          )
      except Exception:
        pass

      save_checkpoint(
        checkpointer=checkpointer,
        step=s,
        tokens_seen=tokens_seen,
        model=model,
        optimizer=optimizer,
        loader=None,
        plan=None,
        zero2_state=zero2_state,
        cfg=cfg,
        rank=rank,
        config_fingerprint=config_fingerprint,
        checkpoint_every=checkpoint_every,
        print_fn=print,
      )

    if rank == 0:
      print(f"[nmoe] RL complete. tokens_seen={tokens_seen:,} steps={cfg.steps}")
    results = {"final_loss": float(last_loss), "tokens_seen": float(tokens_seen)}
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "completed", results)
    return results
  except Exception:
    if exp_tracker is not None and rank == 0:
      exp_tracker.end_run(run_id, "failed")
    raise
  finally:
    checkpointer.close()
    stop_metrics(metrics_ctx)
    if exp_tracker is not None:
      exp_tracker.close()


def main() -> None:
  if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__)
    sys.exit(0)
  if len(sys.argv) < 2:
    print("Usage: python -m nmoe.rl.train <config.toml> [--key=value ...]", file=sys.stderr)
    sys.exit(1)

  with open(sys.argv[1], "rb") as f:
    cfg_dict = tomllib.load(f)

  # Minimal overrides (same pattern as nmoe.train)
  for arg in sys.argv[2:]:
    if arg.startswith("--") and "=" in arg:
      key, val = arg[2:].split("=", 1)
      if val.lower() in ("true", "false"):
        val = val.lower() == "true"
      elif val.lstrip("-").isdigit():
        val = int(val)
      elif val.replace(".", "", 1).lstrip("-").isdigit():
        val = float(val)
      cfg_dict[key] = val

  cfg = Config(**cfg_dict)
  try:
    train(cfg)
  finally:
    runtime.finalize()


if __name__ == "__main__":
  main()
