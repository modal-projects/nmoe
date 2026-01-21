"""Distributed validation for nmoe.rl (8-GPU torchrun path).

Design:
- Initializes distributed (NCCL) via nmoe.runtime.init() to match training.
- Runs a small set of per-rank GPU + collective checks.
- Runs the full nmoe.rl.tests suite on rank0 only (other ranks wait), since
  the unit tests are not written to be per-rank device-index safe.

Usage (single node, 8 GPU):

    # Pick a free port to avoid collisions on shared nodes:
    PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=$PORT -m nmoe.rl.tests_dist

The --master_addr=127.0.0.1 keeps rendezvous local; --master_port=$PORT avoids
the default 29500 which may be in use by other jobs on shared infrastructure.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Callable

# Ensure third_party is on PYTHONPATH before any nmoe imports.
# This allows `torchrun -m nmoe.rl.tests_dist` to work without manual env setup.
_repo_root: Path | None = None
for _p in Path(__file__).resolve().parents:
  if (_p / "third_party").is_dir():
    _repo_root = _p
    break

_extra_paths = []
if _repo_root is not None:
  _extra_paths = [_repo_root / "third_party"]
_existing = os.environ.get("PYTHONPATH", "")
_existing_parts = _existing.split(os.pathsep) if _existing else []

for _p in reversed(_extra_paths):
  if not _p.is_dir():
    continue
  _s = str(_p)
  if _s not in sys.path:
    sys.path.insert(0, _s)
  if _s not in _existing_parts:
    _existing_parts.insert(0, _s)

if _existing_parts:
  os.environ["PYTHONPATH"] = os.pathsep.join(_existing_parts)

import torch
import torch.distributed as dist

from nmoe import runtime


def _print_rank0(rank: int, msg: str) -> None:
  if rank == 0:
    print(msg, flush=True)


def _run_test(rank: int, name: str, fn: Callable[[], None]) -> bool:
  try:
    fn()
    _print_rank0(rank, f"  ✓ {name}")
    return True
  except Exception as e:
    print(f"[rank={rank}] ✗ {name}: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    return False


def _allreduce_min_ok(ok: bool) -> bool:
  t = torch.tensor([1 if ok else 0], device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32)
  dist.all_reduce(t, op=dist.ReduceOp.MIN)
  return bool(int(t.item()))


def _broadcast_i32_from_rank0(v: int) -> int:
  t = torch.tensor([v], device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32)
  dist.broadcast(t, src=0)
  return int(t.item())


def _preflight_check() -> None:
  """Verify critical imports and sandbox before running expensive GPU tests."""
  errors = []

  # Check third_party imports (FlashAttn, etc.)
  try:
    import flash_attn  # noqa: F401
  except ImportError as e:
    errors.append(f"flash_attn: {e}")

  # Check codex sandbox can spawn python
  try:
    from nmoe.rl.tools.codex import CodexConfig, CodexExecutor
    executor = CodexExecutor(CodexConfig(timeout_ms=5000))
    result = executor.exec_bash("python3 -c 'print(1+1)'")
    if not result.success or "2" not in result.stdout:
      errors.append(f"codex sandbox: python3 check failed (success={result.success})")
  except Exception as e:
    errors.append(f"codex sandbox: {e}")

  if errors:
    raise RuntimeError("Preflight check failed:\n  " + "\n  ".join(errors))


def main() -> int:
  # Run preflight before distributed init (avoids wasting GPU resources on bad env).
  _preflight_check()

  rank, world = runtime.init(seed=0)

  if not dist.is_initialized() or world <= 1:
    raise RuntimeError("expected torchrun multi-GPU launch (dist initialized, world_size > 1)")

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  _print_rank0(rank, "=" * 60)
  _print_rank0(rank, "NMOE RL Distributed Test Suite")
  _print_rank0(rank, "=" * 60)
  if world != 8 and rank == 0:
    print(f"[rl/tests_dist] warning: expected world_size=8 (got {world})", flush=True)

  ok = True

  ok &= _run_test(rank, "dist: device assignment matches LOCAL_RANK", lambda: _check_device(local_rank))
  ok &= _run_test(rank, "dist: all_reduce SUM works", lambda: _check_allreduce_sum(rank, world))
  ok &= _run_test(rank, "rl: GPU tensor ops work per-rank", _check_rl_gpu_ops)
  ok &= _run_test(rank, "rl: e2e GRPO step (fwd/bwd/opt)", lambda: _check_rl_e2e_grpo_step(world))
  ok &= _run_test(rank, "rl: e2e GRPO step w/ PERL (adapters-only + IRC)", lambda: _check_rl_e2e_grpo_step_perl(world))

  ok_global = _allreduce_min_ok(ok)
  dist.barrier()

  # Run the full unit suite on rank0 only.
  rc = 0
  if rank == 0:
    from nmoe.rl import tests as rl_tests

    rc = int(rl_tests.run_all_tests())

  rc = _broadcast_i32_from_rank0(rc)
  ok_global = ok_global and (rc == 0)

  dist.barrier()

  if rank == 0:
    print("-" * 60)
    print(f"Results: {'PASS' if ok_global else 'FAIL'} (world={world})")

  runtime.finalize()
  return 0 if ok_global else 1


def _check_device(local_rank: int) -> None:
  dev = torch.cuda.current_device()
  if dev != local_rank:
    raise AssertionError(f"current_device={dev} LOCAL_RANK={local_rank}")


def _check_allreduce_sum(rank: int, world: int) -> None:
  t = torch.tensor([rank], device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int64)
  dist.all_reduce(t, op=dist.ReduceOp.SUM)
  expected = world * (world - 1) // 2
  got = int(t.item())
  if got != expected:
    raise AssertionError(f"all_reduce SUM mismatch: got={got} expected={expected}")


def _check_rl_gpu_ops() -> None:
  # Minimal CUDA coverage for core GRPO math: this should run on every rank's GPU.
  from nmoe.rl.grpo import compute_kl, group_relative_advantages

  device = torch.device("cuda", torch.cuda.current_device())
  log_p = torch.randn(128, device=device, dtype=torch.float32)
  log_ref = torch.randn(128, device=device, dtype=torch.float32)
  kl = compute_kl(log_p, log_ref, kl_type="k3")
  if not (kl.is_cuda and torch.isfinite(kl).all()):
    raise AssertionError("compute_kl must run on CUDA and produce finite values")

  rewards = torch.randn(16, 4, device=device, dtype=torch.float32)
  adv = group_relative_advantages(rewards, normalize_mean=True, normalize_std=False)
  if not (adv.is_cuda and adv.shape == rewards.shape and torch.isfinite(adv).all()):
    raise AssertionError("group_relative_advantages must run on CUDA and preserve shape")


def _check_rl_e2e_grpo_step(world: int) -> None:
  """End-to-end GRPO update step on a tiny MoE model (8-GPU torchrun).

  Purpose: catch kernel-level errors (e.g., illegal memory access) that unit tests
  won't hit, without depending on HF datasets, tokenizers, or the full RL trainer.
  """
  from nmoe.config import Config
  from nmoe.model import Transformer
  from nmoe.opt import build_optimizer, step as opt_step
  from nmoe.rl.grpo import group_relative_advantages, grpo_loss
  from nmoe.rl.rollout import completion_nll_mean
  import inspect

  device = torch.device("cuda", torch.cuda.current_device())
  rank = int(dist.get_rank())

  # Tiny config: still exercises MLA (FlashMLA) + MoE (RDEP).
  #
  # Use BF16 here to avoid depending on the ExpertAdamW (blockscaled) optimizer ABI
  # while still covering the end-to-end autograd + communication path.
  vocab_size = 1024
  eos = vocab_size - 1
  n_seq_local = 2
  seqlen = 32
  topk = 2
  cfg = Config(
    preset="tests_rl_e2e",
    experiment_id="rl_e2e",
    vocab_size=vocab_size,
    eos_token_id=eos,
    dim=512,
    n_layers=2,
    n_dense_layers=1,
    n_heads=4,
    inter_dim=2048,
    moe_inter_dim=512,
    n_routed_experts=8,
    n_activated_experts=topk,
    n_shared_experts=0,
    q_lora_rank=128,
    kv_lora_rank=64,
    max_position_embeddings=64,
    dtype="bf16",
    batch_size=int(world) * int(n_seq_local),  # capacity uses global batch tokens
    seq_len=int(seqlen),
    steps=1,
    resume=False,
    lr_dense=1e-3,
    lr_router=1e-3,
    lr_expert=1e-3,
    weight_decay=0.0,
  )

  torch.manual_seed(1234 + rank)
  model = Transformer(cfg).to(device=device)
  model.init_weights()
  model.train()

  muon_optimizer = None
  opt_out = build_optimizer(model, cfg)
  if not isinstance(opt_out, tuple):
    raise TypeError(f"build_optimizer must return a tuple (got {type(opt_out)})")
  if len(opt_out) == 2:
    optimizer, dense_groups = opt_out
  elif len(opt_out) == 3:
    optimizer, muon_optimizer, dense_groups = opt_out
  else:
    raise TypeError(f"build_optimizer returned unexpected tuple len={len(opt_out)}")
  zero2_state: dict = {}

  # Synthetic fixed-length sequences: [prompt | completion].
  prompt_len = 8
  seqs: list[list[int]] = []
  prompt_lens: list[int] = []
  completion_lens: list[int] = []
  for _ in range(n_seq_local):
    toks = torch.randint(0, vocab_size - 1, (seqlen,), device="cpu", dtype=torch.int64).tolist()
    seqs.append(toks)
    prompt_lens.append(prompt_len)
    completion_lens.append(seqlen - prompt_len)

  # Rewards/advantages: 1 group with G=n_seq_local.
  rewards = torch.randn(1, n_seq_local, device=device, dtype=torch.float32)
  adv = group_relative_advantages(rewards, normalize_mean=True, normalize_std=False).reshape(-1).detach()

  # Old/ref logprobs from the current model (this is enough to exercise the graph).
  with torch.inference_mode():
    nll_old = completion_nll_mean(
      model,
      seqs=seqs,
      prompt_lens=prompt_lens,
      completion_lens=completion_lens,
      pad_id=eos,
      device=device,
      max_length=(seqlen - prompt_len),
    )
    logp_old = (-nll_old).detach()
    logp_ref = logp_old

  nll = completion_nll_mean(
    model,
    seqs=seqs,
    prompt_lens=prompt_lens,
    completion_lens=completion_lens,
    pad_id=eos,
    device=device,
    max_length=(seqlen - prompt_len),
  )
  logp = -nll
  loss, _m = grpo_loss(
    logp_mean=logp,
    logp_mean_old=logp_old,
    logp_mean_ref=logp_ref,
    advantages=adv,
    clip_eps=0.2,
    kl_coef=0.0,
    kl_type="k3",
    use_opsm=False,
  )
  if not torch.isfinite(loss).all():
    raise AssertionError(f"non-finite loss: {loss}")

  w_before = float(model.lm_head.weight.detach().float()[0, 0].item())
  model.zero_grad(set_to_none=True)
  loss.backward()
  torch.cuda.synchronize(device)

  n_opt_args = len(inspect.signature(opt_step).parameters)
  if n_opt_args == 6:
    opt_step(model, optimizer, dense_groups, zero2_state, cfg, world)
  elif n_opt_args == 7:
    opt_step(model, optimizer, muon_optimizer, dense_groups, zero2_state, cfg, world)
  else:
    raise TypeError(f"unexpected opt.step signature with {n_opt_args} parameters")
  torch.cuda.synchronize(device)

  w_after = float(model.lm_head.weight.detach().float()[0, 0].item())
  if w_before == w_after:
    raise AssertionError("expected params to update after optimizer step (lm_head[0,0] unchanged)")


def _check_rl_e2e_grpo_step_perl(world: int) -> None:
  """End-to-end GRPO update step with PERL (L/DoRA Mode A) enabled.

  Verifies:
  - A/B adapter params update (non-trivial gradient path).
  - All non-adapter params are unchanged (PERL adapters-only).
  - IRC metrics stay within abort thresholds (contract guardrails).
  """
  from nmoe.config import Config
  from nmoe.model import Transformer
  from nmoe.opt import build_optimizer, step as opt_step
  from nmoe.perl.apply import apply_ldora
  from nmoe.perl.irc import IrcThresholds, compute_irc_summary
  from nmoe.perl.policy import validate_optimizer_contract
  from nmoe.rl.grpo import group_relative_advantages, grpo_loss
  from nmoe.rl.rollout import completion_nll_mean
  import inspect

  device = torch.device("cuda", torch.cuda.current_device())
  rank = int(dist.get_rank())

  vocab_size = 1024
  eos = vocab_size - 1
  n_seq_local = 2
  seqlen = 32
  prompt_len = 8
  topk = 2
  perl_rank = 32

  cfg = Config(
    preset="tests_rl_e2e_perl",
    experiment_id="rl_e2e_perl",
    vocab_size=vocab_size,
    eos_token_id=eos,
    dim=512,
    n_layers=2,
    n_dense_layers=1,
    n_heads=4,
    inter_dim=2048,
    moe_inter_dim=512,
    n_routed_experts=8,
    n_activated_experts=topk,
    n_shared_experts=0,
    q_lora_rank=128,
    kv_lora_rank=64,
    max_position_embeddings=64,
    dtype="bf16",
    batch_size=int(world) * int(n_seq_local),
    seq_len=int(seqlen),
    steps=1,
    resume=False,
    lr_dense=1e-3,
    lr_router=1e-3,
    lr_expert=0.0,  # Must not update non-adapter expert weights in this test.
    weight_decay=0.0,
    router_bias_update_rate=0.0,
  )

  torch.manual_seed(5678 + rank)
  model = Transformer(cfg).to(device=device)
  model.init_weights()
  model.train()

  perl_adapters, _manifest = apply_ldora(model, rank=perl_rank, freeze_base=True)

  # Adapters-only training policy for this test:
  # - allow adapter params (A,B) to train
  # - keep expert weights requires_grad=True so build_optimizer() can construct expert optimizer,
  #   but use lr_expert=0 so they stay unchanged
  adapter_ids = {id(m.A) for m in perl_adapters.values()} | {id(m.B) for m in perl_adapters.values()}
  expert_params, _dense_params = model.param_sets()
  expert_ids = {id(p) for p in expert_params}
  for _name, p in model.named_parameters():
    p.requires_grad_(id(p) in adapter_ids or id(p) in expert_ids)

  muon_optimizer = None
  opt_out = build_optimizer(model, cfg)
  if not isinstance(opt_out, tuple):
    raise TypeError(f"build_optimizer must return a tuple (got {type(opt_out)})")
  if len(opt_out) == 2:
    optimizer, dense_groups = opt_out
  elif len(opt_out) == 3:
    optimizer, muon_optimizer, dense_groups = opt_out
  else:
    raise TypeError(f"build_optimizer returned unexpected tuple len={len(opt_out)}")
  validate_optimizer_contract(dense_groups, perl_adapters)
  zero2_state: dict = {}

  # Snapshot all params so we can prove "only adapters update".
  before: dict[int, torch.Tensor] = {id(p): p.detach().clone() for p in model.parameters()}

  seqs: list[list[int]] = []
  prompt_lens: list[int] = []
  completion_lens: list[int] = []
  for _ in range(n_seq_local):
    toks = torch.randint(0, vocab_size - 1, (seqlen,), device="cpu", dtype=torch.int64).tolist()
    seqs.append(toks)
    prompt_lens.append(prompt_len)
    completion_lens.append(seqlen - prompt_len)

  rewards = torch.randn(1, n_seq_local, device=device, dtype=torch.float32)
  adv = group_relative_advantages(rewards, normalize_mean=True, normalize_std=False).reshape(-1).detach()

  with torch.inference_mode():
    nll_old = completion_nll_mean(
      model,
      seqs=seqs,
      prompt_lens=prompt_lens,
      completion_lens=completion_lens,
      pad_id=eos,
      device=device,
      max_length=(seqlen - prompt_len),
    )
    logp_old = (-nll_old).detach()
    logp_ref = logp_old

  nll = completion_nll_mean(
    model,
    seqs=seqs,
    prompt_lens=prompt_lens,
    completion_lens=completion_lens,
    pad_id=eos,
    device=device,
    max_length=(seqlen - prompt_len),
  )
  logp = -nll
  loss, _m = grpo_loss(
    logp_mean=logp,
    logp_mean_old=logp_old,
    logp_mean_ref=logp_ref,
    advantages=adv,
    clip_eps=0.2,
    kl_coef=0.0,
    kl_type="k3",
    use_opsm=False,
  )
  if not torch.isfinite(loss).all():
    raise AssertionError(f"non-finite loss: {loss}")

  model.zero_grad(set_to_none=True)
  loss.backward()
  torch.cuda.synchronize(device)

  n_opt_args = len(inspect.signature(opt_step).parameters)
  if n_opt_args == 6:
    opt_step(model, optimizer, dense_groups, zero2_state, cfg, world)
  elif n_opt_args == 7:
    opt_step(model, optimizer, muon_optimizer, dense_groups, zero2_state, cfg, world)
  else:
    raise TypeError(f"unexpected opt.step signature with {n_opt_args} parameters")
  torch.cuda.synchronize(device)

  # Check updates: at least one adapter param changed; no non-adapter param changed.
  changed_adapter = False
  for p in model.parameters():
    p_before = before.get(id(p), None)
    if p_before is None:
      raise AssertionError("missing param snapshot")
    same = torch.equal(p.detach(), p_before)
    is_adapter = id(p) in adapter_ids
    if is_adapter and not same:
      changed_adapter = True
    if (not is_adapter) and (not same):
      raise AssertionError("non-adapter parameter updated under PERL adapters-only policy")
  if not changed_adapter:
    raise AssertionError("expected at least one adapter parameter to update (no A/B changed)")

  # IRC guardrails: abort thresholds must hold (max reduced across ranks).
  thr = IrcThresholds()
  irc = compute_irc_summary(perl_adapters)
  v = torch.tensor([irc.rho, irc.delta_frac, irc.radial_frac], device=device, dtype=torch.float32)
  dist.all_reduce(v, op=dist.ReduceOp.MAX)
  irc_rho, irc_delta, irc_radial = (float(v[0].item()), float(v[1].item()), float(v[2].item()))
  if irc_rho > thr.rho_abort or irc_delta > thr.delta_frac_abort or irc_radial > thr.radial_frac_abort:
    raise AssertionError(
      "IRC exceeded abort thresholds: "
      f"rho={irc_rho:.4f} (abort={thr.rho_abort:.4f}) "
      f"delta_frac={irc_delta:.4f} (abort={thr.delta_frac_abort:.4f}) "
      f"radial_frac={irc_radial:.4f} (abort={thr.radial_frac_abort:.4f})"
    )


if __name__ == "__main__":
  raise SystemExit(main())
