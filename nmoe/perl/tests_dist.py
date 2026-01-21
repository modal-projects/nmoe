"""Distributed tests for nmoe.perl (8-GPU torchrun path).

Run with:
  torchrun --nproc_per_node=8 -m nmoe.perl.tests_dist

This suite is intentionally minimal and targets the training runtime:
- Initializes distributed (NCCL) via nmoe.runtime.init().
- Verifies per-rank device placement and basic collectives.
- Verifies IRC max aggregation can be reduced across ranks.
- Runs the full local PERL suite on rank0 only (other ranks wait).
"""
from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Callable

import torch
from torch import nn
import torch.distributed as dist

from nmoe import runtime
from nmoe.perl.apply import apply_ldora
from nmoe.perl.irc import compute_irc_summary
from nmoe.perl.ldora import LDoRAInit, LDoRALinear


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


def _allreduce_max_f(x: float) -> float:
  t = torch.tensor([x], device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.float32)
  dist.all_reduce(t, op=dist.ReduceOp.MAX)
  return float(t.item())


def _broadcast_i32_from_rank0(rank: int, v: int) -> int:
  t = torch.tensor([v], device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32)
  dist.broadcast(t, src=0)
  return int(t.item())


def main() -> int:
  rank, world = runtime.init(seed=0)

  if not dist.is_initialized() or world <= 1:
    raise RuntimeError("expected torchrun multi-GPU launch (dist initialized, world_size > 1)")

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  _print_rank0(rank, "=" * 60)
  _print_rank0(rank, "NMOE PERL Distributed Test Suite")
  _print_rank0(rank, "=" * 60)
  if world != 8 and rank == 0:
    print(f"[perl/tests_dist] warning: expected world_size=8 (got {world})", flush=True)

  ok = True

  ok &= _run_test(
    rank,
    "dist: device assignment matches LOCAL_RANK",
    lambda: _check_device_assignment(local_rank),
  )

  ok &= _run_test(
    rank,
    "dist: all_reduce SUM works",
    lambda: _check_allreduce_sum(rank, world),
  )

  ok &= _run_test(
    rank,
    "perl: apply_ldora preserves per-rank device",
    lambda: _check_apply_ldora_device(),
  )

  ok &= _run_test(
    rank,
    "perl: IRC max reduces across ranks",
    lambda: _check_irc_max_reduce(rank, world),
  )

  ok &= _run_test(
    rank,
    "perl: merge equivalence (adapter vs merged logits)",
    lambda: _check_merge_equivalence(world),
  )

  # Ensure every rank reaches the next phase.
  ok_global = _allreduce_min_ok(ok)

  dist.barrier()

  # Full local PERL suite on rank0 only; other ranks wait.
  full_ok = 1
  if rank == 0:
    from nmoe.perl import tests as perl_tests

    rc = int(perl_tests.run_all_tests())
    full_ok = 1 if rc == 0 else 0

  full_ok = _broadcast_i32_from_rank0(rank, full_ok)
  ok_global = ok_global and bool(full_ok)

  dist.barrier()

  if rank == 0:
    print("-" * 60)
    print(f"Results: {'PASS' if ok_global else 'FAIL'} (world={world})")

  runtime.finalize()
  return 0 if ok_global else 1


def _check_device_assignment(local_rank: int) -> None:
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


def _check_apply_ldora_device() -> None:
  device = torch.device("cuda", torch.cuda.current_device())
  model = nn.Sequential(nn.Linear(4, 7, bias=False, dtype=torch.bfloat16, device=device))
  adapters, _ = apply_ldora(model, rank=4)
  for _, m in adapters.items():
    if not (m.weight.is_cuda and m.A.is_cuda and m.B.is_cuda and m.g0.is_cuda):
      raise AssertionError("patched adapter params must be CUDA")
    if m.weight.device != device or m.A.device != device or m.B.device != device or m.g0.device != device:
      raise AssertionError("patched adapter params must be on the current rank device")
  x = torch.randn(2, 4, device=device, dtype=torch.bfloat16)
  y = model(x)
  if y.device != device:
    raise AssertionError("forward output must stay on current device")


def _check_irc_max_reduce(rank: int, world: int) -> None:
  # Construct one module per rank with a known per-rank delta_frac.
  #
  # For out=1, in=2, rank=1, eps=0, alpha=rank => scale=1:
  #   W0 = [w, 0], g0=|w|
  #   A=[1,0], B=[1] -> Δ=[1,0], ||Δ||=1
  #   delta_frac = ||Δ||/||W0|| = 1/|w|
  #
  # Pick w = 2^{-rank} so delta_frac = 2^{rank} and the global max is 2^{world-1}.
  device = torch.device("cuda", torch.cuda.current_device())
  init = LDoRAInit(rank=1, alpha=None, eps=0.0)
  lin = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16, device=device)
  m = LDoRALinear.from_linear(lin, init=init, freeze_base=True)
  with torch.no_grad():
    w = math.ldexp(1.0, -rank)  # exact power-of-two, bf16-representable
    m.weight.zero_()
    m.weight[0, 0] = torch.tensor(w, dtype=m.weight.dtype, device=m.weight.device)
    m._reset_g0_from_weight()
    m.A.zero_()
    m.B.zero_()
    m.A[0, 0] = torch.tensor(1.0, dtype=m.A.dtype, device=m.A.device)
    m.B[0, 0] = torch.tensor(1.0, dtype=m.B.dtype, device=m.B.device)

  s = compute_irc_summary({f"rank_{rank}": m}, eps=0.0)
  local = float(s.delta_frac)
  global_max = _allreduce_max_f(local)

  expected_local = float(1 << rank)
  if abs(local - expected_local) > 1e-3:
    raise AssertionError(f"unexpected local delta_frac: got={local} expected={expected_local}")

  expected_global = float(1 << (world - 1))
  if abs(global_max - expected_global) > 1e-3:
    raise AssertionError(f"unexpected global max delta_frac: got={global_max} expected={expected_global}")


def _check_merge_equivalence(world: int) -> None:
  """Check adapter-form vs merged-form logits equivalence under inference dtype.

  Metric: max_{t,vocab} |logits_adapter - logits_merged|.
  """
  from nmoe.config import Config
  from nmoe.model import Transformer

  device = torch.device("cuda", torch.cuda.current_device())

  vocab_size = 512
  eos = vocab_size - 1
  seqlen = 16

  cfg = Config(
    preset="tests_perl_merge",
    experiment_id="perl_merge",
    vocab_size=vocab_size,
    eos_token_id=eos,
    dim=256,
    n_layers=2,
    n_dense_layers=2,  # dense-only (no RDEP / MoE) for a clean merge test
    n_heads=4,
    inter_dim=1024,
    moe_inter_dim=256,
    q_lora_rank=64,
    kv_lora_rank=32,
    max_position_embeddings=64,
    dtype="bf16",
    batch_size=int(world),
    seq_len=int(seqlen),
    steps=1,
    resume=False,
  )

  # P_merge (pinned token batch).
  tokens = torch.tensor(
    [
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, eos],
      [42, 7, 19, 5, 0, 3, 99, 12, 1, 4, 8, 16, 32, 64, 2, eos],
    ],
    device=device,
    dtype=torch.int64,
  )

  torch.manual_seed(0)
  model_adapter = Transformer(cfg).to(device=device).eval()
  model_adapter.init_weights()
  adapters, _manifest = apply_ldora(model_adapter, rank=32, freeze_base=True)
  with torch.no_grad():
    # Make the adapter non-trivial but stay in the small-δ regime.
    for m in adapters.values():
      m.B.normal_(mean=0.0, std=1e-3)

  with torch.inference_mode():
    logits_adapter = model_adapter(tokens).float()

  # Build merged model: same config, replace each adapted linear weight with effective_weight().
  torch.manual_seed(0)
  model_merged = Transformer(cfg).to(device=device).eval()
  model_merged.init_weights()
  sd_src = model_adapter.state_dict()
  sd_dst = model_merged.state_dict()

  with torch.no_grad():
    for k in list(sd_dst.keys()):
      if k.endswith(".weight"):
        path = k[:-len(".weight")]
        m = adapters.get(path, None)
        if m is not None:
          sd_dst[k] = m.effective_weight().to(dtype=sd_dst[k].dtype)
          continue
      if k not in sd_src:
        raise AssertionError(f"missing key in adapter model state_dict: {k}")
      sd_dst[k] = sd_src[k]

  model_merged.load_state_dict(sd_dst, strict=True)
  with torch.inference_mode():
    logits_merged = model_merged(tokens).float()

  max_abs = float((logits_adapter - logits_merged).abs().max().item())
  # Reduce across ranks (should match for dense-only), and assert.
  max_abs = _allreduce_max_f(max_abs)

  eps_merge = 5e-2  # bf16 tolerance (operational; tighten once we have empirical margins)
  if max_abs > eps_merge:
    raise AssertionError(f"merge equivalence failed: max_abs={max_abs:.6f} eps={eps_merge}")


if __name__ == "__main__":
  raise SystemExit(main())
