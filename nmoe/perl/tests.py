"""Tests for nmoe.perl (CPU).

Run with:
  python -m nmoe.perl.tests
"""
from __future__ import annotations

import math
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, List

import torch
from torch import nn


@dataclass
class TestResult:
  name: str
  passed: bool
  error: str | None = None


RESULTS: List[TestResult] = []


def test(name: str):
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


@test("import: nmoe.perl")
def test_import():
  from nmoe.perl import LDoRALinear, apply_ldora, compute_irc_summary, validate_optimizer_contract


@test("ldora: init (B0=0, rank-independent A0 std)")
def test_ldora_init():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m4 = LDoRALinear(512, 64, init=LDoRAInit(rank=4))
  m32 = LDoRALinear(512, 64, init=LDoRAInit(rank=32))

  assert torch.allclose(m4.B, torch.zeros_like(m4.B)), "B0 must be all zeros"
  assert torch.allclose(m32.B, torch.zeros_like(m32.B)), "B0 must be all zeros"

  # A0 std should be close across ranks (same formula).
  s4 = float(m4.A.float().std().item())
  s32 = float(m32.A.float().std().item())
  ratio = s32 / max(1e-12, s4)
  assert 0.85 <= ratio <= 1.15, f"A0 std should be rank-independent (ratio={ratio:.3f})"


@test("ldora: delta=0 matches base linear")
def test_ldora_delta_zero_matches_base():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  lin = nn.Linear(8, 5, bias=False, dtype=torch.bfloat16)
  ld = LDoRALinear.from_linear(lin, init=LDoRAInit(rank=4))

  x = torch.randn(3, 8, dtype=torch.bfloat16)
  y0 = lin(x)
  y1 = ld(x)
  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3), "delta=0 must match base"

@test("ldora: delta=0 matches base linear (with bias)")
def test_ldora_delta_zero_matches_base_with_bias():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  lin = nn.Linear(8, 5, bias=True, dtype=torch.bfloat16)
  ld = LDoRALinear.from_linear(lin, init=LDoRAInit(rank=4))

  x = torch.randn(3, 8, dtype=torch.bfloat16)
  y0 = lin(x)
  y1 = ld(x)
  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3), "delta=0 must match base (bias)"

@test("ldora: forward matches explicit formula (nonzero ΔW)")
def test_ldora_forward_matches_reference():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m = LDoRALinear(16, 12, init=LDoRAInit(rank=4), weight_dtype=torch.float32)
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.1)
    m._reset_g0_from_weight()
    m.A.normal_(mean=0.0, std=0.02)
    m.B.normal_(mean=0.0, std=0.02)

  x = torch.randn(8, 16, dtype=torch.float32)
  y_mod = m(x)

  # Reference: V = W0 + s * (B@A); W = g0 * V / ||V||; y = x W^T
  W0 = m.weight.float()
  A = m.A.float()
  B = m.B.float()
  s = float(m.lora_scale())
  V = W0 + s * (B @ A)
  v_norm = torch.linalg.vector_norm(V, ord=2, dim=1).clamp_min(1e-12)
  W = V * (m.g0.float() / v_norm).unsqueeze(1)
  y_ref = torch.nn.functional.linear(x, W)

  assert torch.allclose(y_mod, y_ref, atol=3e-2, rtol=3e-2), "forward mismatch vs explicit formula"


@test("ldora: bias is not scaled by DoRA factor")
def test_ldora_bias_not_scaled():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m = LDoRALinear(16, 12, init=LDoRAInit(rank=4), bias=True, weight_dtype=torch.float32)
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.1)
    m.bias.normal_(mean=0.0, std=0.1)
    m._reset_g0_from_weight()
    m.A.normal_(mean=0.0, std=0.02)
    m.B.normal_(mean=0.0, std=0.02)

  x = torch.randn(8, 16, dtype=torch.float32)
  y_mod = m(x)

  W_eff = m.effective_weight()
  y_ref = torch.nn.functional.linear(x, W_eff, m.bias)

  assert torch.allclose(y_mod, y_ref, atol=3e-2, rtol=3e-2), "bias must be added after DoRA scaling"


@test("ldora: grads match explicit formula (A,B,x)")
def test_ldora_grads_match_reference():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m = LDoRALinear(16, 12, init=LDoRAInit(rank=4), weight_dtype=torch.float32)
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.1)
    m._reset_g0_from_weight()
    m.A.normal_(mean=0.0, std=0.02)
    m.B.normal_(mean=0.0, std=0.02)

  x = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
  target = torch.randn(8, 12, dtype=torch.float32)

  def loss_fn(y: torch.Tensor) -> torch.Tensor:
    return (y - target).float().pow(2).mean()

  # Module grads
  y_mod = m(x)
  loss_mod = loss_fn(y_mod)
  gA_mod, gB_mod, gx_mod = torch.autograd.grad(loss_mod, [m.A, m.B, x], retain_graph=False, create_graph=False)

  # Reference grads (explicit formula; do not call module helpers)
  W0 = m.weight.float()
  A = m.A.float()
  B = m.B.float()
  s = float(m.lora_scale())
  V = W0 + s * (B @ A)
  v_norm = torch.linalg.vector_norm(V, ord=2, dim=1).clamp_min(1e-12)
  W = V * (m.g0.float() / v_norm).unsqueeze(1)
  y_ref = torch.nn.functional.linear(x, W)
  loss_ref = loss_fn(y_ref)
  gA_ref, gB_ref, gx_ref = torch.autograd.grad(loss_ref, [m.A, m.B, x], retain_graph=False, create_graph=False)

  assert torch.allclose(gx_mod, gx_ref, atol=5e-2, rtol=5e-2), "dL/dx mismatch"
  assert torch.allclose(gA_mod.float(), gA_ref.float(), atol=5e-2, rtol=5e-2), "dL/dA mismatch"
  assert torch.allclose(gB_mod.float(), gB_ref.float(), atol=5e-2, rtol=5e-2), "dL/dB mismatch"


@test("ldora: finite-diff sanity on A,B grads")
def test_ldora_finite_diff_grads():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m = LDoRALinear(8, 6, init=LDoRAInit(rank=2), weight_dtype=torch.float32)
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.1)
    m._reset_g0_from_weight()
    m.A.normal_(mean=0.0, std=0.05)
    m.B.normal_(mean=0.0, std=0.05)

  x = torch.randn(4, 8, dtype=torch.float32)

  def loss() -> torch.Tensor:
    y = m(x)
    return y.float().pow(2).mean()

  gA, gB = torch.autograd.grad(loss(), [m.A, m.B], retain_graph=False, create_graph=False)

  h = 1e-2
  # Check a handful of elements (deterministic indices).
  idx_a = [(0, 0), (0, 3), (1, 2)]
  idx_b = [(0, 0), (2, 1), (5, 0)]

  for i, j in idx_a:
    with torch.no_grad():
      orig = m.A[i, j].item()
      m.A[i, j] = torch.tensor(orig + h, dtype=m.A.dtype)
    lp = float(loss().item())
    with torch.no_grad():
      m.A[i, j] = torch.tensor(orig - h, dtype=m.A.dtype)
    lm = float(loss().item())
    with torch.no_grad():
      m.A[i, j] = torch.tensor(orig, dtype=m.A.dtype)
    fd = (lp - lm) / (2.0 * h)
    ad = float(gA[i, j].float().item())
    denom = max(1e-3, abs(fd), abs(ad))
    assert abs(fd - ad) / denom < 0.35, f"finite-diff dA[{i},{j}] mismatch (fd={fd:.4g}, ad={ad:.4g})"

  for i, j in idx_b:
    with torch.no_grad():
      orig = m.B[i, j].item()
      m.B[i, j] = torch.tensor(orig + h, dtype=m.B.dtype)
    lp = float(loss().item())
    with torch.no_grad():
      m.B[i, j] = torch.tensor(orig - h, dtype=m.B.dtype)
    lm = float(loss().item())
    with torch.no_grad():
      m.B[i, j] = torch.tensor(orig, dtype=m.B.dtype)
    fd = (lp - lm) / (2.0 * h)
    ad = float(gB[i, j].float().item())
    denom = max(1e-3, abs(fd), abs(ad))
    assert abs(fd - ad) / denom < 0.35, f"finite-diff dB[{i},{j}] mismatch (fd={fd:.4g}, ad={ad:.4g})"

@test("cuda: forward/backward matches reference (BF16 ops)")
def test_cuda_forward_backward_matches_reference():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  if not torch.cuda.is_available():
    print("  (skipped: no CUDA)")
    return

  torch.manual_seed(0)
  m = LDoRALinear(32, 24, init=LDoRAInit(rank=8), weight_dtype=torch.bfloat16).cuda()
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.05)
    m._reset_g0_from_weight()
    m.A.normal_(mean=0.0, std=0.02)
    m.B.normal_(mean=0.0, std=0.02)

  x = torch.randn(4, 7, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
  target = torch.randn(4, 7, 24, device="cuda", dtype=torch.float32)

  def loss_fn(y: torch.Tensor) -> torch.Tensor:
    return (y.float() - target).pow(2).mean()

  # Module path
  y_mod = m(x)
  loss_mod = loss_fn(y_mod)
  gA_mod, gB_mod, gx_mod = torch.autograd.grad(loss_mod, [m.A, m.B, x], retain_graph=False, create_graph=False)

  # Reference path (do not call module forward/helpers): same BF16 matmuls, same DoRA scaling.
  W0 = m.weight  # BF16
  z = torch.nn.functional.linear(x, m.A)  # BF16
  delta = torch.nn.functional.linear(z, m.B)  # BF16
  out_v = torch.nn.functional.linear(x, W0) + delta * float(m.lora_scale())

  V = W0.float() + float(m.lora_scale()) * (m.B.float() @ m.A.float())
  v_norm = torch.linalg.vector_norm(V, ord=2, dim=1).clamp_min(1e-12)
  row_scale = (m.g0.float() / v_norm).to(dtype=out_v.dtype)
  y_ref = out_v * row_scale
  loss_ref = loss_fn(y_ref)
  gA_ref, gB_ref, gx_ref = torch.autograd.grad(loss_ref, [m.A, m.B, x], retain_graph=False, create_graph=False)

  assert torch.allclose(y_mod, y_ref, atol=1e-3, rtol=1e-3), "cuda forward mismatch"
  assert torch.allclose(gx_mod, gx_ref, atol=5e-3, rtol=5e-3), "cuda dL/dx mismatch"
  assert torch.allclose(gA_mod, gA_ref, atol=5e-3, rtol=5e-3), "cuda dL/dA mismatch"
  assert torch.allclose(gB_mod, gB_ref, atol=5e-3, rtol=5e-3), "cuda dL/dB mismatch"

@test("cuda: from_linear preserves device placement")
def test_cuda_from_linear_preserves_device():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  if not torch.cuda.is_available():
    print("  (skipped: no CUDA)")
    return

  lin = nn.Linear(8, 5, bias=False, dtype=torch.bfloat16).cuda()
  ld = LDoRALinear.from_linear(lin, init=LDoRAInit(rank=4))
  assert ld.weight.is_cuda and ld.A.is_cuda and ld.B.is_cuda and ld.g0.is_cuda


@test("cuda: from_linear supports bias and preserves device")
def test_cuda_from_linear_bias_preserves_device():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  if not torch.cuda.is_available():
    print("  (skipped: no CUDA)")
    return

  lin = nn.Linear(8, 5, bias=True, dtype=torch.bfloat16).cuda()
  ld = LDoRALinear.from_linear(lin, init=LDoRAInit(rank=4))
  assert ld.weight.is_cuda and ld.A.is_cuda and ld.B.is_cuda and ld.g0.is_cuda
  assert ld.bias is not None and ld.bias.is_cuda

  x = torch.randn(2, 8, device="cuda", dtype=torch.bfloat16)
  y0 = lin(x)
  y1 = ld(x)
  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3), "cuda from_linear(bias) must match base at init"


@test("cuda: apply_ldora preserves device placement")
def test_cuda_apply_ldora_preserves_device():
  from nmoe.perl.apply import apply_ldora

  if not torch.cuda.is_available():
    print("  (skipped: no CUDA)")
    return

  model = nn.Sequential(
    nn.Linear(4, 7, bias=False, dtype=torch.bfloat16),
    nn.ReLU(),
    nn.Linear(7, 3, bias=False, dtype=torch.bfloat16),
  ).cuda()

  adapters, _ = apply_ldora(model, rank=4)
  for _, m in adapters.items():
    assert m.weight.is_cuda and m.A.is_cuda and m.B.is_cuda and m.g0.is_cuda
  x = torch.randn(2, 4, device="cuda", dtype=torch.bfloat16)
  y = model(x)
  assert y.is_cuda and y.shape == (2, 3)


@test("cuda: checkpointing works (non-reentrant)")
def test_cuda_checkpointing_smoke():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  if not torch.cuda.is_available():
    print("  (skipped: no CUDA)")
    return

  torch.manual_seed(0)
  m = LDoRALinear(16, 12, init=LDoRAInit(rank=4), weight_dtype=torch.bfloat16).cuda()
  with torch.no_grad():
    m.weight.normal_(mean=0.0, std=0.05)
    m._reset_g0_from_weight()

  x = torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)

  def f(inp: torch.Tensor) -> torch.Tensor:
    return m(inp)

  y = torch.utils.checkpoint.checkpoint(f, x, use_reentrant=False)
  loss = y.float().pow(2).mean()
  loss.backward()
  assert x.grad is not None and torch.isfinite(x.grad).all()
  assert m.A.grad is not None and torch.isfinite(m.A.grad).all()
  assert m.B.grad is not None and torch.isfinite(m.B.grad).all()


@test("ldora: radial-only ΔW cancels (direction preserved)")
def test_ldora_radial_update_cancels():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  ld = LDoRALinear(2, 2, init=LDoRAInit(rank=1, eps=0.0))
  with torch.no_grad():
    ld.weight.zero_()
    ld.weight[0, 0] = 3.0
    ld.weight[1, 1] = 4.0
    ld._reset_g0_from_weight()
    # ΔW = [[1,0],[0,0]] which is colinear with row0 direction.
    ld.A.zero_()
    ld.A[0, 0] = 1.0
    ld.B.zero_()
    ld.B[0, 0] = 1.0

  W_eff = ld.effective_weight()
  assert torch.allclose(W_eff, ld.weight.float(), atol=1e-6, rtol=0.0), "radial-only update must not change W"


@test("irc: deterministic values on 2x2 example")
def test_irc_values():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear
  from nmoe.perl.irc import compute_irc_summary

  ld = LDoRALinear(2, 2, init=LDoRAInit(rank=1, eps=0.0))
  with torch.no_grad():
    ld.weight.zero_()
    ld.weight[0, 0] = 3.0
    ld.weight[1, 1] = 4.0
    ld._reset_g0_from_weight()
    ld.A.zero_()
    ld.A[0, 0] = 1.0
    ld.B.zero_()
    ld.B[0, 0] = 1.0

  s = compute_irc_summary({"w": ld}, eps=0.0)
  # Row0: ||Δ||/||W0|| = 1/3, radial drift = (4-3)/3, rho = log(3/4).
  assert abs(s.delta_frac - (1.0 / 3.0)) < 1e-6
  assert abs(s.radial_frac - (1.0 / 3.0)) < 1e-6
  assert abs(s.rho - abs(float(torch.log(torch.tensor(3.0 / 4.0)).item()))) < 1e-6


@test("apply: patches all Linear modules and preserves output at init")
def test_apply_ldora_patching():
  from nmoe.perl.apply import apply_ldora

  torch.manual_seed(0)
  model = nn.Sequential(
    nn.Linear(3, 4, bias=False, dtype=torch.bfloat16),
    nn.ReLU(),
    nn.Sequential(
      nn.Linear(4, 5, bias=False, dtype=torch.bfloat16),
      nn.Tanh(),
      nn.Linear(5, 2, bias=False, dtype=torch.bfloat16),
    ),
  )

  x = torch.randn(7, 3, dtype=torch.bfloat16)
  y0 = model(x)
  adapters, manifest = apply_ldora(model, rank=4)
  y1 = model(x)

  assert len(adapters) == 3, f"expected 3 patched linears (got {len(adapters)})"
  assert len(manifest) == 3
  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3), "patched model must match at init (Δ=0)"

  # Base weights should be frozen by default.
  for _, m in adapters.items():
    assert m.weight.requires_grad is False
    assert m.bias is None or m.bias.requires_grad is False


@test("apply: patches bias=True Linear modules and preserves output at init")
def test_apply_ldora_patching_with_bias():
  from nmoe.perl.apply import apply_ldora

  torch.manual_seed(0)
  model = nn.Sequential(
    nn.Linear(3, 4, bias=True, dtype=torch.bfloat16),
    nn.ReLU(),
    nn.Sequential(
      nn.Linear(4, 5, bias=True, dtype=torch.bfloat16),
      nn.Tanh(),
      nn.Linear(5, 2, bias=True, dtype=torch.bfloat16),
    ),
  )

  x = torch.randn(7, 3, dtype=torch.bfloat16)
  y0 = model(x)
  adapters, manifest = apply_ldora(model, rank=4)
  y1 = model(x)

  assert len(adapters) == 3, f"expected 3 patched linears (got {len(adapters)})"
  assert len(manifest) == 3
  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3), "patched model must match at init (Δ=0)"

  for _, m in adapters.items():
    assert m.bias is not None and m.bias.requires_grad is False


@test("policy: optimizer contract (wd=0, base frozen, present)")
def test_policy_optimizer_contract():
  from nmoe.perl.apply import apply_ldora
  from nmoe.perl.policy import validate_optimizer_contract

  class _FakeOpt:
    def __init__(self, param_groups):
      self.param_groups = param_groups

  model = nn.Sequential(nn.Linear(4, 3, bias=False, dtype=torch.bfloat16))
  adapters, _ = apply_ldora(model, rank=4)
  m = next(iter(adapters.values()))

  # Good: A,B present and wd=0, base weight absent.
  validate_optimizer_contract(_FakeOpt([{"params": [m.A, m.B], "weight_decay": 0.0}]), adapters)
  validate_optimizer_contract([{"params": [m.A, m.B], "weight_decay": 0.0}], adapters)

  # Bad: missing adapter param
  try:
    validate_optimizer_contract(_FakeOpt([{"params": [m.A], "weight_decay": 0.0}]), adapters)
    raise AssertionError("expected missing adapter params to raise")
  except ValueError:
    pass

  # Bad: nonzero wd
  try:
    validate_optimizer_contract(_FakeOpt([{"params": [m.A, m.B], "weight_decay": 0.1}]), adapters)
    raise AssertionError("expected nonzero weight_decay on adapters to raise")
  except ValueError:
    pass

  # Bad: base weight included
  try:
    validate_optimizer_contract(_FakeOpt([{"params": [m.A, m.B, m.weight], "weight_decay": 0.0}]), adapters)
    raise AssertionError("expected base weight in optimizer to raise")
  except ValueError:
    pass

  # Bad: base bias included (if present)
  model_b = nn.Sequential(nn.Linear(4, 3, bias=True, dtype=torch.bfloat16))
  adapters_b, _ = apply_ldora(model_b, rank=4)
  mb = next(iter(adapters_b.values()))
  assert mb.bias is not None
  try:
    validate_optimizer_contract(_FakeOpt([{"params": [mb.A, mb.B, mb.bias], "weight_decay": 0.0}]), adapters_b)
    raise AssertionError("expected base bias in optimizer to raise")
  except ValueError:
    pass


@test("irc: max_W(q99_rows) aggregation is max over matrices")
def test_irc_max_aggregation():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear
  from nmoe.perl.irc import compute_irc_summary

  # Two matrices with different known delta_frac.
  a = LDoRALinear(2, 1, init=LDoRAInit(rank=1, eps=0.0))
  b = LDoRALinear(2, 1, init=LDoRAInit(rank=1, eps=0.0))
  with torch.no_grad():
    for m, w in [(a, 2.0), (b, 10.0)]:
      m.weight.zero_()
      m.weight[0, 0] = w
      m._reset_g0_from_weight()
      m.A.zero_()
      m.B.zero_()
      m.A[0, 0] = 1.0
      m.B[0, 0] = 1.0

  s = compute_irc_summary({"a": a, "b": b}, eps=0.0)
  # delta_frac is 1/||W0||, so matrix a should dominate (1/2 > 1/10).
  assert abs(s.delta_frac - 0.5) < 1e-6


@test("irc: brute-force summary matches optimized (random)")
def test_irc_bruteforce_equivalence():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear
  from nmoe.perl.irc import compute_irc_summary

  def q_higher(x: torch.Tensor, q: float) -> torch.Tensor:
    assert x.ndim == 1
    n = int(x.numel())
    k0 = int(math.ceil(q * n)) - 1
    k0 = max(0, min(n - 1, k0))
    return x.kthvalue(k0 + 1).values

  def brute_summary(modules, *, eps: float = 1e-12) -> tuple[float, float, float]:
    rho_max = 0.0
    delta_max = 0.0
    radial_max = 0.0
    for m in modules.values():
      W0 = m.weight.detach().float()
      A = m.A.detach().float()
      B = m.B.detach().float()
      g0 = m.g0.detach().float()
      s = float(m.lora_scale())

      delta = torch.addmm(torch.zeros_like(W0), B, A, beta=0.0, alpha=s)
      V = W0 + delta

      v_norm = torch.linalg.vector_norm(V, ord=2, dim=1).clamp_min(eps)
      delta_norm = torch.linalg.vector_norm(delta, ord=2, dim=1)

      denom = g0 + eps
      rho = torch.log((g0 + eps) / (v_norm + eps))
      delta_frac = delta_norm / denom
      radial_frac = (v_norm - g0).abs() / denom

      rho_q99 = float(q_higher(rho.abs().flatten(), 0.99).item())
      delta_q99 = float(q_higher(delta_frac.flatten(), 0.99).item())
      radial_q99 = float(q_higher(radial_frac.flatten(), 0.99).item())

      rho_max = max(rho_max, rho_q99)
      delta_max = max(delta_max, delta_q99)
      radial_max = max(radial_max, radial_q99)
    return rho_max, delta_max, radial_max

  def make(device: torch.device, *, out_features: int, in_features: int, rank: int) -> LDoRALinear:
    init = LDoRAInit(rank=rank, alpha=None)
    lin = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
      lin.weight.copy_((0.1 * torch.randn_like(lin.weight, dtype=torch.float32)).to(dtype=torch.bfloat16))
    m = LDoRALinear.from_linear(lin, init=init, freeze_base=True)
    with torch.no_grad():
      m.A.copy_((0.05 * torch.randn_like(m.A, dtype=torch.float32)).to(dtype=m.A.dtype))
      m.B.copy_((0.05 * torch.randn_like(m.B, dtype=torch.float32)).to(dtype=m.B.dtype))
    return m

  def run(device: torch.device, *, trials: int) -> None:
    torch.manual_seed(0)
    random.seed(0)
    for _ in range(trials):
      modules = {}
      for j in range(3):
        o = random.randint(3, 257 if j == 0 else 129)
        i = random.randint(3, 257 if j == 0 else 129)
        r = random.randint(1, min(32 if j == 0 else 16, o, i))
        modules[f"m{j}"] = make(device, out_features=o, in_features=i, rank=r)

      ours = compute_irc_summary(modules)
      rho_b, delta_b, radial_b = brute_summary(modules)

      def close(a: float, b: float) -> None:
        # Both sides use float32 math but with different association; allow small drift.
        tol = 2e-4 + 2e-4 * abs(b)
        assert abs(a - b) <= tol, f"mismatch: ours={a} brute={b} tol={tol}"

      close(ours.rho, rho_b)
      close(ours.delta_frac, delta_b)
      close(ours.radial_frac, radial_b)

  run(torch.device("cpu"), trials=20)
  if torch.cuda.is_available():
    run(torch.device("cuda"), trials=20)


@test("ldora: state_dict roundtrip preserves output and g0")
def test_ldora_state_dict_roundtrip():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  init = LDoRAInit(rank=4)
  m0 = LDoRALinear(16, 12, init=init, weight_dtype=torch.bfloat16)
  with torch.no_grad():
    m0.weight.normal_(mean=0.0, std=0.1)
    m0._reset_g0_from_weight()
    m0.A.normal_(mean=0.0, std=0.02)
    m0.B.normal_(mean=0.0, std=0.02)

  x = torch.randn(3, 16, dtype=torch.bfloat16)
  y0 = m0(x)

  state = {k: v.clone() for k, v in m0.state_dict().items()}
  m1 = LDoRALinear(16, 12, init=init, weight_dtype=torch.bfloat16)
  m1.load_state_dict(state, strict=True)
  y1 = m1(x)

  assert torch.allclose(y0, y1, atol=1e-3, rtol=1e-3)
  assert torch.allclose(m0.g0, m1.g0, atol=0.0, rtol=0.0)


@test("ldora: optimizer step updates adapters, not base")
def test_ldora_optimizer_step_updates_only_adapters():
  from nmoe.perl.ldora import LDoRAInit, LDoRALinear

  torch.manual_seed(0)
  m = LDoRALinear(16, 12, init=LDoRAInit(rank=4), weight_dtype=torch.bfloat16)
  m.weight.requires_grad_(False)
  w0 = m.weight.detach().clone()
  g0 = m.g0.detach().clone()

  # Make A-grad nonzero by making B nonzero.
  with torch.no_grad():
    m.A.normal_(mean=0.0, std=0.02)
    m.B.normal_(mean=0.0, std=0.02)
  A0 = m.A.detach().clone()
  B0 = m.B.detach().clone()

  opt = torch.optim.SGD([m.A, m.B], lr=1.0, weight_decay=0.0)
  x = (5.0 * torch.randn(8, 16, dtype=torch.bfloat16))
  target = (5.0 * torch.randn(8, 12, dtype=torch.float32))

  def loss_fn() -> torch.Tensor:
    y = m(x)
    return (y.float() - target).pow(2).mean()

  loss = loss_fn()
  opt.zero_grad(set_to_none=True)
  loss.backward()
  opt.step()

  assert torch.allclose(m.weight, w0, atol=0.0, rtol=0.0), "base weight must remain frozen"
  assert torch.allclose(m.g0, g0, atol=0.0, rtol=0.0), "g0 must remain unchanged"
  assert float((m.A - A0).abs().max().item()) > 0.0, "A must update"
  assert float((m.B - B0).abs().max().item()) > 0.0, "B must update"


def run_all_tests() -> int:
  print("=" * 60)
  print("NMOE PERL Module Test Suite")
  print("=" * 60)

  tests = [
    test_import,
    test_ldora_init,
    test_ldora_delta_zero_matches_base,
    test_ldora_delta_zero_matches_base_with_bias,
    test_ldora_forward_matches_reference,
    test_ldora_bias_not_scaled,
    test_ldora_grads_match_reference,
    test_ldora_finite_diff_grads,
    test_ldora_radial_update_cancels,
    test_irc_values,
    test_policy_optimizer_contract,
    test_irc_max_aggregation,
    test_irc_bruteforce_equivalence,
    test_ldora_state_dict_roundtrip,
    test_ldora_optimizer_step_updates_only_adapters,
    test_apply_ldora_patching,
    test_apply_ldora_patching_with_bias,
    test_cuda_forward_backward_matches_reference,
    test_cuda_from_linear_preserves_device,
    test_cuda_from_linear_bias_preserves_device,
    test_cuda_apply_ldora_preserves_device,
    test_cuda_checkpointing_smoke,
  ]
  for fn in tests:
    fn()

  print("-" * 60)
  passed = sum(1 for r in RESULTS if r.passed)
  failed = sum(1 for r in RESULTS if not r.passed)
  print(f"Results: {passed} passed, {failed} failed")
  if failed:
    print("\nFailed tests:")
    for r in RESULTS:
      if not r.passed:
        print(f"  - {r.name}: {r.error}")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(run_all_tests())
