# SPDX-License-Identifier: Apache-2.0
"""Router microbenchmark: nmoe vs vLLM vs SGLang vs "cuBLASLt/TF32".

This benchmarks *routing only* (gate GEMM + grouped top-k selection) on a single
GPU. It is intended to answer: "how fast could the router be if we used the same
router kernels as vLLM / SGLang, and/or TF32?"

Notes
- No model checkpoints are required; weights/activations are random.
- The vLLM/SGLang baselines are in-tree ports that JIT-compile on first use.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from nmoe.serve.benchmark_router_ops import sglang_moe_fused_gate, vllm_grouped_topk
from nmoe.serve.model import ModelConfig, MoEGate


@dataclass(frozen=True)
class BenchCfg:
  batch: int = 32  # per-rank decode batch for BS=256 @ world_size=8
  hidden: int = 7168
  num_experts: int = 256
  topk: int = 8
  num_groups: int = 8
  topk_groups: int = 4
  route_scale: float = 2.5
  warmup: int = 50
  iters: int = 500


def _require_cuda() -> None:
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required.")


@contextlib.contextmanager
def _tf32(enabled: bool):
  """Context manager for TF32 matmul behavior (best-effort across torch versions)."""
  matmul = torch.backends.cuda.matmul
  if hasattr(matmul, "fp32_precision"):
    orig = matmul.fp32_precision
    matmul.fp32_precision = "tf32" if enabled else "ieee"
    try:
      yield
    finally:
      matmul.fp32_precision = orig
  else:
    orig = matmul.allow_tf32
    matmul.allow_tf32 = bool(enabled)
    try:
      yield
    finally:
      matmul.allow_tf32 = orig


def _bench_ms(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(iters):
    fn()
  end.record()
  torch.cuda.synchronize()
  return float(start.elapsed_time(end)) / float(iters)


def _make_inputs(cfg: BenchCfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
  # Activations are bf16 in our serve path.
  x = torch.randn(cfg.batch, cfg.hidden, device=device, dtype=torch.bfloat16) * 0.01
  w = torch.randn(cfg.num_experts, cfg.hidden, device=device, dtype=torch.float32) * 0.01
  # DeepSeek-V3 gate bias exists for hidden=7168; keep it enabled by default.
  b = torch.randn(cfg.num_experts, device=device, dtype=torch.float32) * 0.01
  return x, w, b


def _nmoe_router(cfg: BenchCfg, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
  gate_cfg = ModelConfig(
    hidden_size=cfg.hidden,
    num_experts=cfg.num_experts,
    num_experts_per_tok=cfg.topk,
    num_expert_groups=cfg.num_groups,
    num_limited_groups=cfg.topk_groups,
    route_scale=cfg.route_scale,
  )
  gate = MoEGate(gate_cfg).to(x.device)
  with torch.no_grad():
    gate.weight.copy_(w)
    if gate.bias is not None:
      gate.bias.copy_(b if b is not None else torch.zeros_like(gate.bias))

  def run():
    # Baseline: disable TF32 so we measure "true FP32" behavior.
    with _tf32(False):
      return gate(x)

  return run


def _torch_router_from_logits(
  cfg: BenchCfg,
  *,
  x: torch.Tensor,
  w: torch.Tensor,
  b: Optional[torch.Tensor],
  tf32: bool,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
  # Replicates MoEGate semantics but allows controlling matmul TF32.
  def run():
    with _tf32(tf32):
      scores = F.linear(x.float(), w, None).sigmoid()
    original_scores = scores
    scores_for_choice = scores if b is None else (scores + b)
    if cfg.num_groups > 1:
      scores_for_choice = scores_for_choice.view(-1, cfg.num_groups, cfg.num_experts // cfg.num_groups)
      group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1) if b is not None else scores_for_choice.amax(dim=-1)
      group_idx = group_scores.topk(cfg.topk_groups, dim=-1)[1]
      mask = torch.ones_like(group_scores, dtype=torch.bool).scatter_(1, group_idx, False)
      scores_for_choice = scores_for_choice.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
    indices = scores_for_choice.topk(cfg.topk, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * cfg.route_scale
    return weights, indices

  return run


def _vllm_grouped_topk_router(
  cfg: BenchCfg,
  *,
  x: torch.Tensor,
  w: torch.Tensor,
  b: torch.Tensor,
) -> Optional[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
  # Ported vLLM grouped_topk kernel (no external imports).
  w_bf16 = w.to(dtype=torch.bfloat16)
  b_bf16 = b.to(dtype=torch.bfloat16)

  def run():
    logits = F.linear(x, w_bf16, None)
    topk_values, topk_indices = vllm_grouped_topk(
      logits,
      num_expert_group=cfg.num_groups,
      topk_group=cfg.topk_groups,
      topk=cfg.topk,
      renormalize=True,
      routed_scaling_factor=cfg.route_scale,
      bias=b_bf16,
      scoring_func=1,  # sigmoid
    )
    return topk_values, topk_indices.to(torch.int64)

  return run


def _sglang_moe_fused_gate_router(
  cfg: BenchCfg,
  *,
  x: torch.Tensor,
  w: torch.Tensor,
  b: torch.Tensor,
) -> Optional[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
  # Ported SGLang moe_fused_gate kernel (no external imports).
  w_bf16 = w.to(dtype=torch.bfloat16)

  def run():
    # Match sglang usage: compute logits, then call the fused gate on float32.
    logits = F.linear(x, w_bf16, None).to(torch.float32)
    topk_weights, topk_ids = sglang_moe_fused_gate(
      logits,
      bias=b,
      num_expert_group=cfg.num_groups,
      topk_group=cfg.topk_groups,
      topk=cfg.topk,
      num_fused_shared_experts=0,
      routed_scaling_factor=cfg.route_scale,
      apply_routed_scaling_factor_on_output=True,
    )
    return topk_weights, topk_ids.to(torch.int64)

  return run


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", type=int, default=0)
  parser.add_argument("--batch", type=int, default=32)
  parser.add_argument("--iters", type=int, default=500)
  parser.add_argument("--warmup", type=int, default=50)
  parser.add_argument(
    "--methods",
    type=str,
    default="nmoe,nmoe_tf32,vllm,sglang",
    help="Comma-separated: nmoe, nmoe_tf32, vllm, sglang",
  )
  args = parser.parse_args()

  _require_cuda()
  device = torch.device(f"cuda:{args.device}")
  torch.cuda.set_device(device)

  cfg = BenchCfg(batch=int(args.batch), iters=int(args.iters), warmup=int(args.warmup))

  x, w, b = _make_inputs(cfg, device)
  assert b is not None

  methods = [m.strip() for m in args.methods.split(",") if m.strip()]
  runners: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = []

  if "nmoe" in methods:
    runners.append(("nmoe_current(fp32_ieee)", _nmoe_router(cfg, x, w, b)))
  if "nmoe_tf32" in methods:
    runners.append(("cublaslt_tf32(fp32)", _torch_router_from_logits(cfg, x=x, w=w, b=b, tf32=True)))
  if "vllm" in methods:
    v = _vllm_grouped_topk_router(cfg, x=x, w=w, b=b)
    assert v is not None
    runners.append(("vllm_grouped_topk(port)", v))
  if "sglang" in methods:
    s = _sglang_moe_fused_gate_router(cfg, x=x, w=w, b=b)
    assert s is not None
    runners.append(("sglang_moe_fused_gate(port)", s))

  name = torch.cuda.get_device_name(device)
  major, minor = torch.cuda.get_device_capability(device)
  print(f"[routerbench] device={device} {name} sm={major}.{minor} torch={torch.__version__}")
  print(f"[routerbench] batch={cfg.batch} hidden={cfg.hidden} experts={cfg.num_experts} topk={cfg.topk} groups={cfg.num_groups} topk_groups={cfg.topk_groups}")

  # Baseline outputs for rough correctness/behavior sanity (ids equality %).
  base_weights: Optional[torch.Tensor] = None
  base_ids: Optional[torch.Tensor] = None

  with torch.inference_mode():
    ok_runners: list[tuple[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]] = []
    for label, fn in runners:
      try:
        w_out, ids_out = fn()
      except Exception as e:
        print(f"[routerbench] {label}: unavailable ({type(e).__name__}: {e})")
        continue
      if base_ids is None:
        base_weights, base_ids = w_out, ids_out
      ok_runners.append((label, fn))
      torch.cuda.synchronize()

    print("")
    print(f"{'method':28s}  {'ms/call':>10s}  {'tok/s':>12s}  {'ids_match%':>10s}")
    print("-" * 70)
    for label, fn in ok_runners:
      ms = _bench_ms(lambda: fn(), warmup=cfg.warmup, iters=cfg.iters)
      tok_s = float(cfg.batch) / (ms / 1000.0)
      ids_match = ""
      if base_ids is not None:
        w_out, ids_out = fn()
        match = (ids_out == base_ids).to(torch.float32).mean().item() * 100.0
        ids_match = f"{match:9.2f}"
      print(f"{label:28s}  {ms:10.4f}  {tok_s:12.1f}  {ids_match:>10s}")


if __name__ == "__main__":
  main()
