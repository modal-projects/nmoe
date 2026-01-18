# SPDX-License-Identifier: Apache-2.0
"""Standalone expert-MLP microbenchmark (decode operating point).

This benchmarks the per-layer MoE expert MLP compute path using the same kernels
as nmoe.serve:
  - DeepGEMM masked grouped FP8 GEMMs (gate-up and down)
  - fused grouped SwiGLU quantization (silu_mul_fp8_grouped)

It also validates numerical accuracy against a BF16 reference implementation:
  y = linear(silu(gate) * up, w2) where [gate, up] = linear(x, w13)

The benchmark is "decode-like" in that:
  - there are many experts with small per-expert token counts (masked_m),
  - sum(masked_m) ~= t_cap * topk (expected local expert hits per rank at EP8).

By default, masked_m is generated synthetically (uniform/zipf/hot). For a more
realistic decode operating point, pass --masked-m-source router and a checkpoint
path; the script will sample expert IDs using DeepSeek-V3 gate weights and
derive the per-rank masked_m distribution from those samples, then measure
expert MLP compute for a worst-case (but non-overflow) rank.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _force_pow2_ceil(x: torch.Tensor) -> torch.Tensor:
  x = torch.clamp(x, min=1e-4)
  return torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), torch.ceil(torch.log2(x)))


@torch.no_grad()
def _quantize_weight_blockwise_fp8_ue8m0(
  w_bf16: torch.Tensor, *, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize BF16 weights to FP8 E4M3FN with UE8M0 (pow2) scales per 128x128 block.

  Args:
    w_bf16: [G, O, I] BF16 weight tensor.
  Returns:
    w_fp8: [G, O, I] float8_e4m3fn
    w_scale: [G, O/128, I/128] float32 pow2 scales
  """
  if w_bf16.dtype != torch.bfloat16 or w_bf16.dim() != 3:
    raise TypeError(f"expected BF16 3D weight [G,O,I], got dtype={w_bf16.dtype} shape={tuple(w_bf16.shape)}")
  g, o, i = w_bf16.shape
  if (o % block) != 0 or (i % block) != 0:
    raise ValueError(f"weight dims must be divisible by {block} (got o={o} i={i})")

  w = w_bf16.float().view(g, o // block, block, i // block, block)
  # Reduce over the two 128-sized inner dims to get one scale per 128x128 block.
  amax = w.abs().amax(dim=(2, 4))  # [G, oB, iB]
  scale = _force_pow2_ceil(amax / 448.0).to(torch.float32)
  w_q = (w / scale[:, :, None, :, None].to(w.dtype)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
  return w_q.view(g, o, i), scale


def _make_masked_m(
  *,
  num_local: int,
  expected_m: int,
  num_pairs: int,
  dist: str,
  zipf_alpha: float,
  seed: int,
) -> torch.Tensor:
  gen = torch.Generator(device="cpu")
  gen.manual_seed(int(seed))

  if dist == "uniform":
    probs = torch.ones((num_local,), dtype=torch.float64)
  elif dist == "zipf":
    probs = 1.0 / torch.arange(1, num_local + 1, dtype=torch.float64).pow(float(zipf_alpha))
  elif dist == "hot":
    probs = torch.ones((num_local,), dtype=torch.float64)
    probs[0] = float(num_local) * 4.0
  else:
    raise ValueError(f"unknown dist: {dist}")
  probs = probs / probs.sum()

  expert_ids = torch.multinomial(probs, num_samples=int(num_pairs), replacement=True, generator=gen)
  counts = torch.bincount(expert_ids, minlength=num_local).to(torch.int32)
  if int(counts.max()) > int(expected_m):
    raise RuntimeError(
      f"masked_m overflow: max_count={int(counts.max())} > expected_m={int(expected_m)} "
      f"(try --expected-m larger or a less-skewed --dist)"
    )
  return counts


def _parse_int_list(s: str) -> list[int]:
  items = []
  for part in s.split(","):
    part = part.strip()
    if not part:
      continue
    items.append(int(part))
  return items


def _resolve_safetensors_shard(ckpt: str, *, rank: int) -> str:
  if os.path.isdir(ckpt):
    cand = os.path.join(ckpt, f"model{int(rank)}-mp8.safetensors")
    if os.path.exists(cand):
      return cand
    # Fall back to any safetensors in the directory.
    for name in sorted(os.listdir(ckpt)):
      if name.endswith(".safetensors"):
        return os.path.join(ckpt, name)
    raise FileNotFoundError(f"no .safetensors files found in ckpt dir: {ckpt}")
  return ckpt


def _find_gate_key(keys: list[str], *, layer: int, suffix: str) -> str:
  # Match either "model.layers.{i}..." or "layers.{i}..." style keys.
  token = f"layers.{int(layer)}{suffix}"
  token_dot = f".{token}"
  for k in keys:
    if k.endswith(token) or (token_dot in k):
      return k
  # Fall back to suffix-only search for better error messages.
  candidates = [k for k in keys if k.endswith(suffix)]
  raise KeyError(f"gate key not found for layer={layer} suffix={suffix}; saw {len(candidates)} keys with suffix")


@torch.no_grad()
def _router_topk_indices_deepseek_v3(
  x_bf16: torch.Tensor,
  *,
  gate_w: torch.Tensor,
  gate_b: torch.Tensor,
  num_groups: int,
  topk_groups: int,
  topk: int,
) -> torch.Tensor:
  # Match MoEGate.forward (DeepSeek-V3 bias-present path) closely.
  # We only need indices, not weights.
  logits = F.linear(x_bf16.float(), gate_w)
  scores = logits.sigmoid()
  scores_for_choice = scores + gate_b
  if int(num_groups) > 1:
    scores_for_choice = scores_for_choice.view(-1, int(num_groups), int(scores_for_choice.size(-1)) // int(num_groups))
    group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = group_scores.topk(int(topk_groups), dim=-1)[1]
    mask = torch.ones_like(group_scores, dtype=torch.bool).scatter_(1, group_idx, False)
    scores_for_choice = scores_for_choice.masked_fill(mask.unsqueeze(-1), float("-inf")).flatten(1)
  return scores_for_choice.topk(int(topk), dim=-1)[1]


@dataclass(frozen=True)
class _Candidate:
  trial: int
  layer: int
  rank: int
  masked_m: torch.Tensor  # [num_local] int32 CPU
  max_count: int
  sum_count: int


@torch.no_grad()
def _sample_router_masked_m_candidates(
  *,
  expected_m: int,
  num_local: int,
  world_size: int,
  t_cap: int,
  topk: int,
  hidden: int,
  x_scale: float,
  ckpt: str,
  ckpt_rank: int,
  gate_layers: list[int],
  trials: int,
  seed: int,
  device: torch.device,
) -> tuple[float, torch.Tensor, dict]:
  """Estimate overflow rate and pick a worst-case masked_m (non-overflow) for benchmarking.

  Overflow is defined as: for a given (trial, layer), any rank has max(masked_m) > expected_m.
  """
  if world_size != 8:
    raise ValueError("router masked_m sampling assumes world_size=8 for DeepSeek-V3 EP8.")
  if int(num_local) * int(world_size) != 256:
    raise ValueError("router masked_m sampling assumes num_experts=256 (DeepSeek-V3).")

  shard_path = _resolve_safetensors_shard(ckpt, rank=ckpt_rank)
  # Lazy import to keep the synthetic path runnable without safetensors.
  from safetensors import safe_open  # type: ignore[import-not-found]

  gate_params: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
  with safe_open(shard_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())
    for layer in gate_layers:
      w_key = _find_gate_key(keys, layer=layer, suffix=".ffn.gate.weight")
      b_key = _find_gate_key(keys, layer=layer, suffix=".ffn.gate.bias")
      w = f.get_tensor(w_key).to(device=device, dtype=torch.float32, non_blocking=True)
      b = f.get_tensor(b_key).to(device=device, dtype=torch.float32, non_blocking=True)
      gate_params[int(layer)] = (w, b)

  total_samples = int(trials) * int(len(gate_layers))
  overflow_samples = 0
  max_any_counts: list[int] = []
  best: _Candidate | None = None

  global_tokens = int(t_cap) * int(world_size)
  for t in range(int(trials)):
    torch.manual_seed(int(seed) + t)
    torch.cuda.manual_seed_all(int(seed) + t)
    # Use an RMS~1.0 activation scale to better approximate real hidden states
    # (RMSNorm output). Very small scales make gate.bias dominate and can create
    # unrealistically skewed routing.
    x = torch.randn((global_tokens, hidden), device=device, dtype=torch.bfloat16) * float(x_scale)
    for layer in gate_layers:
      w, b = gate_params[int(layer)]
      indices = _router_topk_indices_deepseek_v3(
        x,
        gate_w=w,
        gate_b=b,
        num_groups=8,
        topk_groups=4,
        topk=topk,
      )
      flat = indices.reshape(-1)  # [global_tokens*topk]
      dst = flat // int(num_local)
      local = flat - dst * int(num_local)

      # Compute per-rank max count; overflow if any rank exceeds expected_m.
      max_any = 0
      for r in range(int(world_size)):
        local_r = local[dst == r]
        counts_r = torch.bincount(local_r, minlength=int(num_local)).to(torch.int32)
        max_r = int(counts_r.max().item()) if counts_r.numel() else 0
        sum_r = int(counts_r.sum().item()) if counts_r.numel() else 0
        if max_r > max_any:
          max_any = max_r
        if max_r <= int(expected_m):
          # Candidate: maximize max_count, then sum_count (straggler stress).
          score = (max_r, sum_r)
          if best is None or score > (best.max_count, best.sum_count):
            best = _Candidate(
              trial=t,
              layer=int(layer),
              rank=r,
              masked_m=counts_r.cpu(),
              max_count=max_r,
              sum_count=sum_r,
            )

      max_any_counts.append(int(max_any))
      if int(max_any) > int(expected_m):
        overflow_samples += 1

  if best is None:
    raise RuntimeError(f"no non-overflow masked_m candidates found at expected_m={expected_m}")

  max_any_t = torch.tensor(max_any_counts, dtype=torch.float32)
  overflow_rate = float(overflow_samples) / float(total_samples) if total_samples else 0.0
  stats = {
    "samples": total_samples,
    "overflow_samples": overflow_samples,
    "overflow_rate": overflow_rate,
    "max_any_p50": float(torch.quantile(max_any_t, 0.50).item()) if max_any_t.numel() else 0.0,
    "max_any_p90": float(torch.quantile(max_any_t, 0.90).item()) if max_any_t.numel() else 0.0,
    "max_any_p99": float(torch.quantile(max_any_t, 0.99).item()) if max_any_t.numel() else 0.0,
    "max_any_max": float(max_any_t.max().item()) if max_any_t.numel() else 0.0,
    "candidate_trial": best.trial,
    "candidate_layer": best.layer,
    "candidate_rank": best.rank,
    "candidate_max": best.max_count,
    "candidate_sum": best.sum_count,
    "shard_path": shard_path,
  }
  return overflow_rate, best.masked_m.to(dtype=torch.int32), stats


@torch.no_grad()
def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hidden", type=int, default=7168)
  ap.add_argument("--inter", type=int, default=2048)
  ap.add_argument("--num-local", type=int, default=32)
  ap.add_argument("--expected-m", type=int, default=256)
  ap.add_argument("--expected-m-sweep", type=str, default="", help="Comma list, e.g. 128,160,192,256.")
  ap.add_argument("--t-cap", type=int, default=32, help="Per-rank token cap (decode: 32 for BS=256 at world=8).")
  ap.add_argument("--topk", type=int, default=8)
  ap.add_argument("--pairs", type=int, default=-1, help="Total expert pairs; default t_cap*topk.")
  ap.add_argument("--masked-m-source", choices=["synthetic", "router"], default="synthetic")
  ap.add_argument("--dist", choices=["uniform", "zipf", "hot"], default="zipf", help="synthetic masked_m only")
  ap.add_argument("--zipf-alpha", type=float, default=1.3)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--warmup-iters", type=int, default=50)
  ap.add_argument("--iters", type=int, default=200)
  ap.add_argument("--cos-min", type=float, default=0.99)
  ap.add_argument("--rel-l2-max", type=float, default=0.15)
  ap.add_argument("--skip-accuracy", action="store_true")
  ap.add_argument("--ckpt", type=str, default="", help="Checkpoint dir or shard file (router masked_m only).")
  ap.add_argument("--ckpt-rank", type=int, default=0, help="Shard rank to read from (router masked_m only).")
  ap.add_argument("--gate-layers", type=str, default="3,13,23,33,43,53", help="Comma list (router masked_m only).")
  ap.add_argument("--router-trials", type=int, default=200, help="Samples per gate layer (router masked_m only).")
  ap.add_argument("--router-x-scale", type=float, default=1.0, help="Router sampling x scale (router masked_m only).")
  args = ap.parse_args(argv)

  device = torch.device("cuda", 0)
  torch.cuda.set_device(device)

  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")

  hidden = int(args.hidden)
  inter = int(args.inter)
  num_local = int(args.num_local)
  t_cap = int(args.t_cap)
  topk = int(args.topk)
  num_pairs = int(args.pairs) if int(args.pairs) >= 0 else int(t_cap * topk)

  if (hidden % 128) != 0 or (inter % 128) != 0 or ((2 * inter) % 128) != 0:
    raise SystemExit("hidden and inter must be divisible by 128 for FP8 block scaling.")
  expected_ms = [int(args.expected_m)]
  if str(args.expected_m_sweep).strip():
    expected_ms = _parse_int_list(str(args.expected_m_sweep))
  for em in expected_ms:
    if em <= 0 or (em % 16) != 0:
      raise SystemExit("--expected-m and --expected-m-sweep values must be >0 and 16-aligned (DeepGEMM masked kernels).")

  torch.manual_seed(int(args.seed))
  torch.cuda.manual_seed_all(int(args.seed))

  from deep_gemm import m_grouped_fp8_gemm_nt_masked
  from nmoe.serve.kernels.fp8_quant import (
    pack_fp32_ue8m0_scales_to_int,
    quantize_fp8_ue8m0,
    silu_mul_fp8_grouped_packed,
  )

  # Use a single fixed set of weights across expected_m values so the sweep is
  # attributable to expected_m (padding/shape effects), not RNG variance.
  w13_bf16 = torch.randn((num_local, 2 * inter, hidden), device=device, dtype=torch.bfloat16) * 0.02
  w2_bf16 = torch.randn((num_local, hidden, inter), device=device, dtype=torch.bfloat16) * 0.02
  w13_fp8, w13_scale = _quantize_weight_blockwise_fp8_ue8m0(w13_bf16)
  w2_fp8, w2_scale = _quantize_weight_blockwise_fp8_ue8m0(w2_bf16)
  w13_scale_ue8m0 = pack_fp32_ue8m0_scales_to_int(w13_scale, mn=2 * inter, k=hidden, gran_mn_in=128)
  w2_scale_ue8m0 = pack_fp32_ue8m0_scales_to_int(w2_scale, mn=hidden, k=inter, gran_mn_in=128)

  def _time_ms(fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
      fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(iters)

  for expected_m in expected_ms:
    overflow_rate = 0.0
    overflow_stats = {}
    if str(args.masked_m_source) == "router":
      if not str(args.ckpt).strip():
        raise SystemExit("--masked-m-source router requires --ckpt")
      gate_layers = _parse_int_list(str(args.gate_layers))
      _, masked_m_cpu, overflow_stats = _sample_router_masked_m_candidates(
        expected_m=expected_m,
        num_local=num_local,
        world_size=8,
        t_cap=t_cap,
        topk=topk,
        hidden=hidden,
        x_scale=float(args.router_x_scale),
        ckpt=str(args.ckpt),
        ckpt_rank=int(args.ckpt_rank),
        gate_layers=gate_layers,
        trials=int(args.router_trials),
        seed=int(args.seed),
        device=device,
      )
      overflow_rate = float(overflow_stats.get("overflow_rate", 0.0))
    else:
      masked_m_cpu = _make_masked_m(
        num_local=num_local,
        expected_m=expected_m,
        num_pairs=num_pairs,
        dist=str(args.dist),
        zipf_alpha=float(args.zipf_alpha),
        seed=int(args.seed),
      )

    masked_m = masked_m_cpu.to(device=device)

    # Inputs: [num_local, expected_m, hidden] BF16 with rows >= masked_m zeroed.
    x_bf16 = torch.randn((num_local, expected_m, hidden), device=device, dtype=torch.bfloat16) * 0.02
    row = torch.arange(expected_m, device=device, dtype=torch.int32)[None, :]
    valid = row < masked_m[:, None]
    x_bf16.mul_(valid[:, :, None].to(dtype=x_bf16.dtype))

    x_bf16_2d = x_bf16.reshape(-1, hidden).contiguous()
    x_fp8_2d, x_scale_2d = quantize_fp8_ue8m0(x_bf16_2d)
    x_fp8 = x_fp8_2d.view(num_local, expected_m, hidden)
    x_scale = x_scale_2d.view(num_local, expected_m, hidden // 128)
    x_scale_ue8m0 = pack_fp32_ue8m0_scales_to_int(x_scale, mn=expected_m, k=hidden, gran_mn_in=1)

    # Run FP8 path once to produce outputs used for the accuracy check.
    gateup_out = torch.empty((num_local, expected_m, 2 * inter), device=device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_masked(
      (x_fp8, x_scale_ue8m0),
      (w13_fp8, w13_scale_ue8m0),
      gateup_out,
      masked_m,
      expected_m,
      disable_ue8m0_cast=True,
    )
    down_in_q, down_in_scale_ue8m0 = silu_mul_fp8_grouped_packed(gateup_out, masked_m)
    down_out = torch.empty((num_local, expected_m, hidden), device=device, dtype=torch.bfloat16)
    m_grouped_fp8_gemm_nt_masked(
      (down_in_q, down_in_scale_ue8m0),
      (w2_fp8, w2_scale_ue8m0),
      down_out,
      masked_m,
      expected_m,
      disable_ue8m0_cast=True,
    )

    cos = float("nan")
    rel_l2 = float("nan")
    p99_rel = float("nan")
    p99_abs = float("nan")
    max_abs = float("nan")
    if not bool(args.skip_accuracy):
      ref_out = torch.zeros_like(down_out)
      masked_m_host = masked_m_cpu.tolist()
      for e in range(num_local):
        m = int(masked_m_host[e])
        if m == 0:
          continue
        x_e = x_bf16[e, :m, :]
        gateup_ref = F.linear(x_e, w13_bf16[e])
        gate, up = gateup_ref[:, :inter], gateup_ref[:, inter:]
        down_in = F.silu(gate) * up
        y_ref = F.linear(down_in, w2_bf16[e])
        ref_out[e, :m, :].copy_(y_ref)

      valid_mask = valid.to(dtype=torch.bool)
      y = down_out[valid_mask].float()
      y_ref = ref_out[valid_mask].float()
      diff = y - y_ref
      abs_err = diff.abs()
      rel_l2 = float(diff.norm() / (y_ref.norm() + 1e-12))
      y_ref_abs = y_ref.abs()
      rel_err = abs_err / (y_ref_abs + 1e-6)

      cos = float((y * y_ref).sum() / (y.norm() * y_ref.norm() + 1e-12))
      rel_mask = y_ref_abs > 1e-3
      p99_rel = float(torch.quantile(rel_err[rel_mask], 0.99).item()) if bool(rel_mask.any().item()) else 0.0
      p99_abs = float(torch.quantile(abs_err, 0.99).item()) if abs_err.numel() else 0.0
      max_abs = float(abs_err.max().item()) if abs_err.numel() else 0.0

      if cos < float(args.cos_min) or max_abs > 1e-3 or rel_l2 > float(args.rel_l2_max):
        raise SystemExit(
          f"accuracy check failed: cos={cos:.6f} rel_l2={rel_l2:.6f} max_abs={max_abs:.6f} "
          f"(thresholds: cos>={args.cos_min} max_abs<=1e-3 rel_l2<={args.rel_l2_max})"
        )

    # Warm up kernels (quantize + pack + DeepGEMM + fused SwiGLU).
    for _ in range(int(args.warmup_iters)):
      x_fp8_w, x_scale_w = quantize_fp8_ue8m0(x_bf16_2d)
      x_fp8_w = x_fp8_w.view(num_local, expected_m, hidden)
      x_scale_w = x_scale_w.view(num_local, expected_m, hidden // 128)
      x_scale_w_ue8m0 = pack_fp32_ue8m0_scales_to_int(x_scale_w, mn=expected_m, k=hidden, gran_mn_in=1)
      m_grouped_fp8_gemm_nt_masked(
        (x_fp8_w, x_scale_w_ue8m0),
        (w13_fp8, w13_scale_ue8m0),
        gateup_out,
        masked_m,
        expected_m,
        disable_ue8m0_cast=True,
      )
      down_in_q_w, down_in_scale_w = silu_mul_fp8_grouped_packed(gateup_out, masked_m)
      m_grouped_fp8_gemm_nt_masked(
        (down_in_q_w, down_in_scale_w),
        (w2_fp8, w2_scale_ue8m0),
        down_out,
        masked_m,
        expected_m,
        disable_ue8m0_cast=True,
      )
    torch.cuda.synchronize()

    iters = int(args.iters)
    quantize_ms = _time_ms(lambda: quantize_fp8_ue8m0(x_bf16_2d), iters)
    pack_x_scale_ms = _time_ms(
      lambda: pack_fp32_ue8m0_scales_to_int(x_scale, mn=expected_m, k=hidden, gran_mn_in=1),
      iters,
    )
    gateup_ms = _time_ms(
      lambda: m_grouped_fp8_gemm_nt_masked(
        (x_fp8, x_scale_ue8m0),
        (w13_fp8, w13_scale_ue8m0),
        gateup_out,
        masked_m,
        expected_m,
        disable_ue8m0_cast=True,
      ),
      iters,
    )
    swiglu_pack_ms = _time_ms(lambda: silu_mul_fp8_grouped_packed(gateup_out, masked_m), iters)
    down_in_q2, down_in_scale2_ue8m0 = silu_mul_fp8_grouped_packed(gateup_out, masked_m)
    down_ms = _time_ms(
      lambda: m_grouped_fp8_gemm_nt_masked(
        (down_in_q2, down_in_scale2_ue8m0),
        (w2_fp8, w2_scale_ue8m0),
        down_out,
        masked_m,
        expected_m,
        disable_ue8m0_cast=True,
      ),
      iters,
    )
    total_ms = quantize_ms + pack_x_scale_ms + gateup_ms + swiglu_pack_ms + down_ms

    masked = masked_m_cpu.to(dtype=torch.int64)
    print("=== Expert MLP microbench (decode-like) ===", flush=True)
    print(f"hidden={hidden} inter={inter} num_local={num_local} expected_m={expected_m} pairs={num_pairs}", flush=True)
    print(
      f"masked_m: sum={int(masked.sum().item())} max={int(masked.max().item())} "
      f"p50={int(torch.quantile(masked.float(), 0.5).item())} p90={int(torch.quantile(masked.float(), 0.9).item())}",
      flush=True,
    )
    if str(args.masked_m_source) == "router":
      print(
        "router_overflow:"
        f" rate={overflow_rate:.6f}"
        f" samples={int(overflow_stats.get('samples', 0))}"
        f" max_any_p99={float(overflow_stats.get('max_any_p99', 0.0)):.1f}"
        f" max_any_max={float(overflow_stats.get('max_any_max', 0.0)):.1f}"
        f" candidate(layer={int(overflow_stats.get('candidate_layer', -1))}"
        f" trial={int(overflow_stats.get('candidate_trial', -1))}"
        f" rank={int(overflow_stats.get('candidate_rank', -1))}"
        f" max={int(overflow_stats.get('candidate_max', -1))}"
        f" sum={int(overflow_stats.get('candidate_sum', -1))})",
        flush=True,
      )
      print(f"router_shard: {str(overflow_stats.get('shard_path', ''))}", flush=True)

    if bool(args.skip_accuracy):
      print("accuracy: skipped", flush=True)
    else:
      print(
        f"accuracy: cos={cos:.6f} rel_l2={rel_l2:.6f} p99_abs={p99_abs:.6f} p99_rel(|ref|>1e-3)={p99_rel:.6f} max_abs={max_abs:.6f}",
        flush=True,
      )
    print(
      "timing_ms:"
      f" quantize={quantize_ms:.4f}"
      f" pack_x_scale={pack_x_scale_ms:.4f}"
      f" gateup={gateup_ms:.4f}"
      f" swiglu_pack={swiglu_pack_ms:.4f}"
      f" down={down_ms:.4f}"
      f" total={total_ms:.4f}",
      flush=True,
    )


if __name__ == "__main__":
  main()
