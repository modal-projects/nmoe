# SPDX-License-Identifier: Apache-2.0
"""MoE DeepEP correctness when some ranks have T=0 local tokens.

This isolates the core dynamic-disagg requirement for experts-as-a-service:
rank0-owned tokens must get the same MoE output regardless of whether other
ranks also have local tokens, as long as all ranks participate in dispatch/
combine collectives.

Run:
  torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_moe_t0_consistency
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return


def _log(msg: str) -> None:
  if dist.get_rank() == 0:
    print(msg, flush=True)


def _init_random_moe(moe) -> None:
  """Deterministically initialize MoE weights without touching checkpoint I/O."""
  rank = dist.get_rank()
  g = torch.Generator(device="cuda")
  g.manual_seed(1234 + int(rank))

  with torch.no_grad():
    # Gate (replicated): seed must be the same across ranks.
    if rank == 0:
      gw = 0.01 * torch.randn(
        moe.gate.weight.shape,
        device=moe.gate.weight.device,
        dtype=moe.gate.weight.dtype,
        generator=g,
      )
      dist.broadcast(gw, src=0)
      moe.gate.weight.copy_(gw)
      if moe.gate.bias is not None:
        gb = 0.01 * torch.randn(
          moe.gate.bias.shape,
          device=moe.gate.bias.device,
          dtype=moe.gate.bias.dtype,
          generator=g,
        )
        dist.broadcast(gb, src=0)
        moe.gate.bias.copy_(gb)
    else:
      gw = torch.empty_like(moe.gate.weight)
      dist.broadcast(gw, src=0)
      moe.gate.weight.copy_(gw)
      if moe.gate.bias is not None:
        gb = torch.empty_like(moe.gate.bias)
        dist.broadcast(gb, src=0)
        moe.gate.bias.copy_(gb)

    # Experts (sharded): per-rank random is fine (matches real EP layout).
    tmp13 = 0.01 * torch.randn(moe.w13.shape, device=moe.w13.device, dtype=torch.float16, generator=g)
    tmp2 = 0.01 * torch.randn(moe.w2.shape, device=moe.w2.device, dtype=torch.float16, generator=g)
    moe.w13.copy_(tmp13.to(torch.float8_e4m3fn))
    moe.w2.copy_(tmp2.to(torch.float8_e4m3fn))
    moe.w13_scale.fill_(1.0)
    moe.w2_scale.fill_(1.0)

    # Shared expert MLP (replicated): same across ranks.
    for p in moe.shared.parameters():
      if p.dtype.is_floating_point:
        gen_dtype = p.dtype
        if str(p.dtype).startswith("torch.float8"):
          gen_dtype = torch.float16
        if rank == 0:
          t = 0.01 * torch.randn(p.shape, device=p.device, dtype=gen_dtype, generator=g)
          dist.broadcast(t, src=0)
          p.copy_(t.to(p.dtype))
        else:
          t = torch.empty(p.shape, device=p.device, dtype=gen_dtype)
          dist.broadcast(t, src=0)
          p.copy_(t.to(p.dtype))


def main() -> None:
  _maybe_set_cutlass_path()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size != 8:
    raise RuntimeError(f"test_moe_t0_consistency requires world_size=8 (got {world_size})")
  torch.cuda.set_device(rank)
  device = torch.device(f"cuda:{rank}")

  from deep_ep import Buffer
  from nmoe.serve.model import ModelConfig, MoE, init_distributed

  init_distributed(rank, world_size, tp_size=1)
  cfg = ModelConfig(attention_type="mla")

  hidden = int(cfg.hidden_size)
  num_nvl_bytes = max(
    Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(hidden * 2, world_size),
    Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(hidden * 2, world_size),
  )
  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=int(num_nvl_bytes), num_rdma_bytes=0)

  moe = MoE(cfg, buffer, single_node=False).to(device).eval()
  _init_random_moe(moe)
  dist.barrier()

  # Deterministic input tokens.
  T = 32
  if rank == 0:
    torch.manual_seed(2026)
    x0 = torch.randn(T, hidden, device=device, dtype=torch.bfloat16)
  else:
    x0 = torch.empty(T, hidden, device=device, dtype=torch.bfloat16)
  dist.broadcast(x0, src=0)

  with torch.inference_mode():
    # Scenario A: all ranks have tokens.
    y_all = moe(x0, low_latency=False)
    dist.barrier()

    # Scenario B: rank0 has tokens, other ranks have T=0.
    x_t0 = x0 if rank == 0 else torch.empty((0, hidden), device=device, dtype=torch.bfloat16)
    y_t0 = moe(x_t0, low_latency=False)
    dist.barrier()

  if rank == 0:
    if not torch.isfinite(y_all).all() or not torch.isfinite(y_t0).all():
      raise SystemExit("FAIL: MoE produced non-finite outputs (NaN/Inf).")
    diff = (y_all.float() - y_t0.float()).abs()
    if not torch.isfinite(diff).all():
      raise SystemExit("FAIL: diff contains NaN/Inf.")
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    _log("=" * 70)
    _log("MoE T=0 consistency (rank0 output)")
    _log("=" * 70)
    _log(f"T={T}, hidden={hidden}, experts={int(cfg.num_experts)}, topk={int(cfg.num_experts_per_tok)}")
    _log(f"max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")
    tol = float(os.environ.get("NMOE_TEST_TOL", "0.1"))
    if max_abs > tol:
      raise SystemExit(f"FAIL: max_abs_diff={max_abs:.6f} > tol={tol}")
    _log("PASS")

  dist.destroy_process_group()


if __name__ == "__main__":
  main()
