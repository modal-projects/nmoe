# SPDX-License-Identifier: Apache-2.0
"""Compare DeepEP MoE output vs local_mask+all_reduce on a single node.

Acceptance target: on 8 GPUs, DeepEP output matches the single-node local_mask
reference within BF16 tolerance.

Run with:
  torchrun --nproc_per_node=8 -m nmoe.serve.test_moe_deepep_vs_localmask
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


def main() -> None:
  _maybe_set_cutlass_path()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  torch.cuda.set_device(rank)
  device = torch.device(f"cuda:{rank}")

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from deep_ep import Buffer
  from nmoe.serve.ckpt import load_model_config
  from nmoe.serve.model import MoE, init_distributed

  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1")
  cfg = load_model_config(ckpt_path)

  # Use TP=world_size to enable the local_mask path. This is a test-only setup:
  # all ranks run the same token batch, so local_mask+all_reduce is valid.
  init_distributed(rank, world_size, tp_size=world_size)

  hidden = int(cfg.hidden_size)
  num_nvl_bytes = max(
    Buffer.get_dispatch_config(world_size).get_nvl_buffer_size_hint(hidden * 2, world_size),
    Buffer.get_combine_config(world_size).get_nvl_buffer_size_hint(hidden * 2, world_size),
  )
  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=int(num_nvl_bytes), num_rdma_bytes=0, explicitly_destroy=True)

  moe_local = MoE(cfg, buffer, single_node=True).to(device).eval()
  moe_deep = MoE(cfg, buffer, single_node=False).to(device).eval()
  moe_deep.load_state_dict(moe_local.state_dict(), strict=True)

  # Identical input across ranks (local_mask reference requires this).
  T = int(os.environ.get("NMOE_TEST_TOKENS", "256"))
  torch.manual_seed(12345)
  x = torch.randn(T, hidden, device=device, dtype=torch.bfloat16)
  dist.broadcast(x, src=0)

  with torch.inference_mode():
    y_local = moe_local(x, low_latency=False)  # includes all_reduce at end
    dist.barrier()

    y_deep = moe_deep(x, low_latency=False)  # DeepEP combine gives full expert output

    # Correct for shared experts when tp_size>1:
    # - shared MLP is intentionally reduce_output=False (sharded output),
    # - local_mask path all-reduces at the end, DeepEP path does not.
    # Convert DeepEP output to the same "full shared" semantics.
    x2 = x.view(-1, hidden)
    shared_shard = moe_deep.shared(x2)
    shared_full = shared_shard.clone()
    dist.all_reduce(shared_full)
    y_deep_full = y_deep + (shared_full - shared_shard)

  diff = (y_local.float() - y_deep_full.float()).abs()
  max_abs = float(diff.max().item()) if diff.numel() else 0.0
  mean_abs = float(diff.mean().item()) if diff.numel() else 0.0

  local_fail = 0
  if rank == 0:
    _log("=" * 70)
    _log("DeepEP vs local_mask MoE comparison")
    _log("=" * 70)
    _log(f"T={T}, hidden={hidden}, experts={int(cfg.num_experts)}, topk={int(cfg.num_experts_per_tok)}")
    _log(f"max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")

    tol = float(os.environ.get("NMOE_TEST_TOL", "0.1"))
    if max_abs > tol:
      _log(f"FAIL: max_abs_diff={max_abs:.6f} > tol={tol}")
      local_fail = 1
    else:
      _log("PASS")

  fail = torch.tensor([int(local_fail)], device=device, dtype=torch.int32)
  dist.all_reduce(fail, op=dist.ReduceOp.MAX)
  exit_code = int(fail.item())
  dist.barrier()
  buffer.destroy()
  dist.destroy_process_group()
  if exit_code != 0:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
