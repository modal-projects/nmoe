# SPDX-License-Identifier: Apache-2.0
"""Teacher-forcing correctness for serve LOGPROBS policy.

Validates that forcing a known token stream yields the same per-token logprobs
as the normal sampling path under the same sampling parameters.

Run with:
  torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_teacher_forcing_logprobs

This test is rank0-driven (serving semantics):
- Rank 0 submits requests.
- Other ranks participate in every step with T=0 via lockstep.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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


def _wait_for(predicate: Callable[[], bool], *, timeout_s: float) -> None:
  deadline = time.time() + float(timeout_s)
  while time.time() < deadline:
    if predicate():
      return
    time.sleep(0.001)
  raise TimeoutError(f"Timed out after {timeout_s}s")


@dataclass(frozen=True)
class _RunCfg:
  prompt: str
  max_tokens: int
  temperature: float
  top_p: float
  top_k: int
  seed: int


def main() -> None:
  _maybe_set_cutlass_path()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size != 8:
    raise RuntimeError(f"test_teacher_forcing_logprobs requires world_size=8 (got {world_size})")
  torch.cuda.set_device(rank)

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1")
  cfg = load_model_config(ckpt_path)

  # Hard launch requirement: TP=1, EP=world_size.
  init_distributed(rank, world_size, tp_size=1)

  engine_cfg = EngineConfig(
    num_pages=2048,
    page_size=64,
    num_layers=int(cfg.num_layers),
    kv_lora_rank=int(cfg.kv_lora_rank),
    qk_rope_head_dim=int(cfg.qk_rope_head_dim),
    max_batch_size=32,
    max_seq_len=4096,
    max_step_tokens=2048,
    attention_type=str(cfg.attention_type),
    idx_dim=int(cfg.dsa_idx_dim),
    tp_size=1,
  )
  orch_cfg = OrchestratorConfig(
    max_batch_size=32,
    max_prefill_tokens=2048,
    max_seq_len=4096,
    num_pages=2048,
    page_size=64,
    enable_overlap=False,
    enable_chunked_prefill=True,
    chunk_size=512,
    enable_prefix_cache=False,  # Keep execution path identical between sample vs forced run.
    enable_fast_path=True,
  )

  orch = Orchestrator(
    model_config=cfg,
    engine_config=engine_cfg,
    orch_config=orch_cfg,
    rank=rank,
    world_size=world_size,
  )

  # Load weights on all ranks.
  sharded_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(sharded_file):
    missing, unexpected = load_sharded_checkpoint(orch.engine.model, ckpt_path, rank=rank, world_size=world_size)
  else:
    missing, unexpected = load_checkpoint(orch.engine.model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()
  if rank == 0:
    _log(f"Loaded model from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")

  tokenizer = None
  if rank == 0:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

  run = _RunCfg(
    prompt="def fibonacci(n):",
    max_tokens=12,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    seed=1337,
  )

  active = []
  phase: str = "sample"  # then "force", then "done"
  ref_ids: Optional[list[int]] = None
  ref_lp: Optional[list[float]] = None
  failures: list[str] = []
  # Note: We compare sampled-run logprobs vs forced-run logprobs across two
  # separate executions. With FP8 + atomic accumulation in MoE, logits can vary
  # slightly run-to-run even when tokens are identical, so allow some tolerance.
  tol = float(os.environ.get("NMOE_TF_LOGPROB_TOL", "0.25"))

  # CPU tensor for gloo all-reduce (avoid NCCL sync in control path).
  done_local = torch.zeros((1,), device="cpu", dtype=torch.int64)
  exit_code = 0
  try:
    while True:
      if rank == 0 and not active and phase == "sample":
        assert tokenizer is not None
        input_ids = tokenizer.encode(run.prompt, return_tensors="pt", add_special_tokens=False)[0].cpu()
        req = orch.create_request(
          input_ids=input_ids,
          profile_name="rl_sample",
          temperature=run.temperature,
          top_p=run.top_p,
          top_k=run.top_k,
          max_tokens=run.max_tokens,
          seed=run.seed,
        )
        orch.add_request(req)
        active = [req]

      if rank == 0 and not active and phase == "force":
        assert tokenizer is not None
        assert ref_ids is not None and ref_lp is not None
        input_ids = tokenizer.encode(run.prompt, return_tensors="pt", add_special_tokens=False)[0].cpu()
        req = orch.create_forced_request(
          input_ids=input_ids,
          forced_output_ids=ref_ids,
          profile_name="rl_sample",
          temperature=run.temperature,
          top_p=run.top_p,
          top_k=run.top_k,
          max_tokens=len(ref_ids),
          seed=run.seed,
        )
        orch.add_request(req)
        active = [req]

      if rank == 0 and active and active[0].is_finished:
        req = active[0]
        if phase == "sample":
          ref_ids = req.output_ids[:]
          ref_lp = req.output_logprobs[:]
          if not ref_ids:
            failures.append("sample: produced empty output_ids")
          if len(ref_lp) != len(ref_ids):
            failures.append("sample: output_logprobs must align with output_ids")
          phase = "force"
          active = []
          continue

        # Forced phase complete; compare to reference.
        assert ref_ids is not None and ref_lp is not None
        if req.output_ids != ref_ids:
          failures.append("force: output_ids mismatch (teacher-forcing did not reproduce tokens)")
        if len(req.output_logprobs) != len(ref_lp):
          failures.append("force: output_logprobs length mismatch")
        else:
          lp0 = torch.tensor(ref_lp, dtype=torch.float32)
          lp1 = torch.tensor(req.output_logprobs, dtype=torch.float32)
          max_abs = float((lp0 - lp1).abs().max().item()) if lp0.numel() else 0.0
          if max_abs > tol:
            failures.append(f"force: output_logprobs max_abs={max_abs:.3e} (> {tol:g})")
        active = []
        phase = "done"

      done_local.fill_(1 if (rank == 0 and not active and phase == "done") else 0)
      group = getattr(orch, "_lockstep_group", None)
      if group is not None:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX, group=group)
      else:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX)
      done = bool(int(done_local.item()))

      orch._recv_requests()

      any_decode, any_prefill, _any_shutdown = orch._lockstep_any_work()
      if not any_decode and not any_prefill:
        if done:
          break
        time.sleep(0.001)
        continue
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")

    dist.barrier()
    exit_t = torch.zeros((1,), device="cpu", dtype=torch.int64)
    exit_t.fill_(1 if (rank == 0 and bool(failures)) else 0)
    group = getattr(orch, "_lockstep_group", None)
    if group is not None:
      dist.all_reduce(exit_t, op=dist.ReduceOp.MAX, group=group)
    else:
      dist.all_reduce(exit_t, op=dist.ReduceOp.MAX)
    exit_code = int(exit_t.item())

    if rank == 0:
      if failures:
        for f in failures:
          _log(f"[FAIL] {f}")
      else:
        _log("[PASS] teacher-forcing logprobs match sampling path")
  finally:
    orch.shutdown()
    dist.destroy_process_group()

  if exit_code:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
