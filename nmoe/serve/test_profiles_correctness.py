# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness for the 5 serve profiles (single engine path).

Run with:
  torchrun --nproc_per_node=8 -m nmoe.serve.test_profiles_correctness

This test is rank0-driven:
- Rank 0 submits requests.
- Other ranks participate in every step with T=0 (MoE-only) via the lockstep
  coordinator, validating dynamic disaggregation correctness for serving.
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


@dataclass(frozen=True)
class Case:
  name: str
  profile: str
  prompt: str
  max_tokens: int
  prompt_len: Optional[int] = None
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = 0
  seed: Optional[int] = None
  expect_contains: Optional[str] = None


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


def _validate_request(req, *, tokenizer) -> None:
  from nmoe.serve.types import OutputMode

  if req.finish_reason is None:
    raise AssertionError("finish_reason must be set")

  if not req.output_ids:
    raise AssertionError("output_ids must be non-empty")

  mode = req.forward_spec.output_mode
  if mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS):
    if len(req.output_logprobs) != len(req.output_ids):
      raise AssertionError("output_logprobs must align with output_ids")
    if not torch.isfinite(torch.tensor(req.output_logprobs)).all():
      raise AssertionError("output_logprobs must be finite")

  if mode == OutputMode.TOPK_LOGPROBS:
    if len(req.output_topk) != len(req.output_ids):
      raise AssertionError("output_topk must align with output_ids")
    K = int(req.forward_spec.topk)
    for step, row in enumerate(req.output_topk):
      if len(row) != K:
        raise AssertionError(f"output_topk[{step}] expected K={K}, got {len(row)}")

  if mode == OutputMode.LOGITS:
    if req.output_logits is None:
      raise AssertionError("output_logits must be populated in LOGITS mode")
    if req.output_logits.ndim != 2:
      raise AssertionError("output_logits must be [T, vocab]")
    if req.output_logits.size(0) != len(req.output_ids):
      raise AssertionError("output_logits rows must align with output_ids")

    # Greedy sanity: token should be argmax of logits for temperature==0.
    if float(req.sampling_params.temperature) <= 0:
      pred = torch.argmax(req.output_logits, dim=-1).to(torch.int64).tolist()
      if pred != [int(t) for t in req.output_ids]:
        raise AssertionError("LOGITS mode greedy tokens must match argmax(logits)")

  if tokenizer is not None:
    text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
    if getattr(req, "_nmoe_expect_contains", None):
      needle = str(req._nmoe_expect_contains)
      if needle not in text:
        raise AssertionError(
          f"expected output to contain {needle!r}, got: {text!r} "
          f"(finish_reason={req.finish_reason!r}, output_ids={req.output_ids})"
        )


def main() -> None:
  _maybe_set_cutlass_path()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  mode = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
  if mode != "rdep":
    raise RuntimeError(f"test_profiles_correctness requires NMOE_EP_TRANSPORT=rdep (got {mode!r})")
  if world_size != 8:
    raise RuntimeError(f"test_profiles_correctness requires world_size=8 (got {world_size})")
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
    # Keep KV cache modest for correctness tests; DeepSeek-V3 weights are large
    # and the test only needs short contexts (max_seq_len=8192).
    num_pages=512,
    page_size=64,
    num_layers=int(cfg.num_layers),
    kv_lora_rank=int(cfg.kv_lora_rank),
    qk_rope_head_dim=int(cfg.qk_rope_head_dim),
    max_batch_size=64,
    max_seq_len=8192,
    # Keep the inference-RDEP prefill transport capacity modest for tests to
    # avoid OOM during model materialization. The test workload uses at most a
    # single request per rank and chunk_size=512.
    max_step_tokens=1024,
    attention_type=str(cfg.attention_type),
    idx_dim=int(cfg.dsa_idx_dim),
    tp_size=1,  # dynamic disagg mode
  )
  orch_cfg = OrchestratorConfig(
    max_batch_size=64,
    max_prefill_tokens=1024,
    max_seq_len=8192,
    num_pages=512,
    page_size=64,
    enable_overlap=False,  # lockstep multi-rank path
    enable_chunked_prefill=True,
    chunk_size=512,
    enable_prefix_cache=True,
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

  # Warm up on *all ranks* before the rank0-driven dynamic-disagg cases.
  #
  # Why this is required:
  # - The main loop intentionally runs with T=0 on ranks!=0 (MoE-only), which can
  #   reach MoE dispatch/barriers much earlier than rank0.
  # - First-use compilation/autotune (attention, MoE kernels, etc.) can stall
  #   rank0 long enough to trip DeepEP CPU timeouts or RDEP barrier timeouts.
  #
  # Warm up by running a short teacher-forcing request on every rank. Use two
  # prompt lengths: a small one (decode-ish) and a chunked-prefill one to cover
  # the test's paged-prefill path.
  warmup_prompt_lens_env = os.environ.get("NMOE_WARMUP_PROMPT_LENS", "16,1024")
  warmup_prompt_lens = [int(x) for x in warmup_prompt_lens_env.split(",") if x.strip()]
  warmup_max_tokens = int(os.environ.get("NMOE_WARMUP_MAX_TOKENS", "8"))
  warmup_timeout_s = float(os.environ.get("NMOE_WARMUP_TIMEOUT_S", "1200"))

  all_done_t = torch.zeros((1,), device="cpu", dtype=torch.int64)
  group = getattr(orch, "_lockstep_group", None)
  for prompt_len in warmup_prompt_lens:
    warmup_ids = torch.full((int(prompt_len),), 100, dtype=torch.int32, device="cpu")
    warmup_req = orch.create_forced_request(
      input_ids=warmup_ids,
      forced_output_ids=[100] * warmup_max_tokens,  # deterministic; avoids early EOS
      profile_name="production_generate",
      temperature=0.0,
    )
    orch.add_request(warmup_req)

    deadline = time.time() + warmup_timeout_s
    while time.time() < deadline:
      orch._recv_requests()
      any_decode, any_prefill, _any_shutdown = orch._lockstep_any_work()
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")

      all_done_t.fill_(1 if bool(getattr(warmup_req, "is_finished", False)) else 0)
      if group is not None:
        dist.all_reduce(all_done_t, op=dist.ReduceOp.MIN, group=group)
      else:
        dist.all_reduce(all_done_t, op=dist.ReduceOp.MIN)
      if int(all_done_t.item()) == 1:
        break

    if int(all_done_t.item()) != 1:
      raise TimeoutError(f"warmup timed out (prompt_len={prompt_len}, timeout_s={warmup_timeout_s})")
    dist.barrier()

  tokenizer = None
  if rank == 0:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

  cases: list[Case] = [
    Case(
      name="production_generate_paris",
      profile="production_generate",
      prompt="The capital of France is",
      max_tokens=16,
      temperature=0.0,
      expect_contains="Paris",
    ),
    Case(
      name="production_generate_paged_prefill_smoke",
      profile="production_generate",
      prompt="",
      prompt_len=1024,  # > chunk_size so we exercise paged prefill on later chunks
      max_tokens=2,
      temperature=0.0,
    ),
    Case(
      name="online_distill_topk",
      profile="online_distill",
      prompt="2 + 2 =",
      max_tokens=4,
      temperature=0.0,
    ),
    Case(
      name="rl_sample_seeded",
      profile="rl_sample",
      prompt="def fibonacci(n):",
      max_tokens=8,
      temperature=1.0,
      top_p=1.0,
      seed=1234,
    ),
    # Fixed-batching profiles: keep max_tokens small to avoid long decode loops.
    Case(
      name="eval_exact_logits",
      profile="eval_exact",
      prompt="Hello, my name is",
      max_tokens=1,
      temperature=0.0,
    ),
    Case(
      name="offline_distill_logits",
      profile="offline_distill",
      prompt="The meaning of life is",
      max_tokens=1,
      temperature=0.0,
    ),
  ]

  idx = 0
  active: list[object] = []
  failures: list[str] = []
  det_ref: Optional[tuple[list[int], list[float]]] = None
  det_pending: bool = False
  det_input_ids: Optional[torch.Tensor] = None
  det_sampling: Optional[tuple[float, float, int, int]] = None  # (temperature, top_p, top_k, max_tokens)
  det_seed: Optional[int] = None
  # RL determinism contract:
  # - tokens must match exactly for a given seed
  # - logprobs are approximate under FP8/atomics; require tolerance only
  # NOTE: 0.2 is an empirical bound for current FP8+MoE atomics. Tighten only
  # when we have deterministic reductions or an equivalent guarantee.
  det_logprob_tol = float(os.environ.get("NMOE_RL_LOGPROB_TOL", "0.2"))
  # Use CPU tensors for lockstep termination signaling (gloo).
  done_local = torch.zeros((1,), device="cpu", dtype=torch.int64)

  def _active_done() -> bool:
    return all(getattr(r, "is_finished", False) for r in active)

  exit_code = 0
  try:
    # Main lockstep loop (single collective stream; no background threads).
    while True:
      # Rank0: enqueue next case when idle.
      if rank == 0 and not active and idx < len(cases):
        case = cases[idx]
        idx += 1
        _log(f"\n=== Case {idx}/{len(cases)}: {case.name} ({case.profile}) ===")

        if case.prompt_len is not None:
          input_ids = torch.randint(0, 10000, (int(case.prompt_len),), dtype=torch.int64, device="cpu")
        else:
          input_ids = tokenizer.encode(case.prompt, return_tensors="pt", add_special_tokens=False)[0].cpu()
        req = orch.create_request(
          input_ids=input_ids,
          profile_name=case.profile,
          temperature=case.temperature,
          top_p=case.top_p,
          top_k=case.top_k,
          max_tokens=case.max_tokens,
          seed=case.seed,
        )
        # Attach expectation for validation without widening Request surface.
        req._nmoe_expect_contains = case.expect_contains
        orch.add_request(req)
        active = [req]

      # Rank0: validate and clear when done.
      if rank == 0 and active and _active_done():
        req = active[0]
        try:
          _validate_request(req, tokenizer=tokenizer)
          _log(f"[PASS] {req.profile_name} finish_reason={req.finish_reason} tokens={len(req.output_ids)}")
        except Exception as e:
          failures.append(f"{req.profile_name}: {e}")
          _log(f"[FAIL] {req.profile_name}: {e}")
        active = []

        # Determinism check for rl_sample: same seed should reproduce tokens/logprobs.
        if req.profile_name == "rl_sample" and not failures:
          if not det_pending and det_ref is None:
            det_ref = (req.output_ids[:], req.output_logprobs[:])
            det_pending = True
            det_input_ids = req.input_ids.clone()
            det_seed = req.sampling_params.seed
            det_sampling = (
              float(req.sampling_params.temperature),
              float(req.sampling_params.top_p),
              int(req.sampling_params.top_k),
              int(req.sampling_params.max_tokens),
            )
            if det_seed is None:
              failures.append("rl_sample_determinism: seed must be set for determinism test")
              det_pending = False
              det_ref = None
              continue

            input_ids = det_input_ids
            req2 = orch.create_request(
              input_ids=input_ids,
              profile_name="rl_sample",
              temperature=det_sampling[0],
              top_p=det_sampling[1],
              top_k=det_sampling[2],
              max_tokens=det_sampling[3],
              seed=det_seed,
            )
            orch.add_request(req2)
            active = [req2]
          elif det_pending and det_ref is not None:
            # Second run finished; compare.
            ref_ids, ref_lp = det_ref
            if req.output_ids != ref_ids:
              failures.append("rl_sample_determinism: output_ids mismatch across identical seeds")
            if len(req.output_logprobs) != len(ref_lp):
              failures.append("rl_sample_determinism: output_logprobs length mismatch across identical seeds")
            else:
              lp0 = torch.tensor(ref_lp, dtype=torch.float32)
              lp1 = torch.tensor(req.output_logprobs, dtype=torch.float32)
              max_abs = float((lp0 - lp1).abs().max().item()) if lp0.numel() else 0.0
              if max_abs > det_logprob_tol:
                failures.append(
                  f"rl_sample_determinism: output_logprobs max_abs={max_abs:.2e} (> {det_logprob_tol:.3g})"
                )
            det_pending = False

      # Global done broadcast via all-reduce (no background-thread collectives).
      done_local.fill_(1 if (rank == 0 and idx >= len(cases) and not active) else 0)
      group = getattr(orch, "_lockstep_group", None)
      if group is not None:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX, group=group)
      else:
        dist.all_reduce(done_local, op=dist.ReduceOp.MAX)
      done = bool(int(done_local.item()))

      # Feed rank0-submitted requests into the local scheduler on all ranks.
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
      _log("\n=== Summary ===")
      if failures:
        for f in failures:
          _log(f"FAIL: {f}")
      else:
        _log("All profile correctness checks PASSED")
  finally:
    orch.shutdown()
    dist.destroy_process_group()

  if exit_code:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
