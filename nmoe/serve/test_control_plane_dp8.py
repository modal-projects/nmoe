# SPDX-License-Identifier: Apache-2.0
"""DP=8 ownership control-plane correctness (rank0 driver, engine-only).

Validates the CPU-only gloo backchannel used by the rank0 OpenAI HTTP server:
  - rank0 assigns a non-zero owner via MSG_REQUEST_INIT
  - owner executes the normal Orchestrator/Scheduler/Engine path locally
  - owner streams MSG_TOKEN_UPDATE back to rank0
  - rank0 can cancel a remote-owned request via MSG_CANCEL
  - owner can return MSG_ERROR for invalid init

Run:
  torchrun --nproc_per_node=8 --master_port=$MASTER_PORT -m nmoe.serve.test_control_plane_dp8
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

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


class _Proxy:
  def __init__(self, *, uid: int, owner: int) -> None:
    self.uid = int(uid)
    self.owner = int(owner)
    self.sent_t = time.perf_counter()
    self.tokens: list[int] = []
    self.done = False
    self.finish_reason = ""
    self.err: Optional[str] = None


def main() -> None:
  _maybe_set_cutlass_path()

  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  if world_size != 8:
    raise RuntimeError(f"test_control_plane_dp8 requires world_size=8 (got {world_size})")
  torch.cuda.set_device(rank)

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
  from nmoe.serve.control_plane import (
    ControlPlane,
    OUTPUT_MODE_ID_LOGITS,
    OUTPUT_MODE_ID_TOKENS,
    RequestInit,
    finish_reason_id_to_str,
  )
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig
  from nmoe.serve.types import ForwardSpec, OutputMode

  ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1")
  cfg = load_model_config(ckpt_path)
  tokenizer = None
  if rank == 0:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

  init_distributed(rank, world_size, tp_size=1)
  ctrl_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  cp = ControlPlane(rank=rank, world_size=world_size, ctrl_group=ctrl_group)

  engine_cfg = EngineConfig(
    num_pages=2048,
    page_size=64,
    num_layers=int(cfg.num_layers),
    kv_lora_rank=int(cfg.kv_lora_rank),
    qk_rope_head_dim=int(cfg.qk_rope_head_dim),
    max_batch_size=64,
    max_seq_len=8192,
    max_step_tokens=4096,
    attention_type=str(cfg.attention_type),
    idx_dim=int(cfg.dsa_idx_dim),
    tp_size=1,
  )
  orch_cfg = OrchestratorConfig(
    max_batch_size=64,
    max_prefill_tokens=4096,
    max_decode_tokens=64,
    max_seq_len=8192,
    num_pages=2048,
    page_size=64,
    enable_overlap=False,
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
    control_plane=cp,
  )

  sharded_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(sharded_file):
    missing, unexpected = load_sharded_checkpoint(orch.engine.model, ckpt_path, rank=rank, world_size=world_size)
  else:
    missing, unexpected = load_checkpoint(orch.engine.model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()
  if rank == 0:
    _log(f"Loaded model from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")

  lock = threading.Lock()
  proxies: Dict[int, _Proxy] = {}
  done_event = threading.Event()

  def _drive_lockstep_until_all(*, pred, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    # CPU tensor for gloo all-reduce (avoid NCCL sync in control path).
    done_local = torch.zeros((1,), device="cpu", dtype=torch.int64)
    while time.time() < deadline:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        raise RuntimeError("Unexpected shutdown observed during test.")
      if not any_decode and not any_prefill:
        time.sleep(0.001)
        continue
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")

      done_local.fill_(1 if pred() else 0)
      group = getattr(orch, "_lockstep_group", None)
      if group is not None:
        dist.all_reduce(done_local, op=dist.ReduceOp.MIN, group=group)
      else:
        dist.all_reduce(done_local, op=dist.ReduceOp.MIN)
      if bool(int(done_local.item())):
        return
      time.sleep(0.001)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for predicate.")

  def _drive_lockstep_until_rank0_done(*, timeout_s: float) -> None:
    """Drive steps until rank0 sets done_event (broadcast via MAX)."""
    deadline = time.time() + float(timeout_s)
    done_t = torch.zeros((1,), device="cpu", dtype=torch.int64)
    group = getattr(orch, "_lockstep_group", None)
    while time.time() < deadline:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        return
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")

      done_t.fill_(1 if (rank == 0 and done_event.is_set()) else 0)
      if group is not None:
        dist.all_reduce(done_t, op=dist.ReduceOp.MAX, group=group)
      else:
        dist.all_reduce(done_t, op=dist.ReduceOp.MAX)
      if bool(int(done_t.item())):
        return
      time.sleep(0.001)
    raise TimeoutError(f"Timed out after {timeout_s}s waiting for rank0 completion.")

  def _maybe_done() -> None:
    # Signal when all tracked proxies are done.
    with lock:
      if proxies and all(p.done for p in proxies.values()):
        done_event.set()

  def _on_token_update(batch) -> None:
    uids = batch.uids.tolist()
    toks = batch.tokens.tolist()
    flags = batch.uflags.tolist()
    with lock:
      for uid, tok, fl in zip(uids, toks, flags, strict=False):
        uid = int(uid)
        p = proxies.get(uid)
        if p is None:
          continue
        done = bool(int(fl) & 0x1)
        rid = int((int(fl) >> 1) & 0x7) if done else 0
        if int(tok) >= 0:
          p.tokens.append(int(tok))
        if done:
          p.done = True
          p.finish_reason = finish_reason_id_to_str(rid)
    _maybe_done()

  def _on_error(uid: int, msg: str) -> None:
    with lock:
      p = proxies.get(int(uid))
      if p is None:
        return
      p.done = True
      p.finish_reason = "error"
      p.err = str(msg)
    _maybe_done()

  # Warm up on all ranks (prevents DeepEP CPU timeout when rank0 has T=0).
  #
  # Use an explicit lockstep driver (no background orchestrator thread). This
  # keeps the test deterministic and avoids relying on multi-threaded dist ops.
  # Warm up with a realistic prompt (broadcast from rank0 over the gloo control
  # group) so all ranks execute an identical prefill+decode path.
  if rank == 0:
    assert tokenizer is not None
    warmup_ids = tokenizer.encode("The capital of France is", return_tensors="pt", add_special_tokens=False)[0]
    warmup_ids = warmup_ids.to(torch.int32).cpu().contiguous()
    warmup_len = torch.tensor([int(warmup_ids.numel())], dtype=torch.int64, device="cpu")
  else:
    warmup_len = torch.zeros((1,), dtype=torch.int64, device="cpu")
  dist.broadcast(warmup_len, src=0, group=ctrl_group)
  if rank != 0:
    warmup_ids = torch.empty((int(warmup_len.item()),), dtype=torch.int32, device="cpu")
  dist.broadcast(warmup_ids, src=0, group=ctrl_group)
  warmup_req = orch.create_request(
    input_ids=warmup_ids,
    profile_name="production_generate",
    temperature=0.0,
    max_tokens=2,
  )
  if not orch.try_add_request(warmup_req, timeout=0.0):
    raise RuntimeError("warmup request rejected (queue full)")
  _drive_lockstep_until_all(pred=lambda: warmup_req.is_finished, timeout_s=900.0)
  dist.barrier()
  if rank == 0:
    _log("Warmup complete.")

  # Start control-plane threads after warmup.
  if rank == 0:
    cp.start_rank0(on_token_update=_on_token_update, on_error=_on_error)
  else:

    def _on_request_init(init: RequestInit) -> None:
      if int(init.uid) % int(world_size) != int(rank):
        cp.enqueue_error(uid=int(init.uid), msg="wrong owner for uid")
        return
      ok, err = orch.validate_request_bounds(int(init.prompt_len), int(init.max_tokens))
      if not ok:
        cp.enqueue_error(uid=int(init.uid), msg=err)
        return
      if int(init.output_mode_id) != OUTPUT_MODE_ID_TOKENS:
        cp.enqueue_error(uid=int(init.uid), msg="unsupported output_mode")
        return
      fs = ForwardSpec(output_mode=OutputMode.TOKENS, topk=int(init.topk))
      seed = None if int(init.seed_or_minus1) < 0 else int(init.seed_or_minus1)
      req = orch.create_request(
        input_ids=init.input_ids,
        profile_name="production_generate",
        uid=int(init.uid),
        forward_spec=fs,
        max_tokens=int(init.max_tokens),
        temperature=float(init.temperature),
        top_p=float(init.top_p),
        top_k=int(init.top_k),
        seed=seed,
      )
      accepted = orch.try_add_request(req, timeout=0.0)
      if not accepted:
        cp.enqueue_error(uid=int(init.uid), msg="owner queue full")

    cp.start_worker(
      on_request_init=_on_request_init,
      on_cancel=lambda uid: orch.cancel(int(uid)),
      on_shutdown=lambda: orch.request_stop(),
    )

  try:
    # Rank0: emit the three test cases.
    uid1 = 1  # owner = 1
    owner1 = uid1 % world_size
    uid2 = 2  # owner = 2
    owner2 = uid2 % world_size
    uid3 = 3  # owner = 3
    owner3 = uid3 % world_size

    if rank == 0:
      # Case 1: rank0 -> non-zero owner (stream tokens until done).
      proxies[uid1] = _Proxy(uid=uid1, owner=owner1)
      # Real prompt to validate semantic correctness under DP ownership + T=0 ranks.
      prompt = "The capital of France is"
      assert tokenizer is not None
      input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)[0].to(torch.int32).cpu()
      cp.send_request_init(
        owner=owner1,
        uid=uid1,
        output_mode_id=OUTPUT_MODE_ID_TOKENS,
        topk=0,
        max_tokens=16,
        top_k=0,
        seed_or_minus1=-1,
        temperature=0.0,
        top_p=1.0,
        input_ids=input_ids,
      )

      # Case 2: cancellation of a remote-owned request.
      proxies[uid2] = _Proxy(uid=uid2, owner=owner2)
      input_ids2 = torch.full((256,), 100, dtype=torch.int32, device="cpu")
      cp.send_request_init(
        owner=owner2,
        uid=uid2,
        output_mode_id=OUTPUT_MODE_ID_TOKENS,
        topk=0,
        max_tokens=128,
        top_k=0,
        seed_or_minus1=-1,
        temperature=0.0,
        top_p=1.0,
        input_ids=input_ids2,
      )

      # Case 3: invalid init triggers MSG_ERROR (unsupported output_mode).
      proxies[uid3] = _Proxy(uid=uid3, owner=owner3)
      input_ids3 = torch.full((32,), 100, dtype=torch.int32, device="cpu")
      cp.send_request_init(
        owner=owner3,
        uid=uid3,
        output_mode_id=OUTPUT_MODE_ID_LOGITS,
        topk=0,
        max_tokens=1,
        top_k=0,
        seed_or_minus1=-1,
        temperature=0.0,
        top_p=1.0,
        input_ids=input_ids3,
      )

    # Drive lockstep steps on ALL ranks until rank0 sees completion.
    cancel_sent = False
    cancel_deadline = time.time() + 30.0
    end_deadline = time.time() + 240.0
    done_t = torch.zeros((1,), device="cpu", dtype=torch.int64)
    group = getattr(orch, "_lockstep_group", None)

    while time.time() < end_deadline:
      orch._recv_requests()
      any_decode, any_prefill, any_shutdown = orch._lockstep_any_work()
      if any_shutdown:
        break
      if any_decode:
        orch._lockstep_run_phase("decode")
      if any_prefill:
        orch._lockstep_run_phase("prefill")

      if rank == 0 and not cancel_sent:
        with lock:
          got_first = uid2 in proxies and len(proxies[uid2].tokens) >= 1
        if got_first or time.time() >= cancel_deadline:
          cp.send_cancel(owner=owner2, uid=uid2)
          cancel_sent = True

      done_t.fill_(1 if (rank == 0 and done_event.is_set()) else 0)
      if group is not None:
        dist.all_reduce(done_t, op=dist.ReduceOp.MAX, group=group)
      else:
        dist.all_reduce(done_t, op=dist.ReduceOp.MAX)
      if bool(int(done_t.item())):
        break
      time.sleep(0.001)

    if not bool(int(done_t.item())):
      raise TimeoutError("Timed out waiting for control-plane requests to finish.")

    if rank == 0:
      with lock:
        p1 = proxies[uid1]
        p2 = proxies[uid2]
        p3 = proxies[uid3]

      if not p1.tokens:
        raise AssertionError("uid1 produced no tokens")
      if p1.finish_reason not in ("stop", "length"):
        raise AssertionError(f"uid1 unexpected finish_reason={p1.finish_reason!r}")
      assert tokenizer is not None
      out = tokenizer.decode(p1.tokens, skip_special_tokens=True)
      if "Paris" not in out:
        raise AssertionError(f"uid1 expected Paris, got: {out!r} (tokens={p1.tokens})")

      if p2.finish_reason != "cancelled":
        raise AssertionError(f"uid2 expected cancelled, got {p2.finish_reason!r}")

      if p3.finish_reason != "error" or not p3.err:
        raise AssertionError("uid3 expected MSG_ERROR")

      _log("PASS")

  finally:
    orch.shutdown()
    cp.close(timeout_s=30.0)
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
