# SPDX-License-Identifier: Apache-2.0
"""Correctness parity vs DeepSeek reference implementation.

This is a *semantic* test: it fails if our logits diverge from the reference,
even if everything is finite and shapes look right.

Requirements:
- Run on SM100 (B200) with 8 GPUs (mp8 reference checkpoint).
- Run under torchrun:
    torchrun --nproc_per_node=8 -m nmoe.serve.test_correctness_reference

Configuration (required via env):
- NMOE_MODEL_PATH: reference HF checkpoint dir (e.g. /data/models/DeepSeek-V3.2-Speciale)
- NMOE_OUR_MODEL_PATH: nmoe EP8-TP1 dir (e.g. /data/models/DeepSeek-V3.2-Speciale-ep8-tp1)
- NMOE_REFERENCE_DIR: reference python dir (e.g. $NMOE_MODEL_PATH/inference)
- NMOE_REFERENCE_CONFIG: reference json (e.g. $NMOE_REFERENCE_DIR/config_671B_v3.2.json)
- NMOE_REFERENCE_MP_DIR: mp8 shard dir (e.g. /data/models/DeepSeek-V3.2-Speciale-mp8)

Optional:
- NMOE_REFERENCE_MAX_INPUTS: limit number of test inputs (default: 2)
- NMOE_REFERENCE_PHASE: auto|reference|compare (default: auto)
- NMOE_REFERENCE_LOGITS_PATH: where rank0 saves reference logits (default: /tmp/nmoe_reference_logits.pt)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import unittest
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _Paths:
  reference_model_path: str
  our_model_path: str
  reference_dir: str
  reference_config: str
  reference_mp_dir: str


def _require_env(name: str) -> str:
  v = os.environ.get(name, "").strip()
  if not v:
    raise unittest.SkipTest(f"{name} is required for this test (heavy correctness suite).")
  return v


def _paths() -> _Paths:
  reference_model_path = _require_env("NMOE_MODEL_PATH")
  our_model_path = _require_env("NMOE_OUR_MODEL_PATH")
  reference_dir = os.environ.get("NMOE_REFERENCE_DIR", f"{reference_model_path}/inference").strip()
  reference_config = os.environ.get("NMOE_REFERENCE_CONFIG", f"{reference_dir}/config_671B_v3.2.json").strip()
  reference_mp_dir = _require_env("NMOE_REFERENCE_MP_DIR")
  return _Paths(
    reference_model_path=reference_model_path,
    our_model_path=our_model_path,
    reference_dir=reference_dir,
    reference_config=reference_config,
    reference_mp_dir=reference_mp_dir,
  )


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _minor = torch.cuda.get_device_capability()
  return major == 10


def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  cand = "/workspace/nmoe/third_party/DeepGEMM/third-party/cutlass"
  if os.path.isdir(cand):
    os.environ["CUTLASS_PATH"] = cand


def _init_dist() -> tuple[int, int]:
  # torchrun exports LOCAL_RANK. Bind device *before* NCCL init to avoid device
  # guessing that can hang collectives in shared debug containers.
  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  torch.cuda.set_device(local_rank)
  if not dist.is_initialized():
    # torchrun sets MASTER_ADDR/PORT and env://.
    dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  return rank, world_size


def _alloc_our_kv_caches(*, num_layers: int, num_blocks: int, idx_dim: int, device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
  kv_caches = [torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(num_layers)]
  idx_k_caches = [torch.zeros(num_blocks, 64, idx_dim, dtype=torch.bfloat16, device=device) for _ in range(num_layers)]
  return kv_caches, idx_k_caches


def _run_reference_logits_last(tokens: torch.Tensor, *, paths: _Paths, rank: int, world_size: int) -> torch.Tensor:
  _maybe_set_cutlass_path()
  # Import reference model from the checkpoint.
  ref_dir = Path(paths.reference_dir)
  if not ref_dir.is_dir():
    raise unittest.SkipTest(f"NMOE_REFERENCE_DIR not found: {ref_dir}")

  # Patch the reference's `from kernel import ...` to use our torch implementation
  # (TileLang/TVM cannot currently JIT for SM100a in this container).
  from nmoe.serve import ref_kernel_torch as torch_kernel
  prev_kernel = sys.modules.get("kernel")
  sys.modules["kernel"] = torch_kernel

  sys.path.insert(0, str(ref_dir))
  try:
    from model import Transformer, ModelArgs  # type: ignore
  except Exception as e:
    if prev_kernel is None:
      sys.modules.pop("kernel", None)
    else:
      sys.modules["kernel"] = prev_kernel
    raise RuntimeError(f"Failed to import reference model from {ref_dir}: {e}") from e

  cfg_path = Path(paths.reference_config)
  if not cfg_path.is_file():
    sys.path.remove(str(ref_dir))
    if prev_kernel is None:
      sys.modules.pop("kernel", None)
    else:
      sys.modules["kernel"] = prev_kernel
    raise unittest.SkipTest(f"NMOE_REFERENCE_CONFIG not found: {cfg_path}")
  try:
    with cfg_path.open() as f:
      args_dict = json.load(f)
  except Exception:
    sys.path.remove(str(ref_dir))
    if prev_kernel is None:
      sys.modules.pop("kernel", None)
    else:
      sys.modules["kernel"] = prev_kernel
    raise

  torch.set_default_dtype(torch.bfloat16)
  model = Transformer(ModelArgs(**args_dict)).cuda().eval()

  ckpt_file = Path(paths.reference_mp_dir) / f"model{rank}-mp{world_size}.safetensors"
  if not ckpt_file.is_file():
    raise unittest.SkipTest(f"Reference shard not found: {ckpt_file}")

  from safetensors.torch import load_model
  load_model(model, str(ckpt_file))

  with torch.inference_mode():
    logits = model.forward(tokens, start_pos=0)  # [B, V] last token only

  # Cleanup.
  out = logits.float().cpu()
  del logits
  del model
  gc.collect()
  torch.cuda.empty_cache()
  sys.path.remove(str(ref_dir))
  if prev_kernel is None:
    sys.modules.pop("kernel", None)
  else:
    sys.modules["kernel"] = prev_kernel
  return out


def _run_our_logits_last(tokens: torch.Tensor, *, paths: _Paths, rank: int, world_size: int) -> torch.Tensor:
  _maybe_set_cutlass_path()
  mode = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
  if mode != "rdep":
    raise RuntimeError(f"test_correctness_reference requires NMOE_EP_TRANSPORT=rdep (got {mode!r})")

  # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
  # barrier/collective timeouts.
  master_port = os.environ.get("MASTER_PORT", "0")
  os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
  os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

  from nmoe.serve.ckpt import load_checkpoint, load_model_config
  from nmoe.serve.engine import Engine, EngineConfig

  device = torch.device(f"cuda:{rank}")
  cfg = load_model_config(paths.our_model_path)
  engine_cfg = EngineConfig(
    # Keep all allocations minimal: this test only runs short prefill (S~8).
    num_pages=8,
    page_size=64,
    num_layers=int(cfg.num_layers),
    kv_lora_rank=int(getattr(cfg, "kv_lora_rank", 0)),
    qk_rope_head_dim=int(getattr(cfg, "qk_rope_head_dim", 0)),
    max_batch_size=32,
    max_seq_len=512,
    max_step_tokens=256,
    attention_type=str(getattr(cfg, "attention_type", "dsa")),
    idx_dim=int(getattr(cfg, "dsa_idx_dim", 128)),
    tp_size=1,
  )
  engine = Engine(cfg, engine_cfg, rank=rank, world_size=world_size)

  missing, unexpected = load_checkpoint(engine.model, paths.our_model_path, rank=rank, world_size=world_size, cfg=cfg)
  if missing:
    raise RuntimeError(f"load_checkpoint missing keys: {sorted(missing)[:10]}")

  B, S = tokens.shape
  positions = torch.arange(S, device=device, dtype=torch.int64).unsqueeze(0).expand(B, -1)
  out_loc = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1)

  # 2 blocks is enough for <=128 tokens.
  num_blocks = 2
  kv_caches, idx_k_caches = _alloc_our_kv_caches(
    num_layers=cfg.num_layers,
    num_blocks=num_blocks,
    idx_dim=cfg.dsa_idx_dim,
    device=device,
  )
  block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1)
  cache_seqlens = torch.full((B,), S, device=device, dtype=torch.int32)

  with torch.inference_mode():
    logits = engine.model(
      tokens.to(device),
      positions,
      kv_caches=kv_caches,
      idx_k_caches=idx_k_caches,
      block_table=block_table,
      cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[int(S)] * int(B),
      out_loc=out_loc,
      # Reference forward is a prefill over S>1 tokens; ensure we do not take
      # the decode (low-latency) MoE dispatch path, which assumes decode-like
      # shapes and can trip DeepEP invariants.
      prefill_mode="dense",
    )
  return logits[:, -1, :].float().cpu()


class TestCorrectnessReference(unittest.TestCase):
  def test_logits_argmax_matches_reference(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires SM100 (B200)")

    paths = _paths()
    rank, world_size = _init_dist()
    if world_size != 8:
      raise unittest.SkipTest(f"requires world_size=8 for mp8 reference (got {world_size})")

    mode = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
    if mode != "rdep":
      raise RuntimeError(f"test_correctness_reference requires NMOE_EP_TRANSPORT=rdep (got {mode!r})")

    # Deterministic token inputs (avoid tokenizer/template confounders).
    test_inputs: List[List[int]] = [
      [1, 100, 1000, 10000, 50000, 60000, 1234, 5678],
    ]

    out_path = os.environ.get("NMOE_REFERENCE_LOGITS_PATH", "/tmp/nmoe_reference_logits.pt")
    phase_env = os.environ.get("NMOE_REFERENCE_PHASE", "auto").strip().lower()
    if phase_env not in ("auto", "reference", "compare"):
      raise ValueError(f"Invalid NMOE_REFERENCE_PHASE={phase_env!r}; expected auto|reference|compare")
    if phase_env == "auto":
      phase = "compare" if os.path.exists(out_path) else "reference"
    else:
      phase = phase_env

    if phase == "reference":
      ref_logits_all: List[torch.Tensor] = []
      for ids in test_inputs:
        tokens = torch.tensor([ids], dtype=torch.long, device=f"cuda:{rank}")
        ref_logits_all.append(_run_reference_logits_last(tokens, paths=paths, rank=rank, world_size=world_size))
        gc.collect()
        torch.cuda.empty_cache()

      if rank == 0:
        payload = {"test_inputs": test_inputs, "ref_last_logits": ref_logits_all}
        torch.save(payload, out_path)
        print(f"[reference] saved logits to: {out_path}", flush=True)
      return

    # compare
    if not os.path.exists(out_path):
      raise RuntimeError(
        f"[compare] missing reference logits: {out_path}. "
        f"Run with NMOE_REFERENCE_PHASE=reference first."
      )
    payload = torch.load(out_path, map_location="cpu")
    saved_inputs = payload.get("test_inputs")
    ref_logits_all = payload.get("ref_last_logits")
    if not isinstance(saved_inputs, list) or not isinstance(ref_logits_all, list):
      raise RuntimeError(f"[compare] invalid payload in {out_path}")
    if saved_inputs != test_inputs:
      raise RuntimeError(f"[compare] payload inputs differ from test inputs (payload={saved_inputs}, test={test_inputs})")

    our_logits_all: List[torch.Tensor] = []
    for ids in test_inputs:
      tokens = torch.tensor([ids], dtype=torch.long, device=f"cuda:{rank}")
      our_logits_all.append(_run_our_logits_last(tokens, paths=paths, rank=rank, world_size=world_size))
      gc.collect()
      torch.cuda.empty_cache()

    if rank == 0:
      # Save both sides for debugging (large but actionable).
      out_debug = os.environ.get("NMOE_REFERENCE_COMPARE_DUMP", "/tmp/nmoe_compare_logits.pt")
      torch.save(
        {
          "test_inputs": test_inputs,
          "ref_last_logits": ref_logits_all,
          "our_last_logits": our_logits_all,
        },
        out_debug,
      )
      for i, (ref, ours) in enumerate(zip(ref_logits_all, our_logits_all)):
        self.assertEqual(tuple(ref.shape), tuple(ours.shape), f"shape mismatch on input_{i}: {ref.shape} vs {ours.shape}")
        ref_argmax = int(ref[0].argmax().item())
        our_argmax = int(ours[0].argmax().item())
        if ref_argmax != our_argmax:
          ref_top5 = ref[0].topk(5).indices.tolist()
          our_top5 = ours[0].topk(5).indices.tolist()
          max_abs = float((ref - ours).abs().max().item())
          mean_abs = float((ref - ours).abs().mean().item())
          raise AssertionError(
            f"argmax mismatch on input_{i}: ref={ref_argmax} ours={our_argmax}\n"
            f"  ref_top5={ref_top5}\n"
            f"  our_top5={our_top5}\n"
            f"  max_abs_diff={max_abs:.6f} mean_abs_diff={mean_abs:.6f}\n"
            f"  reference_logits={out_path}\n"
            f"  compare_dump={out_debug}"
          )


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestCorrectnessReference)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  try:
    if dist.is_initialized():
      dist.destroy_process_group()
  except Exception:
    pass
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
