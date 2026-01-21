# SPDX-License-Identifier: Apache-2.0
"""Warmup utilities for nmoe.serve.

Why this exists:
- First-token execution triggers JIT compilation (DeepGEMM, CuTeDSL, custom CUDA),
  which can stall one rank long enough to trip DeepEP's CPU timeout on other ranks
  that reach MoE dispatch earlier (common when some ranks have T=0).
- Production serving should warm up before declaring readiness.
"""

from __future__ import annotations

import time

import torch


def warmup_orchestrator_local(
  orch,
  *,
  prompt_len: int = 256,
  max_tokens: int = 2,
  timeout_s: float = 600.0,
) -> None:
  """Warm up kernels by running a short local request on every rank.

  Must be called on all ranks (each rank enqueues its own request).
  """
  # Use stable token IDs to avoid adversarial/random sequences triggering
  # non-finite activations or kernel corner-cases during compilation warmup.
  input_ids = torch.full((int(prompt_len),), 100, dtype=torch.int32, device="cpu")
  req = orch.create_request(
    input_ids=input_ids,
    profile_name="production_generate",
    temperature=0.0,
    max_tokens=int(max_tokens),
  )
  accepted = orch.try_add_request(req, timeout=0.0)
  if not accepted:
    raise RuntimeError("warmup request rejected (queue full)")

  deadline = time.time() + float(timeout_s)
  while not req.is_finished and time.time() < deadline:
    time.sleep(0.01)
  if not req.is_finished:
    raise TimeoutError(f"warmup timed out after {timeout_s}s")
