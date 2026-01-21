# SPDX-License-Identifier: Apache-2.0
"""Fast unit test for long-context request bounds (no GPU required).

This validates the server-facing contract for long context lengths:
- Prompt lengths up to ~160k are accepted when within max_seq_len and KV pages.
- Requests that cannot fit in KV pages fail fast with an actionable error.
"""

from __future__ import annotations

from nmoe.serve.orchestrator import OrchestratorConfig, validate_request_bounds_cfg


def main() -> None:
  cfg = OrchestratorConfig(
    max_seq_len=163840,
    max_prompt_tokens=159744,
    max_output_tokens=4096,
    num_pages=4096,
    page_size=64,
  )

  ok, err = validate_request_bounds_cfg(cfg, prompt_tokens=128000, max_tokens=16)
  if not ok:
    raise AssertionError(f"expected accept 128k+16, got error: {err}")

  ok, err = validate_request_bounds_cfg(cfg, prompt_tokens=159744, max_tokens=4096)
  if ok:
    raise AssertionError("expected reject when prompt+output exceeds max_seq_len")
  if "Total sequence length" not in err:
    raise AssertionError(f"unexpected error: {err!r}")

  cfg_small = OrchestratorConfig(
    max_seq_len=163840,
    max_prompt_tokens=159744,
    max_output_tokens=4096,
    num_pages=128,  # intentionally too small
    page_size=64,
  )
  ok, err = validate_request_bounds_cfg(cfg_small, prompt_tokens=32000, max_tokens=16)
  if ok:
    raise AssertionError("expected reject when KV pages are insufficient")
  if "Insufficient KV pages" not in err:
    raise AssertionError(f"unexpected error: {err!r}")

  print("PASS")


if __name__ == "__main__":
  main()

