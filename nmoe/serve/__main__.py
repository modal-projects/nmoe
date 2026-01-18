# SPDX-License-Identifier: Apache-2.0
"""Main entry point for nmoe.serve."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="nmoe.serve - DeepSeek-V3 inference server")

  # Model
  parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
  parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer (default: model-path)")

  # Server
  parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
  parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

  # Hardware
  parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1, help="Tensor parallel size")
  parser.add_argument("--device", type=str, default="cuda:0", help="Device (single GPU mode)")

  # Memory
  parser.add_argument("--num-pages", type=int, default=4096, help="Number of KV cache pages")
  parser.add_argument("--page-size", type=int, default=64, help="Tokens per page")
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory fraction")

  # Scheduling
  parser.add_argument("--max-batch-size", type=int, default=256, help="Max batch size")
  parser.add_argument("--max-prefill-tokens", type=int, default=8192, help="Max prefill tokens per batch")
  parser.add_argument(
    "--moe-expected-m",
    type=int,
    default=256,
    help="MoE expected_m capacity per local expert (decode fast path; must be multiple of 16)",
  )
  # DeepSeek V3 family supports long context (128k–161k depending on version).
  # Use 0 to auto-select the checkpoint's supported max context length.
  parser.add_argument("--max-seq-len", type=int, default=0, help="Max sequence length (0=use checkpoint)")
  parser.add_argument("--disable-chunked-prefill", action="store_true", help="Disable chunked prefill")
  parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size for chunked prefill")

  # Request limits (P1 hardening)
  # Use 0 to auto-set to (max_seq_len - max_output_tokens).
  parser.add_argument("--max-prompt-tokens", type=int, default=0, help="Max prompt tokens per request (0=auto)")
  parser.add_argument("--max-output-tokens", type=int, default=4096, help="Max output tokens per request")
  parser.add_argument("--max-pending-requests", type=int, default=1024, help="Max pending requests (backpressure)")

  # Options
  parser.add_argument("--enable-overlap", action="store_true", help="Enable CPU/GPU overlap scheduling")
  parser.add_argument(
    "--enable-cuda-graph",
    action="store_true",
    help="Enable CUDA-graph replay for greedy decode TOKENS (MLA-only; requires NMOE_DEEPEP_LOW_LATENCY=1).",
  )
  parser.add_argument("--disable-prefix-cache", action="store_true", help="Disable prefix caching")

  # Attention type
  parser.add_argument(
    "--attention-type", type=str, default="auto",
    choices=["auto", "dsa", "mla"],
    help="Attention type: auto (detect from checkpoint), dsa (Speciale sparse), or mla (V3-0324 dense)"
  )

  return parser.parse_args()


def main() -> int:
  args = parse_args()

  # nmoe/serve production contract (management): TP=1/DP=8/EP=8, world_size=8.
  # We require torchrun because DeepEP collectives and dynamic disaggregation
  # assume all 8 ranks are present and synchronized for every step (T=0 allowed).
  rank_env = os.environ.get("RANK")
  world_size_env = os.environ.get("WORLD_SIZE")
  if rank_env is None or world_size_env is None:
    raise RuntimeError(
      "nmoe.serve must be launched via torchrun with world_size=8.\n"
      "Example:\n"
      "  torchrun --nproc_per_node=8 --master_port=29530 -m nmoe.serve ..."
    )

  rank = int(rank_env)
  world_size = int(world_size_env)
  local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
  if world_size != 8:
    raise RuntimeError(f"nmoe.serve requires WORLD_SIZE=8 (got {world_size}).")
  if int(args.tensor_parallel_size) != 1:
    raise RuntimeError(f"nmoe.serve requires --tensor-parallel-size=1 (got {args.tensor_parallel_size}).")

  # Only rank 0 prints startup info (avoid log spam)
  def log(msg: str) -> None:
    if rank == 0:
      print(msg, flush=True)

  log(f"nmoe.serve starting (world_size={world_size})...")
  log(f"  Model: {args.model_path}")
  log(f"  Host: {args.host}:{args.port} (rank 0 only)")
  log(
    f"  Limits (requested): max_prompt_tokens={args.max_prompt_tokens} (0=auto), "
    f"max_output_tokens={args.max_output_tokens}, max_pending={args.max_pending_requests}"
  )

  # Import here to avoid slow startup
  from transformers import AutoTokenizer

  from nmoe.serve.api import create_app
  from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  # Initialize distributed (torchrun sets MASTER_ADDR/MASTER_PORT; use env:// init).
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
  if dist.get_world_size() != world_size:
    raise RuntimeError(f"World size mismatch: env={world_size}, dist={dist.get_world_size()}")
  if dist.get_rank() != rank:
    raise RuntimeError(f"Rank mismatch: env={rank}, dist={dist.get_rank()}")

  init_distributed(rank, world_size, tp_size=1)

  # CPU-only control plane for rank0<->owner p2p (no NCCL ops; no new collectives per step).
  from nmoe.serve.control_plane import ControlPlane, RequestInit

  ctrl_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  control_plane = ControlPlane(rank=rank, world_size=world_size, ctrl_group=ctrl_group)

  # Device: use LOCAL_RANK for multi-GPU per node
  device = torch.device(f"cuda:{local_rank}")
  torch.cuda.set_device(device)

  if rank == 0:
    print(f"  Device: {device} (LOCAL_RANK={local_rank})", flush=True)
    print(f"  CUDA: {torch.cuda.get_device_name(device)}", flush=True)

  # Load tokenizer (all ranks need it for token counting in distributed scenarios)
  tokenizer_path = args.tokenizer_path or args.model_path
  log(f"Loading tokenizer from {tokenizer_path}...")
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

  # Model config (auto-detect MLA vs DSA from checkpoint, or use explicit --attention-type)
  if args.attention_type == "auto":
    model_config = load_model_config(args.model_path)
    log(f"  Attention type: {model_config.attention_type} (auto-detected)")
  else:
    model_config = ModelConfig(attention_type=args.attention_type)
    log(f"  Attention type: {args.attention_type}")

  # Serving-time bounds must not exceed the checkpoint's supported max context.
  max_seq_len = int(args.max_seq_len) if int(args.max_seq_len) > 0 else int(model_config.max_seq_len)
  if max_seq_len > int(model_config.max_seq_len):
    raise RuntimeError(
      f"--max-seq-len={max_seq_len} exceeds checkpoint max_seq_len={int(model_config.max_seq_len)}. "
      f"Use --max-seq-len <= {int(model_config.max_seq_len)}."
    )
  max_output_tokens = int(args.max_output_tokens)
  if max_output_tokens < 0:
    raise RuntimeError(f"--max-output-tokens must be >= 0 (got {max_output_tokens})")
  max_prompt_tokens = int(args.max_prompt_tokens)
  if max_prompt_tokens <= 0:
    max_prompt_tokens = max(0, max_seq_len - max_output_tokens)
  if max_prompt_tokens + max_output_tokens > max_seq_len:
    raise RuntimeError(
      f"Invalid bounds: max_prompt_tokens({max_prompt_tokens}) + max_output_tokens({max_output_tokens}) "
      f"> max_seq_len({max_seq_len})."
    )
  log(
    f"  Limits (effective): max_seq_len={max_seq_len}, max_prompt_tokens={max_prompt_tokens}, "
    f"max_output_tokens={max_output_tokens}"
  )

  moe_expected_m = int(args.moe_expected_m)
  if moe_expected_m <= 0 or (moe_expected_m % 16) != 0:
    raise RuntimeError(f"--moe-expected-m must be > 0 and a multiple of 16 (got {moe_expected_m}).")

  # Engine config
  engine_config = EngineConfig(
    num_pages=args.num_pages,
    page_size=args.page_size,
    num_layers=model_config.num_layers,
    kv_lora_rank=model_config.kv_lora_rank,
    qk_rope_head_dim=model_config.qk_rope_head_dim,
    max_batch_size=args.max_batch_size,
    max_seq_len=max_seq_len,
    moe_expected_m=moe_expected_m,
    attention_type=model_config.attention_type,
    idx_dim=model_config.dsa_idx_dim,  # Only used for DSA
    tp_size=1,
  )

  # Orchestrator config
  orch_config = OrchestratorConfig(
    max_batch_size=args.max_batch_size,
    max_prefill_tokens=args.max_prefill_tokens,
    max_seq_len=max_seq_len,
    num_pages=args.num_pages,
    page_size=args.page_size,
    enable_overlap=args.enable_overlap,
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=args.chunk_size,
    enable_prefix_cache=not args.disable_prefix_cache,
    enable_cuda_graph=bool(args.enable_cuda_graph),
    # Request limits (P1 hardening)
    max_prompt_tokens=max_prompt_tokens,
    max_output_tokens=max_output_tokens,
    max_pending_requests=args.max_pending_requests,
  )

  # Create orchestrator (creates engine and model)
  log("Creating orchestrator...")
  orchestrator = Orchestrator(
    model_config=model_config,
    engine_config=engine_config,
    orch_config=orch_config,
    rank=rank,
    world_size=world_size,
    control_plane=control_plane,
  )

  # Load checkpoint
  log(f"Loading checkpoint from {args.model_path}...")
  shard_path = os.path.join(args.model_path, f"model{rank}-mp{world_size}.safetensors")
  if os.path.exists(shard_path):
    missing, unexpected = load_sharded_checkpoint(
      orchestrator.engine.model,
      args.model_path,
      rank=rank,
      world_size=world_size,
    )
  else:
    missing, unexpected = load_checkpoint(
      orchestrator.engine.model,
      args.model_path,
      rank=rank,
      world_size=world_size,
      cfg=model_config,
    )
  if rank == 0:
    if missing:
      print(f"  Missing keys: {len(missing)}", flush=True)
      for k in sorted(missing)[:10]:
        print(f"    {k}", flush=True)
    if unexpected:
      print(f"  Unexpected keys: {len(unexpected)}", flush=True)
      for k in sorted(unexpected)[:10]:
        print(f"    {k}", flush=True)

  # Synchronize all ranks before starting server
  if world_size > 1:
    dist.barrier()
    log("All ranks synchronized, ready to serve.")

  # Warm up kernels on all ranks so the first real request doesn't trigger long
  # JIT compilation stalls (which can trip DeepEP CPU timeouts for T=0 ranks).
  from nmoe.serve.warmup import warmup_orchestrator_local

  if rank == 0:
    log("Warming up (compilation + kernel cache)...")
  warmup_orchestrator_local(orchestrator, prompt_len=min(256, args.max_prompt_tokens), max_tokens=2, timeout_s=900.0)
  if rank == 0:
    log("Warmup complete.")

  # --- Rank-specific execution paths ---
  #
  # Rank 0: Runs HTTP server (uvicorn) with orchestrator in background thread
  # Ranks 1-7: Run orchestrator.run() directly (blocking, lockstep with rank 0)
  #
  # This ensures:
  # - Only one process binds to host:port (no port collision)
  # - All ranks participate in every DeepEP collective via lockstep loop

  if rank == 0:
    # Rank 0: HTTP server mode
    app = create_app(orchestrator, tokenizer, model_name="deepseek-v3", control_plane=control_plane)

    import uvicorn

    print(f"[Rank 0] Starting HTTP server on {args.host}:{args.port}...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
  else:
    # Ranks 1-7: Worker mode (no HTTP, just orchestrator loop)
    print(f"[Rank {rank}] Starting worker mode (lockstep with rank 0)...", flush=True)
    from nmoe.serve.types import ForwardSpec, OutputMode

    def _on_request_init(init: RequestInit) -> None:
      if int(init.uid) % int(world_size) != int(rank):
        control_plane.enqueue_error(uid=int(init.uid), msg="wrong owner for uid")
        return
      ok, err = orchestrator.validate_request_bounds(int(init.prompt_len), int(init.max_tokens))
      if not ok:
        control_plane.enqueue_error(uid=int(init.uid), msg=err)
        return
      if int(init.output_mode_id) != 0:
        control_plane.enqueue_error(uid=int(init.uid), msg="unsupported output_mode for v0.1 HTTP")
        return
      fs = ForwardSpec(output_mode=OutputMode.TOKENS, topk=int(init.topk))
      seed = None if int(init.seed_or_minus1) < 0 else int(init.seed_or_minus1)
      req = orchestrator.create_request(
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
      accepted = orchestrator.try_add_request(req, timeout=0.0)
      if not accepted:
        control_plane.enqueue_error(uid=int(init.uid), msg="owner queue full")

    def _on_cancel(uid: int) -> None:
      uid = int(uid)
      # If cancellation arrives before the request is known locally, emit a final
      # DONE update so rank0 can close out proxy state immediately.
      if uid not in orchestrator._uid_to_req:
        control_plane.enqueue_token_updates([uid], [-1], [1 | (3 << 1)])  # DONE + CANCELLED
      orchestrator.cancel(uid)

    control_plane.start_worker(
      on_request_init=_on_request_init,
      on_cancel=_on_cancel,
    )
    try:
      orchestrator.run()  # Blocking - participates in lockstep loop
    except KeyboardInterrupt:
      print(f"[Rank {rank}] Interrupted, shutting down...", flush=True)
    finally:
      orchestrator.shutdown()

  return 0


if __name__ == "__main__":
  sys.exit(main())
