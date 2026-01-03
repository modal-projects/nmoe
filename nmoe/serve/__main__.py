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
  parser.add_argument("--max-seq-len", type=int, default=32768, help="Max sequence length")
  parser.add_argument("--disable-chunked-prefill", action="store_true", help="Disable chunked prefill")
  parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size for chunked prefill")

  # Options
  parser.add_argument("--enable-overlap", action="store_true", help="Enable CPU/GPU overlap scheduling")
  parser.add_argument("--disable-prefix-cache", action="store_true", help="Disable prefix caching")

  # Attention type
  parser.add_argument(
    "--attention-type", type=str, default="dsa",
    choices=["dsa", "mla"],
    help="Attention type: dsa (Speciale sparse) or mla (V3-0324 dense)"
  )

  return parser.parse_args()


def main() -> int:
  args = parse_args()

  print(f"nmoe.serve starting...")
  print(f"  Model: {args.model_path}")
  print(f"  Host: {args.host}:{args.port}")
  print(f"  TP: {args.tensor_parallel_size}")

  # Import here to avoid slow startup
  from transformers import AutoTokenizer

  from nmoe.serve.api import create_app
  from nmoe.serve.ckpt import load_checkpoint, load_sharded_checkpoint
  from nmoe.serve.engine import EngineConfig
  from nmoe.serve.model import ModelConfig, init_distributed
  from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig

  # Initialize distributed if TP > 1
  rank = 0
  world_size = args.tensor_parallel_size

  if world_size > 1:
    if not dist.is_initialized():
      dist.init_process_group(
        backend="nccl",
        init_method=os.environ.get("MASTER_ADDR", "tcp://127.0.0.1:29500"),
        world_size=world_size,
        rank=int(os.environ.get("RANK", "0")),
      )
    rank = dist.get_rank()

  init_distributed(rank, world_size)

  # Device
  device = torch.device(f"cuda:{rank}" if world_size > 1 else args.device)
  torch.cuda.set_device(device)

  print(f"  Device: {device}")
  print(f"  CUDA: {torch.cuda.get_device_name(device)}")

  # Load tokenizer
  tokenizer_path = args.tokenizer_path or args.model_path
  print(f"Loading tokenizer from {tokenizer_path}...")
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

  # Model config (set attention_type from args)
  model_config = ModelConfig(attention_type=args.attention_type)
  print(f"  Attention type: {args.attention_type}")

  # Engine config
  engine_config = EngineConfig(
    num_pages=args.num_pages,
    page_size=args.page_size,
    num_layers=model_config.num_layers,
    kv_lora_rank=model_config.kv_lora_rank,
    qk_rope_head_dim=model_config.qk_rope_head_dim,
    max_batch_size=args.max_batch_size,
    max_seq_len=args.max_seq_len,
    attention_type=args.attention_type,
    idx_dim=model_config.dsa_idx_dim,  # Only used for DSA
  )

  # Orchestrator config
  orch_config = OrchestratorConfig(
    max_batch_size=args.max_batch_size,
    max_prefill_tokens=args.max_prefill_tokens,
    max_seq_len=args.max_seq_len,
    num_pages=args.num_pages,
    page_size=args.page_size,
    enable_overlap=args.enable_overlap,
    enable_chunked_prefill=not args.disable_chunked_prefill,
    chunk_size=args.chunk_size,
    enable_prefix_cache=not args.disable_prefix_cache,
  )

  # Create orchestrator (creates engine and model)
  print("Creating orchestrator...")
  orchestrator = Orchestrator(
    model_config=model_config,
    engine_config=engine_config,
    orch_config=orch_config,
    rank=rank,
    world_size=world_size,
  )

  # Load checkpoint
  print(f"Loading checkpoint from {args.model_path}...")
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
  if missing:
    print(f"  Missing keys: {len(missing)}")
    for k in sorted(missing)[:10]:
      print(f"    {k}")
  if unexpected:
    print(f"  Unexpected keys: {len(unexpected)}")
    for k in sorted(unexpected)[:10]:
      print(f"    {k}")

  # Create FastAPI app
  app = create_app(orchestrator, tokenizer, model_name="deepseek-v3")

  # Run server
  import uvicorn

  print(f"Starting server on {args.host}:{args.port}...")
  uvicorn.run(app, host=args.host, port=args.port, log_level="info")

  return 0


if __name__ == "__main__":
  sys.exit(main())
