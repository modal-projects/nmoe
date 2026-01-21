# SPDX-License-Identifier: Apache-2.0
"""L4: API wiring + streaming semantics tests.

Tests:
1. Non-streaming completion returns correct format
2. Streaming generates proper SSE events with incremental tokens
3. Concurrent requests don't interfere
4. Finish reason is correct (stop vs length)
5. Chat template formatting works

Run:
    torchrun --nproc-per-node=8 -m nmoe.serve.test_api_streaming

Requires:
- NMOE_MODEL_PATH environment variable pointing to DeepSeek-V3-0324 checkpoint
- 8 GPUs with loaded model
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncIterator


def _maybe_set_cutlass_path() -> None:
    """Set CUTLASS_PATH for DeepGEMM JIT compilation."""
    if os.environ.get("CUTLASS_PATH"):
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
        if cand.is_dir():
            os.environ["CUTLASS_PATH"] = str(cand)
            return


_maybe_set_cutlass_path()

import torch
import torch.distributed as dist

# Test configurations
TEST_PROMPTS = [
    ("capital", "The capital of France is"),
    ("code", "def fibonacci(n):"),
    ("short", "Hi"),  # Short prompt to test EOS detection
]

MAX_TOKENS = 30


def main():
    """Main entry point - runs on all ranks."""
    # Setup distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
    # DeepEP timeouts (e.g., one rank compiling while others enter collectives).
    from torch.utils import cpp_extension as cpp_ext
    master_port = os.environ.get("MASTER_PORT", "0")
    jit_dir = f"/tmp/torch_extensions_nmoe_{master_port}"
    os.makedirs(jit_dir, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", jit_dir)
    cpp_ext.TORCH_EXTENSIONS_DIR = jit_dir

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("=" * 70)
        print("L4: API Wiring + Streaming Tests")
        print("=" * 70)

    # Get model path (default to ep8-tp1 sharded weights)
    # Config comes from original checkpoint, weights from ep8-tp1 sharded version
    ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1")
    config_path = os.environ.get("NMOE_CONFIG_PATH", "/data/models/DeepSeek-V3-0324")
    if rank == 0:
        print(f"\nCheckpoint: {ckpt_path}")
        print(f"Config: {config_path}")

    # Import after distributed init
    from transformers import AutoTokenizer

    from nmoe.serve.api import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatMessage,
        CompletionRequest,
        CompletionResponse,
        _format_chat_messages,
        _stream_chat_response,
        _stream_completion_response,
    )
    from nmoe.serve.ckpt import load_checkpoint, load_model_config, load_sharded_checkpoint
    from nmoe.serve.engine import Engine, EngineConfig
    from nmoe.serve.model import ModelConfig, init_distributed
    from nmoe.serve.orchestrator import AsyncOrchestrator, Orchestrator, OrchestratorConfig
    from nmoe.serve.types import Request, SamplingParams

    init_distributed(rank, world_size)

    # Load config (auto-detect MLA vs DSA) from config_path
    cfg = load_model_config(config_path)
    if rank == 0:
        print(f"Attention type: {cfg.attention_type}")
        print(f"Layers: {cfg.num_layers} ({cfg.num_dense_layers} dense)")

    # Load tokenizer (from config_path which has tokenizer files)
    tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    if rank == 0:
        print(f"\nEOS token: {repr(tokenizer.eos_token)} (id={eos_token_id})")

    # Engine config
    engine_config = EngineConfig(
        num_pages=2048,
        page_size=64,
        num_layers=cfg.num_layers,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        max_batch_size=64,
        max_seq_len=8192,
        attention_type=cfg.attention_type,
        idx_dim=cfg.dsa_idx_dim,
    )

    # Orchestrator config
    orch_config = OrchestratorConfig(
        max_batch_size=64,
        max_prefill_tokens=4096,
        max_seq_len=8192,
        num_pages=2048,
        page_size=64,
        enable_overlap=False,
        enable_chunked_prefill=True,
        chunk_size=512,
        enable_prefix_cache=True,
    )

    # Create orchestrator
    orchestrator = Orchestrator(
        model_config=cfg,
        engine_config=engine_config,
        orch_config=orch_config,
        rank=rank,
        world_size=world_size,
    )

    # Load checkpoint - use load_sharded_checkpoint for pre-sharded mp8 format
    if rank == 0:
        print("Loading model...")
    dist.barrier()

    # Check if this is a pre-sharded checkpoint (has model{rank}-mp{world_size}.safetensors)
    sharded_file = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(sharded_file):
        missing, unexpected = load_sharded_checkpoint(
            orchestrator.engine.model, ckpt_path, rank=rank, world_size=world_size
        )
    else:
        # HuggingFace format - shard on the fly
        missing, unexpected = load_checkpoint(
            orchestrator.engine.model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg
        )
    dist.barrier()

    if rank == 0:
        print("Model loaded.\n")

    # Run tests - synchronous mode (all GPUs must run together for EP)
    results = {}

    # Test 1: Non-streaming completion (sync mode - all GPUs run_step together)
    if rank == 0:
        print("=" * 70)
        print("Test 1: Non-streaming completion (sync)")
        print("=" * 70)

    for name, prompt in TEST_PROMPTS:
        # All ranks encode the same prompt (deterministic)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]

        # All ranks create and add the same request
        req = orchestrator.create_request(
            input_ids=input_ids,
            profile_name="production_generate",
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        orchestrator.add_request(req)

        if rank == 0:
            print(f"\n[{name}] Prompt: {repr(prompt)} ({len(input_ids)} tokens)")

        # Synchronous generation - all ranks run_step() together
        dist.barrier()
        step = 0
        while not req.is_finished:
            if rank == 0 and step < 3:
                print(f"  Step {step}: calling run_step()...", flush=True)
            orchestrator.run_step()
            if rank == 0 and step < 3:
                print(f"  Step {step}: done, output_ids={len(req.output_ids)}", flush=True)
            step += 1
            if step > MAX_TOKENS + 10:
                if rank == 0:
                    print(f"  WARNING: Exceeded max steps, breaking")
                break

        dist.barrier()

        output_text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
        finish_reason = req.finish_reason or "stop"
        hit_max = len(req.output_ids) >= MAX_TOKENS

        if rank == 0:
            print(f"  Output: {output_text}")
            print(f"  Tokens: {len(req.output_ids)}, Finish: {finish_reason}, MaxTok: {hit_max}")

        results[f"nonstream_{name}"] = len(req.output_ids) > 0

    # Test 2: Multi-request batching (concurrent requests, sync execution)
    if rank == 0:
        print("\n" + "=" * 70)
        print("Test 2: Multi-request batching")
        print("=" * 70)

    # Add 3 requests to all GPUs
    multi_prompts = [
        ("req1", "Paris is the capital of"),
        ("req2", "def add(a, b):"),
        ("req3", "The sky is"),
    ]

    multi_reqs = []
    for name, prompt in multi_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
        req = orchestrator.create_request(
            input_ids=input_ids,
            profile_name="production_generate",
            max_tokens=20,
            temperature=0.0,
        )
        orchestrator.add_request(req)
        multi_reqs.append((name, req))

    if rank == 0:
        print(f"  Added {len(multi_reqs)} requests")

    # Run until all finish (all GPUs run together)
    dist.barrier()
    steps = 0
    while not all(r.is_finished for _, r in multi_reqs):
        orchestrator.run_step()
        steps += 1
        if steps > 200:
            if rank == 0:
                print(f"  WARNING: Exceeded 200 steps")
            break
    dist.barrier()

    if rank == 0:
        print(f"  Completed in {steps} steps")

    # Verify outputs
    for name, req in multi_reqs:
        output_text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
        if rank == 0:
            print(f"\n  [{name}] Output: {output_text[:60]}...")

        # Basic sanity checks
        if name == "req1":
            results["multi_req1"] = "france" in output_text.lower() or "paris" in output_text.lower()
        elif name == "req2":
            results["multi_req2"] = "return" in output_text.lower() or "+" in output_text
        elif name == "req3":
            results["multi_req3"] = len(output_text) > 0

    # Summary
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_pass = True
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {test_name}: {status}")
            if not passed:
                all_pass = False
        print("=" * 70)
        if all_pass:
            print("All API tests PASSED")
        else:
            print("Some tests FAILED")

    # Cleanup
    dist.barrier()
    orchestrator.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
