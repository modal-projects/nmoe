# SPDX-License-Identifier: Apache-2.0
"""End-to-end benchmark using the real serving stack.

Uses: Orchestrator -> Scheduler -> Engine -> Model
This is what actually runs in production.

LMSYS targets (per 8x H100 node):
- Prefill: 52.3k tok/s
- Decode: 22.3k tok/s
"""

import os
import time
from pathlib import Path


def _maybe_set_cutlass_path() -> None:
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


def benchmark_throughput(orchestrator, num_requests: int, input_len: int, output_len: int, rank: int):
    """Benchmark request throughput through the full serving stack."""
    from nmoe.serve.types import Request, SamplingParams, ForwardSpec, OutputMode

    if rank == 0:
        print(f"\n  Config: {num_requests} reqs x {input_len} input + {output_len} output")

    # Create requests
    requests = []
    for i in range(num_requests):
        input_ids = torch.randint(0, 10000, (input_len,))  # CPU tensor required
        req = orchestrator.create_request(
            input_ids=input_ids,
            profile_name="production_generate",
            max_tokens=output_len,
            temperature=0.0,  # greedy
        )
        requests.append(req)

    # Add all requests
    for req in requests:
        orchestrator.add_request(req)

    # Run until all complete
    torch.cuda.synchronize()
    if orchestrator.world_size > 1:
        dist.barrier()

    start = time.perf_counter()

    completed = 0
    while completed < num_requests:
        orchestrator.run_step()

        # Check for completed requests
        finished = orchestrator.finished_requests
        completed += len(finished)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate metrics
    total_input_tokens = num_requests * input_len
    total_output_tokens = num_requests * output_len
    total_tokens = total_input_tokens + total_output_tokens

    prefill_tok_s = total_input_tokens / elapsed
    decode_tok_s = total_output_tokens / elapsed
    total_tok_s = total_tokens / elapsed

    if rank == 0:
        print(f"    Time: {elapsed:.2f}s")
        print(f"    Prefill: {prefill_tok_s:,.0f} tok/s")
        print(f"    Decode:  {decode_tok_s:,.0f} tok/s")
        print(f"    Total:   {total_tok_s:,.0f} tok/s")

    return prefill_tok_s, decode_tok_s


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from nmoe.serve.model import ModelConfig, init_distributed
    from nmoe.serve.engine import EngineConfig
    from nmoe.serve.orchestrator import Orchestrator, OrchestratorConfig
    from nmoe.serve.ckpt import load_checkpoint

    init_distributed(rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("nmoe.serve E2E Benchmark")
        print(f"GPUs: {world_size}")
        print("=" * 60)
        print("\nTargets (per 8x H100 node):")
        print("  Prefill: 52,300 tok/s")
        print("  Decode:  22,300 tok/s")

    # Configs - V3-0324 uses MLA (not DSA)
    model_config = ModelConfig(
        num_layers=61,
        num_dense_layers=3,
        attention_type="mla",  # V3-0324 uses standard MLA
    )

    engine_config = EngineConfig(
        num_pages=4096,
        page_size=64,
        num_layers=model_config.num_layers,
        kv_lora_rank=model_config.kv_lora_rank,
        qk_rope_head_dim=model_config.qk_rope_head_dim,
        max_batch_size=256,
        max_seq_len=32768,
        attention_type="mla",  # Match model
    )

    orch_config = OrchestratorConfig(
        max_batch_size=256,
        max_prefill_tokens=16384,  # LMSYS uses 16384 per device
        max_decode_tokens=4096,
        max_seq_len=32768,
        num_pages=4096,
        page_size=64,
        enable_overlap=False,  # Start simple
        enable_cuda_graph=False,  # TODO
    )

    if rank == 0:
        print(f"\nCreating orchestrator...")

    orchestrator = Orchestrator(
        model_config=model_config,
        engine_config=engine_config,
        orch_config=orch_config,
        rank=rank,
        world_size=world_size,
    )

    # Load pre-converted mp8 checkpoint
    ckpt_path = "/data/models/DeepSeek-V3-0324-mp8"
    if rank == 0:
        print(f"Loading mp8 checkpoint from {ckpt_path}...")

    from safetensors.torch import safe_open
    fpath = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    state_dict = {}
    with safe_open(fpath, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    missing, unexpected = orchestrator.engine.model.load_state_dict(state_dict, strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    dist.barrier()

    if rank == 0:
        print("Model loaded.\n")

    # Warmup
    if rank == 0:
        print("Warming up...")
    benchmark_throughput(orchestrator, num_requests=2, input_len=128, output_len=16, rank=rank)

    # Benchmark configs (matching LMSYS conditions)
    if rank == 0:
        print("\n" + "=" * 60)
        print("THROUGHPUT BENCHMARK")
        print("=" * 60)

    configs = [
        # (num_requests, input_len, output_len)
        (1, 512, 32),      # baseline single request
        (4, 512, 32),      # small batch
        (16, 512, 32),     # medium batch
        (32, 1024, 64),    # larger
        (64, 2000, 100),   # closer to LMSYS decode config
        # (256, 2000, 100),  # LMSYS decode config - may OOM
    ]

    for num_req, input_len, output_len in configs:
        try:
            benchmark_throughput(orchestrator, num_req, input_len, output_len, rank)
        except Exception as e:
            if rank == 0:
                print(f"    FAILED: {e}")

    # Summary
    if rank == 0:
        print("\n" + "=" * 60)
        print("OPTIMIZATION STATUS")
        print("=" * 60)
        print(f"  CUDA Graphs: {'ON' if orch_config.enable_cuda_graph else 'OFF (TODO)'}")
        print(f"  Overlap:     {'ON' if orch_config.enable_overlap else 'OFF'}")
        print("=" * 60)

    orchestrator.shutdown()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
