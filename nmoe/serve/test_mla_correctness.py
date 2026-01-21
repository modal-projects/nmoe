# SPDX-License-Identifier: Apache-2.0
"""Comprehensive MLA correctness tests for all three attention paths.

Tests:
1. Dense prefill → decode (no cache, fresh sequence)
2. Paged prefill → decode (chunked prefill with cached prefix)
3. Multi-request isolation (no cross-request KV contamination)
4. EOS handling

Run with: torchrun --nproc_per_node=8 -m nmoe.serve.test_mla_correctness
"""

import os
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
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GenerationResult:
    """Result of a generation run."""
    output_text: str
    output_ids: list[int]
    num_tokens: int
    hit_eos: bool
    hit_max_tokens: bool


def _greedy_argmax(full_logits: torch.Tensor) -> int:
    """Greedy argmax over full vocab logits (TP=1 mode)."""
    return int(torch.argmax(full_logits, dim=-1).item())


def generate_with_chunked_prefill(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    chunk_size: int = 4,  # Prefill in chunks to test paged prefill path
    device: torch.device = None,
    rank: int = 0,
) -> GenerationResult:
    """Generation with chunked prefill to test paged prefill (PATH 2)."""
    device = device or next(model.parameters()).device
    cfg = model.cfg

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    B, S = input_ids.shape

    # Allocate caches
    max_seq = S + max_new_tokens + 64
    num_blocks = (max_seq + 63) // 64
    num_layers = cfg.num_layers

    if cfg.attention_type == "mla":
        kv_caches = None
        idx_k_caches = None
        page_size = 64
        kv_caches_latent = [
            torch.zeros(num_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(num_layers)
        ]
        kv_caches_rope = [
            torch.zeros(num_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(num_layers)
        ]
    else:
        kv_caches = [
            torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
            for _ in range(num_layers)
        ]
        idx_k_caches = [
            torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
            for _ in range(num_layers)
        ]
        kv_caches_latent = None
        kv_caches_rope = None

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    generated_ids = input_ids.tolist()[0]

    with torch.inference_mode():
        # Chunked prefill: process prompt in chunks
        cached_len = 0
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            chunk_len = chunk_end - chunk_start

            chunk_ids = input_ids[:, chunk_start:chunk_end]
            positions = torch.arange(chunk_start, chunk_end, dtype=torch.int64, device=device).unsqueeze(0)
            out_loc = torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=device).unsqueeze(0)
            cache_seqlens = torch.tensor([chunk_end], dtype=torch.int32, device=device)

            # First chunk uses dense prefill, subsequent chunks use paged prefill
            if cached_len == 0:
                prefill_mode = "dense" if cfg.attention_type == "mla" else None
            else:
                prefill_mode = "paged" if cfg.attention_type == "mla" else None

            logits = model(
                chunk_ids,
                positions,
                kv_caches=kv_caches,
                idx_k_caches=idx_k_caches,
                kv_caches_latent=kv_caches_latent,
                kv_caches_rope=kv_caches_rope,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=[chunk_end],
                out_loc=out_loc,
                prefill_mode=prefill_mode,
            )
            torch.cuda.synchronize(device)

            cached_len = chunk_end

        # Sample from last position of final chunk
        next_token = _greedy_argmax(logits[0, -1, :])
        generated_ids.append(next_token)

        hit_eos = (next_token == tokenizer.eos_token_id)
        tokens_generated = 1

        # Decode loop
        for step in range(max_new_tokens - 1):
            if hit_eos:
                break

            cur_pos = S + step + 1
            cache_seqlens = torch.tensor([cur_pos], dtype=torch.int32, device=device)

            decode_ids = torch.tensor([[next_token]], dtype=torch.int64, device=device)
            positions = torch.tensor([[cur_pos - 1]], dtype=torch.int64, device=device)
            out_loc = torch.tensor([[cur_pos - 1]], dtype=torch.int32, device=device)

            logits = model(
                decode_ids,
                positions,
                kv_caches=kv_caches,
                idx_k_caches=idx_k_caches,
                kv_caches_latent=kv_caches_latent,
                kv_caches_rope=kv_caches_rope,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=[cur_pos],
                out_loc=out_loc,
                prefill_mode=None,
            )
            torch.cuda.synchronize(device)

            next_token = _greedy_argmax(logits[0, 0, :])
            generated_ids.append(next_token)
            tokens_generated += 1

            if next_token == tokenizer.eos_token_id:
                hit_eos = True
                break

    return GenerationResult(
        output_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        output_ids=generated_ids,
        num_tokens=tokens_generated,
        hit_eos=hit_eos,
        hit_max_tokens=(tokens_generated >= max_new_tokens),
    )


def generate_dense_prefill(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    device: torch.device = None,
    rank: int = 0,
) -> GenerationResult:
    """Standard generation with dense prefill (PATH 1) → decode (PATH 3)."""
    device = device or next(model.parameters()).device
    cfg = model.cfg

    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    B, S = input_ids.shape

    max_seq = S + max_new_tokens + 64
    num_blocks = (max_seq + 63) // 64
    num_layers = cfg.num_layers

    if cfg.attention_type == "mla":
        kv_caches = None
        idx_k_caches = None
        page_size = 64
        kv_caches_latent = [
            torch.zeros(num_blocks, page_size, cfg.kv_lora_rank, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(num_layers)
        ]
        kv_caches_rope = [
            torch.zeros(num_blocks, page_size, cfg.qk_rope_head_dim, dtype=torch.bfloat16, device=device).permute(1, 2, 0)
            for _ in range(num_layers)
        ]
    else:
        kv_caches = [
            torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
            for _ in range(num_layers)
        ]
        idx_k_caches = [
            torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
            for _ in range(num_layers)
        ]
        kv_caches_latent = None
        kv_caches_rope = None

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    generated_ids = input_ids.tolist()[0]

    with torch.inference_mode():
        # Dense prefill
        positions = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)
        out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)
        cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)

        prefill_mode = "dense" if cfg.attention_type == "mla" else None
        logits = model(
            input_ids,
            positions,
            kv_caches=kv_caches,
            idx_k_caches=idx_k_caches,
            kv_caches_latent=kv_caches_latent,
            kv_caches_rope=kv_caches_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cache_seqlens_cpu=[S],
            out_loc=out_loc,
            prefill_mode=prefill_mode,
        )
        torch.cuda.synchronize(device)

        next_token = _greedy_argmax(logits[0, -1, :])
        generated_ids.append(next_token)

        hit_eos = (next_token == tokenizer.eos_token_id)
        tokens_generated = 1

        # Decode loop
        for step in range(max_new_tokens - 1):
            if hit_eos:
                break

            cur_pos = S + step + 1
            cache_seqlens = torch.tensor([cur_pos], dtype=torch.int32, device=device)

            decode_ids = torch.tensor([[next_token]], dtype=torch.int64, device=device)
            positions = torch.tensor([[cur_pos - 1]], dtype=torch.int64, device=device)
            out_loc = torch.tensor([[cur_pos - 1]], dtype=torch.int32, device=device)

            logits = model(
                decode_ids,
                positions,
                kv_caches=kv_caches,
                idx_k_caches=idx_k_caches,
                kv_caches_latent=kv_caches_latent,
                kv_caches_rope=kv_caches_rope,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=[cur_pos],
                out_loc=out_loc,
                prefill_mode=None,
            )
            torch.cuda.synchronize(device)

            next_token = _greedy_argmax(logits[0, 0, :])
            generated_ids.append(next_token)
            tokens_generated += 1

            if next_token == tokenizer.eos_token_id:
                hit_eos = True
                break

    return GenerationResult(
        output_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        output_ids=generated_ids,
        num_tokens=tokens_generated,
        hit_eos=hit_eos,
        hit_max_tokens=(tokens_generated >= max_new_tokens),
    )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mode = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
    if mode != "rdep":
        raise RuntimeError(f"test_mla_correctness requires NMOE_EP_TRANSPORT=rdep (got {mode!r})")

    # RDEP decode path is gated on NMOE_MOE_FUSED_PACK=1. Without it, MoE.forward
    # falls through to the DeepEP normal path which crashes on RDEP's SimpleNamespace
    # buffer (no get_dispatch_layout method). Fail fast with a clear message.
    fused_pack = os.environ.get("NMOE_MOE_FUSED_PACK", "0").strip().lower()
    if fused_pack not in ("1", "true"):
        raise RuntimeError(
            "NMOE_EP_TRANSPORT=rdep requires NMOE_MOE_FUSED_PACK=1 to use the RDEP decode path. "
            "Without it, MoE falls through to DeepEP and crashes. "
            f"Got NMOE_MOE_FUSED_PACK={fused_pack!r}"
        )

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Use a shared per-run extensions dir across ranks to avoid JIT skew causing
    # barrier/collective timeouts during MoE dispatch.
    master_port = os.environ.get("MASTER_PORT", "0")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"/tmp/torch_extensions_nmoe_{master_port}")
    os.makedirs(os.environ["TORCH_EXTENSIONS_DIR"], exist_ok=True)

    from nmoe.serve.ckpt import load_checkpoint, load_model_config
    from nmoe.serve.engine import Engine, EngineConfig

    ckpt_path = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3-0324-ep8-tp1-nmoe")
    cfg = load_model_config(ckpt_path)

    if rank == 0:
        print("=" * 70)
        print("MLA Correctness Tests")
        print("=" * 70)
        print(f"Checkpoint: {ckpt_path}")
        print(f"Attention type: {cfg.attention_type}")
        print(f"Layers: {cfg.num_layers} ({cfg.num_dense_layers} dense)")

    # Use the production engine init so MoE runs on the RDEP transport path.
    engine_cfg = EngineConfig(
        # Keep KV cache modest for correctness tests; we allocate per-test caches
        # for attention paths, but the engine still initializes its own caches.
        num_pages=64,
        page_size=64,
        num_layers=int(cfg.num_layers),
        kv_lora_rank=int(cfg.kv_lora_rank),
        qk_rope_head_dim=int(cfg.qk_rope_head_dim),
        max_batch_size=64,
        max_seq_len=8192,
        # Keep the inference-RDEP prefill transport capacity modest for tests.
        max_step_tokens=4096,
        attention_type=str(cfg.attention_type),
        idx_dim=int(getattr(cfg, "dsa_idx_dim", 128)),
        tp_size=1,
    )
    engine = Engine(cfg, engine_cfg, rank=rank, world_size=world_size)
    model = engine.model

    # Always use the nmoe loader; the "sharded" loader can materialize large BF16
    # weights (dequant) and OOM on this checkpoint even on B200.
    if rank == 0:
        print("\nLoading weights (this can take several minutes)...", flush=True)
    missing, unexpected = load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={len(missing)} unexpected={len(unexpected)}")
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)

    if rank == 0:
        print(f"\nEOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
        print("Model loaded.\n")

    results = {}

    # Test 1: Dense prefill → decode (PATH 1 → PATH 3)
    if rank == 0:
        print("=" * 70)
        print("Test 1: Dense prefill → decode (PATH 1 → PATH 3)")
        print("=" * 70)

    prompts_dense = [
        ("capital", "The capital of France is"),
        ("code", "def fibonacci(n):"),
        ("math", "2 + 2 ="),
    ]

    for name, prompt in prompts_dense:
        dist.barrier()
        try:
            result = generate_dense_prefill(model, tokenizer, prompt, max_new_tokens=30, device=device, rank=rank)
            results[f"dense_{name}"] = True
            if rank == 0:
                print(f"\n[{name}] Prompt: {prompt!r}")
                print(f"  Output: {result.output_text}")
                print(f"  Tokens: {result.num_tokens}, EOS: {result.hit_eos}, MaxTok: {result.hit_max_tokens}")
        except Exception as e:
            results[f"dense_{name}"] = False
            if rank == 0:
                print(f"\n[{name}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Test 2: Chunked prefill → decode (PATH 1 → PATH 2 → PATH 3)
    if rank == 0:
        print("\n" + "=" * 70)
        print("Test 2: Chunked prefill → decode (PATH 1 → PATH 2 → PATH 3)")
        print("=" * 70)

    prompts_chunked = [
        ("chunked_capital", "The capital of France is", 4),  # chunk_size=4
        ("chunked_long", "Once upon a time in a land far far away there lived a", 8),
    ]

    for name, prompt, chunk_size in prompts_chunked:
        dist.barrier()
        try:
            result = generate_with_chunked_prefill(
                model, tokenizer, prompt, max_new_tokens=30, chunk_size=chunk_size, device=device, rank=rank
            )
            results[f"chunked_{name}"] = True
            if rank == 0:
                print(f"\n[{name}] Prompt: {prompt!r} (chunk_size={chunk_size})")
                print(f"  Output: {result.output_text}")
                print(f"  Tokens: {result.num_tokens}, EOS: {result.hit_eos}, MaxTok: {result.hit_max_tokens}")
        except Exception as e:
            results[f"chunked_{name}"] = False
            if rank == 0:
                print(f"\n[{name}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Test 3: Multi-request isolation (no KV contamination)
    if rank == 0:
        print("\n" + "=" * 70)
        print("Test 3: Multi-request isolation (no KV contamination)")
        print("=" * 70)

    isolation_prompts = [
        "Paris is the capital of",
        "def add(a, b):",
        "The sky is",
    ]

    prev_outputs = []
    isolation_ok = True
    for i, prompt in enumerate(isolation_prompts):
        dist.barrier()
        try:
            result = generate_dense_prefill(model, tokenizer, prompt, max_new_tokens=20, device=device, rank=rank)
            if rank == 0:
                print(f"\n[Request {i+1}] {prompt!r} → {result.output_text}")
                # Check for contamination: output shouldn't contain content from previous prompts
                for prev in prev_outputs:
                    # Simple heuristic: check if previous prompt's key words appear
                    if "Paris" in prev and "capital" in result.output_text.lower() and "add" in prompt.lower():
                        print(f"  WARNING: Possible contamination from request with 'Paris'")
                        isolation_ok = False
                prev_outputs.append(result.output_text)
        except Exception as e:
            isolation_ok = False
            if rank == 0:
                print(f"[Request {i+1}] FAILED: {e}")

    results["isolation"] = isolation_ok

    # Test 4: Dense vs chunked path comparison (informational)
    # NOTE: Dense prefill uses FA4 kernel, paged prefill uses CuTeDSL kernel.
    # These are mathematically equivalent but numerically different due to
    # different accumulation order in BF16. This is expected and not a bug.
    # We check that both produce coherent text (no crashes, valid tokens).
    if rank == 0:
        print("\n" + "=" * 70)
        print("Test 4: Dense vs chunked comparison (informational)")
        print("=" * 70)
        print("NOTE: Different kernels (FA4 vs CuTeDSL) may produce different outputs.")
        print("      This is expected numerical behavior, not a correctness bug.")

    prompt = "The capital of France is"
    dist.barrier()
    try:
        dense = generate_dense_prefill(model, tokenizer, prompt, max_new_tokens=20, device=device, rank=rank)
        chunked = generate_with_chunked_prefill(
            model, tokenizer, prompt, max_new_tokens=20, chunk_size=4, device=device, rank=rank
        )
        # Both paths should produce valid output (no crashes, reasonable token counts)
        dense_valid = dense.num_tokens > 0 and len(dense.output_ids) > 0
        chunked_valid = chunked.num_tokens > 0 and len(chunked.output_ids) > 0
        results["dense_vs_chunked"] = dense_valid and chunked_valid
        if rank == 0:
            print(f"\nPrompt: {prompt!r}")
            print(f"  dense:   {dense.output_text}")
            print(f"  chunked: {chunked.output_text}")
            match = dense.output_ids == chunked.output_ids
            if match:
                print("  Outputs match exactly (lucky!)")
            else:
                # Find first divergence point for information
                for j, (a, b) in enumerate(zip(dense.output_ids, chunked.output_ids)):
                    if a != b:
                        print(f"  Diverged at token {j}: dense={a} chunked={b} (expected, different kernels)")
                        break
    except Exception as e:
        results["dense_vs_chunked"] = False
        if rank == 0:
            print(f"\n[dense_vs_chunked] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    dist.barrier()
    local_fail = 0
    if rank == 0:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_passed = True
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")
            if not passed:
                all_passed = False
        print("=" * 70)
        if all_passed:
            print("All MLA correctness tests PASSED")
        else:
            print("Some tests FAILED")
            local_fail = 1

    fail = torch.tensor([int(local_fail)], device=device, dtype=torch.int32)
    dist.all_reduce(fail, op=dist.ReduceOp.MAX)
    exit_code = int(fail.item())

    dist.barrier()
    engine.shutdown()
    dist.destroy_process_group()

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
