# SPDX-License-Identifier: Apache-2.0
"""Compare nmoe/serve/model.py against the reference implementation.

This test validates that our implementation produces the same outputs as
the DeepSeek reference implementation in /data/models/.../inference/.

Run with: torchrun --nproc_per_node=8 -m nmoe.serve.test_reference_comparison

Strategy:
1. Run reference model, save logits to file
2. Run our model, compare logits
"""

import json
import os
import sys
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


CKPT_PATH = os.environ.get("NMOE_MODEL_PATH", "/data/models/DeepSeek-V3.2-Speciale")
CKPT_PATH_MP8 = os.environ.get("NMOE_MODEL_PATH_MP", "/data/models/DeepSeek-V3.2-Speciale-mp8")
CONFIG_PATH = os.environ.get("NMOE_REFERENCE_CONFIG", f"{CKPT_PATH}/inference/config_671B_v3.2.json")
REFERENCE_DIR = os.environ.get("NMOE_REFERENCE_DIR", f"{CKPT_PATH}/inference")
GOLDEN_OUTPUT_PATH = os.environ.get("NMOE_REFERENCE_GOLDEN", "/tmp/reference_logits.pt")

# Test inputs - use specific tokens to ensure reproducibility
TEST_INPUTS = [
  [1, 100, 1000, 10000],  # Simple numeric sequence
  [1, 2, 3, 4, 5, 6, 7, 8],  # Sequential
]


def run_reference_model(rank: int, world_size: int) -> torch.Tensor:
  """Run the reference implementation and return logits."""
  sys.path.insert(0, REFERENCE_DIR)
  from model import Transformer, ModelArgs
  from safetensors.torch import load_model

  with open(CONFIG_PATH) as f:
    args = ModelArgs(**json.load(f))

  torch.set_default_dtype(torch.bfloat16)

  with torch.device("cuda"):
    model = Transformer(args)

  # Load pre-sharded weights
  ckpt_file = f"{CKPT_PATH_MP8}/model{rank}-mp{world_size}.safetensors"
  if rank == 0:
    print(f"Loading reference model from {ckpt_file}")
  load_model(model, ckpt_file)
  model.eval()

  # Run forward pass
  # Note: reference model returns logits for LAST position only: [B, V]
  all_logits = []
  for input_ids in TEST_INPUTS:
    tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
      logits = model.forward(tokens, start_pos=0)  # [B, V] - last position only
    all_logits.append(logits.cpu())

  sys.path.remove(REFERENCE_DIR)
  return all_logits


def run_our_model(rank: int, world_size: int) -> torch.Tensor:
  """Run our implementation and return logits."""
  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Full model config
  cfg = ModelConfig(num_layers=61, num_dense_layers=3)

  # DeepEP buffer
  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)

  device = torch.device(f"cuda:{rank}")
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  if rank == 0:
    print(f"Loading our model from {CKPT_PATH}")
  load_checkpoint(model, CKPT_PATH, rank=rank, world_size=world_size, cfg=cfg)

  # Run forward pass
  all_logits = []
  for input_ids in TEST_INPUTS:
    B, S = 1, len(input_ids)
    tokens = torch.tensor([input_ids], dtype=torch.int64, device=device)
    positions = torch.arange(S, dtype=torch.int64, device=device).unsqueeze(0)

    # Create KV caches
    num_blocks = 2
    kv_caches = [
      torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
      for _ in range(cfg.num_layers)
    ]
    idx_k_caches = [
      torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
      for _ in range(cfg.num_layers)
    ]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
    out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

    with torch.inference_mode():
      logits = model(
        tokens, positions,
        kv_caches=kv_caches, idx_k_caches=idx_k_caches,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[S], out_loc=out_loc,
      )
    # Return only last position logits to match reference: [B, V]
    all_logits.append(logits[:, -1, :].cpu())

  return all_logits


def compare_logits(ref_logits: list, our_logits: list, rank: int) -> dict:
  """Compare logits and return metrics."""
  results = {}

  for i, (ref, ours) in enumerate(zip(ref_logits, our_logits)):
    # Compare shapes
    if ref.shape != ours.shape:
      results[f"input_{i}"] = {
        "status": "FAIL",
        "reason": f"Shape mismatch: ref={ref.shape}, ours={ours.shape}"
      }
      continue

    # Compare values
    ref_f = ref.float()
    ours_f = ours.float()

    # Absolute difference
    abs_diff = (ref_f - ours_f).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Relative difference (avoid div by zero)
    rel_diff = abs_diff / (ref_f.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Top-k comparison (most important for generation)
    # Logits are now [B, V] not [B, S, V]
    ref_topk = ref_f[0, :].topk(10)
    ours_topk = ours_f[0, :].topk(10)
    topk_match = torch.equal(ref_topk.indices, ours_topk.indices)
    argmax_match = ref_f[0, :].argmax() == ours_f[0, :].argmax()

    # Determine pass/fail
    # Loose tolerance for FP8 quantization differences
    passed = max_abs_diff < 1.0 and argmax_match

    results[f"input_{i}"] = {
      "status": "PASS" if passed else "FAIL",
      "max_abs_diff": max_abs_diff,
      "mean_abs_diff": mean_abs_diff,
      "max_rel_diff": max_rel_diff,
      "mean_rel_diff": mean_rel_diff,
      "argmax_match": bool(argmax_match),
      "topk_match": bool(topk_match),
      "ref_argmax": int(ref_f[0, :].argmax()),
      "ours_argmax": int(ours_f[0, :].argmax()),
      "ref_top5": ref_topk.indices[:5].tolist(),
      "ours_top5": ours_topk.indices[:5].tolist(),
    }

  return results


def main():
  # Initialize distributed
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  if rank == 0:
    print("=" * 60)
    print("Reference Comparison Test")
    print("=" * 60)
    print(f"World size: {world_size}")
    print(f"Test inputs: {TEST_INPUTS}")

  dist.barrier()

  # Step 1: Run reference model
  if rank == 0:
    print("\n[Step 1] Running reference implementation...")

  ref_logits = run_reference_model(rank, world_size)

  if rank == 0:
    print(f"  Reference logits shapes: {[l.shape for l in ref_logits]}")
    torch.save(ref_logits, GOLDEN_OUTPUT_PATH)
    print(f"  Saved to {GOLDEN_OUTPUT_PATH}")

  dist.barrier()

  # Clear GPU memory before loading our model
  del ref_logits
  torch.cuda.empty_cache()

  # Need to re-init distributed for our model (it has its own init_distributed)
  # Actually, we should not destroy - just continue with same process group

  # Step 2: Run our model
  if rank == 0:
    print("\n[Step 2] Running our implementation...")

  our_logits = run_our_model(rank, world_size)

  if rank == 0:
    print(f"  Our logits shapes: {[l.shape for l in our_logits]}")

  dist.barrier()

  # Step 3: Compare (only on rank 0)
  if rank == 0:
    print("\n[Step 3] Comparing outputs...")

    ref_logits = torch.load(GOLDEN_OUTPUT_PATH)
    results = compare_logits(ref_logits, our_logits, rank)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_passed = True
    for name, result in results.items():
      status = result["status"]
      if status == "FAIL":
        all_passed = False

      print(f"\n{name}: {status}")
      if "reason" in result:
        print(f"  {result['reason']}")
      else:
        print(f"  max_abs_diff: {result['max_abs_diff']:.6f}")
        print(f"  mean_abs_diff: {result['mean_abs_diff']:.6f}")
        print(f"  argmax_match: {result['argmax_match']}")
        print(f"  ref_argmax: {result['ref_argmax']}, ours_argmax: {result['ours_argmax']}")
        print(f"  ref_top5: {result['ref_top5']}")
        print(f"  ours_top5: {result['ours_top5']}")

    print("\n" + "=" * 60)
    if all_passed:
      print("ALL TESTS PASSED")
    else:
      print("SOME TESTS FAILED")
    print("=" * 60)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
