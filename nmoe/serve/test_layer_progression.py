# SPDX-License-Identifier: Apache-2.0
"""Test model with increasing number of layers to find where output breaks."""

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


def test_layers(num_layers: int):
  """Test model with given number of layers."""
  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  cfg = ModelConfig(num_layers=num_layers, num_dense_layers=3)

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=0, world_size=1, cfg=cfg)

  # Test input
  B, S = 1, 8
  input_ids = torch.tensor([[1, 100, 1000, 10000, 50000, 100000, 1234, 5678]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

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

  with torch.no_grad():
    logits = model(
      input_ids, positions,
      kv_caches=kv_caches, idx_k_caches=idx_k_caches,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc,
    )

  has_nan = torch.isnan(logits).any().item()
  amax = logits.abs().max().item()
  top5 = logits[0, -1, :].topk(5)
  argmax = logits[0, -1, :].argmax().item()

  # Check if output looks reasonable
  # EOS token is 1, so argmax=1 suggests broken output
  is_eos = argmax == 1

  return {
    "num_layers": num_layers,
    "has_nan": has_nan,
    "amax": amax,
    "argmax": argmax,
    "is_eos": is_eos,
    "top5_indices": top5.indices.tolist(),
    "top5_values": [f"{v:.2f}" for v in top5.values.tolist()],
  }


def main():
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0)

  print("=" * 80)
  print("Layer Progression Test - Finding where output breaks")
  print("=" * 80)

  # Test with increasing layers
  # 1 = embed only (dense)
  # 3 = all dense
  # 4 = 3 dense + 1 MoE
  # ...
  # 61 = full model

  layer_configs = [1, 3, 4, 5, 8, 16, 32, 48, 61]

  results = []
  for n in layer_configs:
    print(f"\nTesting {n} layers...")
    try:
      result = test_layers(n)
      results.append(result)
      status = "BROKEN (EOS)" if result["is_eos"] else "OK"
      print(f"  Layers={n}: {status}, argmax={result['argmax']}, top5={result['top5_indices']}")
    except Exception as e:
      print(f"  Layers={n}: ERROR - {e}")
      results.append({"num_layers": n, "error": str(e)})

    # Clear CUDA cache between tests
    torch.cuda.empty_cache()

  print("\n" + "=" * 80)
  print("Summary")
  print("=" * 80)
  for r in results:
    if "error" in r:
      print(f"  {r['num_layers']} layers: ERROR")
    else:
      status = "BROKEN" if r["is_eos"] else "OK"
      print(f"  {r['num_layers']} layers: {status} (argmax={r['argmax']})")

  dist.destroy_process_group()


if __name__ == "__main__":
  main()
