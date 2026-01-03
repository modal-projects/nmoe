# SPDX-License-Identifier: Apache-2.0
"""Debug MoE multi-GPU divergence."""

import os
from pathlib import Path

def _maybe_set_cutlass_path():
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
import torch.nn.functional as F
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed, MoEGate
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  cfg = ModelConfig(num_layers=4, num_dense_layers=3)

  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  input_ids = torch.tensor([[671, 6102, 294, 8760, 344]], device=device)
  S = 5
  positions = torch.arange(S, device=device).unsqueeze(0)
  kv_caches = [torch.zeros(1, 64, 1, 656, dtype=torch.uint8, device=device) for _ in range(cfg.num_layers)]
  idx_k_caches = [torch.zeros(1, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device) for _ in range(cfg.num_layers)]
  block_table = torch.arange(1, dtype=torch.int32, device=device).unsqueeze(0)
  cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
  out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

  # Get the MoE layer (layer 3)
  moe = model.layers[3].ffn

  # Ensure freqs_cis is initialized
  model._ensure_freqs(device)

  # Run up to layer 2 to get inputs to layer 3
  with torch.no_grad():
    freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(1, S, -1)
    x = model.embed(input_ids)

    for i in range(3):
      x = model.layers[i](
        x, freqs,
        kv_cache=kv_caches[i],
        idx_k_cache=idx_k_caches[i],
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[S],
        out_loc=out_loc,
        positions=positions,
      )

    # Now we have x ready for layer 3
    x_norm = model.layers[3].ffn_norm(x + model.layers[3].attn(
      model.layers[3].attn_norm(x), freqs,
      kv_cache=kv_caches[3], idx_k_cache=idx_k_caches[3],
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc, positions=positions,
    ))

  # Check input to MoE
  if rank == 0:
    print("=" * 60)
    print("MoE Debug")
    print("=" * 60)

  x_moe = x_norm.view(-1, cfg.hidden_size)  # [T, hidden]
  T = x_moe.size(0)

  # Check 1: x_moe should be identical across ranks
  x_mean = x_moe.float().mean()
  x_means = [torch.empty_like(x_mean) for _ in range(world_size)]
  dist.all_gather(x_means, x_mean.contiguous())
  if rank == 0:
    print(f"\n1. MoE input (x_moe) mean across ranks: {[f'{m:.6f}' for m in x_means]}")
    print(f"   All same: {all(abs(float(m) - float(x_means[0])) < 1e-4 for m in x_means)}")

  # Check 2: Gate weights should be identical (replicated)
  gate_mean = moe.gate.weight.float().mean()
  gate_means = [torch.empty_like(gate_mean) for _ in range(world_size)]
  dist.all_gather(gate_means, gate_mean.contiguous())
  if rank == 0:
    print(f"\n2. Gate weight mean across ranks: {[f'{m:.6f}' for m in gate_means]}")
    print(f"   All same: {all(abs(float(m) - float(gate_means[0])) < 1e-4 for m in gate_means)}")

  # Check 3: Gate output (scores and indices) should be identical
  with torch.no_grad():
    weights, indices = moe.gate(x_moe)

  indices_first = indices[0]  # First token's expert indices
  all_indices = [torch.empty_like(indices_first) for _ in range(world_size)]
  dist.all_gather(all_indices, indices_first.contiguous())
  if rank == 0:
    print(f"\n3. Expert indices for first token:")
    for r, idx in enumerate(all_indices):
      print(f"   Rank {r}: {idx.tolist()}")
    print(f"   All same: {all(torch.equal(idx, all_indices[0]) for idx in all_indices)}")

  weights_first = weights[0]
  all_weights = [torch.empty_like(weights_first) for _ in range(world_size)]
  dist.all_gather(all_weights, weights_first.contiguous())
  if rank == 0:
    print(f"\n4. Expert weights for first token:")
    for r, w in enumerate(all_weights):
      print(f"   Rank {r}: {[f'{v:.4f}' for v in w.tolist()]}")

  # Check 4: Run MoE forward and compare outputs
  with torch.no_grad():
    moe_out = moe(x_norm.view(-1, cfg.hidden_size).unsqueeze(0).view_as(x_norm))

  moe_out_flat = moe_out.view(-1, cfg.hidden_size)
  moe_out_mean = moe_out_flat.float().mean()
  moe_out_means = [torch.empty_like(moe_out_mean) for _ in range(world_size)]
  dist.all_gather(moe_out_means, moe_out_mean.contiguous())
  if rank == 0:
    print(f"\n5. MoE output mean across ranks: {[f'{m:.6f}' for m in moe_out_means]}")
    print(f"   All same: {all(abs(float(m) - float(moe_out_means[0])) < 1e-4 for m in moe_out_means)}")

  # Check 5: Compare first few elements of MoE output
  moe_out_sample = moe_out_flat[0, :10].float()
  all_samples = [torch.empty_like(moe_out_sample) for _ in range(world_size)]
  dist.all_gather(all_samples, moe_out_sample.contiguous())
  if rank == 0:
    print(f"\n6. MoE output first 10 values for token 0:")
    for r, s in enumerate(all_samples):
      print(f"   Rank {r}: {[f'{v:.4f}' for v in s.tolist()]}")
    if not torch.equal(all_samples[0], all_samples[1]):
      diff = (all_samples[0] - all_samples[1]).abs()
      print(f"   Diff R0-R1: max={diff.max():.6f}, mean={diff.mean():.6f}")

  # Check 6: Shared experts output
  with torch.no_grad():
    shared_out = moe.shared(x_moe)

  shared_mean = shared_out.float().mean()
  shared_means = [torch.empty_like(shared_mean) for _ in range(world_size)]
  dist.all_gather(shared_means, shared_mean.contiguous())
  if rank == 0:
    print(f"\n7. Shared experts output mean across ranks: {[f'{m:.6f}' for m in shared_means]}")
    print(f"   All same: {all(abs(float(m) - float(shared_means[0])) < 1e-4 for m in shared_means)}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
