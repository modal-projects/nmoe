# SPDX-License-Identifier: Apache-2.0
"""Compare single-GPU vs multi-GPU at each module step."""

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
import torch.distributed as dist


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  # Use 1 layer for detailed debugging
  cfg = ModelConfig(num_layers=1, num_dense_layers=1)

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

  if rank == 0:
    print("=" * 70)
    print(f"Single-GPU vs Multi-GPU Module Comparison (world_size={world_size})")
    print("=" * 70)

  # Simple test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  # Ensure freqs are computed
  model._ensure_freqs(device)
  freqs = model.freqs_cis.index_select(0, positions.view(-1).to(torch.int64)).view(B, S, -1)

  def gather_and_print(name, tensor):
    """Gather stats from all ranks and print on rank 0."""
    # For column-parallel outputs, we need to concatenate
    # For row-parallel inputs, all ranks have the same tensor

    mean = tensor.float().mean()
    amax = tensor.float().abs().max()

    all_means = [torch.empty_like(mean) for _ in range(world_size)]
    all_amax = [torch.empty_like(amax) for _ in range(world_size)]

    dist.all_gather(all_means, mean.contiguous())
    dist.all_gather(all_amax, amax.contiguous())

    if rank == 0:
      mean_val = sum(float(m) for m in all_means) / world_size
      amax_val = max(float(a) for a in all_amax)
      print(f"  {name}: mean={mean_val:.6f}, amax={amax_val:.4f}")

  # ===== EMBEDDING =====
  if rank == 0:
    print("\n--- Embedding ---")

  with torch.no_grad():
    x = model.embed(input_ids)
  gather_and_print("embed_out", x)

  # ===== ATTN NORM =====
  if rank == 0:
    print("\n--- Attention Norm ---")

  with torch.no_grad():
    x_norm = model.layers[0].attn_norm(x)
  gather_and_print("attn_norm", x_norm)

  # ===== ATTENTION =====
  if rank == 0:
    print("\n--- Attention ---")

  attn = model.layers[0].attn

  with torch.no_grad():
    # Set up cache for attention
    num_blocks = 1
    kv_cache = torch.zeros(num_blocks, 64, 1, 656, dtype=torch.uint8, device=device)
    idx_k_cache = torch.zeros(num_blocks, 64, cfg.dsa_idx_dim, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device=device)
    out_loc = torch.arange(S, dtype=torch.int32, device=device).unsqueeze(0)

    attn_out = attn(
      x_norm, freqs,
      kv_cache=kv_cache, idx_k_cache=idx_k_cache,
      block_table=block_table, cache_seqlens=cache_seqlens,
      cache_seqlens_cpu=[S], out_loc=out_loc, positions=positions,
    )
  gather_and_print("attn_out", attn_out)

  # ===== RESIDUAL =====
  with torch.no_grad():
    x_after_attn = x + attn_out
  gather_and_print("x_after_attn", x_after_attn)

  # ===== FFN NORM =====
  if rank == 0:
    print("\n--- FFN Norm ---")

  with torch.no_grad():
    x_ffn_norm = model.layers[0].ffn_norm(x_after_attn)
  gather_and_print("x_ffn_norm", x_ffn_norm)

  # ===== FFN W1 (gate) =====
  if rank == 0:
    print("\n--- FFN ---")

  ffn = model.layers[0].ffn
  with torch.no_grad():
    w1_out = ffn.w1(x_ffn_norm)
    w3_out = ffn.w3(x_ffn_norm)
  gather_and_print("ffn.w1 (gate)", w1_out)
  gather_and_print("ffn.w3 (up)", w3_out)

  # ===== GATED OUTPUT =====
  import torch.nn.functional as F
  with torch.no_grad():
    gate_up = F.silu(w1_out.float()) * w3_out.float()
  gather_and_print("gate*up", gate_up)

  # ===== FFN W2 (down) =====
  with torch.no_grad():
    # Need to convert back and apply w2
    # w2 is row-parallel, so output is already reduced
    ffn_out = ffn.w2(gate_up.to(x_ffn_norm.dtype))
  gather_and_print("ffn_out", ffn_out)

  # ===== FINAL OUTPUT =====
  if rank == 0:
    print("\n--- Final Output ---")

  with torch.no_grad():
    x_final = x_after_attn + ffn_out
    x_normed = model.norm(x_final)
    logits = model.lm_head(x_normed.float())

    if world_size > 1:
      all_logits = [torch.empty_like(logits) for _ in range(world_size)]
      dist.all_gather(all_logits, logits)
      logits = torch.cat(all_logits, dim=-1)

  gather_and_print("x_final", x_final)
  gather_and_print("logits", logits)

  if rank == 0:
    argmax = logits[0, -1, :].argmax().item()
    top5 = logits[0, -1, :].topk(5)
    print(f"\n  Argmax: {argmax}")
    print(f"  Top 5: {top5.indices.tolist()}")
    print(f"  Top 5 logits: {[f'{v:.2f}' for v in top5.values.tolist()]}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
