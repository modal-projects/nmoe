# SPDX-License-Identifier: Apache-2.0
"""Compare MoE output between single-GPU and multi-GPU paths."""

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

  from nmoe.serve.model import ModelConfig, MoE, MoEGate, MLP, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer
  from safetensors.torch import safe_open
  from glob import glob
  import re
  from collections import defaultdict

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, world_size)

  cfg = ModelConfig()

  # Create MoE layer
  hidden_bytes = cfg.hidden_size * 2
  dispatch_config = Buffer.get_dispatch_config(world_size)
  combine_config = Buffer.get_combine_config(world_size)
  num_nvl_bytes = max(
    dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
    combine_config.get_nvl_buffer_size_hint(hidden_bytes, world_size),
  )

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=0)
  moe = MoE(cfg, buffer).to(device)
  moe.eval()

  # Load layer 3 weights manually
  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  files = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))

  n_local_experts = cfg.num_experts // world_size
  expert_start = rank * n_local_experts
  expert_end = expert_start + n_local_experts

  expert_weights = defaultdict(dict)
  expert_scales = defaultdict(dict)
  gate_weight = None
  gate_bias = None

  for fpath in files:
    with safe_open(fpath, framework="pt", device="cpu") as f:
      for name in f.keys():
        if "layers.3.mlp.gate.weight" in name:
          gate_weight = f.get_tensor(name)
        if "layers.3.mlp.gate.e_score_correction_bias" in name:
          gate_bias = f.get_tensor(name)

        # Expert weights for layer 3
        match = re.search(r"layers\.3\.mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(weight_scale_inv|weight)", name)
        if match:
          expert_idx = int(match.group(1))
          proj = match.group(2)
          suffix = match.group(3)
          key = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]

          # Only load experts for this rank
          if expert_idx >= expert_start and expert_idx < expert_end:
            local_idx = expert_idx - expert_start
            if suffix == "weight":
              expert_weights[local_idx][key] = f.get_tensor(name)
            else:
              expert_scales[local_idx][key] = f.get_tensor(name)

        # Shared experts
        if "layers.3.mlp.shared_experts.gate_proj.weight" in name:
          moe.shared.w1.weight.data = f.get_tensor(name).to(device)
        if "layers.3.mlp.shared_experts.down_proj.weight" in name:
          moe.shared.w2.weight.data = f.get_tensor(name).to(device)
        if "layers.3.mlp.shared_experts.up_proj.weight" in name:
          moe.shared.w3.weight.data = f.get_tensor(name).to(device)

  # Load gate weights
  moe.gate.weight.data = gate_weight.to(device).to(torch.bfloat16)
  if gate_bias is not None and moe.gate.bias is not None:
    moe.gate.bias.data = gate_bias.to(device).to(torch.float32)

  # Stack expert weights
  w1_list, w2_list, w3_list = [], [], []
  w1_scale_list, w2_scale_list, w3_scale_list = [], [], []

  for local_idx in range(n_local_experts):
    w1_list.append(expert_weights[local_idx]["w1"])
    w2_list.append(expert_weights[local_idx]["w2"])
    w3_list.append(expert_weights[local_idx]["w3"])
    if local_idx in expert_scales:
      w1_scale_list.append(expert_scales[local_idx].get("w1"))
      w2_scale_list.append(expert_scales[local_idx].get("w2"))
      w3_scale_list.append(expert_scales[local_idx].get("w3"))

  w1_stacked = torch.stack(w1_list, dim=0)
  w2_stacked = torch.stack(w2_list, dim=0)
  w3_stacked = torch.stack(w3_list, dim=0)
  w13_stacked = torch.cat([w1_stacked, w3_stacked], dim=1)

  moe.w13.data = w13_stacked.to(device)
  moe.w2.data = w2_stacked.to(device)

  if w1_scale_list and w1_scale_list[0] is not None:
    w1_scale_stacked = torch.stack(w1_scale_list, dim=0)
    w2_scale_stacked = torch.stack(w2_scale_list, dim=0)
    w3_scale_stacked = torch.stack(w3_scale_list, dim=0)
    w13_scale_stacked = torch.cat([w1_scale_stacked, w3_scale_stacked], dim=1)

    # UE8M0 conversion
    w13_scale_stacked = torch.pow(2.0, torch.ceil(torch.log2(w13_scale_stacked.abs().clamp(min=1e-12))))
    w2_scale_stacked = torch.pow(2.0, torch.ceil(torch.log2(w2_scale_stacked.abs().clamp(min=1e-12))))

    moe.w13_scale.data = w13_scale_stacked.to(device)
    moe.w2_scale.data = w2_scale_stacked.to(device)

  dist.barrier()

  # Create test input
  torch.manual_seed(42)
  x = torch.randn(5, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  # Broadcast x from rank 0 to ensure identical input
  dist.broadcast(x, src=0)

  if rank == 0:
    print("=" * 60)
    print("MoE Single vs Multi GPU Path Comparison")
    print("=" * 60)
    print(f"Input: shape={x.shape}, mean={x.mean():.6f}")

  # Run MoE
  with torch.no_grad():
    out = moe(x.unsqueeze(0)).squeeze(0)

  # Gather outputs
  out_mean = out.float().mean()
  out_sample = out[0, :20].float()  # First 20 values of first token

  out_means = [torch.empty_like(out_mean) for _ in range(world_size)]
  dist.all_gather(out_means, out_mean.contiguous())

  out_samples = [torch.empty_like(out_sample) for _ in range(world_size)]
  dist.all_gather(out_samples, out_sample.contiguous())

  if rank == 0:
    print(f"\nMulti-GPU MoE output (world_size={world_size}):")
    print(f"  Mean: {out_mean:.6f}")
    print(f"  Sample (token 0, first 20 values): {[f'{v:.4f}' for v in out_sample.tolist()]}")

    # Now we need to compare with single-GPU
    # For single-GPU, we would need to load ALL experts
    print("\n--- To compare with single-GPU: ---")
    print("Run with world_size=1 separately and compare output values")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
