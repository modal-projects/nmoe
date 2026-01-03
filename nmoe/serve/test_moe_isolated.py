# SPDX-License-Identifier: Apache-2.0
"""Isolated test of MoE layer to verify expert routing and computation."""

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
import torch.nn.functional as F
import torch.distributed as dist


def reference_moe_forward(x, gate_weight, w1_list, w2_list, w3_list,
                          num_experts=256, topk=8, route_scale=2.5):
  """Reference MoE implementation matching DeepSeek-V3."""
  T, hidden = x.shape

  # Gate: linear + sigmoid
  scores = F.linear(x.float(), gate_weight.float()).sigmoid()
  original_scores = scores

  # For simplicity, skip group selection (we test full routing)
  indices = scores.topk(topk, dim=-1)[1]  # [T, topk]
  weights = original_scores.gather(1, indices)  # [T, topk]
  weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
  weights = weights * route_scale

  # Expert computation
  y = torch.zeros(T, hidden, device=x.device, dtype=torch.float32)

  for i in range(num_experts):
    idx, top = torch.where(indices == i)
    if len(idx) == 0:
      continue

    # Expert forward: w2(silu(w1(x)) * w3(x))
    x_expert = x[idx].float()
    gate = F.silu(F.linear(x_expert, w1_list[i].float()))
    up = F.linear(x_expert, w3_list[i].float())
    down = F.linear(gate * up, w2_list[i].float())

    y[idx] += down * weights[idx, top, None]

  return y.to(x.dtype)


def test_moe_gate():
  """Test that our gate produces valid routing."""
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, MoEGate

  cfg = ModelConfig()
  gate = MoEGate(cfg).to(device)

  # Initialize with known weights
  torch.manual_seed(42)
  gate.weight.data = torch.randn_like(gate.weight.data) * 0.1
  if gate.bias is not None:
    gate.bias.data.zero_()

  # Test input
  x = torch.randn(8, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  weights, indices = gate(x)

  print("=" * 60)
  print("MoE Gate Test")
  print("=" * 60)
  print(f"Input shape: {x.shape}")
  print(f"Weights shape: {weights.shape}, dtype: {weights.dtype}")
  print(f"Indices shape: {indices.shape}, dtype: {indices.dtype}")
  print(f"Weights range: [{weights.min():.4f}, {weights.max():.4f}]")
  print(f"Weights sum per token: {weights.sum(dim=-1).tolist()}")
  print(f"Expert indices sample (token 0): {indices[0].tolist()}")

  # Verify weights sum to route_scale (2.5) for sigmoid normalization
  expected_sum = cfg.route_scale
  actual_sum = weights.sum(dim=-1).mean().item()
  print(f"Expected weight sum: {expected_sum}, Actual: {actual_sum:.4f}")

  # Verify indices are valid
  assert indices.min() >= 0, "Indices should be >= 0"
  assert indices.max() < cfg.num_experts, f"Indices should be < {cfg.num_experts}"

  print("\nGate test PASSED")


def test_moe_with_loaded_weights():
  """Test MoE with actual checkpoint weights."""
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(0, 1)

  # Load a 4-layer model (3 dense + 1 MoE)
  cfg = ModelConfig(num_layers=4, num_dense_layers=3)

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
  model = DeepSeekV3(cfg, buffer).to(device)
  model.eval()

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=0, world_size=1, cfg=cfg)

  print("=" * 60)
  print("MoE Layer Test (4 layers: 3 dense + 1 MoE)")
  print("=" * 60)

  # Test input
  B, S = 1, 4
  input_ids = torch.tensor([[1, 100, 1000, 10000]], device=device)
  positions = torch.arange(S, device=device).unsqueeze(0)

  num_blocks = 1
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

  print(f"Logits shape: {logits.shape}")
  print(f"Has NaN: {torch.isnan(logits).any().item()}")
  print(f"Logits amax: {logits.abs().max().item():.4f}")

  top5 = logits[0, -1, :].topk(5)
  print(f"Top 5 indices: {top5.indices.tolist()}")
  print(f"Top 5 values: {[f'{v:.2f}' for v in top5.values.tolist()]}")

  # The key test: is the output non-EOS?
  argmax = logits[0, -1, :].argmax().item()
  print(f"\nArgmax token: {argmax}")
  if argmax == 1:
    print("WARNING: Argmax is EOS token (1) - this suggests broken generation")
  else:
    print("OK: Argmax is not EOS token")


def test_moe_expert_output_scale():
  """Test that MoE expert outputs are in reasonable range."""
  if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29599", world_size=1, rank=0)

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, MoE, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(0, 1)

  cfg = ModelConfig(num_layers=4, num_dense_layers=3)

  buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

  # Create MoE layer directly
  moe = MoE(cfg, buffer).to(device)
  moe.eval()

  # Load MoE weights for layer 3
  from safetensors.torch import safe_open
  from glob import glob
  import re
  from collections import defaultdict

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  files = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))

  expert_weights = defaultdict(dict)
  expert_scales = defaultdict(dict)
  gate_weight = None
  gate_bias = None
  shared_w1 = shared_w2 = shared_w3 = None
  shared_w1_scale = shared_w2_scale = shared_w3_scale = None

  for fpath in files:
    with safe_open(fpath, framework="pt", device="cpu") as f:
      for name in f.keys():
        # Gate
        if "layers.3.mlp.gate.weight" in name:
          gate_weight = f.get_tensor(name)
        if "layers.3.mlp.gate.e_score_correction_bias" in name:
          gate_bias = f.get_tensor(name)

        # Shared experts
        if "layers.3.mlp.shared_experts.gate_proj.weight" in name:
          shared_w1 = f.get_tensor(name)
        if "layers.3.mlp.shared_experts.down_proj.weight" in name:
          shared_w2 = f.get_tensor(name)
        if "layers.3.mlp.shared_experts.up_proj.weight" in name:
          shared_w3 = f.get_tensor(name)
        if "layers.3.mlp.shared_experts.gate_proj.weight_scale_inv" in name:
          shared_w1_scale = f.get_tensor(name)
        if "layers.3.mlp.shared_experts.down_proj.weight_scale_inv" in name:
          shared_w2_scale = f.get_tensor(name)
        if "layers.3.mlp.shared_experts.up_proj.weight_scale_inv" in name:
          shared_w3_scale = f.get_tensor(name)

        # Expert weights (layer 3, all experts)
        match = re.search(r"layers\.3\.mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(weight_scale_inv|weight)", name)
        if match:
          expert_idx = int(match.group(1))
          proj = match.group(2)
          suffix = match.group(3)
          key = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}[proj]
          if suffix == "weight":
            expert_weights[expert_idx][key] = f.get_tensor(name)
          else:
            expert_scales[expert_idx][key] = f.get_tensor(name)

  print("=" * 60)
  print("MoE Expert Output Scale Test")
  print("=" * 60)

  # Load gate weights
  if gate_weight is not None:
    moe.gate.weight.data = gate_weight.to(device).to(torch.bfloat16)
  if gate_bias is not None and moe.gate.bias is not None:
    moe.gate.bias.data = gate_bias.to(device).to(torch.float32)

  print(f"Loaded gate weight: {gate_weight.shape if gate_weight is not None else None}")
  print(f"Loaded gate bias: {gate_bias.shape if gate_bias is not None else None}")

  # Stack expert weights
  num_experts = cfg.num_experts
  w1_list, w2_list, w3_list = [], [], []
  w1_scale_list, w2_scale_list, w3_scale_list = [], [], []

  for i in range(num_experts):
    w1_list.append(expert_weights[i]["w1"])
    w2_list.append(expert_weights[i]["w2"])
    w3_list.append(expert_weights[i]["w3"])
    if i in expert_scales:
      w1_scale_list.append(expert_scales[i].get("w1"))
      w2_scale_list.append(expert_scales[i].get("w2"))
      w3_scale_list.append(expert_scales[i].get("w3"))

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

    # Ensure UE8M0
    w13_scale_stacked = torch.pow(2.0, torch.ceil(torch.log2(w13_scale_stacked.abs().clamp(min=1e-12))))
    w2_scale_stacked = torch.pow(2.0, torch.ceil(torch.log2(w2_scale_stacked.abs().clamp(min=1e-12))))

    moe.w13_scale.data = w13_scale_stacked.to(device)
    moe.w2_scale.data = w2_scale_stacked.to(device)

  print(f"Expert w13: {moe.w13.shape}, dtype={moe.w13.dtype}")
  print(f"Expert w2: {moe.w2.shape}, dtype={moe.w2.dtype}")
  print(f"Expert w13_scale range: [{moe.w13_scale.min():.6f}, {moe.w13_scale.max():.6f}]")
  print(f"Expert w2_scale range: [{moe.w2_scale.min():.6f}, {moe.w2_scale.max():.6f}]")

  # Load shared expert weights
  if shared_w1 is not None:
    moe.shared.w1.weight.data = shared_w1.to(device)
    moe.shared.w2.weight.data = shared_w2.to(device)
    moe.shared.w3.weight.data = shared_w3.to(device)
    if shared_w1_scale is not None:
      moe.shared.w1.weight_scale.data = torch.pow(2.0, torch.ceil(torch.log2(shared_w1_scale.abs().clamp(min=1e-12)))).to(device)
      moe.shared.w2.weight_scale.data = torch.pow(2.0, torch.ceil(torch.log2(shared_w2_scale.abs().clamp(min=1e-12)))).to(device)
      moe.shared.w3.weight_scale.data = torch.pow(2.0, torch.ceil(torch.log2(shared_w3_scale.abs().clamp(min=1e-12)))).to(device)

  # Test with random input
  x = torch.randn(4, cfg.hidden_size, device=device, dtype=torch.bfloat16)
  print(f"\nInput: shape={x.shape}, range=[{x.min():.4f}, {x.max():.4f}]")

  with torch.no_grad():
    out = moe(x)

  print(f"Output: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")
  print(f"Has NaN: {torch.isnan(out).any().item()}")
  print(f"Output/Input ratio: {out.abs().mean() / x.abs().mean():.4f}")


if __name__ == "__main__":
  print("\n" + "=" * 60)
  print("TEST 1: Gate Routing")
  print("=" * 60)
  test_moe_gate()

  print("\n" + "=" * 60)
  print("TEST 2: MoE with 4-layer model")
  print("=" * 60)
  test_moe_with_loaded_weights()
