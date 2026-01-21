# SPDX-License-Identifier: Apache-2.0
"""Compare MoE routing logic against reference implementation."""

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


def reference_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,  # Raw logits from gate linear
    correction_bias: torch.Tensor,  # Gate bias
    topk: int = 8,
    num_expert_group: int = 8,
    topk_group: int = 4,
    route_scale: float = 2.5,
    renormalize: bool = True,
):
  """Reference grouped top-k routing (matches SGLang biased_grouped_topk_impl)."""
  num_tokens = hidden_states.shape[0]
  num_experts = gating_output.shape[-1]
  experts_per_group = num_experts // num_expert_group

  # Step 1: Sigmoid scores
  scores = gating_output.sigmoid()

  # Step 2: Add bias for selection (NOT for weights)
  scores_for_choice = scores + correction_bias.unsqueeze(0)

  # Step 3: Group selection - sum of top-2 experts per group
  scores_grouped = scores_for_choice.view(num_tokens, num_expert_group, experts_per_group)
  group_scores = scores_grouped.topk(2, dim=-1)[0].sum(dim=-1)  # [num_tokens, num_groups]

  # Step 4: Select top-k groups
  group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]  # [num_tokens, topk_group]

  # Step 5: Create mask for selected groups
  group_mask = torch.zeros_like(group_scores)
  group_mask.scatter_(1, group_idx, 1)
  score_mask = (
    group_mask.unsqueeze(-1)
    .expand(num_tokens, num_expert_group, experts_per_group)
    .reshape(num_tokens, num_experts)
  )

  # Step 6: Mask out non-selected groups (use -inf for topk)
  tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

  # Step 7: Select top-k experts
  _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

  # Step 8: Get weights from ORIGINAL scores (not biased)
  topk_weights = scores.gather(1, topk_ids)

  # Step 9: Renormalize
  if renormalize:
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

  return topk_weights, topk_ids


def our_grouped_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int = 8,
    num_groups: int = 8,
    topk_groups: int = 4,
    route_scale: float = 2.5,
):
  """Our routing implementation (from model.py MoEGate)."""
  num_experts = weight.shape[0]
  experts_per_group = num_experts // num_groups

  scores = F.linear(x, weight).sigmoid()
  original_scores = scores
  scores_for_choice = scores + bias

  # Group selection
  scores_for_choice = scores_for_choice.view(-1, num_groups, experts_per_group)
  group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
  group_idx = group_scores.topk(topk_groups, dim=-1)[1]
  mask = torch.ones_like(group_scores, dtype=torch.bool).scatter_(1, group_idx, False)
  scores_for_choice = scores_for_choice.masked_fill(mask.unsqueeze(-1), float("-inf"))
  scores_for_choice = scores_for_choice.flatten(1)

  indices = scores_for_choice.topk(topk, dim=-1)[1]
  weights = original_scores.gather(1, indices)
  weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
  weights = weights * route_scale
  return weights, indices


def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint
  from deep_ep import Buffer

  init_distributed(rank, world_size)

  # Load model to get real gate weights
  cfg = ModelConfig(num_layers=4, num_dense_layers=3)  # Layer 3 is first MoE

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

  ckpt_path = "/data/models/DeepSeek-V3.2-Speciale"
  load_checkpoint(model, ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
  dist.barrier()

  if rank == 0:
    print("=" * 70)
    print("MoE Routing Comparison Test")
    print("=" * 70)

  # Get layer 3's MoE gate
  moe_layer = model.layers[3].ffn
  gate = moe_layer.gate

  # Test input
  torch.manual_seed(42)
  x = torch.randn(4, cfg.hidden_size, device=device, dtype=torch.bfloat16)

  with torch.no_grad():
    # Our implementation
    our_weights, our_indices = gate(x)

    # Reference implementation (using same gate weights)
    gating_output = F.linear(x, gate.weight)
    ref_weights, ref_indices = reference_grouped_topk(
      x,
      gating_output,
      gate.bias,
      topk=cfg.num_experts_per_tok,
      num_expert_group=cfg.num_expert_groups,
      topk_group=cfg.num_limited_groups,
      route_scale=cfg.route_scale,
    )

  if rank == 0:
    print(f"\n=== Token 0 Expert Selection ===")
    print(f"Our indices:  {our_indices[0].tolist()}")
    print(f"Ref indices:  {ref_indices[0].tolist()}")
    print(f"Indices match: {set(our_indices[0].tolist()) == set(ref_indices[0].tolist())}")

    print(f"\n=== Token 0 Weights ===")
    print(f"Our weights:  {our_weights[0].tolist()}")
    print(f"Ref weights (scaled): {(ref_weights[0] * cfg.route_scale).tolist()}")

    # Sort both by index for comparison
    our_sorted = sorted(zip(our_indices[0].tolist(), our_weights[0].tolist()))
    ref_sorted = sorted(zip(ref_indices[0].tolist(), (ref_weights[0] * cfg.route_scale).tolist()))
    print(f"\nSorted by expert ID:")
    print(f"Our:  {our_sorted}")
    print(f"Ref:  {ref_sorted}")

    # Check all tokens
    print(f"\n=== All Tokens Summary ===")
    for t in range(x.shape[0]):
      our_set = set(our_indices[t].tolist())
      ref_set = set(ref_indices[t].tolist())
      match = our_set == ref_set
      print(f"Token {t}: {'✓' if match else '✗'} (our: {sorted(our_set)[:4]}..., ref: {sorted(ref_set)[:4]}...)")

    # Detailed weight comparison
    print(f"\n=== Weight Statistics ===")
    our_weights_sum = our_weights.sum(dim=-1)
    ref_weights_sum = ref_weights.sum(dim=-1)
    print(f"Our weights sum: {our_weights_sum.tolist()}")
    print(f"Ref weights sum (before scale): {ref_weights_sum.tolist()}")
    print(f"Ref weights sum (after scale): {(ref_weights_sum * cfg.route_scale).tolist()}")

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
