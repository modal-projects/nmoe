from __future__ import annotations

import torch
import torch.distributed as dist


def stripe_moe_tensor_axis0_32_to_striped_inplace(t: torch.Tensor, *, rank: int, world_size: int) -> None:
  """Convert contiguous expert layout -> striped layout, in-place.

  Contract:
  - world_size must be 8
  - t.shape[0] must be 32 (num_local experts)

  Semantics:
  - Before: local expert i corresponds to global expert (rank * 32 + i)
  - After:  local expert i corresponds to global expert (rank + world_size * i)

  Implementation:
  - Each rank sends 4 experts to each destination rank:
      src_idx = [dst, dst+8, dst+16, dst+24]
    and the destination stores those 4 experts at:
      dst_idx = [4*src, 4*src+1, 4*src+2, 4*src+3]
  """
  if world_size != 8:
    raise RuntimeError(f"striped expert placement requires world_size=8 (got {world_size})")
  if t.dim() < 1 or int(t.shape[0]) != 32:
    raise RuntimeError(f"expected axis0 size 32 for MoE experts (got shape={tuple(t.shape)})")
  if not dist.is_initialized():
    raise RuntimeError("torch.distributed must be initialized to stripe experts")

  device = t.device
  dtype = t.dtype
  # New tensor holds the striped ordering. We overwrite `t` only after all sends complete.
  new = torch.empty_like(t)

  # Fill local contribution (source == dest == rank).
  self_idx = torch.tensor([rank, rank + 8, rank + 16, rank + 24], device=device, dtype=torch.int64)
  new[(4 * rank) : (4 * rank + 4)].copy_(t.index_select(0, self_idx))

  recv = torch.empty((4, *t.shape[1:]), device=device, dtype=dtype)
  for step in range(1, world_size):
    dst = (rank + step) % world_size
    src = (rank - step) % world_size

    send_idx = torch.tensor([dst, dst + 8, dst + 16, dst + 24], device=device, dtype=torch.int64)
    send = t.index_select(0, send_idx).contiguous()

    reqs = dist.batch_isend_irecv(
      [
        dist.P2POp(dist.isend, send, dst),
        dist.P2POp(dist.irecv, recv, src),
      ]
    )
    for r in reqs:
      r.wait()

    new[(4 * src) : (4 * src + 4)].copy_(recv)

  # Ensure all ranks finished exchanging before we overwrite model params.
  dist.barrier()
  t.copy_(new)
  dist.barrier()


def stripe_moe_experts_inplace(model: torch.nn.Module, *, rank: int, world_size: int) -> None:
  """Apply striped expert placement to all MoE expert tensors in the model."""
  from nmoe.serve.model import MoE  # local import to avoid module cycles

  for m in model.modules():
    if not isinstance(m, MoE):
      continue
    stripe_moe_tensor_axis0_32_to_striped_inplace(m.w13.data, rank=rank, world_size=world_size)
    stripe_moe_tensor_axis0_32_to_striped_inplace(m.w2.data, rank=rank, world_size=world_size)
    stripe_moe_tensor_axis0_32_to_striped_inplace(m.w13_scale.data, rank=rank, world_size=world_size)
    stripe_moe_tensor_axis0_32_to_striped_inplace(m.w2_scale.data, rank=rank, world_size=world_size)
    # Invalidate packed scale caches (recomputed lazily on first use).
    if hasattr(m, "_w13_scale_ue8m0"):
      m._w13_scale_ue8m0 = torch.empty(0, dtype=torch.int32, device=m.w13_scale.device)  # type: ignore[attr-defined]
    if hasattr(m, "_w2_scale_ue8m0"):
      m._w2_scale_ue8m0 = torch.empty(0, dtype=torch.int32, device=m.w2_scale.device)  # type: ignore[attr-defined]
