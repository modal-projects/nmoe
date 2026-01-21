# SPDX-License-Identifier: Apache-2.0
"""FlashMLA sparse attention output matches a dense torch reference.

This is the most direct test we can run on B200 to catch integration bugs:
packing format, indices semantics, and softmax scaling must produce the same
latent attention output as an explicit attention computation over the selected
keys.
"""

from __future__ import annotations

import math
import unittest

import torch


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 10


def _unpack_kv_cache(kv_cache: torch.Tensor, *, T: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Unpack FlashMLA FP8 KV cache into (latent, rope) BF16 tensors.

  Args:
    kv_cache: [num_blocks, 64, 1, 656] uint8
    T: number of tokens to unpack from the front of the flattened cache

  Returns:
    latent: [T, 512] bf16
    rope:   [T,  64] bf16
  """
  flat = kv_cache.view(-1, 656)[:T]  # [T,656] uint8

  fp8_bytes = flat[:, :512]
  scales_bytes = flat[:, 512:528]
  rope_bytes = flat[:, 528:]

  fp8 = fp8_bytes.contiguous().view(torch.float8_e4m3fn).view(T, 512)
  scales = scales_bytes.contiguous().view(torch.float32).view(T, 4)  # per 128 tile
  rope = rope_bytes.contiguous().view(torch.bfloat16).view(T, 64)

  # Dequant latent per tile: fp8 * scale
  tiles = []
  for t in range(4):
    lo, hi = t * 128, (t + 1) * 128
    tiles.append(fp8[:, lo:hi].float() * scales[:, t : t + 1])
  latent = torch.cat(tiles, dim=1).to(torch.bfloat16)
  return latent, rope


class TestFlashMlaDenseEquivalence(unittest.TestCase):
  def test_flashmla_matches_dense_reference(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires SM100 (B200)")

    torch.cuda.set_device(0)
    device = torch.device("cuda")

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    B = 1
    S_q = 4
    H_q = 128
    H_kv = 1
    D_qk = 576  # latent(512)+rope(64)
    D_v = 512
    topk = 64
    block_size = 64

    # Create a small cache containing T=64 tokens.
    T = 64
    num_blocks = 1
    kv_cache = torch.empty(num_blocks, block_size, H_kv, 656, device=device, dtype=torch.uint8)
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([T], device=device, dtype=torch.int32)

    # Construct latent+rope and pack in the same layout as model.py.
    latent = (torch.randn(T, 512, device=device, dtype=torch.bfloat16) / 10).contiguous()
    rope = (torch.randn(T, 64, device=device, dtype=torch.bfloat16) / 10).contiguous()

    # Pack.
    packed = torch.empty((T, 656), device=device, dtype=torch.uint8)
    latent_f = latent.float()
    scales = []
    q_bytes = []
    for tile in range(4):
      lo, hi = tile * 128, (tile + 1) * 128
      tile_f = latent_f[:, lo:hi]
      sf = tile_f.abs().amax(dim=-1).clamp(min=1e-8) / 448.0
      scales.append(sf)
      tile_q = (tile_f / sf.unsqueeze(-1)).to(torch.float8_e4m3fn)
      q_bytes.append(tile_q.view(torch.uint8))
    packed[:, :512] = torch.cat(q_bytes, dim=1)
    packed[:, 512:528] = torch.stack(scales, dim=1).to(torch.float32).view(torch.uint8).view(T, 16)
    packed[:, 528:] = rope.view(torch.uint8).view(T, 128)
    kv_cache.view(-1, 656)[:T].copy_(packed)

    # Query: random BF16.
    q = (torch.randn(B, S_q, H_q, D_qk, device=device, dtype=torch.bfloat16) / 10).contiguous()

    # Indices: attend to the last 64 tokens (identity physical layout).
    indices = torch.arange(T - topk, T, device=device, dtype=torch.int32).view(1, 1, topk).expand(B, S_q, topk).contiguous()

    metadata, num_splits = get_mla_metadata(
      cache_seqlens=cache_seqlens,
      num_q_tokens_per_head_k=S_q * H_q // H_kv,
      num_heads_k=H_kv,
      num_heads_q=H_q,
      is_fp8_kvcache=True,
      topk=topk,
    )

    out_flash, _lse = flash_mla_with_kvcache(
      q,
      kv_cache,
      block_table,
      cache_seqlens,
      head_dim_v=D_v,
      tile_scheduler_metadata=metadata,
      num_splits=num_splits,
      softmax_scale=1.0 / math.sqrt(D_qk),
      causal=False,
      is_fp8_kvcache=True,
      indices=indices,
    )

    # Dense reference over selected keys.
    latent_u, rope_u = _unpack_kv_cache(kv_cache, T=T)
    k_full = torch.cat([latent_u, rope_u], dim=-1).float()  # [T,576]
    v_full = latent_u.float()  # [T,512]

    # Gather K/V per query from indices.
    idx = indices.to(torch.int64)  # [B,S,topk]
    k_sel = k_full.index_select(0, idx.view(-1)).view(B, S_q, topk, D_qk)
    v_sel = v_full.index_select(0, idx.view(-1)).view(B, S_q, topk, D_v)

    scores = torch.einsum("bshd,bskd->bshk", q.float(), k_sel) * (1.0 / math.sqrt(D_qk))
    probs = scores.softmax(dim=-1)
    out_dense = torch.einsum("bshk,bskc->bshc", probs, v_sel).to(torch.bfloat16)

    # FlashMLA is a different implementation but should be numerically close.
    torch.testing.assert_close(out_flash, out_dense, rtol=5e-2, atol=5e-2)


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFlashMlaDenseEquivalence)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())

