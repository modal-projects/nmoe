# SPDX-License-Identifier: Apache-2.0
"""DSA indexer correctness vs DeepSeek reference algorithm (naive torch).

This test directly targets the most common "everything is finite but output is
garbage" failure mode: wrong sparse indices -> wrong attention -> bad logits.

We compute top-k indices two ways:
1) Our production path inside `DsaFlashMla` (capture `indices` passed to FlashMLA)
2) A faithful torch implementation of the reference indexer algorithm using:
   - non-interleaved RoPE for indexer
   - Hadamard transform
   - FP8 act_quant with UE8M0 rounding (scale_fmt != None)
   - fp8_index scoring semantics (relu(dot(fp8_k, fp8_q)) * q_s, sum heads, * k_s)

Run (in debug pod):
  NMOE_MODEL_PATH=/data/models/DeepSeek-V3.2-Speciale python -m nmoe.serve.test_correctness_indexer_vs_ref_fp8
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from nmoe.serve.ckpt import load_checkpoint
from nmoe.serve.model import DsaFlashMla, ModelConfig, apply_rotary_emb, precompute_freqs_cis, rotate_activation
from nmoe.serve.ref_kernel_torch import act_quant as ref_act_quant, fp8_index as ref_fp8_index


def _is_sm100() -> bool:
  if not torch.cuda.is_available():
    return False
  major, _ = torch.cuda.get_device_capability()
  return major == 10


class TestIndexerVsReferenceFP8(unittest.TestCase):
  def test_indexer_topk_matches_reference_algorithm(self) -> None:
    if not _is_sm100():
      raise unittest.SkipTest("requires SM100 (B200)")

    model_path = os.environ.get("NMOE_MODEL_PATH", "").strip()
    if not model_path:
      raise unittest.SkipTest("NMOE_MODEL_PATH is required")

    torch.cuda.set_device(0)
    device = torch.device("cuda")

    # Keep this small: we only need one layer, short context.
    cfg = ModelConfig(num_layers=1, num_dense_layers=1, dsa_topk=2048)
    attn = DsaFlashMla(cfg, layer_idx=0).to(device).eval()

    class _Wrap(torch.nn.Module):
      def __init__(self, attn_mod: torch.nn.Module) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].attn = attn_mod

    wrap = _Wrap(attn).to(device).eval()

    # Load only layer 0 weights (mapped into layers.0.attn.*).
    missing, unexpected = load_checkpoint(wrap, model_path, rank=0, world_size=1, cfg=cfg, strict=False)
    if missing:
      raise RuntimeError(f"missing keys: {sorted(missing)[:10]}")
    _ = unexpected

    # Inputs.
    B, S = 1, 32
    x = (torch.randn(B, S, cfg.hidden_size, device=device, dtype=torch.bfloat16) / 10).contiguous()
    positions = torch.arange(S, device=device, dtype=torch.int64).unsqueeze(0)
    freqs_all = precompute_freqs_cis(cfg, device=device)
    freqs = freqs_all.index_select(0, positions.view(-1)).view(B, S, -1)
    out_loc = torch.arange(S, device=device, dtype=torch.int32).unsqueeze(0)

    # Caches (single block is enough for S<=64).
    kv_cache = torch.zeros(1, 64, 1, 656, device=device, dtype=torch.uint8)
    idx_k_cache = torch.zeros(1, 64, cfg.dsa_idx_dim, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros((B, 1), device=device, dtype=torch.int32)  # block 0
    cache_seqlens = torch.tensor([S], device=device, dtype=torch.int32)
    cache_seqlens_cpu = [S]

    captured = {}

    def _fake_get_mla_metadata(**kwargs):
      md = torch.empty((1, 1), device=device, dtype=torch.int32)
      return md, torch.tensor([1], device=device, dtype=torch.int32)

    def _fake_flash_mla_with_kvcache(
      q,
      k_cache,
      block_table_,
      cache_seqlens_,
      head_dim_v,
      tile_scheduler_metadata,
      num_splits,
      *,
      softmax_scale,
      causal,
      is_fp8_kvcache,
      indices,
    ):
      captured["indices"] = indices.detach().clone()
      # Return zeros with correct shapes.
      out = torch.zeros((B, S, attn.num_local_heads, head_dim_v), device=device, dtype=q.dtype)
      lse = torch.zeros((B, attn.num_local_heads, S), device=device, dtype=torch.float32)
      return out, lse

    import flash_mla  # type: ignore
    with patch.object(flash_mla, "get_mla_metadata", new=_fake_get_mla_metadata):
      with patch.object(flash_mla, "flash_mla_with_kvcache", new=_fake_flash_mla_with_kvcache):
        _ = attn(
          x,
          freqs,
          kv_cache=kv_cache,
          idx_k_cache=idx_k_cache,
          block_table=block_table,
          cache_seqlens=cache_seqlens,
          cache_seqlens_cpu=cache_seqlens_cpu,
          out_loc=out_loc,
          positions=positions,
        )

    ours = captured.get("indices")
    self.assertIsNotNone(ours, "failed to capture FlashMLA indices")
    ours = ours  # [B,S,topk] physical (block*64+off); here block=0 => logical ids.

    # === Reference algorithm (naive torch) ===
    # MLA shared latent.
    q_latent = attn.q_norm(attn.wq_a(x))  # [B,S,q_lora_rank]

    # Indexer Q: [B,S,H,D]
    q = attn.wq_idx(q_latent).view(B, S, attn.n_idx_heads, attn.idx_dim)
    q_pe, q_nope = torch.split(q, [cfg.qk_rope_head_dim, attn.idx_dim - cfg.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_emb(q_pe, freqs, interleaved=False)
    q = torch.cat([q_pe, q_nope], dim=-1)

    # Indexer K: [B,S,D]
    k = attn.k_norm(attn.wk_idx(x))
    k_pe, k_nope = torch.split(k, [cfg.qk_rope_head_dim, attn.idx_dim - cfg.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs, interleaved=False).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    # Hadamard.
    q = rotate_activation(q.to(torch.bfloat16))
    k = rotate_activation(k.to(torch.bfloat16))

    # Quantize (UE8M0 because scale_fmt != None).
    q_fp8, q_scale = ref_act_quant(q.contiguous(), 128, scale_fmt="ue8m0")
    k_fp8, k_scale = ref_act_quant(k.contiguous(), 128, scale_fmt="ue8m0")

    # weights = weights_proj(x.float()) * H^-0.5
    weights = attn.w_idx(x.float()).view(B, S, attn.n_idx_heads) * (attn.n_idx_heads ** -0.5)
    # weights = weights.unsqueeze(-1) * q_scale * softmax_scale
    softmax_scale = (attn.idx_dim**-0.5)
    q_s = weights.unsqueeze(-1) * q_scale * softmax_scale

    # Reference uses causal mask in prefill.
    mask = torch.full((S, S), float("-inf"), device=device).triu_(1)
    score = ref_fp8_index(q_fp8.contiguous(), q_s.contiguous(), k_fp8.contiguous(), k_scale.contiguous())
    score = score + mask  # [B,S,S]
    topk = score.topk(min(attn.topk, S), dim=-1).indices.to(torch.int32)  # logical indices

    # Compare: since ctx_len == S and block_table==0, our physical ids == logical ids.
    ours_top = ours[:, :, :S].to(torch.int32)

    # Strongest signal: top-1 should match for most query positions.
    ref_top1 = topk[:, :, 0]
    ours_top1 = ours_top[:, :, 0]
    top1_match = (ref_top1 == ours_top1).float().mean().item()
    self.assertGreaterEqual(top1_match, 0.75, f"top1 match too low: {top1_match:.2f}")

    # Also require substantial overlap in top-8 for later tokens.
    k8 = min(8, S)
    ref8 = topk[:, :, :k8]
    ours8 = ours_top[:, :, :k8]
    overlap = 0.0
    for i in range(S):
      a = set(int(x) for x in ref8[0, i].tolist())
      b = set(int(x) for x in ours8[0, i].tolist())
      overlap += len(a.intersection(b)) / float(k8)
    overlap /= float(S)
    self.assertGreaterEqual(overlap, 0.6, f"top8 overlap too low: {overlap:.2f}")


def main() -> int:
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestIndexerVsReferenceFP8)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  raise SystemExit(main())
