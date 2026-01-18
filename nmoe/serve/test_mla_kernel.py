# SPDX-License-Identifier: Apache-2.0
"""Test CuTeDSL MLA kernel integration."""

from __future__ import annotations

import math
import unittest

import torch
import cutlass
import cutlass.torch as cutlass_torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


def _create_mla_tensor(shape: tuple, dtype: torch.dtype, device: torch.device, is_paged: bool = False) -> torch.Tensor:
    """Create tensor with correct layout for CuTeDSL MLA kernel.

    Following grouped.py pattern: create with base layout then permute to get
    the required stride pattern (leading_dim=1 needs stride=1).

    For MLA kernel:
    - Q tensors: shape (H, D, B) with D (dim 1) having stride=1
    - KV cache: shape (page_size, D, pages) with D (dim 1) having stride=1
    """
    # Generic helper: create tensor with stride[leading_dim]=1
    def make_with_leading_dim(target_shape: tuple, leading_dim: int) -> torch.Tensor:
        """Create tensor where stride[leading_dim]=1."""
        # Reorder so leading_dim is last (has stride=1 in row-major), then permute back
        perm = list(range(len(target_shape)))
        perm.remove(leading_dim)
        perm.append(leading_dim)
        inv_perm = [perm.index(i) for i in range(len(target_shape))]
        reordered_shape = tuple(target_shape[p] for p in perm)
        t = torch.randn(reordered_shape, dtype=dtype, device=device) * 0.1
        return t.permute(*inv_perm)

    if is_paged:
        # (page_size, D, pages) with stride[1]=1
        return make_with_leading_dim(shape, 1)
    else:
        # (H, D, B) with stride[1]=1
        return make_with_leading_dim(shape, 1)


class TestMlaKernel(unittest.TestCase):
    """Test the _CompiledMlaKernel wrapper."""

    @unittest.skipUnless(_is_sm100(), "Requires SM100 (B200)")
    def test_kernel_compiles_and_runs(self):
        """Test that the kernel compiles and produces valid output."""
        from nmoe.serve.mla import _CompiledMlaKernel

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        # Config
        B = 2  # batch size
        H = 128  # num heads
        L = 512  # latent dim
        R = 64   # rope dim
        page_size = 64
        num_pages = 4
        seq_len = page_size * num_pages

        # Create kernel
        kernel = _CompiledMlaKernel(
            num_heads=H,
            max_batch=B,
            max_seq_len=seq_len,
            page_size=page_size,
            device=device,
            c_latent=_create_mla_tensor((page_size, L, num_pages), torch.bfloat16, device, is_paged=True),
            c_rope=_create_mla_tensor((page_size, R, num_pages), torch.bfloat16, device, is_paged=True),
        )

        # Create inputs with correct layout
        q_latent = _create_mla_tensor((H, L, B), torch.bfloat16, device)
        q_rope = _create_mla_tensor((H, R, B), torch.bfloat16, device)

        # Page table: [num_pages, B] with stride[0]=1
        # Create (B, num_pages) then permute (without contiguous!)
        page_table = torch.empty(B, num_pages, dtype=torch.int32, device=device)
        for i in range(num_pages):
            page_table[:, i] = i
        page_table = page_table.permute(1, 0)  # (num_pages, B) with stride (1, num_pages)

        # Cache sequence lengths
        cache_seqs = torch.full((B,), seq_len, dtype=torch.int32, device=device)

        # Softmax scale
        softmax_scale = 1.0 / math.sqrt(L + R)

        # Run kernel
        out, lse = kernel(q_latent, q_rope, page_table, cache_seqs, softmax_scale)

        # Check outputs
        self.assertEqual(out.shape, (H, L, B))
        self.assertEqual(lse.shape, (H, B))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(lse.dtype, torch.float32)

        # Check no NaN/Inf
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")
        self.assertFalse(torch.isnan(lse).any(), "LSE contains NaN")

        print(f"✓ Kernel output shape: {out.shape}")
        print(f"✓ Kernel output stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
        print(f"✓ LSE stats: min={lse.min():.4f}, max={lse.max():.4f}")

    @unittest.skipUnless(_is_sm100(), "Requires SM100 (B200)")
    def test_kernel_different_configs(self):
        """Test kernel works with different configurations (separate kernels per config)."""
        from nmoe.serve.mla import _CompiledMlaKernel

        device = torch.device("cuda:0")
        H, L, R = 128, 512, 64
        page_size = 64

        # Test different batch/page configurations (each requires its own kernel)
        for B, num_pages in [(2, 2), (4, 4), (8, 8)]:
            kernel = _CompiledMlaKernel(
                num_heads=H, max_batch=B, max_seq_len=page_size * num_pages,
                page_size=page_size, device=device,
                c_latent=_create_mla_tensor((page_size, L, num_pages), torch.bfloat16, device, is_paged=True),
                c_rope=_create_mla_tensor((page_size, R, num_pages), torch.bfloat16, device, is_paged=True),
            )

            # Create inputs with correct layout
            q_latent = _create_mla_tensor((H, L, B), torch.bfloat16, device)
            q_rope = _create_mla_tensor((H, R, B), torch.bfloat16, device)

            # Page table with stride[0]=1
            page_table = torch.empty(B, num_pages, dtype=torch.int32, device=device)
            for i in range(num_pages):
                page_table[:, i] = i
            page_table = page_table.permute(1, 0)
            cache_seqs = torch.full((B,), page_size * num_pages, dtype=torch.int32, device=device)

            out, lse = kernel(q_latent, q_rope, page_table, cache_seqs, 1.0 / math.sqrt(576))

            self.assertEqual(out.shape, (H, L, B))
            self.assertFalse(torch.isnan(out).any())
            print(f"✓ B={B}, pages={num_pages}: output shape {out.shape}")


class TestMlaModule(unittest.TestCase):
    """Test the MLA nn.Module (requires model weights - skip if not available)."""

    @unittest.skipUnless(_is_sm100(), "Requires SM100 (B200)")
    def test_mla_forward_shapes(self):
        """Test MLA module forward pass with mock weights."""
        # This test would require mock model config and weights
        # For now, just verify imports work
        from nmoe.serve.mla import MLA, _CompiledMlaKernel
        print("✓ MLA and _CompiledMlaKernel imported successfully")


def main() -> int:
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestMlaKernel))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestMlaModule))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
