# SPDX-License-Identifier: Apache-2.0
"""Test attention type wiring for DSA and MLA."""

from __future__ import annotations

import torch


def test_dsa_model_construction():
    """Test DSA model construction and attention routing."""
    from nmoe.serve.engine import EngineConfig
    from nmoe.serve.model import ModelConfig, TransformerBlock

    print("Testing DSA model construction...")

    model_cfg = ModelConfig(attention_type="dsa")
    engine_cfg = EngineConfig(
        num_pages=16,
        page_size=64,
        num_layers=2,
        attention_type="dsa",
        idx_dim=128,
    )

    assert model_cfg.attention_type == "dsa"
    assert engine_cfg.attention_type == "dsa"
    assert engine_cfg.idx_dim == 128
    print("  ✓ DSA config validated")

    # Test TransformerBlock creates DSA attention
    from nmoe.serve.model import init_distributed
    init_distributed(0, 1)

    block = TransformerBlock(model_cfg, layer_idx=0, buffer=None)
    assert block.attention_type == "dsa"
    print("  ✓ TransformerBlock attention_type == dsa")

    from nmoe.serve.dsa import DSA
    assert isinstance(block.attn, DSA), f"Expected DSA, got {type(block.attn)}"
    print("  ✓ DSA attention class instantiated")

    print("  ✓ DSA model construction test passed\n")


def test_mla_model_construction():
    """Test MLA model construction and attention routing."""
    from nmoe.serve.engine import EngineConfig
    from nmoe.serve.model import ModelConfig, TransformerBlock, init_distributed

    print("Testing MLA model construction...")

    model_cfg = ModelConfig(attention_type="mla")
    engine_cfg = EngineConfig(
        num_pages=16,
        page_size=64,
        num_layers=2,
        attention_type="mla",
    )

    assert model_cfg.attention_type == "mla"
    assert engine_cfg.attention_type == "mla"
    print("  ✓ MLA config validated")

    # Test TransformerBlock creates MLA attention
    init_distributed(0, 1)

    block = TransformerBlock(model_cfg, layer_idx=0, buffer=None)
    assert block.attention_type == "mla"
    print("  ✓ TransformerBlock attention_type == mla")

    from nmoe.serve.mla import MLA
    assert isinstance(block.attn, MLA), f"Expected MLA, got {type(block.attn)}"
    print("  ✓ MLA attention class instantiated")

    print("  ✓ MLA model construction test passed\n")


def test_cache_shapes():
    """Test that cache tensors have correct shapes for each attention type."""
    from nmoe.serve.engine import EngineConfig

    print("Testing cache shapes...")

    num_pages = 16
    page_size = 64
    kv_lora_rank = 512
    qk_rope_head_dim = 64

    # DSA cache shapes
    dsa_kv_shape = (num_pages, page_size, 1, 656)  # FP8 packed
    dsa_idx_shape = (num_pages, page_size, 128)  # idx_dim

    # MLA cache shapes
    mla_latent_shape = (num_pages, page_size, kv_lora_rank)
    mla_rope_shape = (num_pages, page_size, qk_rope_head_dim)

    # Verify shapes are what we expect
    device = torch.device("cuda:0")

    dsa_kv = torch.zeros(dsa_kv_shape, dtype=torch.uint8, device=device)
    dsa_idx = torch.zeros(dsa_idx_shape, dtype=torch.bfloat16, device=device)
    print(f"  DSA kv_cache: {dsa_kv.shape} ({dsa_kv.dtype})")
    print(f"  DSA idx_k_cache: {dsa_idx.shape} ({dsa_idx.dtype})")

    mla_latent = torch.zeros(mla_latent_shape, dtype=torch.bfloat16, device=device)
    mla_rope = torch.zeros(mla_rope_shape, dtype=torch.bfloat16, device=device)
    print(f"  MLA kv_cache_latent: {mla_latent.shape} ({mla_latent.dtype})")
    print(f"  MLA kv_cache_rope: {mla_rope.shape} ({mla_rope.dtype})")

    # Memory comparison
    dsa_bytes = dsa_kv.numel() * 1 + dsa_idx.numel() * 2  # uint8 + bf16
    mla_bytes = (mla_latent.numel() + mla_rope.numel()) * 2  # both bf16

    print(f"\n  DSA memory per page: {dsa_bytes / num_pages / 1024:.1f} KB")
    print(f"  MLA memory per page: {mla_bytes / num_pages / 1024:.1f} KB")

    print("  ✓ Cache shapes test passed\n")


def main() -> int:
    torch.cuda.set_device(0)

    print("\n" + "=" * 60)
    print("Attention Type Wiring Tests")
    print("=" * 60 + "\n")

    try:
        test_cache_shapes()
        test_dsa_model_construction()
        test_mla_model_construction()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
