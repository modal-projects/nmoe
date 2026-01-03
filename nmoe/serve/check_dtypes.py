# SPDX-License-Identifier: Apache-2.0
"""Check weight dtypes."""

import torch
import torch.distributed as dist

def main():
  dist.init_process_group(backend="nccl")
  rank = dist.get_rank()

  from nmoe.serve.model import ModelConfig, DeepSeekV3, init_distributed
  from nmoe.serve.ckpt import load_checkpoint

  device = torch.device(f"cuda:{rank}")
  torch.cuda.set_device(device)
  init_distributed(rank, 8)

  cfg = ModelConfig(num_layers=1, num_dense_layers=1)
  model = DeepSeekV3(cfg, buffer=None).to(device)
  load_checkpoint(model, "/data/models/DeepSeek-V3.2-Speciale", rank=rank, world_size=8, cfg=cfg)

  if rank == 0:
    attn = model.layers[0].attn
    print("=== Attention Layer Weights ===")
    print(f"wkv_a: shape={attn.wkv_a.weight.shape}, dtype={attn.wkv_a.weight.dtype}")
    print(f"  has scale_inv: {hasattr(attn.wkv_a, 'weight_scale_inv')}")
    if hasattr(attn.wkv_a, 'weight_scale_inv') and attn.wkv_a.weight_scale_inv is not None:
      print(f"  scale_inv: shape={attn.wkv_a.weight_scale_inv.shape}")
    print()
    print(f"wq_a: shape={attn.wq_a.weight.shape}, dtype={attn.wq_a.weight.dtype}")
    print(f"wq_b: shape={attn.wq_b.weight.shape}, dtype={attn.wq_b.weight.dtype}")
    print(f"wkv_b: shape={attn.wkv_b.weight.shape}, dtype={attn.wkv_b.weight.dtype}")
    print(f"wo: shape={attn.wo.weight.shape}, dtype={attn.wo.weight.dtype}")

  dist.barrier()
  dist.destroy_process_group()

if __name__ == "__main__":
  main()
