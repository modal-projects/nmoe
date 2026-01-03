# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading for DeepSeek-V3.2 models with FP8 experts."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from safetensors.torch import safe_open

from nmoe.serve.model import ModelConfig


def _is_ue8m0(scale: torch.Tensor) -> bool:
  """Check if scale tensor is already in UE8M0 format (all power-of-2)."""
  log2 = torch.log2(scale.abs().clamp(min=1e-12))
  return torch.allclose(log2, log2.round(), atol=1e-5)


def _ensure_ue8m0_scale(scale: torch.Tensor) -> torch.Tensor:
  """Round scale values up to the nearest power-of-two (UE8M0 requirement)."""
  scale = scale.abs().clamp(min=1e-12)
  log2 = torch.log2(scale)
  return torch.pow(2.0, torch.ceil(log2)).to(scale.dtype)


def _requantize_fp8_for_ue8m0(
  weight_fp8: torch.Tensor,
  scale_old: torch.Tensor,
  block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Requantize FP8 weight to use UE8M0 (power-of-2) scales.

  SM100 DeepGEMM requires weight scales to be powers of 2.
  When the checkpoint has non-pow2 scales, we must:
    1. Dequantize: w_bf16 = w_fp8 * scale_old
    2. Compute pow2 scale from max abs value per block
    3. Requantize: w_fp8_new = clamp(w_bf16 / scale_pow2, -448, 448)

  This preserves the dequantized weight as closely as FP8 allows.

  Handles non-128-aligned dimensions by padding weight to match scale tiles.

  Args:
    weight_fp8: FP8 weight tensor [out_features, in_features]
    scale_old: Original scales [out_tiles, in_tiles] (may be non-pow2)
    block_size: Tile size for block-wise quantization (default 128)

  Returns:
    (weight_fp8_new, scale_pow2): Requantized weight and power-of-2 scale
  """
  if _is_ue8m0(scale_old):
    return weight_fp8, scale_old

  out_features, in_features = weight_fp8.shape
  out_tiles, in_tiles = scale_old.shape

  # Compute padded dimensions to match scale tiles
  out_padded = out_tiles * block_size
  in_padded = in_tiles * block_size

  # Pad weight if needed
  w_float = weight_fp8.float()
  if out_padded != out_features or in_padded != in_features:
    w_padded = torch.zeros(out_padded, in_padded, dtype=torch.float32)
    w_padded[:out_features, :in_features] = w_float
    w_float = w_padded

  # Reshape weight to tiles: [out_tiles, block_size, in_tiles, block_size]
  w_tiled = w_float.view(out_tiles, block_size, in_tiles, block_size)

  # Dequantize: multiply each tile by its scale
  # scale_old: [out_tiles, in_tiles] -> [out_tiles, 1, in_tiles, 1]
  w_dequant = w_tiled * scale_old.view(out_tiles, 1, in_tiles, 1)

  # Compute new pow2 scale from max abs per tile
  # Find amax per tile: [out_tiles, in_tiles]
  amax = w_dequant.abs().amax(dim=(1, 3))
  amax = amax.clamp(min=1e-12)

  # Compute UE8M0 scale: 2^ceil(log2(amax / 448))
  FP8_MAX = 448.0
  log2_scale = torch.log2(amax / FP8_MAX)
  scale_pow2 = torch.pow(2.0, torch.ceil(log2_scale))

  # Requantize with new scale
  w_requant = w_dequant / scale_pow2.view(out_tiles, 1, in_tiles, 1)
  w_requant = w_requant.clamp(-FP8_MAX, FP8_MAX)

  # Remove padding and convert back to FP8
  w_requant = w_requant.view(out_padded, in_padded)[:out_features, :in_features]
  w_requant = w_requant.contiguous().to(torch.float8_e4m3fn)

  return w_requant, scale_pow2


def _build_state_dict_for_rank(
  ckpt_path: str,
  *,
  rank: int,
  world_size: int,
  cfg: ModelConfig,
) -> Dict[str, torch.Tensor]:
  """Build a per-rank state_dict in our *mp* format, including UE8M0 fixes."""
  num_layers = int(cfg.num_layers)
  num_experts = int(cfg.num_experts)
  n_local_experts = num_experts // world_size
  expert_start = rank * n_local_experts
  expert_end = expert_start + n_local_experts

  files = sorted(glob(os.path.join(ckpt_path, "*.safetensors")))
  if not files:
    raise ValueError(f"No safetensors files found in {ckpt_path}")

  raw_state_dict: Dict[str, torch.Tensor] = {}
  expert_weights: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
  expert_scales: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

  for fpath in files:
    with safe_open(fpath, framework="pt", device="cpu") as f:
      for hf_name in f.keys():
        # Skip MTP layer 61 (unused) and any layers outside cfg.num_layers.
        m = re.search(r"layers\.(\d+)\.", hf_name)
        if m is not None:
          layer_idx = int(m.group(1))
          if layer_idx == 61 or layer_idx >= num_layers:
            continue

        # Experts (not shared) are sharded by EP; collect for stacking.
        if _is_expert_weight(hf_name):
          info = _parse_expert_info(hf_name)
          if info is None:
            continue
          layer_idx, expert_idx, weight_type, suffix = info
          if layer_idx >= num_layers:
            continue
          if expert_idx < expert_start or expert_idx >= expert_end:
            continue

          tensor = f.get_tensor(hf_name)
          local_idx = expert_idx - expert_start
          if suffix == "weight":
            expert_weights[layer_idx][local_idx][weight_type] = tensor
          else:
            expert_scales[layer_idx][local_idx][weight_type] = tensor
          continue

        # Shared experts behave like dense weights but live under a different HF prefix.
        if "shared_experts" in hf_name:
          tensor = f.get_tensor(hf_name)
          internal_name, shard_dim = _map_name(hf_name)
          raw_state_dict[internal_name] = _shard_tensor(tensor, shard_dim, rank, world_size)
          continue

        tensor = f.get_tensor(hf_name)
        internal_name, shard_dim = _map_name(hf_name)
        raw_state_dict[internal_name] = _shard_tensor(tensor, shard_dim, rank, world_size)

  # Requantize FP8 weights (non-expert) with non-pow2 scales.
  state_dict: Dict[str, torch.Tensor] = {}
  for name, tensor in raw_state_dict.items():
    if name.endswith(".weight_scale_inv"):
      weight_name = name.replace(".weight_scale_inv", ".weight")
      if weight_name in raw_state_dict:
        weight = raw_state_dict[weight_name]
        if weight.dtype == torch.float8_e4m3fn and not _is_ue8m0(tensor):
          w_new, s_new = _requantize_fp8_for_ue8m0(weight, tensor)
          state_dict[weight_name] = w_new
          state_dict[name] = s_new
          continue
      state_dict[name] = tensor if _is_ue8m0(tensor) else _ensure_ue8m0_scale(tensor)
    elif name.endswith(".weight"):
      if name not in state_dict:
        state_dict[name] = tensor
    else:
      state_dict[name] = tensor

  # Stack experts into DeepGEMM format, fixing UE8M0 scales as needed.
  for layer_idx in sorted(expert_weights.keys()):
    layer_experts = expert_weights[layer_idx]
    layer_scales = expert_scales[layer_idx]

    w1_list, w2_list, w3_list = [], [], []
    w1_scale_list, w2_scale_list, w3_scale_list = [], [], []
    expect_scales: Optional[bool] = None

    for local_idx in range(n_local_experts):
      if local_idx not in layer_experts:
        raise ValueError(f"Missing expert {local_idx} in layer {layer_idx}")
      exp = layer_experts[local_idx]
      sc = layer_scales.get(local_idx, {})

      w1 = exp["w1"]
      w2 = exp["w2"]
      w3 = exp["w3"]
      s1 = sc.get("w1")
      s2 = sc.get("w2")
      s3 = sc.get("w3")

      this_has_scales = (s1 is not None) or (s2 is not None) or (s3 is not None)
      if expect_scales is None:
        expect_scales = this_has_scales
      elif expect_scales != this_has_scales:
        raise ValueError(f"Inconsistent expert scale presence in layer {layer_idx} (expert {local_idx}).")
      if this_has_scales and (s1 is None or s2 is None or s3 is None):
        raise ValueError(f"Missing expert scale(s) in layer {layer_idx} (expert {local_idx}).")

      if s1 is not None and not _is_ue8m0(s1):
        if w1.dtype == torch.float8_e4m3fn:
          w1, s1 = _requantize_fp8_for_ue8m0(w1, s1)
        else:
          s1 = _ensure_ue8m0_scale(s1)
      if s2 is not None and not _is_ue8m0(s2):
        if w2.dtype == torch.float8_e4m3fn:
          w2, s2 = _requantize_fp8_for_ue8m0(w2, s2)
        else:
          s2 = _ensure_ue8m0_scale(s2)
      if s3 is not None and not _is_ue8m0(s3):
        if w3.dtype == torch.float8_e4m3fn:
          w3, s3 = _requantize_fp8_for_ue8m0(w3, s3)
        else:
          s3 = _ensure_ue8m0_scale(s3)

      w1_list.append(w1)
      w2_list.append(w2)
      w3_list.append(w3)
      if expect_scales:
        w1_scale_list.append(s1)
        w2_scale_list.append(s2)
        w3_scale_list.append(s3)

    w1_stacked = torch.stack(w1_list, dim=0)
    w2_stacked = torch.stack(w2_list, dim=0)
    w3_stacked = torch.stack(w3_list, dim=0)
    w13_stacked = torch.cat([w1_stacked, w3_stacked], dim=1)

    state_dict[f"layers.{layer_idx}.ffn.w13"] = w13_stacked
    state_dict[f"layers.{layer_idx}.ffn.w2"] = w2_stacked

    if expect_scales:
      w13_scale = torch.cat([torch.stack(w1_scale_list), torch.stack(w3_scale_list)], dim=1)
      w2_scale = torch.stack(w2_scale_list)
      state_dict[f"layers.{layer_idx}.ffn.w13_scale"] = w13_scale
      state_dict[f"layers.{layer_idx}.ffn.w2_scale"] = w2_scale

  return state_dict


# HuggingFace name -> (internal name suffix, shard dim)
# dim=0: shard on output, dim=1: shard on input, dim=None: replicate
WEIGHT_MAP = {
  "embed_tokens": ("embed", None),  # Replicate embedding across ranks
  "input_layernorm": ("attn_norm", None),
  "post_attention_layernorm": ("ffn_norm", None),
  "q_proj": ("wq", 0),
  "q_a_proj": ("wq_a", None),
  "q_a_layernorm": ("q_norm", None),
  "q_b_proj": ("wq_b", 0),
  "kv_a_proj_with_mqa": ("wkv_a", None),
  "kv_a_layernorm": ("kv_norm", None),
  "kv_b_proj": ("wkv_b", 0),
  "o_proj": ("wo", 1),
  "gate": ("gate", None),
  "gate_proj": ("w1", 0),  # dense MLP gate -> w1
  "down_proj": ("w2", 1),  # dense MLP down -> w2
  "up_proj": ("w3", 0),    # dense MLP up -> w3
  "norm": ("norm", None),
  "lm_head": ("lm_head", 0),  # Keep as lm_head (model uses self.lm_head)
  "e_score_correction_bias": ("bias", None),
}


def _map_name(hf_name: str) -> Tuple[str, Optional[int]]:
  """Map HuggingFace weight name to internal name and shard dimension."""
  if hf_name.startswith("model."):
    hf_name = hf_name[len("model."):]

  hf_name = hf_name.replace("self_attn", "attn")
  hf_name = hf_name.replace("mlp", "ffn")
  hf_name = hf_name.replace("shared_experts", "shared")  # MoE shared experts

  # DSA indexer: flatten indexer.X -> X (e.g., attn.indexer.wq_b -> attn.wq_idx)
  hf_name = hf_name.replace(".indexer.wq_b", ".wq_idx")
  hf_name = hf_name.replace(".indexer.wk", ".wk_idx")
  hf_name = hf_name.replace(".indexer.weights_proj", ".w_idx")
  hf_name = hf_name.replace(".indexer.k_norm", ".k_norm")

  parts = hf_name.split(".")
  # Iterate from end to match leaf components first (e.g., e_score_correction_bias before gate)
  for i in range(len(parts) - 1, -1, -1):
    part = parts[i]
    if part in WEIGHT_MAP:
      new_part, dim = WEIGHT_MAP[part]
      parts[i] = new_part
      return ".".join(parts), dim

  return hf_name, None


def _shard_tensor(
  tensor: torch.Tensor,
  dim: Optional[int],
  rank: int,
  world_size: int,
) -> torch.Tensor:
  """Shard tensor for tensor parallelism."""
  if dim is None or world_size == 1:
    return tensor
  size = tensor.size(dim)
  assert size % world_size == 0, f"Dim {dim} size {size} not divisible by {world_size}"
  shard_size = size // world_size
  return tensor.narrow(dim, rank * shard_size, shard_size).contiguous()


def _is_expert_weight(name: str) -> bool:
  """Check if name is an expert weight (not shared_experts)."""
  return "experts" in name and "shared_experts" not in name


def _parse_expert_info(name: str) -> Optional[Tuple[int, int, str, str]]:
  """Parse layer_idx, expert_idx, weight_type, suffix from expert weight name.

  Returns (layer_idx, expert_idx, weight_type, suffix) or None if not expert weight.
  weight_type is 'w1', 'w2', or 'w3'
  suffix is 'weight' or 'weight_scale_inv'
  """
  # Pattern: layers.X.{mlp,ffn}.experts.Y.{gate_proj,down_proj,up_proj}.{weight,weight_scale_inv}
  pattern = r"layers\.(\d+)\.(?:mlp|ffn)\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(weight_scale_inv|weight)"
  match = re.search(pattern, name)
  if not match:
    return None

  layer_idx = int(match.group(1))
  expert_idx = int(match.group(2))
  proj_type = match.group(3)
  suffix = match.group(4)

  # Map HF names to our naming
  type_map = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
  weight_type = type_map[proj_type]

  return layer_idx, expert_idx, weight_type, suffix


def load_checkpoint(
  model: torch.nn.Module,
  ckpt_path: str,
  rank: int = 0,
  world_size: int = 1,
  cfg: Optional[ModelConfig] = None,
  strict: bool = False,
) -> Tuple[set, set]:
  """
  Load HuggingFace checkpoint into model with FP8 expert stacking.

  Expert weights are loaded and stacked into the format expected by DeepGEMM:
  - w13: [num_local, 2*inter, hidden] FP8 (concat of w1=gate and w3=up)
  - w2: [num_local, hidden, inter] FP8 (down)
  """
  cfg = cfg or ModelConfig()
  state_dict = _build_state_dict_for_rank(ckpt_path, rank=rank, world_size=world_size, cfg=cfg)

  missing, unexpected = model.load_state_dict(state_dict, strict=False)

  if strict and (missing or unexpected):
    raise RuntimeError(
      f"Checkpoint mismatch:\n"
      f"  Missing: {missing}\n"
      f"  Unexpected: {unexpected}"
    )

  return set(missing), set(unexpected)


def convert_hf_to_sharded(
  hf_ckpt_path: str,
  save_path: str,
  world_size: int,
  cfg: Optional[ModelConfig] = None,
) -> None:
  """
  Convert HuggingFace checkpoint to sharded format with stacked experts.

  Creates one safetensors file per rank: model{rank}-mp{world_size}.safetensors
  """
  from safetensors.torch import save_file

  cfg = cfg or ModelConfig()
  os.makedirs(save_path, exist_ok=True)

  for rank in range(world_size):
    print(f"Processing rank {rank}/{world_size}...")
    state_dict = _build_state_dict_for_rank(hf_ckpt_path, rank=rank, world_size=world_size, cfg=cfg)
    out_path = os.path.join(save_path, f"model{rank}-mp{world_size}.safetensors")
    save_file(state_dict, out_path)
    print(f"Saved {out_path}")


def load_sharded_checkpoint(
  model: torch.nn.Module,
  ckpt_path: str,
  rank: int = 0,
  world_size: int = 1,
) -> Tuple[set, set]:
  """Load pre-sharded checkpoint (from convert_hf_to_sharded)."""
  fname = f"model{rank}-mp{world_size}.safetensors"
  fpath = os.path.join(ckpt_path, fname)

  if not os.path.exists(fpath):
    raise FileNotFoundError(f"Sharded checkpoint not found: {fpath}")

  state_dict = {}
  with safe_open(fpath, framework="pt", device="cpu") as f:
    for name in f.keys():
      state_dict[name] = f.get_tensor(name)

  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  return set(missing), set(unexpected)


def _cli() -> int:
  import argparse

  ap = argparse.ArgumentParser(description="nmoe.serve checkpoint utilities")
  sub = ap.add_subparsers(dest="cmd", required=True)

  ap_dl = sub.add_parser("download", help="Download a HF repo snapshot to a local directory")
  ap_dl.add_argument("--repo", required=True, help="HF repo id (e.g. deepseek-ai/DeepSeek-V3-0324) or a local dir")
  ap_dl.add_argument("--out", required=True, help="Local output dir (recommend /data/models/<name>)")
  ap_dl.add_argument("--revision", default=None, help="HF revision/tag/commit")
  ap_dl.add_argument("--cache-dir", default=None, help="HF cache dir (optional)")
  ap_dl.add_argument("--symlinks", action="store_true", help="Use symlinks into HF cache (faster, less space)")

  ap_cv = sub.add_parser("convert-hf-to-mp", help="Convert a HF safetensors dir to nmoe mp shards")
  ap_cv.add_argument("--hf", required=True, help="Local HF checkpoint dir containing *.safetensors")
  ap_cv.add_argument("--out", required=True, help="Output dir for mp shards (e.g. /data/models/<name>-mp8)")
  ap_cv.add_argument("--world-size", type=int, required=True, help="Number of mp shards to produce")
  ap_cv.add_argument("--attention-type", choices=["dsa", "mla"], default="dsa", help="Model attention type")
  ap_cv.add_argument("--num-layers", type=int, default=ModelConfig.num_layers, help="Number of transformer layers")
  ap_cv.add_argument("--num-dense-layers", type=int, default=ModelConfig.num_dense_layers, help="Number of dense layers")
  ap_cv.add_argument("--num-experts", type=int, default=ModelConfig.num_experts, help="Total MoE experts")

  ap_dc = sub.add_parser("download-and-convert", help="Download from HF then convert to mp shards")
  ap_dc.add_argument("--repo", required=True, help="HF repo id (or local dir)")
  ap_dc.add_argument("--out-hf", required=True, help="Local HF snapshot dir (e.g. /data/models/<name>)")
  ap_dc.add_argument("--out-mp", required=True, help="Output mp shard dir (e.g. /data/models/<name>-mp8)")
  ap_dc.add_argument("--world-size", type=int, required=True, help="Number of mp shards to produce")
  ap_dc.add_argument("--revision", default=None, help="HF revision/tag/commit")
  ap_dc.add_argument("--cache-dir", default=None, help="HF cache dir (optional)")
  ap_dc.add_argument("--symlinks", action="store_true", help="Use symlinks into HF cache (faster, less space)")
  ap_dc.add_argument("--attention-type", choices=["dsa", "mla"], default="dsa", help="Model attention type")
  ap_dc.add_argument("--num-layers", type=int, default=ModelConfig.num_layers, help="Number of transformer layers")
  ap_dc.add_argument("--num-dense-layers", type=int, default=ModelConfig.num_dense_layers, help="Number of dense layers")
  ap_dc.add_argument("--num-experts", type=int, default=ModelConfig.num_experts, help="Total MoE experts")

  args = ap.parse_args()

  if args.cmd == "download":
    from nmoe.serve.hf_download import snapshot_download_to_dir

    path = snapshot_download_to_dir(
      args.repo,
      local_dir=args.out,
      revision=args.revision,
      cache_dir=args.cache_dir,
      local_dir_use_symlinks=bool(args.symlinks),
    )
    print(path)
    return 0

  if args.cmd == "convert-hf-to-mp":
    cfg = ModelConfig(
      attention_type=args.attention_type,
      num_layers=int(args.num_layers),
      num_dense_layers=int(args.num_dense_layers),
      num_experts=int(args.num_experts),
    )
    convert_hf_to_sharded(args.hf, args.out, world_size=int(args.world_size), cfg=cfg)
    return 0

  if args.cmd == "download-and-convert":
    from nmoe.serve.hf_download import snapshot_download_to_dir

    hf_dir = snapshot_download_to_dir(
      args.repo,
      local_dir=args.out_hf,
      revision=args.revision,
      cache_dir=args.cache_dir,
      local_dir_use_symlinks=bool(args.symlinks),
    )
    cfg = ModelConfig(
      attention_type=args.attention_type,
      num_layers=int(args.num_layers),
      num_dense_layers=int(args.num_dense_layers),
      num_experts=int(args.num_experts),
    )
    convert_hf_to_sharded(hf_dir, args.out_mp, world_size=int(args.world_size), cfg=cfg)
    return 0

  raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
  raise SystemExit(_cli())
