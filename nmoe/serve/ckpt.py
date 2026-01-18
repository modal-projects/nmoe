# SPDX-License-Identifier: Apache-2.0
"""Checkpoint loading for DeepSeek-V3 family models with FP8 experts."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple

import torch
from safetensors.torch import safe_open

from nmoe.serve.model import ModelConfig


def load_model_config(ckpt_path: str) -> ModelConfig:
    """Load ModelConfig from checkpoint, auto-detecting MLA vs DSA.

    Reads config.json from the checkpoint directory and returns a ModelConfig
    with the appropriate attention_type and model parameters.

    Detection logic:
      - If config.json contains "index_head_dim" -> DSA (V3.2-Speciale)
      - Otherwise -> MLA (V3-0324, Kimi-K2, etc.)

    Args:
        ckpt_path: Path to checkpoint directory containing config.json

    Returns:
        ModelConfig with fields populated from checkpoint config.
    """
    config_path = Path(ckpt_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {ckpt_path}")

    with open(config_path) as f:
        hf_cfg = json.load(f)

    # Detect attention type
    is_dsa = "index_head_dim" in hf_cfg
    attention_type: Literal["dsa", "mla"] = "dsa" if is_dsa else "mla"

    rope_scaling = hf_cfg.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
      rope_scaling = {}

    def _first_int(*keys: str, default: int) -> int:
      for k in keys:
        v = hf_cfg.get(k)
        if isinstance(v, int):
          return int(v)
      return int(default)

    def _first_float(*keys: str, default: float) -> float:
      for k in keys:
        v = hf_cfg.get(k)
        if isinstance(v, (int, float)):
          return float(v)
      return float(default)

    max_seq_len = _first_int(
      "max_position_embeddings",
      "max_seq_len",
      "model_max_length",
      default=ModelConfig().max_seq_len,
    )
    # YaRN/rope scaling metadata (optional; defaults are safe for V3-family MLA).
    orig = rope_scaling.get("original_max_position_embeddings")
    original_seq_len = int(orig) if isinstance(orig, int) else _first_int("original_max_position_embeddings", default=ModelConfig().original_seq_len)

    rf = rope_scaling.get("factor")
    rope_factor = float(rf) if isinstance(rf, (int, float)) else _first_float("rope_factor", default=ModelConfig().rope_factor)

    bf = rope_scaling.get("beta_fast")
    beta_fast = float(bf) if isinstance(bf, (int, float)) else _first_float("beta_fast", default=ModelConfig().beta_fast)

    bs = rope_scaling.get("beta_slow")
    beta_slow = float(bs) if isinstance(bs, (int, float)) else _first_float("beta_slow", default=ModelConfig().beta_slow)

    ms = rope_scaling.get("mscale")
    mscale = float(ms) if isinstance(ms, (int, float)) else _first_float("mscale", default=ModelConfig().mscale)

    # Some DeepSeek-V3 family variants set LoRA ranks to null to indicate
    # "disabled" (use the non-LoRA projection path instead).
    q_lora_rank = hf_cfg.get("q_lora_rank", ModelConfig().q_lora_rank)
    q_lora_rank = 0 if q_lora_rank is None else int(q_lora_rank)
    kv_lora_rank = hf_cfg.get("kv_lora_rank", ModelConfig().kv_lora_rank)
    kv_lora_rank = 0 if kv_lora_rank is None else int(kv_lora_rank)

    return ModelConfig(
        attention_type=attention_type,
        # Core dimensions
        vocab_size=hf_cfg.get("vocab_size", 129280),
        hidden_size=hf_cfg.get("hidden_size", 7168),
        intermediate_size=hf_cfg.get("intermediate_size", 18432),
        num_layers=hf_cfg.get("num_hidden_layers", 61),
        num_dense_layers=hf_cfg.get("first_k_dense_replace", 3),
        # Attention
        num_heads=hf_cfg.get("num_attention_heads", 128),
        q_lora_rank=int(q_lora_rank),
        kv_lora_rank=int(kv_lora_rank),
        qk_nope_head_dim=hf_cfg.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=hf_cfg.get("qk_rope_head_dim", 64),
        v_head_dim=hf_cfg.get("v_head_dim", 128),
        # RoPE / context length
        rope_theta=_first_float("rope_theta", default=ModelConfig().rope_theta),
        rope_factor=float(rope_factor),
        max_seq_len=int(max_seq_len),
        original_seq_len=int(original_seq_len),
        beta_fast=float(beta_fast),
        beta_slow=float(beta_slow),
        mscale=float(mscale),
        # DSA indexer (only meaningful when attention_type="dsa")
        dsa_n_idx_heads=hf_cfg.get("index_n_heads", 64),
        dsa_idx_dim=hf_cfg.get("index_head_dim", 128),
        dsa_topk=hf_cfg.get("index_topk", 2048),
        # MoE
        num_experts=hf_cfg.get("n_routed_experts", 256),
        num_shared_experts=hf_cfg.get("n_shared_experts", 1),
        num_experts_per_tok=hf_cfg.get("num_experts_per_tok", 8),
        moe_intermediate_size=hf_cfg.get("moe_intermediate_size", 2048),
    )


def _is_ue8m0(scale: torch.Tensor) -> bool:
  """Check if scale tensor is already in UE8M0 format (all power-of-2)."""
  log2 = torch.log2(scale.abs().clamp(min=1e-12))
  return torch.allclose(log2, log2.round(), atol=1e-5)


def _ensure_ue8m0_scale(scale: torch.Tensor) -> torch.Tensor:
  """Round scale values up to the nearest power-of-two (UE8M0 requirement)."""
  scale = scale.abs().clamp(min=1e-12)
  log2 = torch.log2(scale)
  return torch.pow(2.0, torch.ceil(log2)).to(scale.dtype)


_FP8_E4M3_MAX: float = 448.0


def _quantize_bf16_weight_to_fp8_ue8m0(
  weight: torch.Tensor,
  *,
  block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a BF16/FP16/FP32 weight matrix to FP8 + UE8M0 scales.

  Returns:
    (weight_fp8, scale): weight_fp8 has dtype float8_e4m3fn and shape [out, in];
    scale has dtype float32 and shape [out_tiles, in_tiles] where each scale is
    a power-of-two (UE8M0).

  Contract:
    Dequantization is `weight_fp8.float() * scale`.
  """
  if weight.dim() != 2:
    raise ValueError(f"Expected 2D weight, got shape={tuple(weight.shape)}")

  out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
  out_tiles = (out_features + block_size - 1) // block_size
  in_tiles = (in_features + block_size - 1) // block_size
  out_padded = out_tiles * block_size
  in_padded = in_tiles * block_size

  w = weight.to(torch.float32).contiguous()
  if out_padded != out_features or in_padded != in_features:
    w_pad = torch.zeros((out_padded, in_padded), dtype=torch.float32)
    w_pad[:out_features, :in_features] = w
    w = w_pad

  w_tiled = w.view(out_tiles, block_size, in_tiles, block_size)
  amax = w_tiled.abs().amax(dim=(1, 3))  # [out_tiles, in_tiles]
  amax = amax.clamp(min=1e-12)

  log2_scale = torch.log2(amax / float(_FP8_E4M3_MAX))
  scale = torch.pow(2.0, torch.ceil(log2_scale)).to(torch.float32)

  w_q = (w_tiled / scale.view(out_tiles, 1, in_tiles, 1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
  w_q = w_q.view(out_padded, in_padded)[:out_features, :in_features]
  w_fp8 = w_q.contiguous().to(torch.float8_e4m3fn)
  return w_fp8, scale


def _should_quantize_dense_weight_to_fp8(name: str) -> bool:
  # Gate/router must stay FP32 for numerical stability.
  if name.endswith(".ffn.gate.weight") or name.endswith(".ffn.gate.bias"):
    return False
  if name == "lm_head.weight" or name == "embed.weight":
    return False

  # MLA attention projections.
  if name.endswith(".attn.wq.weight"):
    return True
  if name.endswith(".attn.wq_a.weight"):
    return True
  if name.endswith(".attn.wq_b.weight"):
    return True
  if name.endswith(".attn.wkv_a.weight"):
    return True
  if name.endswith(".attn.wkv_b.weight"):
    return True
  if name.endswith(".attn.wo.weight"):
    return True

  # Dense MLP + shared experts MLP.
  if name.endswith(".ffn.w1.weight") or name.endswith(".ffn.w2.weight") or name.endswith(".ffn.w3.weight"):
    return True
  if name.endswith(".ffn.shared.w1.weight") or name.endswith(".ffn.shared.w2.weight") or name.endswith(".ffn.shared.w3.weight"):
    return True

  return False


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
  tp_size: int = 1,
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

  tp_size = int(tp_size)
  if tp_size not in (1, world_size):
    raise ValueError(
      "convert_hf_to_sharded/load_checkpoint expects tp_size to be either 1 "
      "(TP disabled; dense weights replicated) or world_size (pure TP sharding). "
      f"Got tp_size={tp_size}, world_size={world_size}."
    )
  tp_rank = int(rank) if tp_size == world_size else 0

  for fpath in files:
    with safe_open(fpath, framework="pt", device="cpu") as f:
      for hf_name in f.keys():
        # HF stores per-layer rotary_emb.inv_freq buffers; our MLA path uses
        # a single precomputed freqs table instead.
        if hf_name.endswith("rotary_emb.inv_freq"):
          continue

        # Skip MTP layer 61 (unused) and any layers outside cfg.num_layers.
        m = re.search(r"layers\.(\d+)\.", hf_name)
        if m is not None:
          layer_idx = int(m.group(1))
          if layer_idx == 61 or layer_idx >= num_layers:
            continue

        # The DeepSeek gate has an optional correction bias. Our implementation
        # only uses a gate bias for the 7168-hidden checkpoints; skip it for
        # smaller variants (e.g., Moonlight 16B) to avoid unexpected keys.
        if hf_name.endswith("e_score_correction_bias") and int(cfg.hidden_size) != 7168:
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
          raw_state_dict[internal_name] = _shard_tensor(tensor, shard_dim, tp_rank, tp_size)
          continue

        tensor = f.get_tensor(hf_name)
        internal_name, shard_dim = _map_name(hf_name)
        raw_state_dict[internal_name] = _shard_tensor(tensor, shard_dim, tp_rank, tp_size)

  # HF checkpoints may store dense weights in BF16/FP16/FP32. Our inference stack
  # expects FP8 weights + UE8M0 scales for the DeepGEMM-backed modules.
  for name, tensor in list(raw_state_dict.items()):
    if not name.endswith(".weight"):
      continue
    if not _should_quantize_dense_weight_to_fp8(name):
      continue
    if tensor.dtype == torch.float8_e4m3fn:
      continue
    w_fp8, scale = _quantize_bf16_weight_to_fp8_ue8m0(tensor)
    raw_state_dict[name] = w_fp8
    raw_state_dict[name.replace(".weight", ".weight_scale_inv")] = scale

  # Gate is computed in FP32 for stability; match the model parameter dtype.
  for name, tensor in list(raw_state_dict.items()):
    if name.endswith(".ffn.gate.weight") or name.endswith(".ffn.gate.bias"):
      raw_state_dict[name] = tensor.to(torch.float32)

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

      # BF16 experts (HF-style) do not have scales; generate FP8 + UE8M0 scales.
      if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn or w3.dtype != torch.float8_e4m3fn:
        w1, s1 = _quantize_bf16_weight_to_fp8_ue8m0(w1)
        w2, s2 = _quantize_bf16_weight_to_fp8_ue8m0(w2)
        w3, s3 = _quantize_bf16_weight_to_fp8_ue8m0(w3)

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
        w1_scale_list.append(s1.to(torch.float32))
        w2_scale_list.append(s2.to(torch.float32))
        w3_scale_list.append(s3.to(torch.float32))

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
  "lm_head": ("lm_head", None),  # Replicate across ranks (no TP sharding)
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
  tp_size: int = 1,
) -> Tuple[set, set]:
  """
  Load HuggingFace checkpoint into model with FP8 expert stacking.

  Expert weights are loaded and stacked into the format expected by DeepGEMM:
  - w13: [num_local, 2*inter, hidden] FP8 (concat of w1=gate and w3=up)
  - w2: [num_local, hidden, inter] FP8 (down)
  """
  cfg = cfg or ModelConfig()
  state_dict = _build_state_dict_for_rank(
    ckpt_path, rank=rank, world_size=world_size, tp_size=int(tp_size), cfg=cfg
  )

  missing, unexpected = model.load_state_dict(state_dict, strict=False)

  if strict and (missing or unexpected):
    raise RuntimeError(
      f"Checkpoint mismatch:\n"
      f"  Missing: {missing}\n"
      f"  Unexpected: {unexpected}"
    )

  # Inference fast-path: convert MoE shared MLP weights to BF16 fused form and
  # drop FP8 copies. This is a load-time transform (not a new checkpoint
  # format); the BF16 weights are kept as non-persistent buffers.
  if hasattr(model, "convert_shared_mlps_to_bf16_"):
    model.convert_shared_mlps_to_bf16_()  # type: ignore[attr-defined]
  # Inference fast-path: convert small-MN attention/indexer FP8Linear modules to
  # BF16 weights to avoid activation quantize+pack overhead on decode.
  if hasattr(model, "convert_attention_fp8linears_to_bf16_"):
    model.convert_attention_fp8linears_to_bf16_()  # type: ignore[attr-defined]

  return set(missing), set(unexpected)


def convert_hf_to_sharded(
  hf_ckpt_path: str,
  save_path: str,
  world_size: int,
  cfg: Optional[ModelConfig] = None,
  tp_size: int = 1,
) -> None:
  """
  Convert HuggingFace checkpoint to sharded format with stacked experts.

  Creates one safetensors file per rank: model{rank}-mp{world_size}.safetensors
  """
  from safetensors.torch import save_file

  cfg = cfg or ModelConfig()
  os.makedirs(save_path, exist_ok=True)

  for rank in range(world_size):
    print(f"Processing rank {rank}/{world_size}...", flush=True)
    state_dict = _build_state_dict_for_rank(
      hf_ckpt_path, rank=rank, world_size=world_size, tp_size=int(tp_size), cfg=cfg
    )
    out_path = os.path.join(save_path, f"model{rank}-mp{world_size}.safetensors")
    save_file(state_dict, out_path)
    print(f"Saved {out_path}", flush=True)


def load_sharded_checkpoint(
  model: torch.nn.Module,
  ckpt_path: str,
  rank: int = 0,
  world_size: int = 1,
) -> Tuple[set, set]:
  """Load pre-sharded checkpoint (from convert_hf_to_sharded).

  Note: lm_head is replicated (not vocab-parallel). Legacy checkpoints may
  contain vocab-sharded lm_head; in that case each rank reconstructs locally
  by concatenating shards from disk (no collective).
  """

  fname = f"model{rank}-mp{world_size}.safetensors"
  fpath = os.path.join(ckpt_path, fname)

  if not os.path.exists(fpath):
    raise FileNotFoundError(f"Sharded checkpoint not found: {fpath}")

  state_dict = {}
  with safe_open(fpath, framework="pt", device="cpu") as f:
    for name in f.keys():
      state_dict[name] = f.get_tensor(name)

  # Reconstruct vocab-sharded lm_head from disk (legacy format) iff needed.
  if world_size > 1 and "lm_head.weight" in state_dict:
    vocab_size: Optional[int] = None
    cfg_path = os.path.join(ckpt_path, "config.json")
    if os.path.exists(cfg_path):
      try:
        with open(cfg_path) as f:
          vocab_size = int(json.load(f).get("vocab_size"))
      except Exception:
        vocab_size = None

    local_rows = int(state_dict["lm_head.weight"].shape[0])
    should_concat = False
    if vocab_size is not None:
      if local_rows == vocab_size:
        should_concat = False
      elif local_rows * world_size == vocab_size:
        should_concat = True
      else:
        raise RuntimeError(
          "lm_head.weight shape is inconsistent with config.json vocab_size: "
          f"local_rows={local_rows}, world_size={world_size}, vocab_size={vocab_size}"
        )
    else:
      # Best-effort fallback for legacy checkpoints without config.json.
      should_concat = True

    if should_concat:
      shards = []
      for r in range(world_size):
        shard_path = os.path.join(ckpt_path, f"model{r}-mp{world_size}.safetensors")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
          shards.append(f.get_tensor("lm_head.weight"))
      state_dict["lm_head.weight"] = torch.cat(shards, dim=0)  # [vocab, hidden]

  missing, unexpected = model.load_state_dict(state_dict, strict=False)
  if hasattr(model, "convert_shared_mlps_to_bf16_"):
    model.convert_shared_mlps_to_bf16_()  # type: ignore[attr-defined]
  if hasattr(model, "convert_attention_fp8linears_to_bf16_"):
    model.convert_attention_fp8linears_to_bf16_()  # type: ignore[attr-defined]

  placement = os.environ.get("NMOE_EXPERT_PLACEMENT", "contiguous").strip().lower()
  if placement == "striped":
    if world_size != 8:
      raise RuntimeError(f"NMOE_EXPERT_PLACEMENT=striped requires world_size=8 (got {world_size}).")
    from nmoe.serve.expert_placement import stripe_moe_experts_inplace

    stripe_moe_experts_inplace(model, rank=rank, world_size=world_size)

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

  ap_cv = sub.add_parser("convert-hf-to-nmoe", help="Convert a HF safetensors dir to nmoe EP shards (for nmoe.serve)")
  ap_cv.add_argument("--hf", required=True, help="Local HF checkpoint dir containing *.safetensors and config.json")
  ap_cv.add_argument("--out", required=True, help="Output dir for nmoe shards (e.g. /data/models/<name>-ep8-tp1)")
  ap_cv.add_argument("--world-size", type=int, required=True, help="Number of mp shards to produce")

  ap_dc = sub.add_parser("download-and-convert", help="Download from HF then convert to nmoe EP shards")
  ap_dc.add_argument("--repo", required=True, help="HF repo id (or local dir)")
  ap_dc.add_argument("--out-hf", required=True, help="Local HF snapshot dir (e.g. /data/models/<name>)")
  ap_dc.add_argument("--out-nmoe", required=True, help="Output nmoe shard dir (e.g. /data/models/<name>-ep8-tp1)")
  ap_dc.add_argument("--world-size", type=int, required=True, help="Number of mp shards to produce")
  ap_dc.add_argument("--revision", default=None, help="HF revision/tag/commit")
  ap_dc.add_argument("--cache-dir", default=None, help="HF cache dir (optional)")
  ap_dc.add_argument("--symlinks", action="store_true", help="Use symlinks into HF cache (faster, less space)")

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

  if args.cmd == "convert-hf-to-nmoe":
    cfg = load_model_config(args.hf)
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
    cfg = load_model_config(hf_dir)
    convert_hf_to_sharded(hf_dir, args.out_nmoe, world_size=int(args.world_size), cfg=cfg)
    return 0

  raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
  raise SystemExit(_cli())
