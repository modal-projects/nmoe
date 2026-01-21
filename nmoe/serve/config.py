# SPDX-License-Identifier: Apache-2.0
"""Configuration for nmoe.serve."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal, Optional

from nmoe.serve.types import ForwardSpec, OutputMode


class DisaggMode(Enum):
  """How to split prefill and decode across replicas."""
  NONE = auto()        # Keep request on one replica end-to-end
  FULL = auto()        # Prefill on prefill replicas, decode on decode replicas
  DECODE_ONLY = auto() # Prefill locally, transfer to decode pool


class BatchingMode(Enum):
  """How to batch requests."""
  CONTINUOUS = auto()  # Opportunistic continuous batching
  FIXED = auto()       # Fixed microbatching (deterministic ordering)


@dataclass(frozen=True)
class Profile:
  """Execution profile controlling scheduler and orchestrator behavior."""
  name: str
  disagg: DisaggMode
  output_mode: OutputMode
  batching: BatchingMode
  streaming: bool
  deterministic: bool
  prefix_cache: Literal["aggressive", "normal", "disabled"]

  def to_forward_spec(self, topk: int = 10) -> ForwardSpec:
    """Convert profile to ForwardSpec for engine."""
    return ForwardSpec(
      output_mode=self.output_mode,
      return_hidden_states=False,
      topk=topk,
    )


# Pre-defined profiles
PROFILES: dict[str, Profile] = {
  "production_generate": Profile(
    name="production_generate",
    disagg=DisaggMode.FULL,
    output_mode=OutputMode.TOKENS,
    batching=BatchingMode.CONTINUOUS,
    streaming=True,
    deterministic=False,
    prefix_cache="normal",
  ),
  "online_distill": Profile(
    name="online_distill",
    disagg=DisaggMode.NONE,  # Keep on one replica unless KV transfer is net-win
    output_mode=OutputMode.TOPK_LOGPROBS,
    batching=BatchingMode.CONTINUOUS,
    streaming=False,
    deterministic=True,
    prefix_cache="aggressive",
  ),
  "rl_sample": Profile(
    name="rl_sample",
    disagg=DisaggMode.DECODE_ONLY,
    output_mode=OutputMode.LOGPROBS,
    batching=BatchingMode.CONTINUOUS,
    streaming=False,
    deterministic=True,  # Per-request RNG
    prefix_cache="normal",
  ),
  "eval_exact": Profile(
    name="eval_exact",
    disagg=DisaggMode.NONE,
    output_mode=OutputMode.LOGITS,
    batching=BatchingMode.FIXED,
    streaming=False,
    deterministic=True,
    prefix_cache="normal",  # Prefix cache is bitwise-identical
  ),
  "offline_distill": Profile(
    name="offline_distill",
    disagg=DisaggMode.NONE,  # Only if improves tokens/s
    output_mode=OutputMode.LOGITS,
    batching=BatchingMode.FIXED,
    streaming=False,
    deterministic=False,
    prefix_cache="aggressive",
  ),
}


@dataclass
class MlaKvLayout:
  """FlashMLA KV cache layout (DeepSeek-V3 style)."""
  kv_lora_rank: int = 512
  qk_rope_head_dim: int = 64
  num_layers: int = 61
  page_size: int = 16  # Tokens per page
  dtype_size: int = 2  # bf16

  @property
  def kv_dim(self) -> int:
    """Compressed KV dimension per token."""
    return self.kv_lora_rank + self.qk_rope_head_dim

  def bytes_per_page(self) -> int:
    """Total bytes for one page across all layers."""
    return self.page_size * self.kv_dim * self.dtype_size * self.num_layers

  def bytes_per_token(self) -> int:
    """Bytes per token across all layers."""
    return self.kv_dim * self.dtype_size * self.num_layers


@dataclass
class ReplicaConfig:
  """Configuration for a single replica (prefill or decode)."""
  replica_id: int
  gpus: list[int]  # GPU IDs for this replica
  role: Literal["prefill", "decode", "both"]
  node: str = "local"  # For multi-node

  @property
  def world_size(self) -> int:
    """Number of GPUs in this replica."""
    return len(self.gpus)


@dataclass
class ServeConfig:
  """Top-level configuration for serve system."""
  # Model
  model_path: str
  model_family: Literal["deepseek", "glm", "minimax", "qwen"] = "deepseek"

  # Replicas - no defaults, must be explicitly configured
  replicas: list[ReplicaConfig] = field(default_factory=list)

  # KV cache layout (depends on model)
  kv_layout: MlaKvLayout = field(default_factory=MlaKvLayout)

  # Scheduling
  max_batch_size: int = 256
  max_prefill_tokens: int = 8192
  max_seq_len: int = 32768

  # Memory
  gpu_memory_utilization: float = 0.9
  num_pages: int = 0  # 0 = auto-calculate

  # Distributed - require explicit config, no defaults
  master_addr: str = field(
    default_factory=lambda: os.environ.get("MASTER_ADDR", "")
  )
  master_port: int = field(
    default_factory=lambda: int(os.environ.get("MASTER_PORT", "0"))
  )

  # API
  host: str = field(
    default_factory=lambda: os.environ.get("SERVE_HOST", "")
  )
  port: int = field(
    default_factory=lambda: int(os.environ.get("SERVE_PORT", "0"))
  )

  # Observability
  metrics_port: int = field(
    default_factory=lambda: int(os.environ.get("METRICS_PORT", "0"))
  )

  def validate(self) -> None:
    """Validate configuration. Raises ValueError if invalid."""
    if not self.model_path:
      raise ValueError("model_path is required")
    if not self.replicas:
      raise ValueError("replicas must be configured")
    if not self.master_addr:
      raise ValueError("master_addr must be set via config or MASTER_ADDR env")
    if not self.master_port:
      raise ValueError("master_port must be set via config or MASTER_PORT env")
    if not self.host:
      raise ValueError("host must be set via config or SERVE_HOST env")
    if not self.port:
      raise ValueError("port must be set via config or SERVE_PORT env")

    # Validate replicas
    all_gpus = set()
    for r in self.replicas:
      for g in r.gpus:
        if g in all_gpus:
          raise ValueError(f"GPU {g} assigned to multiple replicas")
        all_gpus.add(g)

  @property
  def prefill_replicas(self) -> list[ReplicaConfig]:
    """Replicas that handle prefill."""
    return [r for r in self.replicas if r.role in ("prefill", "both")]

  @property
  def decode_replicas(self) -> list[ReplicaConfig]:
    """Replicas that handle decode."""
    return [r for r in self.replicas if r.role in ("decode", "both")]

  @classmethod
  def from_toml(cls, path: str) -> "ServeConfig":
    """Load configuration from TOML file."""
    import tomllib
    with open(path, "rb") as f:
      data = tomllib.load(f)

    replicas = []
    for r in data.get("replicas", []):
      replicas.append(ReplicaConfig(
        replica_id=r["replica_id"],
        gpus=r["gpus"],
        role=r["role"],
        node=r.get("node", "local"),
      ))

    kv_data = data.get("kv_layout", {})
    kv_layout = MlaKvLayout(
      kv_lora_rank=kv_data.get("kv_lora_rank", 512),
      qk_rope_head_dim=kv_data.get("qk_rope_head_dim", 64),
      num_layers=kv_data.get("num_layers", 61),
      page_size=kv_data.get("page_size", 16),
    )

    return cls(
      model_path=data["model_path"],
      model_family=data.get("model_family", "deepseek"),
      replicas=replicas,
      kv_layout=kv_layout,
      max_batch_size=data.get("max_batch_size", 256),
      max_prefill_tokens=data.get("max_prefill_tokens", 8192),
      max_seq_len=data.get("max_seq_len", 32768),
      gpu_memory_utilization=data.get("gpu_memory_utilization", 0.9),
      num_pages=data.get("num_pages", 0),
      master_addr=data.get("master_addr", os.environ.get("MASTER_ADDR", "")),
      master_port=data.get("master_port", int(os.environ.get("MASTER_PORT", "0"))),
      host=data.get("host", os.environ.get("SERVE_HOST", "")),
      port=data.get("port", int(os.environ.get("SERVE_PORT", "0"))),
      metrics_port=data.get("metrics_port", int(os.environ.get("METRICS_PORT", "0"))),
    )
