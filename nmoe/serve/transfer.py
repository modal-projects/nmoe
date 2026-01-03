# SPDX-License-Identifier: Apache-2.0
"""NIXL-based KV cache transfer for disaggregated serving."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import torch
from nixl._api import nixl_agent, nixl_agent_config

if TYPE_CHECKING:
  from nmoe.serve.cache import KvCache, MlaKvLayout


@dataclass
class TransferConfig:
  """Configuration for KV transfer."""
  backend: str = "UCX"
  timeout_ms: int = 5000
  poll_interval_ms: int = 1


@dataclass
class PeerTensorInfo:
  """One tensor participating in page transfers."""
  base_addr: int
  gpu_id: int
  num_blocks: int
  bytes_per_block: int


class KvHandle:
  """Handle to registered KV tensors with per-page prepped descriptors."""

  def __init__(self, agent: NixlAgent, tensors: list[torch.Tensor]) -> None:
    self.agent = agent
    self.tensors = tensors
    self._reg_handle = None
    self._prepped_handle = None
    self.num_blocks: int = 0
    self.num_tensors: int = 0
    self._tensor_infos: list[PeerTensorInfo] = []

  def register(self) -> None:
    """Register tensors and create a prepped descriptor list.

    Descriptor ordering is tensor-major:
      desc_index = tensor_idx * num_blocks + block_idx
    """
    if self._prepped_handle is not None:
      return

    if not self.tensors:
      raise ValueError("No tensors provided for KV transfer.")

    num_blocks = int(self.tensors[0].size(0))
    for t in self.tensors:
      if int(t.size(0)) != num_blocks:
        raise ValueError("All KV transfer tensors must share the same num_blocks (dim0).")
      if t.device.type != "cuda":
        raise ValueError("KV transfer tensors must be CUDA tensors.")
      if not t.is_contiguous():
        raise ValueError("KV transfer tensors must be contiguous for page-strided descriptors.")

    self.num_blocks = num_blocks
    self.num_tensors = len(self.tensors)

    self._reg_handle = self.agent._agent.register_memory(self.tensors)

    descs: list[tuple[int, int, int]] = []
    self._tensor_infos = []
    for t in self.tensors:
      base = int(t.data_ptr())
      gpu_id = int(t.get_device())
      bytes_per_block = int(t.stride(0) * t.element_size())
      self._tensor_infos.append(
        PeerTensorInfo(
          base_addr=base,
          gpu_id=gpu_id,
          num_blocks=num_blocks,
          bytes_per_block=bytes_per_block,
        )
      )
      for i in range(num_blocks):
        descs.append((base + i * bytes_per_block, bytes_per_block, gpu_id))

    self._prepped_handle = self.agent._agent.prep_xfer_dlist("NIXL_INIT_AGENT", descs, "VRAM")

  def release(self) -> None:
    """Release NIXL resources."""
    if self._prepped_handle is not None:
      self._prepped_handle.release()
      self._prepped_handle = None
    if self._reg_handle is not None:
      self.agent._agent.deregister_memory(self._reg_handle)
      self._reg_handle = None

  def get_tensor_infos(self) -> list[PeerTensorInfo]:
    """Get tensor infos for peer exchange (must call register() first)."""
    if self._prepped_handle is None:
      raise RuntimeError("KV handle not registered.")
    return list(self._tensor_infos)


class NixlAgent:
  """NIXL agent using prepped descriptor pattern for efficient transfers."""

  def __init__(self, name: str, config: TransferConfig) -> None:
    self.name = name
    self.config = config
    self._agent = nixl_agent(name, nixl_agent_config(backends=[config.backend]))
    self._remote_prepped: dict[str, object] = {}

  def get_metadata(self) -> bytes:
    """Get agent metadata for peer exchange."""
    return self._agent.get_agent_metadata()

  def add_peer(self, metadata: bytes, tensor_infos: list[PeerTensorInfo]) -> str:
    """Add peer and prep remote descriptors (tensor-major ordering)."""
    peer_name = self._agent.add_remote_agent(metadata)
    if not tensor_infos:
      raise ValueError("Peer tensor_infos is empty.")
    # Validate consistent num_blocks across tensors.
    num_blocks = int(tensor_infos[0].num_blocks)
    for info in tensor_infos:
      if int(info.num_blocks) != num_blocks:
        raise ValueError("Peer tensor_infos must share the same num_blocks.")
    descs: list[tuple[int, int, int]] = []
    for info in tensor_infos:
      for i in range(int(info.num_blocks)):
        descs.append((int(info.base_addr) + i * int(info.bytes_per_block), int(info.bytes_per_block), int(info.gpu_id)))
    self._remote_prepped[peer_name] = self._agent.prep_xfer_dlist(peer_name, descs, "VRAM")
    return peer_name

  def remove_peer(self, peer_name: str) -> None:
    """Remove peer."""
    if peer_name in self._remote_prepped:
      self._remote_prepped[peer_name].release()
      del self._remote_prepped[peer_name]
      self._agent.remove_remote_agent(peer_name)

  async def transfer_pages(
    self,
    local_handle: KvHandle,
    peer_name: str,
    local_pages: List[int],
    remote_pages: List[int],
    direction: str = "WRITE",
    notif: bytes = b"",
  ) -> None:
    """Transfer pages (block indices) across all registered tensors."""
    if local_handle._prepped_handle is None:
      raise RuntimeError("Local KV handle is not registered.")

    if len(local_pages) != len(remote_pages):
      raise ValueError("local_pages and remote_pages must have the same length.")

    num_blocks = int(local_handle.num_blocks)
    num_tensors = int(local_handle.num_tensors)

    def expand(pages: List[int]) -> np.ndarray:
      idx: list[int] = []
      for t in range(num_tensors):
        base = t * num_blocks
        for p in pages:
          idx.append(base + int(p))
      return np.array(idx, dtype=np.int32)

    xfer = self._agent.make_prepped_xfer(
      direction,
      local_handle._prepped_handle,
      expand(local_pages),
      self._remote_prepped[peer_name],
      expand(remote_pages),
      notif,
    )
    state = self._agent.transfer(xfer)
    if state == "ERR":
      xfer.release()
      raise RuntimeError("Transfer post failed")
    await self._wait(xfer)

  async def _wait(self, xfer: object) -> None:
    """Poll until completion."""
    poll = self.config.poll_interval_ms / 1000.0
    timeout = self.config.timeout_ms / 1000.0
    elapsed = 0.0
    while elapsed < timeout:
      state = self._agent.check_xfer_state(xfer)
      if state == "DONE":
        xfer.release()
        return
      if state == "ERR":
        xfer.release()
        raise RuntimeError("Transfer failed")
      await asyncio.sleep(poll)
      elapsed += poll
    xfer.release()
    raise TimeoutError(f"Transfer timed out after {timeout}s")

  def get_notifications(self) -> dict[str, list[bytes]]:
    """Get pending notifications."""
    return self._agent.get_new_notifs()

  def send_notification(self, peer_name: str, msg: bytes) -> None:
    """Send notification to peer."""
    self._agent.send_notif(peer_name, msg)

  def shutdown(self) -> None:
    """Shutdown agent."""
    for peer in list(self._remote_prepped.keys()):
      self.remove_peer(peer)


class KvTransferManager:
  """Manages KV cache transfers between prefill and decode replicas."""

  def __init__(
    self,
    name: str,
    kv: Union[torch.Tensor, Sequence[torch.Tensor], "KvCache"],
    config: Optional[TransferConfig] = None,
  ) -> None:
    self._kv = kv
    self.config = config or TransferConfig()
    self.agent = NixlAgent(name, self.config)
    self._handle: Optional[KvHandle] = None

  def _tensors(self) -> list[torch.Tensor]:
    if isinstance(self._kv, torch.Tensor):
      return [self._kv]
    # KvCache fallback (legacy): transfer the single backing k_cache tensor.
    if hasattr(self._kv, "k_cache"):
      return [self._kv.k_cache]  # type: ignore[attr-defined]
    return list(self._kv)

  def initialize(self) -> None:
    """Initialize and register local KV cache."""
    self._handle = KvHandle(self.agent, self._tensors())
    self._handle.register()

  def get_metadata(self) -> bytes:
    """Get agent metadata for peer exchange."""
    return self.agent.get_metadata()

  def get_tensor_infos(self) -> list[PeerTensorInfo]:
    """Get local tensor infos for peer exchange."""
    if self._handle is None:
      raise RuntimeError("Transfer manager not initialized.")
    return self._handle.get_tensor_infos()

  def add_peer(self, metadata: bytes, tensor_infos: list[PeerTensorInfo]) -> str:
    """Add peer for transfer."""
    return self.agent.add_peer(metadata, tensor_infos)

  async def send_pages(
    self,
    peer_name: str,
    local_pages: List[int],
    remote_pages: Optional[List[int]] = None,
    notif: str = "",
  ) -> None:
    """Send pages to peer (prefill → decode)."""
    await self.agent.transfer_pages(
      self._handle,
      peer_name,
      local_pages,
      remote_pages or local_pages,
      "WRITE",
      notif.encode(),
    )

  async def recv_pages(
    self,
    peer_name: str,
    local_pages: List[int],
    remote_pages: Optional[List[int]] = None,
  ) -> None:
    """Receive pages from peer (decode ← prefill)."""
    await self.agent.transfer_pages(
      self._handle,
      peer_name,
      local_pages,
      remote_pages or local_pages,
      "READ",
    )

  def get_notifications(self) -> dict[str, list[bytes]]:
    """Get pending notifications."""
    return self.agent.get_notifications()

  def send_notification(self, peer_name: str, msg: str) -> None:
    """Send notification to peer."""
    self.agent.send_notification(peer_name, msg.encode())

  def shutdown(self) -> None:
    """Shutdown and release resources."""
    if self._handle is not None:
      self._handle.release()
      self._handle = None
    self.agent.shutdown()
