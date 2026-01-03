# SPDX-License-Identifier: Apache-2.0
"""KV cache management for FlashMLA with radix prefix tree."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Optional

import torch


def _fnv1a64(tokens: tuple[int, ...]) -> int:
  """Deterministic 64-bit FNV-1a over token ids (page fingerprint)."""
  h = 1469598103934665603  # offset basis
  for t in tokens:
    x = int(t) & 0xFFFFFFFFFFFFFFFF
    h ^= x
    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
  return h


@dataclass
class MlaKvLayout:
  """FlashMLA KV cache memory layout."""
  num_blocks: int       # Total pages in GPU memory
  block_size: int = 64  # Tokens per page
  h_kv: int = 1         # KV heads (1 for MQA mode)
  head_dim: int = 576   # 512 (NoPE/value) + 64 (RoPE)
  dtype: torch.dtype = torch.bfloat16

  @property
  def bytes_per_block(self) -> int:
    """Bytes per page block."""
    return self.block_size * self.h_kv * self.head_dim * self.dtype.itemsize

  @property
  def bytes_per_token(self) -> int:
    """Bytes per token."""
    return self.h_kv * self.head_dim * self.dtype.itemsize

  def block_byte_offset(self, block_idx: int) -> int:
    """Byte offset for a specific block in the cache tensor."""
    return block_idx * self.bytes_per_block


@dataclass
class CacheHandle:
  """Handle to a prefix cache entry."""
  cached_len: int
  node: Optional["RadixNode"] = None


class RadixNode:
  """Node in the radix prefix tree for KV cache sharing (page-granular)."""
  _counter: int = 0

  def __init__(self, timestamp: Optional[float] = None) -> None:
    self.children: dict[int, RadixNode] = {}
    self._parent: Optional[RadixNode] = None
    self.ref_count: int = 0
    self.uuid = RadixNode._counter
    RadixNode._counter += 1
    self.timestamp = timestamp or time.monotonic()
    self._key: list[int] = []
    self._page_tokens: list[tuple[int, ...]] = []
    self._value: torch.Tensor = torch.empty(0, dtype=torch.int32)
    self._length: int = 0

  def set_key_value(self, key: list[int], page_tokens: list[tuple[int, ...]], value: torch.Tensor) -> None:
    assert len(key) == len(value)
    assert len(page_tokens) == len(key)
    self._key = key
    self._page_tokens = page_tokens
    self._value = value
    self._length = len(key)

  def set_parent(self, parent: "RadixNode") -> None:
    self._parent = parent
    if len(self._key) > 0:
      parent.children[self._key[0]] = self

  @property
  def length(self) -> int:
    return self._length

  @property
  def parent(self) -> "RadixNode":
    assert self._parent is not None
    return self._parent

  @property
  def value(self) -> torch.Tensor:
    return self._value

  def is_root(self) -> bool:
    return self._parent is None

  def is_leaf(self) -> bool:
    return len(self.children) == 0

  def get_match_len(self, tokens: list[int], start: int) -> int:
    raise NotImplementedError("RadixNode.get_match_len is page-granular; use get_page_match_len.")

  def get_page_match_len(self, page_hashes: list[int], page_tokens: list[tuple[int, ...]], start: int) -> int:
    min_len = min(len(self._key), len(page_hashes) - start)
    for i in range(min_len):
      j = start + i
      if self._key[i] != page_hashes[j]:
        return i
      # Collision-safe: verify actual tokens for this page.
      if self._page_tokens[i] != page_tokens[j]:
        return i
    return min_len

  def split_at(self, pos: int) -> "RadixNode":
    assert 0 < pos < self.length
    parent = self.parent
    new_node = RadixNode(self.timestamp)
    new_node.set_key_value(self._key[:pos], self._page_tokens[:pos], self._value[:pos])
    new_node.set_parent(parent)
    new_node.ref_count = self.ref_count
    self.set_key_value(self._key[pos:], self._page_tokens[pos:], self._value[pos:])
    self.set_parent(new_node)
    return new_node

  def __lt__(self, other: "RadixNode") -> bool:
    return self.timestamp < other.timestamp


class RadixCache:
  """Radix tree for prefix caching with LRU eviction."""

  def __init__(self, device: torch.device, page_size: int) -> None:
    self.device = device
    self.page_size = int(page_size)
    self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
    self.root = RadixNode()
    self.root.ref_count = 1
    self.evictable_size = 0
    self.protected_size = 0

  def _pages(self, input_ids: torch.Tensor) -> tuple[list[int], list[tuple[int, ...]]]:
    """Return (page_hashes, page_tokens) for full pages only."""
    assert input_ids.device.type == "cpu", "Prefix cache requires CPU input_ids."
    T = int(input_ids.numel())
    P = T // self.page_size
    if P <= 0:
      return [], []
    tokens = [int(x) for x in input_ids[: P * self.page_size].tolist()]
    pages: list[tuple[int, ...]] = []
    hashes: list[int] = []
    for p in range(P):
      lo = p * self.page_size
      hi = lo + self.page_size
      pt = tuple(tokens[lo:hi])
      pages.append(pt)
      hashes.append(_fnv1a64(pt))
    return hashes, pages

  def match_prefix(self, input_ids: torch.Tensor) -> tuple[CacheHandle, torch.Tensor]:
    page_hashes, page_tokens = self._pages(input_ids)
    if not page_hashes:
      return CacheHandle(0, self.root), self.empty_tensor

    node, prefix_pages = self._walk(page_hashes, page_tokens)
    if prefix_pages == 0:
      return CacheHandle(0, self.root), self.empty_tensor
    value_list: list[torch.Tensor] = []
    matched_node = node
    while not node.is_root():
      value_list.append(node.value)
      node = node.parent
    value_list.reverse()
    return CacheHandle(prefix_pages * self.page_size, matched_node), torch.cat(value_list)

  def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    page_hashes, page_tokens = self._pages(input_ids)
    if not page_hashes:
      return 0

    _require_page_ids = (indices.dtype == torch.int32 and indices.device.type == "cpu")
    assert _require_page_ids, "Prefix cache indices must be CPU int32 page ids."
    assert len(indices) == len(page_hashes), "indices must contain one page id per full page in input_ids."

    node, prefix_pages = self._walk(page_hashes, page_tokens)
    if prefix_pages < len(page_hashes):
      new_node = RadixNode()
      new_node.set_key_value(
        page_hashes[prefix_pages:],
        page_tokens[prefix_pages:],
        indices[prefix_pages:],
      )
      new_node.set_parent(node)
      self.evictable_size += new_node.length
    return prefix_pages * self.page_size

  def lock(self, handle: CacheHandle) -> None:
    if handle.node is None:
      return
    node = handle.node
    while not node.is_root():
      if node.ref_count == 0:
        self.evictable_size -= node.length
        self.protected_size += node.length
      node.ref_count += 1
      node = node.parent

  def unlock(self, handle: CacheHandle) -> None:
    if handle.node is None:
      return
    node = handle.node
    while not node.is_root():
      node = node.parent
      node.ref_count -= 1
      assert node.ref_count >= 0
      if node.ref_count == 0:
        self.evictable_size += node.length
        self.protected_size -= node.length

  def evict(self, size: int) -> torch.Tensor:
    if size == 0:
      return self.empty_tensor
    assert size <= self.evictable_size, \
      f"Cannot evict {size}, only {self.evictable_size} evictable"
    leaves = self._collect_evictable_leaves()
    heapq.heapify(leaves)
    evicted: list[torch.Tensor] = []
    evicted_size = 0
    while evicted_size < size:
      assert leaves, f"Need {size}, only evicted {evicted_size}"
      node = heapq.heappop(leaves)
      assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
      evicted_size += node.length
      evicted.append(node.value)
      self.evictable_size -= node.length
      parent = node.parent
      del parent.children[node._key[0]]
      if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
        heapq.heappush(leaves, parent)
    return torch.cat(evicted)

  def _walk(self, page_hashes: list[int], page_tokens: list[tuple[int, ...]]) -> tuple[RadixNode, int]:
    prefix_pages = 0
    node = self.root
    timestamp = time.monotonic()
    while prefix_pages < len(page_hashes):
      page_hash = page_hashes[prefix_pages]
      if page_hash not in node.children:
        return node, prefix_pages
      node = node.children[page_hash]
      match_len = node.get_page_match_len(page_hashes, page_tokens, prefix_pages)
      prefix_pages += match_len
      if match_len != node.length:
        node = node.split_at(match_len)
        return node, prefix_pages
      node.timestamp = timestamp
    return node, prefix_pages

  def _collect_evictable_leaves(self) -> list[RadixNode]:
    leaves: list[RadixNode] = []
    stack = [self.root]
    while stack:
      node = stack.pop()
      if node.is_leaf():
        if node.ref_count == 0 and not node.is_root():
          leaves.append(node)
      else:
        stack.extend(node.children.values())
    return leaves


class PageAllocator:
  """Free-list allocator for KV cache pages."""

  def __init__(self, num_pages: int, device: torch.device) -> None:
    self.device = device
    self.num_pages = num_pages
    self._free = torch.arange(num_pages, dtype=torch.int32, device=device)

  @property
  def available(self) -> int:
    return len(self._free)

  def allocate(self, n: int) -> torch.Tensor:
    assert n <= len(self._free), f"Cannot allocate {n}, only {len(self._free)} free"
    allocated = self._free[:n]
    self._free = self._free[n:]
    return allocated

  def free(self, indices: torch.Tensor) -> None:
    if len(indices) > 0:
      self._free = torch.cat([self._free, indices.to(self.device)])


class KvCache:
  """
  FlashMLA-compatible KV cache with paged allocation and prefix caching.

  The cache tensor has shape (num_blocks, block_size, h_kv, head_dim) and
  is compatible with flash_mla_with_kvcache().
  """

  def __init__(
    self,
    layout: MlaKvLayout,
    device: torch.device,
    enable_prefix_cache: bool = True,
  ) -> None:
    self.layout = layout
    self.device = device

    # Metadata (page ids / prefix tree) is CPU-owned to avoid GPU sync on list/tensor
    # conversions in the scheduler hot path.
    meta_device = torch.device("cpu")
    self.allocator = PageAllocator(layout.num_blocks, meta_device)
    self.radix = RadixCache(meta_device, layout.block_size) if enable_prefix_cache else None

    # Main KV cache tensor - FlashMLA format
    self.k_cache = torch.zeros(
      layout.num_blocks,
      layout.block_size,
      layout.h_kv,
      layout.head_dim,
      dtype=layout.dtype,
      device=device,
    )

  @property
  def available(self) -> int:
    evictable = self.radix.evictable_size if self.radix else 0
    return self.allocator.available + evictable

  def match_prefix(self, input_ids: torch.Tensor) -> tuple[CacheHandle, torch.Tensor]:
    if self.radix is None:
      return CacheHandle(0, None), torch.empty(0, dtype=torch.int32, device="cpu")
    return self.radix.match_prefix(input_ids)

  def lock(self, handle: CacheHandle) -> None:
    if self.radix:
      self.radix.lock(handle)

  def unlock(self, handle: CacheHandle) -> None:
    if self.radix:
      self.radix.unlock(handle)

  def allocate(self, n: int) -> torch.Tensor:
    if n <= self.allocator.available:
      return self.allocator.allocate(n)
    need = n - self.allocator.available
    if self.radix:
      evicted = self.radix.evict(need)
      self.allocator.free(evicted)
    return self.allocator.allocate(n)

  def free(self, indices: torch.Tensor) -> None:
    self.allocator.free(indices)

  def insert_and_free(
    self,
    handle: CacheHandle,
    input_ids: torch.Tensor,
    indices: torch.Tensor,
  ) -> None:
    if self.radix is None:
      self.allocator.free(indices)
      return

    page_size = int(self.layout.block_size)
    full_pages = int(input_ids.numel()) // page_size
    if full_pages > 0:
      # Only cache full pages to avoid sharing partial pages that would be overwritten.
      input_ids_full = input_ids[: full_pages * page_size]
      pages_full = indices[:full_pages]
      self.radix.insert_prefix(input_ids_full, pages_full)

    # Release the original matched prefix lock (if any).
    self.radix.unlock(handle)

    # Free tail pages (prompt remainder + generated) that we did not insert into cache.
    tail = indices[full_pages:]
    self.allocator.free(tail)

  def get_block_data(self, block_idx: int) -> torch.Tensor:
    """Get data for a single block (for NIXL transfer)."""
    return self.k_cache[block_idx]

  def get_blocks_data(self, block_indices: torch.Tensor) -> torch.Tensor:
    """Get data for multiple blocks (for NIXL transfer)."""
    return self.k_cache[block_indices]

  def set_blocks_data(self, block_indices: torch.Tensor, data: torch.Tensor) -> None:
    """Set data for multiple blocks (from NIXL transfer)."""
    self.k_cache[block_indices] = data
