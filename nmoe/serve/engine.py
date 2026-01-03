# SPDX-License-Identifier: Apache-2.0
"""Inference engine for DeepSeekV3 with FlashMLA + DeepEP."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import tempfile
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deep_ep import Buffer
from nmoe.serve.model import DeepSeekV3, ModelConfig, init_distributed
from nmoe.serve.types import Batch, ForwardOutput, Request, SamplingParams
from nmoe.serve.types import OutputMode

@dataclass
class BatchSamplingArgs:
    """Prepared sampling arguments for a batch."""
    temperatures: Optional[torch.Tensor]
    top_k: Optional[list[int]]
    top_p: Optional[list[float]]
    seeds: Optional[list[Optional[int]]]


# -----------------------------------------------------------------------------
# Context for model forward
# -----------------------------------------------------------------------------


@dataclass
class EngineContext:
    """Context passed to model during forward."""
    page_size: int
    _batch: Optional[Batch] = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch"
        return self._batch

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward not allowed"
        try:
            self._batch = batch
            yield
        finally:
            self._batch = None


_CTX: Optional[EngineContext] = None


def set_ctx(ctx: EngineContext) -> None:
    global _CTX
    _CTX = ctx


def get_ctx() -> EngineContext:
    assert _CTX is not None, "Context not set"
    return _CTX


# -----------------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------------


class Sampler:
    """Token sampler with temperature, top-k, top-p support."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """Prepare sampling args from batch requests."""
        reqs = batch.reqs
        # Check if all greedy
        if all(r.sampling_params.temperature <= 0 for r in reqs):
            return BatchSamplingArgs(None, None, None, None)

        MIN_T = 1e-5
        temps = torch.tensor(
            [max(r.sampling_params.temperature, MIN_T) for r in reqs],
            dtype=torch.float32,
            device=self.device,
        )
        top_k = [int(r.sampling_params.top_k) for r in reqs]
        if not any(k > 0 for k in top_k):
            top_k = None
        top_p = [float(r.sampling_params.top_p) for r in reqs]
        if not any(p < 1.0 for p in top_p):
            top_p = None
        seeds = [r.sampling_params.seed for r in reqs]
        return BatchSamplingArgs(temps, top_k, top_p, seeds)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """Sample next tokens from logits."""
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)
        return self._sample_with_params(logits, args)

    def _sample_with_params(
        self, logits: torch.Tensor, args: BatchSamplingArgs
    ) -> torch.Tensor:
        B = logits.size(0)
        # Temperature
        logits = logits / args.temperatures.unsqueeze(-1)

        # Top-k per request
        if args.top_k is not None:
            for i in range(B):
                k = int(args.top_k[i])
                if k > 0:
                    k = min(k, logits.size(-1))
                    thresh = torch.topk(logits[i], k)[0][-1]
                    logits[i] = logits[i].masked_fill(logits[i] < thresh, float('-inf'))

        # Top-p per request
        if args.top_p is not None:
            for i in range(B):
                p = float(args.top_p[i])
                if p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits[i], descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    sorted_logits[mask] = float('-inf')
                    logits[i] = sorted_logits.scatter(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)

        # Sample with optional per-request seeds
        if args.seeds and any(s is not None for s in args.seeds):
            tokens = torch.empty(B, dtype=torch.int64, device=self.device)
            for i in range(B):
                if args.seeds[i] is not None:
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(args.seeds[i])
                    tokens[i] = torch.multinomial(probs[i], 1, generator=gen)
                else:
                    tokens[i] = torch.multinomial(probs[i], 1)
            return tokens
        return torch.multinomial(probs, 1).squeeze(-1)


# -----------------------------------------------------------------------------
# Engine configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineConfig:
    """Configuration for inference engine."""
    num_pages: int
    page_size: int = 64
    num_layers: int = 61
    kv_lora_rank: int = 512
    qk_rope_head_dim: int = 64
    max_batch_size: int = 256
    max_seq_len: int = 32768
    # Max tokens per scheduler step (used for buffer sizing in future CUDA-graph work).
    # For non-chunked prefill, this should be >= the largest prompt segment we admit per batch.
    max_step_tokens: int = 16384

    # Attention type: "dsa" (Speciale) or "mla" (V3-0324, Kimi-K2)
    attention_type: str = "dsa"

    # DSA-specific (only used when attention_type="dsa")
    idx_dim: int = 128  # DSA indexer dimension

    # Tensor parallelism size for attention heads. Default 1 for TP=1/EP=N mode.
    # Set to world_size for full TP (splits attention heads across GPUs).
    tp_size: int = 1


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------


class Engine:
    """
    Inference engine for DeepSeekV3.

    Owns model, KV caches, and provides forward_batch() for the scheduler.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model_config = model_config
        self.engine_config = engine_config
        self.rank = rank
        self.world_size = world_size

        # Device and stream
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()

        # Distributed init
        init_distributed(rank, world_size, tp_size=engine_config.tp_size)
        self._init_deep_ep()

        # Model
        self.model = DeepSeekV3(model_config, self.buffer).to(self.device)
        self.model.eval()

        # KV caches: format depends on attention_type
        # DSA: kv_caches [num_pages, page_size, 1, 656] uint8 + idx_k_caches [num_pages, page_size, idx_dim] bf16
        # MLA: kv_caches_latent [num_pages, page_size, kv_lora_rank] bf16 + kv_caches_rope [num_pages, page_size, qk_rope_head_dim] bf16
        self.attention_type = engine_config.attention_type

        if self.attention_type == "dsa":
            self.kv_caches: List[torch.Tensor] = []
            self.idx_k_caches: List[torch.Tensor] = []
            for _ in range(engine_config.num_layers):
                kv = torch.zeros(
                    engine_config.num_pages,
                    engine_config.page_size,
                    1,
                    656,
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.kv_caches.append(kv)
                idx_k = torch.zeros(
                    engine_config.num_pages,
                    engine_config.page_size,
                    engine_config.idx_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.idx_k_caches.append(idx_k)
        else:
            # MLA: separate latent and rope caches
            self.kv_caches_latent: List[torch.Tensor] = []
            self.kv_caches_rope: List[torch.Tensor] = []
            for _ in range(engine_config.num_layers):
                latent = torch.zeros(
                    engine_config.num_pages,
                    engine_config.page_size,
                    engine_config.kv_lora_rank,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.kv_caches_latent.append(latent)
                rope = torch.zeros(
                    engine_config.num_pages,
                    engine_config.page_size,
                    engine_config.qk_rope_head_dim,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self.kv_caches_rope.append(rope)

        # Page table: [max_batch_size, max_pages_per_seq]
        max_pages = (engine_config.max_seq_len + engine_config.page_size - 1) // engine_config.page_size
        self.page_table = torch.zeros(
            engine_config.max_batch_size,
            max_pages,
            dtype=torch.int32,
            device=self.device,
        )

        # Context and sampler
        self.ctx = EngineContext(page_size=engine_config.page_size)
        set_ctx(self.ctx)
        self.sampler = Sampler(self.device)

        # Decode CUDA graph cache (optional; enabled by orchestrator).
        self.enable_cuda_graph: bool = False
        self._decode_graphs: dict[int, "_DecodeGraph"] = {}

    def _init_deep_ep(self) -> None:
        """Initialize DeepEP buffer for MoE."""
        # DeepEP requires a process group even for single-GPU
        if not dist.is_initialized():
            if self.world_size == 1:
                tmp = tempfile.NamedTemporaryFile(prefix="nmoe_pg_", suffix=".tmp", delete=False)
                tmp.close()
                init_method = f"file://{tmp.name}"
            else:
                master_addr = os.environ.get("MASTER_ADDR", "")
                master_port = os.environ.get("MASTER_PORT", "")
                if master_addr.startswith(("tcp://", "file://")):
                    init_method = master_addr
                elif master_addr and master_port:
                    init_method = f"tcp://{master_addr}:{master_port}"
                else:
                    raise ValueError(
                        "Distributed init requires MASTER_ADDR/MASTER_PORT (or MASTER_ADDR as full init_method)."
                    )
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=self.world_size,
                rank=self.rank,
            )
        self.buffer = Buffer(
            group=dist.group.WORLD,
            num_nvl_bytes=max(
                Buffer.get_dispatch_config(self.world_size).get_nvl_buffer_size_hint(
                    int(self.model_config.hidden_size) * 2, self.world_size
                ),
                Buffer.get_combine_config(self.world_size).get_nvl_buffer_size_hint(
                    int(self.model_config.hidden_size) * 2, self.world_size
                ),
            ),
            num_rdma_bytes=0,
        )

    def forward_batch(self, batch: Batch) -> ForwardOutput:
        """
        Execute forward pass on batch.

        Returns logits and sampled next tokens (GPU + async CPU copy).
        """
        with torch.cuda.stream(self.stream):
            with self.ctx.forward_batch(batch):
                _require_batch = (
                    batch.input_ids is not None
                    and batch.positions is not None
                    and batch.out_loc is not None
                )
                if not _require_batch:
                    raise ValueError("Batch tensors must be prepared by scheduler.")
                B, S = batch.input_ids.shape
                block_table = batch.block_table if batch.block_table is not None else self._build_block_table(batch)
                cache_seqlens_cpu = batch.cache_seqlens_cpu
                if cache_seqlens_cpu is None:
                    cache_seqlens_cpu = [int(r.cached_len + S) for r in batch.reqs]
                cache_seqlens = batch.cache_seqlens
                if cache_seqlens is None:
                    cache_seqlens = torch.tensor(cache_seqlens_cpu, dtype=torch.int32, device=self.device)

                # MLA path: prefill_mode determines attention kernel
                # - "dense": FA4 for initial prefill (no cached KV)
                # - "paged": CuTeDSL token-parallel prefill (cache hit / later chunks)
                # - None: CuTeDSL decode (S==1)
                if self.attention_type == "dsa":
                    prefill_mode = None
                elif batch.is_prefill:
                    prefill_mode = "paged" if batch.reqs[0].cached_len > 0 else "dense"
                else:
                    prefill_mode = None

                logits = self._forward_model(
                    batch.input_ids,
                    batch.positions,
                    out_loc=batch.out_loc,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    cache_seqlens_cpu=cache_seqlens_cpu,
                    prefill_mode=prefill_mode,
                )

            # Sample (must be TP-correct for vocab-parallel lm_head)
            sample_args = self.sampler.prepare(batch)
            # Get last-token logits for decode, or last per sequence for prefill
            sample_logits = logits[:, -1, :] if logits.dim() == 3 else logits

            # Enforce a single output mode per batch (profile contract).
            output_mode = batch.reqs[0].forward_spec.output_mode
            for r in batch.reqs[1:]:
                if r.forward_spec.output_mode != output_mode:
                    raise ValueError("Mixed output_mode in one batch is not supported.")

            if output_mode not in (OutputMode.TOKENS, OutputMode.LOGITS):
                raise NotImplementedError(
                    f"OutputMode={output_mode} is not implemented in Engine yet. "
                    "Supported: TOKENS (serving), LOGITS (teacher/analysis)."
                )

            full_logits: Optional[torch.Tensor] = None
            if self.world_size > 1 and (output_mode == OutputMode.LOGITS or sample_args.temperatures is not None):
                # If we need full-vocab logits (e.g., distillation, non-greedy sampling), all-gather shards.
                full_logits = self._all_gather_vocab_shards(sample_logits)

            # Optional CUDA graph fast path: decode + greedy only.
            if (
                self.enable_cuda_graph
                and batch.is_decode
                and output_mode == OutputMode.TOKENS
                and sample_args.temperatures is None
                and batch.block_table is not None
                and batch.cache_seqlens is not None
            ):
                g = self._decode_graphs.get(B)
                if g is None:
                    g = _DecodeGraph(self, B)
                    self._decode_graphs[B] = g
                local_max, local_gid, copy_event = g.run(
                    input_ids=batch.input_ids,
                    positions=batch.positions,
                    out_loc=batch.out_loc,
                    block_table=batch.block_table,
                    cache_seqlens=batch.cache_seqlens,
                )
                if self.world_size == 1:
                    next_tokens_gpu = local_gid.to(torch.int32)
                else:
                    next_tokens_gpu = self._tp_greedy_from_local(local_max, local_gid).to(torch.int32)
                next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
                return ForwardOutput(
                    logits=sample_logits,  # unused by orchestrator in TOKENS mode
                    next_tokens_gpu=next_tokens_gpu,
                    next_tokens_cpu=next_tokens_cpu,
                    copy_event=copy_event,
                )

            if self.world_size > 1 and sample_args.temperatures is None:
                # Greedy decode can be done without full logits by reducing local argmax.
                next_tokens_gpu = self._tp_greedy_argmax(sample_logits).to(torch.int32)
            else:
                # Sampling requires full vocab logits.
                if full_logits is None:
                    full_logits = sample_logits
                next_tokens_gpu = self.sampler.sample(full_logits, sample_args).to(torch.int32)

            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
            copy_event = torch.cuda.Event()
            copy_event.record()

        return ForwardOutput(
            logits=full_logits if output_mode == OutputMode.LOGITS else sample_logits,
            next_tokens_gpu=next_tokens_gpu,
            next_tokens_cpu=next_tokens_cpu,
            copy_event=copy_event,
        )


    def _forward_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        out_loc: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_seqlens_cpu: Optional[list[int]],
        prefill_mode: Optional[str],
    ) -> torch.Tensor:
        # Forward (cache format depends on attention_type). Keep this as the single
        # call-site used by both eager and CUDA-graph decode.
        if self.attention_type == "dsa":
            return self.model(
                input_ids,
                positions,
                kv_caches=self.kv_caches,
                idx_k_caches=self.idx_k_caches,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cache_seqlens_cpu=cache_seqlens_cpu,
                out_loc=out_loc,
            )
        return self.model(
            input_ids,
            positions,
            kv_caches_latent=self.kv_caches_latent,
            kv_caches_rope=self.kv_caches_rope,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            out_loc=out_loc,
            prefill_mode=prefill_mode,
        )

    def _all_gather_vocab_shards(self, local_logits: torch.Tensor) -> torch.Tensor:
        """All-gather vocab-parallel logits along the last dim."""
        if self.world_size == 1:
            return local_logits
        if not dist.is_initialized():
            raise RuntimeError("TP all-gather requires distributed initialized.")
        parts = [torch.empty_like(local_logits) for _ in range(self.world_size)]
        dist.all_gather(parts, local_logits.contiguous())
        return torch.cat(parts, dim=-1)

    def _tp_greedy_argmax(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Greedy argmax over vocab-parallel shards, returning global token ids."""
        if self.world_size == 1:
            return torch.argmax(local_logits, dim=-1)
        if not dist.is_initialized():
            raise RuntimeError("TP sampling requires distributed initialized.")
        B, v_shard = local_logits.shape
        vocab_size = int(self.model_config.vocab_size)
        if v_shard * self.world_size != vocab_size:
            raise ValueError(f"Vocab sharding mismatch: {v_shard}*{self.world_size} != {vocab_size}")

        start = self.rank * v_shard
        local_max, local_idx = torch.max(local_logits, dim=-1)  # [B], [B]
        local_gid = local_idx.to(torch.int64) + int(start)

        return self._tp_greedy_from_local(local_max, local_gid)

    def _tp_greedy_from_local(self, local_max: torch.Tensor, local_gid: torch.Tensor) -> torch.Tensor:
        """Reduce per-rank (local_max, local_gid) pairs to global token ids (tie: smallest id)."""
        if self.world_size == 1:
            return local_gid
        vocab_size = int(self.model_config.vocab_size)
        gathered_vals = [torch.empty_like(local_max) for _ in range(self.world_size)]
        gathered_gids = [torch.empty_like(local_gid) for _ in range(self.world_size)]
        dist.all_gather(gathered_vals, local_max.contiguous())
        dist.all_gather(gathered_gids, local_gid.contiguous())

        vals = torch.stack(gathered_vals, dim=0)  # [W,B]
        gids = torch.stack(gathered_gids, dim=0)  # [W,B]
        gmax = torch.max(vals, dim=0).values  # [B]

        # Tie-break deterministically by smallest global token id among max logits.
        mask = vals == gmax.unsqueeze(0)
        big = torch.full_like(gids, vocab_size + 1)
        candidates = torch.where(mask, gids, big)
        return torch.min(candidates, dim=0).values.to(torch.int64)

    def _build_block_table(self, batch: Batch) -> torch.Tensor:
        """Build block table tensor for this batch."""
        B = batch.size
        max_pages = self.page_table.size(1)
        block_table = torch.zeros((B, max_pages), dtype=torch.int32, device=self.device)
        for i, req in enumerate(batch.reqs):
            if not req.page_ids:
                continue
            num_pages = len(req.page_ids)
            block_table[i, :num_pages] = torch.tensor(
                req.page_ids,
                dtype=torch.int32,
                device=self.device,
            )
        return block_table

    def shutdown(self) -> None:
        """Clean up resources."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def kv_transfer_tensors(self) -> list[torch.Tensor]:
        """Return the per-layer paged KV tensors for NIXL transfer.

        Ordering is stable and tensor-major. All tensors share dim0=num_pages so
        transfer can use page indices as block ids.

        NOTE: This is a plumbing surface only; orchestrator/scheduler decide
        which page ids to move for disaggregation.
        """
        if self.attention_type == "dsa":
            # Transfer both FlashMLA packed KV and indexer K cache.
            out: list[torch.Tensor] = []
            out.extend(self.kv_caches)
            out.extend(self.idx_k_caches)
            return out
        out = []
        out.extend(self.kv_caches_latent)
        out.extend(self.kv_caches_rope)
        return out

    @property
    def vocab_size(self) -> int:
        return self.model_config.vocab_size

    @property
    def num_layers(self) -> int:
        return self.model_config.num_layers


class _DecodeGraph:
    """CUDA graph for decode+greedy for a fixed batch size (MLA-only).

    Captures the full forward + TP greedy argmax on static buffers and replays.

    NOTE: This is intentionally narrow: it is the only path that matters for
    LMSYS-grade decode throughput, and it avoids feature creep. DSA graphs are
    left explicit TODO (needs CPU-free cache_seqlens path first).
    """

    def __init__(self, engine: Engine, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if engine.attention_type == "dsa":
            raise NotImplementedError("CUDA graphs for DSA decode are not implemented (MLA-only for now).")
        self.engine = engine
        self.B = int(batch_size)
        self.device = engine.device
        self.stream = engine.stream

        max_pages = (engine.engine_config.max_seq_len + engine.engine_config.page_size - 1) // engine.engine_config.page_size

        # Static input buffers.
        self.input_ids = torch.empty((self.B, 1), dtype=torch.int64, device=self.device)
        self.positions = torch.empty((self.B, 1), dtype=torch.int64, device=self.device)
        self.out_loc = torch.empty((self.B, 1), dtype=torch.int32, device=self.device)
        self.block_table = torch.empty((self.B, max_pages), dtype=torch.int32, device=self.device)
        self.cache_seqlens = torch.empty((self.B,), dtype=torch.int32, device=self.device)

        # Captured outputs.
        self.local_max = torch.empty((self.B,), dtype=torch.float32, device=self.device)
        self.local_gid = torch.empty((self.B,), dtype=torch.int64, device=self.device)
        self.graph = torch.cuda.CUDAGraph()

        self._capture()

    def _capture(self) -> None:
        # Ensure static inputs are initialized (avoid NaNs in capture).
        self.input_ids.zero_()
        self.positions.zero_()
        self.out_loc.zero_()
        self.block_table.zero_()
        self.cache_seqlens.fill_(1)

        torch.cuda.synchronize(self.device)
        with torch.cuda.stream(self.stream):
            # Warmup forwards outside capture to stabilize allocator.
            for _ in range(3):
                logits = self.engine._forward_model(
                    self.input_ids,
                    self.positions,
                    out_loc=self.out_loc,
                    block_table=self.block_table,
                    cache_seqlens=self.cache_seqlens,
                    cache_seqlens_cpu=None,
                    prefill_mode=None,
                )
                _ = logits[:, -1, :]
            torch.cuda.synchronize(self.device)

            self.graph.capture_begin()
            logits = self.engine._forward_model(
                self.input_ids,
                self.positions,
                out_loc=self.out_loc,
                block_table=self.block_table,
                cache_seqlens=self.cache_seqlens,
                cache_seqlens_cpu=None,
                prefill_mode=None,
            )
            sample_logits = logits[:, -1, :]
            v_shard = int(sample_logits.size(-1))
            local_max, local_idx = torch.max(sample_logits, dim=-1)
            start = int(self.engine.rank) * v_shard
            local_gid = local_idx.to(torch.int64) + int(start)
            self.local_max.copy_(local_max)
            self.local_gid.copy_(local_gid)
            self.graph.capture_end()

    def run(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        self.input_ids.copy_(input_ids)
        self.positions.copy_(positions)
        self.out_loc.copy_(out_loc)
        self.block_table.copy_(block_table)
        self.cache_seqlens.copy_(cache_seqlens)

        self.graph.replay()

        copy_event = torch.cuda.Event()
        copy_event.record(self.stream)
        return self.local_max, self.local_gid, copy_event
