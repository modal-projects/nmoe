# SPDX-License-Identifier: Apache-2.0
"""Inference engine for DeepSeekV3 with FlashMLA + expert-parallel transport."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import socket
import tempfile
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

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
        # Deterministic per-request RNG: use (seed + step_idx) so a request
        # produces a stable token stream across steps regardless of batching.
        seeds: list[Optional[int]] = []
        for r in reqs:
            if r.sampling_params.seed is None:
                seeds.append(None)
                continue
            step_idx = len(r.output_ids)  # 0 for first sampled token, then increments
            seeds.append(int(r.sampling_params.seed) + int(step_idx))
        return BatchSamplingArgs(temps, top_k, top_p, seeds)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """Sample next tokens from logits."""
        tokens, _, _, _ = self.sample_with_aux(logits, args)
        return tokens

    def sample_with_aux(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        *,
        return_logprobs: bool = False,
        return_topk: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Sample tokens and optionally return logprobs / top-k logprobs.

        Returns:
          tokens: [B] int64
          token_logprobs: [B] float32 (optional)
          topk_ids: [B, K] int64 (optional)
          topk_logprobs: [B, K] float32 (optional)
        """
        want_topk = int(return_topk) > 0
        if args.temperatures is None:
            tokens = torch.argmax(logits, dim=-1)
            if not (return_logprobs or want_topk):
                return tokens, None, None, None
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_lp = log_probs.gather(1, tokens.view(-1, 1)).squeeze(1) if return_logprobs else None
            if want_topk:
                topk_lp, topk_ids = torch.topk(log_probs, int(return_topk), dim=-1)
                return tokens, token_lp, topk_ids.to(torch.int64), topk_lp
            return tokens, token_lp, None, None
        tokens, token_lp, topk_ids, topk_lp = self._sample_with_params(
            logits, args, return_logprobs=return_logprobs, return_topk=int(return_topk)
        )
        return tokens, token_lp, topk_ids, topk_lp

    def logprobs_for_tokens(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        tokens: torch.Tensor,
        *,
        return_topk: int = 0,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute logprobs (and optional top-k logprobs) for provided token ids.

        This applies the same temperature / top-k / top-p transforms as sampling,
        but does not draw tokens. It is used for teacher-forcing requests where
        tokens are forced externally but logprobs must match the serving policy.

        Args:
          logits: [B, V] float tensor
          args: batch sampling args (temperatures/top_k/top_p; seeds ignored)
          tokens: [B] int64 token ids to score
          return_topk: if >0, also return topk_ids/topk_logprobs under the same policy.

        Returns:
          token_logprobs: [B] float32
          topk_ids: [B, K] int64 (optional)
          topk_logprobs: [B, K] float32 (optional)
        """
        if tokens.ndim != 1:
            raise ValueError("tokens must be [B]")
        if logits.ndim != 2:
            raise ValueError("logits must be [B, V]")
        if logits.size(0) != tokens.size(0):
            raise ValueError("logits/tokens batch mismatch")

        want_topk = int(return_topk) > 0
        if args.temperatures is None:
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_lp = log_probs.gather(1, tokens.view(-1, 1)).squeeze(1)
            if want_topk:
                topk_lp, topk_ids = torch.topk(log_probs, int(return_topk), dim=-1)
                return token_lp, topk_ids.to(torch.int64), topk_lp
            return token_lp, None, None

        # Apply the same transforms as _sample_with_params (without seeds).
        logits2 = logits / args.temperatures.unsqueeze(-1)

        B = logits2.size(0)
        if args.top_k is not None:
            for i in range(B):
                k = int(args.top_k[i])
                if k > 0:
                    k = min(k, logits2.size(-1))
                    thresh = torch.topk(logits2[i], k)[0][-1]
                    logits2[i] = logits2[i].masked_fill(logits2[i] < thresh, float("-inf"))

        if args.top_p is not None:
            for i in range(B):
                p = float(args.top_p[i])
                if p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits2[i], descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    sorted_logits[mask] = float("-inf")
                    logits2[i] = sorted_logits.scatter(0, sorted_idx, sorted_logits)

        log_probs = F.log_softmax(logits2.float(), dim=-1)
        token_lp = log_probs.gather(1, tokens.view(-1, 1)).squeeze(1)
        if want_topk:
            topk_lp, topk_ids = torch.topk(log_probs, int(return_topk), dim=-1)
            return token_lp, topk_ids.to(torch.int64), topk_lp
        return token_lp, None, None

    def _sample_with_params(
        self,
        logits: torch.Tensor,
        args: BatchSamplingArgs,
        *,
        return_logprobs: bool,
        return_topk: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()

        # Sample with optional per-request seeds
        if args.seeds and any(s is not None for s in args.seeds):
            tokens = torch.empty(B, dtype=torch.int64, device=self.device)
            for i in range(B):
                if args.seeds[i] is not None:
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(args.seeds[i])
                    # Deterministic multinomial sampling: torch.multinomial can be
                    # nondeterministic on CUDA even with a seeded generator.
                    #
                    # Use CDF sampling via cumsum + searchsorted instead.
                    u = torch.rand((1,), device=self.device, dtype=torch.float32, generator=gen)
                    cdf = probs[i].cumsum(dim=-1)
                    cdf = cdf / cdf[-1].clamp_min(1e-12)
                    tokens[i] = torch.searchsorted(cdf, u, right=False)[0].to(torch.int64)
                else:
                    tokens[i] = torch.multinomial(probs[i], 1)
        else:
            tokens = torch.multinomial(probs, 1).squeeze(-1)

        token_lp = log_probs.gather(1, tokens.view(-1, 1)).squeeze(1) if return_logprobs else None
        if int(return_topk) > 0:
            topk_lp, topk_ids = torch.topk(log_probs, int(return_topk), dim=-1)
            return tokens, token_lp, topk_ids.to(torch.int64), topk_lp
        return tokens, token_lp, None, None


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
    # MoE masked-GEMM capacity per local expert (decode fast path). Must be a multiple
    # of 16; lower is faster but increases overflow risk (dropped expert pairs).
    moe_expected_m: int = 256

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
        mode = os.environ.get("NMOE_EP_TRANSPORT", "rdep").strip().lower()
        if mode == "deepep":
            self._init_deep_ep()
        elif mode == "rdep":
            import types as _types
            self.buffer = _types.SimpleNamespace()
            self._init_rdep_infer_transport()
        else:
            raise ValueError(f"Invalid NMOE_EP_TRANSPORT={mode!r}; expected 'rdep' or 'deepep'.")

        probe_flag = os.environ.get("NMOE_RDEP_LOAD_PROBE", "0") in ("1", "true", "True")
        self._rdep_load_probe_remaining: int = int(os.environ.get("NMOE_RDEP_LOAD_PROBE_STEPS", "1")) if probe_flag else 0

        # MoE decode grouped-capacity (masked GEMM fast path). Stored on the buffer so
        # MoE layers can read it without threading EngineConfig through the model.
        setattr(self.buffer, "_nmoe_masked_gemm_expected_m", int(engine_config.moe_expected_m))

        # Model
        self.model = DeepSeekV3(model_config, self.buffer).to(self.device)
        self.model.eval()

        # KV caches: format depends on attention_type
        # DSA: kv_caches [num_pages, page_size, 1, 656] uint8 + idx_k_caches [num_pages, page_size, idx_dim] bf16
        # MLA: kv_caches_latent [page_size, kv_lora_rank, num_pages] bf16 (CuTeDSL layout, kv_lora_rank stride=1)
        #      kv_caches_rope   [page_size, qk_rope_head_dim, num_pages] bf16 (CuTeDSL layout, rope_dim stride=1)
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
            # MLA: separate latent and rope caches (CuTeDSL layout; no per-layer staging copies).
            self.kv_caches_latent: List[torch.Tensor] = []
            self.kv_caches_rope: List[torch.Tensor] = []
            for _ in range(engine_config.num_layers):
                # Allocate KV caches in CuTeDSL layout with:
                #   - shape (page_size, D, num_pages)
                #   - stride(1)=1 (feature dimension contiguous)
                #   - pages are contiguous blocks in memory (for TMA-friendly access)
                #
                # Achieve this by allocating a contiguous (num_pages, page_size, D) tensor
                # and permuting to (page_size, D, num_pages).
                latent = (
                    torch.zeros(
                        engine_config.num_pages,
                        engine_config.page_size,
                        engine_config.kv_lora_rank,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    .permute(1, 2, 0)
                )
                self.kv_caches_latent.append(latent)
                rope = (
                    torch.zeros(
                        engine_config.num_pages,
                        engine_config.page_size,
                        engine_config.qk_rope_head_dim,
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    .permute(1, 2, 0)
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

        # Pinned CPU ring for async D2H token materialization (used by lockstep overlap).
        # Keep this internal and tiny; ring slots are selected by the orchestrator.
        self._cpu_ring_size: int = 2
        self._next_tokens_cpu_ring: list[torch.Tensor] = [
            torch.empty((engine_config.max_batch_size,), dtype=torch.int64, pin_memory=True)
            for _ in range(self._cpu_ring_size)
        ]
        # MoE overflow counter (dropped expert pairs due to expected_m clipping).
        # This is copied D2H once per step and logged by the orchestrator.
        self._moe_overflow_gpu = torch.zeros((1,), dtype=torch.int32, device=self.device)
        setattr(self.buffer, "_nmoe_moe_overflow_gpu", self._moe_overflow_gpu)
        self._moe_overflow_cpu_ring: list[torch.Tensor] = [
            torch.empty((1,), dtype=torch.int32, pin_memory=True)
            for _ in range(self._cpu_ring_size)
        ]

    def _init_rdep_infer_transport(self) -> None:
        """Initialize inference-RDEP transport for both decode and prefill.

        This replaces DeepEP notify/dispatch/combine with CUDA-IPC + tag-barrier
        transport. RDEP is the default transport for serve.
        """
        if self.world_size != 8:
            raise RuntimeError(f"RDEP inference transport requires world_size=8 (got {self.world_size}).")
        if int(self.engine_config.tp_size) != 1:
            raise RuntimeError(f"RDEP inference transport requires tp_size=1 (got {self.engine_config.tp_size}).")

        hidden = int(self.model_config.hidden_size)
        num_experts = int(self.model_config.num_experts)
        if num_experts != 256:
            raise RuntimeError(f"RDEP inference transport v0 expects num_experts=256 (got {num_experts}).")
        if (num_experts % self.world_size) != 0:
            raise RuntimeError("num_experts must be divisible by world_size.")
        n_local = num_experts // self.world_size
        topk = int(getattr(self.model_config, "num_experts_per_tok", 8))
        if topk != 8:
            raise RuntimeError(f"RDEP inference transport v0 expects topk=8 (got {topk}).")

        placement = os.environ.get("NMOE_EXPERT_PLACEMENT", "contiguous").strip().lower()
        if placement not in ("contiguous", "striped"):
            raise RuntimeError(f"Invalid NMOE_EXPERT_PLACEMENT={placement!r} (expected 'contiguous' or 'striped').")
        if placement == "striped" and int(self.world_size) != 8:
            raise RuntimeError(f"NMOE_EXPERT_PLACEMENT=striped requires world_size=8 (got {self.world_size}).")
        expert_placement = 1 if placement == "striped" else 0

        # Decode launch requirement for this week: global BS=256 => T_cap=32 per rank.
        t_cap_decode = 32
        if self.world_size * t_cap_decode != 256:
            raise RuntimeError("internal error: expected world_size=8, t_cap=32 => BS=256")

        from nmoe.csrc import rdep as _C
        from nmoe.csrc import infer_ipc
        from nmoe.rdep import Rdep

        # Bootstrap IPC barrier signal tables using the existing BF16 RDEP IPC init.
        # We keep a reference so the underlying allocation isn't freed.
        self._rdep_barrier = Rdep(dim=hidden, n_local=n_local, topk=topk, profile="bf16", capacity=4096)
        # Keep CUDA-IPC slabs alive for the lifetime of the Engine.
        self._infer_ipc_slabs: list[torch.Tensor] = []

        def _all_gather_mem_handle(t: torch.Tensor) -> np.ndarray:
            h_local = torch.from_numpy(_C.ipc_get_mem_handle(int(t.data_ptr()))).to(device=self.device, dtype=torch.uint8)
            all_h = [torch.empty_like(h_local) for _ in range(int(self.world_size))]
            dist.all_gather(all_h, h_local)
            return np.concatenate([x.detach().cpu().numpy() for x in all_h], axis=0)

        def _init_channel(*, channel: int, t_cap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            slab, barrier, recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_w, ret_y, slab_off = infer_ipc.alloc_infer_ipc_slab_fp8(
                int(self.world_size), int(t_cap), int(hidden), int(topk)
            )
            y_out = torch.empty((t_cap, hidden), device=self.device, dtype=torch.bfloat16)
            send_mask = torch.empty((t_cap,), device=self.device, dtype=torch.uint8)

            # One CUDA-IPC handle per peer (slab base) instead of per-tensor handles.
            slab_h = _all_gather_mem_handle(slab)
            self._infer_ipc_slabs.append(slab)

            _C.infer_init_ipc_slab_fp8_local(
                int(channel),
                int(self.rank),
                int(self.world_size),
                int(t_cap),
                int(hidden),
                int(topk),
                int(n_local),
                int(expert_placement),
                slab_h,
                slab_off.cpu().numpy(),
                slab.data_ptr(),
            )
            return barrier, recv_x_fp8, recv_x_scale, recv_topk_idx, recv_topk_w, ret_y, y_out, send_mask

        # Channel 0: decode (fixed BS=256 operating point).
        _bar0, xq0, xs0, ti0, tw0, ry0, y0, m0 = _init_channel(channel=0, t_cap=int(t_cap_decode))

        # Channel 1: prefill (token budget cap). This is not graph-captured.
        t_cap_prefill = int(self.engine_config.max_step_tokens)
        if t_cap_prefill <= 0:
            raise RuntimeError("EngineConfig.max_step_tokens must be > 0 for RDEP prefill transport.")
        _bar1, xq1, xs1, ti1, tw1, ry1, y1, m1 = _init_channel(channel=1, t_cap=int(t_cap_prefill))

        # Attach inference-RDEP buffers to the model EP transport object.
        setattr(self.buffer, "_nmoe_ep_transport", "rdep")
        setattr(self.buffer, "_nmoe_rdep_t_cap", int(t_cap_decode))  # legacy decode alias
        setattr(self.buffer, "_nmoe_rdep_prefill_t_cap", int(t_cap_prefill))

        setattr(self.buffer, "_nmoe_rdep_recv_x_fp8", xq0)
        setattr(self.buffer, "_nmoe_rdep_recv_x_scale", xs0)
        setattr(self.buffer, "_nmoe_rdep_recv_topk_idx", ti0)
        setattr(self.buffer, "_nmoe_rdep_recv_topk_w", tw0)
        setattr(self.buffer, "_nmoe_rdep_ret_y", ry0)
        setattr(self.buffer, "_nmoe_rdep_y_out", y0)
        setattr(self.buffer, "_nmoe_rdep_send_mask", m0)

        setattr(self.buffer, "_nmoe_rdep_prefill_recv_x_fp8", xq1)
        setattr(self.buffer, "_nmoe_rdep_prefill_recv_x_scale", xs1)
        setattr(self.buffer, "_nmoe_rdep_prefill_recv_topk_idx", ti1)
        setattr(self.buffer, "_nmoe_rdep_prefill_recv_topk_w", tw1)
        setattr(self.buffer, "_nmoe_rdep_prefill_ret_y", ry1)
        setattr(self.buffer, "_nmoe_rdep_prefill_y_out", y1)
        setattr(self.buffer, "_nmoe_rdep_prefill_send_mask", m1)

    def _init_deep_ep(self) -> None:
        """Initialize DeepEP buffer for MoE."""
        try:
            from deep_ep import Buffer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "NMOE_EP_TRANSPORT=deepep requested, but deep_ep is not available in this environment."
            ) from e
        def _is_multi_node(*, world_size: int) -> bool:
            if world_size <= 1 or not dist.is_initialized():
                return False
            try:
                ctrl_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
                host = socket.gethostname()
                hosts: list[str] = ["" for _ in range(world_size)]
                dist.all_gather_object(hosts, host, group=ctrl_group)
                return len(set(hosts)) > 1
            except Exception:
                # Best-effort. If detection fails, prefer single-node behavior (LL off).
                return False

        self.ll_max_dispatch_tokens_per_rank: int = 0
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
        hidden = int(self.model_config.hidden_size)
        num_experts = int(self.model_config.num_experts)
        if self.world_size > 1 and (num_experts % self.world_size) != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by world_size ({self.world_size})")
        num_local_experts = (num_experts // self.world_size) if self.world_size > 1 else num_experts

        num_nvl_bytes = max(
            Buffer.get_dispatch_config(self.world_size).get_nvl_buffer_size_hint(hidden * 2, self.world_size),
            Buffer.get_combine_config(self.world_size).get_nvl_buffer_size_hint(hidden * 2, self.world_size),
        )

        # EP decode wants DeepEP low-latency dispatch/combine. This requires:
        # - low_latency_mode=True
        # - an RDMA buffer sized via DeepEP’s size hint
        # - num_qps_per_rank == num_local_experts
        ll_env = os.environ.get("NMOE_DEEPEP_LOW_LATENCY", "auto").strip().lower()
        if ll_env in ("1", "true", "yes"):
            ll_mode = "1"
        elif ll_env in ("0", "false", "no"):
            ll_mode = "0"
        elif ll_env == "auto":
            ll_mode = "auto"
        else:
            raise ValueError(f"Invalid NMOE_DEEPEP_LOW_LATENCY={ll_env!r}; expected 0/1/auto.")

        enable_ll = False
        if self.world_size > 1 and int(self.engine_config.tp_size) == 1:
            if ll_mode == "1":
                enable_ll = True
            elif ll_mode == "auto":
                enable_ll = _is_multi_node(world_size=int(self.world_size))

        if enable_ll:
            # Max tokens dispatched per rank in a decode step; must be the same on all ranks.
            # EngineConfig.max_batch_size is a per-rank cap (scheduler enforces this in
            # schedule_decode). Use it directly for DeepEP LL buffer sizing.
            self.ll_max_dispatch_tokens_per_rank = int(self.engine_config.max_batch_size)
            if self.ll_max_dispatch_tokens_per_rank <= 0:
                raise ValueError("EngineConfig.max_batch_size must be > 0 for DeepEP low-latency mode.")
            # DeepEP internode_ll uses FINISHED_SUM_TAG=1024; do not exceed.
            if self.ll_max_dispatch_tokens_per_rank > 1024:
                raise ValueError(
                    f"DeepEP low-latency requires max_dispatch_tokens_per_rank <= 1024 "
                    f"(got {self.ll_max_dispatch_tokens_per_rank})."
                )
            num_rdma_bytes = int(
                Buffer.get_low_latency_rdma_size_hint(
                    int(self.ll_max_dispatch_tokens_per_rank), hidden, self.world_size, num_experts
                )
            )
            self.buffer = Buffer(
                group=dist.group.WORLD,
                num_nvl_bytes=int(num_nvl_bytes),
                num_rdma_bytes=int(num_rdma_bytes),
                low_latency_mode=True,
                allow_nvlink_for_low_latency_mode=True,  # Use NVLink path on single-node
                num_qps_per_rank=int(num_local_experts),
                explicitly_destroy=True,
            )
            # Internal-only: let MoE read the sizing without threading EngineConfig through.
            self.buffer._nmoe_ll_max_dispatch_tokens_per_rank = int(self.ll_max_dispatch_tokens_per_rank)
        else:
            self.buffer = Buffer(
                group=dist.group.WORLD,
                num_nvl_bytes=int(num_nvl_bytes),
                num_rdma_bytes=0,
                low_latency_mode=False,
                explicitly_destroy=True,
            )

    def forward_batch(self, batch: Batch, *, copy_slot: Optional[int] = None) -> ForwardOutput:
        """Execute one forward pass for a scheduled batch.

        Returns:
          ForwardOutput with sampled tokens (and optional aux outputs depending on profile).
        """
        with torch.inference_mode():
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

                    # Enforce a single output mode per batch (profile contract) early
                    # so we can steer MoE kernel choices before the forward.
                    output_mode = batch.reqs[0].forward_spec.output_mode
                    for r in batch.reqs[1:]:
                        if r.forward_spec.output_mode != output_mode:
                            raise ValueError("Mixed output_mode in one batch is not supported.")

                    # Allow fused MoE pack only for TOKENS mode. LOGPROBS/TOPK/LOGITS
                    # profiles prefer deterministic routing/packing order to keep
                    # auxiliary outputs stable under FP8/atomics.
                    setattr(self.buffer, "_nmoe_allow_fused_pack", output_mode == OutputMode.TOKENS)
                    # Reset per-step overflow counter before the model forward.
                    if self._moe_overflow_gpu is not None:
                        self._moe_overflow_gpu.zero_()

                    # Prepare sampling args (CPU-only).
                    sample_args = self.sampler.prepare(batch)

                    # Optional CUDA graph fast path: decode + greedy TOKENS only.
                    #
                    # This is the only path that can plausibly close the launch-overhead gap for
                    # LMSYS-grade decode throughput. Keep it narrow and fail-fast.
                    transport = getattr(self.buffer, "_nmoe_ep_transport", "deepep")
                    do_rdep_probe = (
                        transport == "rdep"
                        and self._rdep_load_probe_remaining > 0
                        and bool(batch.is_decode)
                    )
                    use_graph = (
                        self.enable_cuda_graph
                        and batch.is_decode
                        and output_mode == OutputMode.TOKENS
                        and sample_args.temperatures is None
                        and int(S) == 1
                    )
                    if do_rdep_probe:
                        # Probe uses CUDA events and is not graph-capturable.
                        use_graph = False
                    if use_graph:
                        # For RDEP decode transport, graph decode is only supported at the
                        # fixed BS=256 operating point (t_cap=32 per rank). This avoids
                        # graph capture with uneven local batches (T=0 participants),
                        # which would otherwise deadlock barrier-based EP.
                        if transport == "rdep":
                            t_cap = int(getattr(self.buffer, "_nmoe_rdep_t_cap", 0))
                            if int(B) != int(t_cap):
                                use_graph = False
                        if self.attention_type == "dsa":
                            raise NotImplementedError("CUDA graph decode is MLA-only (attention_type=dsa unsupported).")
                        if transport == "deepep" and not bool(getattr(self.buffer, "low_latency_mode", False)):
                            raise RuntimeError(
                                "CUDA graph decode requires DeepEP low-latency mode. "
                                "Set NMOE_DEEPEP_LOW_LATENCY=1 (or enable multi-node auto-detect)."
                            )
                        if batch.block_table is None or batch.cache_seqlens is None:
                            raise RuntimeError("CUDA graph decode requires scheduler to provide block_table/cache_seqlens.")

                        g = self._decode_graphs.get(B)
                        if g is None:
                            g = _DecodeGraph(self, B)
                            self._decode_graphs[B] = g
                        _local_max, local_gid = g.run(
                            input_ids=batch.input_ids,
                            positions=batch.positions,
                            out_loc=batch.out_loc,
                            block_table=batch.block_table,
                            cache_seqlens=batch.cache_seqlens,
                        )
                        next_tokens_gpu = local_gid.to(torch.int64)
                        if copy_slot is not None:
                            slot = int(copy_slot) % self._cpu_ring_size
                            dst = self._next_tokens_cpu_ring[slot]
                            dst[:B].copy_(next_tokens_gpu, non_blocking=True)
                            next_tokens_cpu = dst[:B]
                            moe_overflow_cpu = self._moe_overflow_cpu_ring[slot]
                            moe_overflow_cpu.copy_(self._moe_overflow_gpu, non_blocking=True)
                        else:
                            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
                            moe_overflow_cpu = self._moe_overflow_gpu.to("cpu", non_blocking=True)
                        copy_event = torch.cuda.Event()
                        copy_event.record(self.stream)
                        # logits are unused in TOKENS mode; keep as a tiny placeholder tensor.
                        logits_placeholder = torch.empty((0,), device=self.device, dtype=torch.float32)
                        return ForwardOutput(
                            logits=logits_placeholder,
                            next_tokens_gpu=next_tokens_gpu,
                            next_tokens_cpu=next_tokens_cpu,
                            moe_overflow_cpu=moe_overflow_cpu,
                            copy_event=copy_event,
                        )

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

                    if do_rdep_probe:
                        # Allocate/reset per-step probe state on the buffer. MoE layers
                        # accumulate expert load and barrier timings into these.
                        if not hasattr(self.buffer, "_nmoe_rdep_probe_load"):
                            setattr(
                                self.buffer,
                                "_nmoe_rdep_probe_load",
                                torch.zeros((), device=self.device, dtype=torch.int64),
                            )
                        self.buffer._nmoe_rdep_probe_load.zero_()  # type: ignore[attr-defined]
                        self.buffer._nmoe_rdep_probe_barrier_events = []  # type: ignore[attr-defined]

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
                # Get last-token logits for decode, or last per sequence for prefill
                sample_logits = logits[:, -1, :] if logits.dim() == 3 else logits

                if output_mode not in (
                    OutputMode.TOKENS,
                    OutputMode.LOGPROBS,
                    OutputMode.TOPK_LOGPROBS,
                    OutputMode.LOGITS,
                ):
                    raise NotImplementedError(f"Unknown OutputMode={output_mode}")

                # With replicated lm_head (no TP), each rank has full vocab - no gather needed
                full_logits: Optional[torch.Tensor] = sample_logits

            # Sampling and any D2H copies must be enqueued on the same stream as the
            # model forward. Otherwise, the default stream can read logits before
            # self.stream finishes producing them, leading to nondeterminism and
            # incorrect outputs (masked by CUDA_LAUNCH_BLOCKING=1).
            with torch.cuda.stream(self.stream):
                # With replicated lm_head (no TP), each rank samples locally.
                want_logprobs = output_mode in (OutputMode.LOGPROBS, OutputMode.TOPK_LOGPROBS)
                want_topk = output_mode == OutputMode.TOPK_LOGPROBS
                topk = int(batch.reqs[0].forward_spec.topk) if want_topk else 0
                if want_topk and topk <= 0:
                    raise ValueError("TOPK_LOGPROBS requires ForwardSpec.topk > 0")

                (
                    next_tokens_i64,
                    token_logprobs,
                    topk_ids,
                    topk_logprobs,
                ) = self.sampler.sample_with_aux(
                    full_logits,
                    sample_args,
                    return_logprobs=want_logprobs,
                    return_topk=topk,
                )
                next_tokens_gpu = next_tokens_i64.to(torch.int64)

                if copy_slot is not None and output_mode == OutputMode.TOKENS:
                    slot = int(copy_slot) % self._cpu_ring_size
                    dst = self._next_tokens_cpu_ring[slot]
                    dst[:B].copy_(next_tokens_gpu, non_blocking=True)
                    next_tokens_cpu = dst[:B]
                    moe_overflow_cpu = self._moe_overflow_cpu_ring[slot]
                    moe_overflow_cpu.copy_(self._moe_overflow_gpu, non_blocking=True)
                else:
                    next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
                    moe_overflow_cpu = self._moe_overflow_gpu.to("cpu", non_blocking=True)
                next_logprobs_gpu = token_logprobs
                next_logprobs_cpu = token_logprobs.to("cpu", non_blocking=True) if token_logprobs is not None else None
                next_topk_ids_gpu = topk_ids.to(torch.int32) if topk_ids is not None else None
                next_topk_logprobs_gpu = topk_logprobs
                next_topk_ids_cpu = (
                    next_topk_ids_gpu.to("cpu", non_blocking=True) if next_topk_ids_gpu is not None else None
                )
                next_topk_logprobs_cpu = (
                    next_topk_logprobs_gpu.to("cpu", non_blocking=True) if next_topk_logprobs_gpu is not None else None
                )
                logits_cpu = full_logits.to("cpu", non_blocking=True) if output_mode == OutputMode.LOGITS else None

                copy_event = torch.cuda.Event()
                copy_event.record(self.stream)

            if do_rdep_probe:
                # Ensure all probe events have completed before reading timings.
                torch.cuda.synchronize(self.device)
                load = int(getattr(self.buffer, "_nmoe_rdep_probe_load").item())
                ev_pairs = getattr(self.buffer, "_nmoe_rdep_probe_barrier_events", [])
                barrier_ms = 0.0
                for ev0, ev1 in ev_pairs:
                    barrier_ms += float(ev0.elapsed_time(ev1))
                print(
                    f"[RDEP_LOAD_PROBE][rank{self.rank}] decode B_local={int(B)} "
                    f"load_sum(masked_m)={load} barrier_ms={barrier_ms:.3f} "
                    f"n_barriers={len(ev_pairs)}",
                    flush=True,
                )
                self._rdep_load_probe_remaining -= 1
                # Free event list promptly (avoid holding onto many CUDA events).
                self.buffer._nmoe_rdep_probe_barrier_events = []  # type: ignore[attr-defined]

        return ForwardOutput(
            logits=full_logits if output_mode == OutputMode.LOGITS else sample_logits,
            logits_cpu=logits_cpu,
            next_tokens_gpu=next_tokens_gpu,
            next_tokens_cpu=next_tokens_cpu,
            moe_overflow_cpu=moe_overflow_cpu,
            next_logprobs_gpu=next_logprobs_gpu,
            next_logprobs_cpu=next_logprobs_cpu,
            next_topk_ids_gpu=next_topk_ids_gpu,
            next_topk_logprobs_gpu=next_topk_logprobs_gpu,
            next_topk_ids_cpu=next_topk_ids_cpu,
            next_topk_logprobs_cpu=next_topk_logprobs_cpu,
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
        """Clean up resources.

        Note: The distributed process group lifecycle is owned by the caller
        (e.g. `torchrun`, benchmark harness, or server). The Engine must not
        unilaterally destroy it.
        """
        buf = getattr(self, "buffer", None)
        if buf is None:
            return
        if bool(getattr(buf, "explicitly_destroy", False)):
            # Ensure no in-flight kernels use DeepEP buffers at destruction time.
            try:
                self.stream.synchronize()
            except Exception:
                pass
            try:
                comm_stream = buf.get_comm_stream()  # type: ignore[attr-defined]
                comm_stream.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            buf.destroy()

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
        # Optional debug guard: catch accidental torch.distributed/NCCL collectives
        # inside CUDA graph capture. NCCL ops are not permitted when stream is
        # capturing, and failures can be non-local/hard to attribute.
        saved_dist_fns = None
        if os.environ.get("NMOE_DEBUG_DIST_IN_CAPTURE", "0") in ("1", "true", "True"):
            import functools
            import traceback

            def _wrap_dist_fn(name: str, fn):  # type: ignore[no-untyped-def]
                @functools.wraps(fn)
                def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                    if torch.cuda.is_current_stream_capturing():
                        stack = "".join(traceback.format_stack(limit=64))
                        raise RuntimeError(
                            f"torch.distributed.{name} called during CUDA graph capture (NCCL not capture-safe).\n{stack}"
                        )
                    return fn(*args, **kwargs)

                return _wrapped

            import torch.distributed as _dist

            saved_dist_fns = {}
            for _name in (
                "all_reduce",
                "all_gather",
                "all_gather_object",
                "broadcast",
                "broadcast_object_list",
                "barrier",
                "reduce_scatter",
                "reduce_scatter_tensor",
                "all_to_all",
                "all_to_all_single",
                "gather",
                "scatter",
            ):
                if hasattr(_dist, _name):
                    _fn = getattr(_dist, _name)
                    saved_dist_fns[_name] = _fn
                    setattr(_dist, _name, _wrap_dist_fn(_name, _fn))

            # Also guard low-level functional collectives if present.
            try:
                import torch.distributed._functional_collectives as _fc  # type: ignore

                for _name in dir(_fc):
                    if _name.startswith("_"):
                        continue
                    _fn = getattr(_fc, _name)
                    if not callable(_fn):
                        continue
                    # Only wrap known collective-like APIs; be conservative.
                    if any(
                        _name.startswith(p)
                        for p in (
                            "all_reduce",
                            "all_gather",
                            "broadcast",
                            "reduce_scatter",
                            "all_to_all",
                            "barrier",
                        )
                    ):
                        saved_dist_fns[f"_fc.{_name}"] = _fn
                        setattr(_fc, _name, _wrap_dist_fn(f"_functional_collectives.{_name}", _fn))
            except Exception:
                pass

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
            try:
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
                start = 0 if int(self.engine.engine_config.tp_size) == 1 else int(self.engine.rank) * v_shard
                local_gid = local_idx.to(torch.int64) + start
                self.local_max.copy_(local_max)
                self.local_gid.copy_(local_gid)
            finally:
                self.graph.capture_end()
                if saved_dist_fns is not None:
                    import torch.distributed as _dist

                    for _name, _fn in saved_dist_fns.items():
                        if _name.startswith("_fc."):
                            continue
                        setattr(_dist, _name, _fn)
                    # Best-effort restore functional collectives.
                    try:
                        import torch.distributed._functional_collectives as _fc  # type: ignore

                        for _name, _fn in saved_dist_fns.items():
                            if not _name.startswith("_fc."):
                                continue
                            setattr(_fc, _name[len("_fc.") :], _fn)
                    except Exception:
                        pass

    def run(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Always use the stream the graph was captured on.
        with torch.cuda.stream(self.stream):
            self.input_ids.copy_(input_ids)
            self.positions.copy_(positions)
            self.out_loc.copy_(out_loc)
            self.block_table.copy_(block_table)
            self.cache_seqlens.copy_(cache_seqlens)

            self.graph.replay()

        return self.local_max, self.local_gid
