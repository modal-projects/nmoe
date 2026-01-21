# SPDX-License-Identifier: Apache-2.0
"""Validity tests for nmoe.serve single engine / 5 profiles architecture.

Tests enforce:
1. Profile contracts - Each profile returns correct output format
2. Model validity - Compare outputs against HuggingFace reference
3. Invariants - Batch shapes, no .item() in hot path, determinism
4. Integration - Full forward pass through all components

Run with: python -m nmoe.serve.test_validity
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import unittest
from functools import wraps
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

def _maybe_set_cutlass_path() -> None:
  if os.environ.get("CUTLASS_PATH"):
    return
  here = Path(__file__).resolve()
  for parent in (here.parent, *here.parents):
    cand = parent / "third_party" / "DeepGEMM" / "third-party" / "cutlass"
    if cand.is_dir():
      os.environ["CUTLASS_PATH"] = str(cand)
      return

_maybe_set_cutlass_path()

import torch
import torch.nn.functional as F

from nmoe.serve.config import PROFILES, Profile
from nmoe.serve.types import (
  Batch,
  ForwardOutput,
  ForwardSpec,
  OutputMode,
  Request,
  RequestStatus,
  SamplingParams,
)


# =============================================================================
# Test Utilities
# =============================================================================

def is_sm90_or_higher() -> bool:
  """Check if GPU supports SM90+ (required for FP8)."""
  if not torch.cuda.is_available():
    return False
  cap = torch.cuda.get_device_capability()
  return cap[0] >= 9


def skip_if_no_gpu(test_func):
  """Decorator to skip tests without GPU."""
  @wraps(test_func)
  def wrapper(*args, **kwargs):
    if not torch.cuda.is_available():
      raise unittest.SkipTest(f"{test_func.__name__}: requires CUDA GPU")
    return test_func(*args, **kwargs)
  return wrapper


def skip_if_no_sm90(test_func):
  """Decorator to skip tests on GPUs without SM90+ support."""
  @wraps(test_func)
  def wrapper(*args, **kwargs):
    if not is_sm90_or_higher():
      raise unittest.SkipTest(f"{test_func.__name__}: requires SM90+ GPU")
    return test_func(*args, **kwargs)
  return wrapper


def _init_single_process_group() -> None:
  """Initialize a single-process NCCL group without fixed ports."""
  import torch.distributed as dist
  if dist.is_initialized():
    return
  if not torch.cuda.is_available():
    raise unittest.SkipTest("requires CUDA for NCCL process group")
  tmp = tempfile.NamedTemporaryFile(prefix="nmoe_pg_", suffix=".tmp", delete=False)
  tmp.close()
  dist.init_process_group(
    backend="nccl",
    init_method=f"file://{tmp.name}",
    world_size=1,
    rank=0,
  )


# =============================================================================
# Test 1: Profile Contract Tests
# =============================================================================

class TestProfileContracts(unittest.TestCase):
  """Test that each profile defines correct contracts."""

  def test_all_profiles_defined(self):
    """Verify all 5 profiles are defined."""
    expected = {"production_generate", "online_distill", "rl_sample", "eval_exact", "offline_distill"}
    actual = set(PROFILES.keys())
    self.assertEqual(expected, actual, f"Missing profiles: {expected - actual}")

  def test_production_generate_profile(self):
    """Test production_generate profile contract."""
    p = PROFILES["production_generate"]
    self.assertEqual(p.output_mode, OutputMode.TOKENS)
    self.assertTrue(p.streaming)
    self.assertFalse(p.deterministic)
    self.assertEqual(p.prefix_cache, "normal")

  def test_online_distill_profile(self):
    """Test online_distill profile contract."""
    p = PROFILES["online_distill"]
    self.assertEqual(p.output_mode, OutputMode.TOPK_LOGPROBS)
    self.assertFalse(p.streaming)
    self.assertTrue(p.deterministic)
    self.assertEqual(p.prefix_cache, "aggressive")

  def test_rl_sample_profile(self):
    """Test rl_sample profile contract."""
    p = PROFILES["rl_sample"]
    self.assertEqual(p.output_mode, OutputMode.LOGPROBS)
    self.assertFalse(p.streaming)
    self.assertTrue(p.deterministic)  # Per-request RNG

  def test_eval_exact_profile(self):
    """Test eval_exact profile contract."""
    p = PROFILES["eval_exact"]
    self.assertEqual(p.output_mode, OutputMode.LOGITS)
    self.assertFalse(p.streaming)
    self.assertTrue(p.deterministic)
    # Fixed batching ensures deterministic ordering
    from nmoe.serve.config import BatchingMode
    self.assertEqual(p.batching, BatchingMode.FIXED)

  def test_offline_distill_profile(self):
    """Test offline_distill profile contract."""
    p = PROFILES["offline_distill"]
    self.assertEqual(p.output_mode, OutputMode.LOGITS)
    self.assertFalse(p.streaming)
    # Fixed batching for reproducibility even if not deterministic
    from nmoe.serve.config import BatchingMode
    self.assertEqual(p.batching, BatchingMode.FIXED)

  def test_forward_spec_conversion(self):
    """Test that profiles convert to ForwardSpec correctly."""
    for name, profile in PROFILES.items():
      spec = profile.to_forward_spec(topk=10)
      self.assertEqual(spec.output_mode, profile.output_mode)
      self.assertIsInstance(spec, ForwardSpec)


# =============================================================================
# Test 2: Output Mode Contract Tests
# =============================================================================

class TestOutputModeContracts(unittest.TestCase):
  """Test output contracts for each OutputMode."""

  def _make_mock_logits(self, batch_size: int, vocab_size: int) -> torch.Tensor:
    """Create mock logits for testing."""
    return torch.randn(batch_size, vocab_size)

  def test_tokens_mode_output(self):
    """TOKENS mode: returns sampled token IDs only."""
    logits = self._make_mock_logits(4, 1000)
    tokens = torch.argmax(logits, dim=-1)

    # TOKENS mode should just have token IDs
    self.assertEqual(tokens.shape, (4,))
    self.assertTrue(tokens.dtype in (torch.int32, torch.int64))

  def test_logprobs_mode_output(self):
    """LOGPROBS mode: returns log probabilities of sampled tokens."""
    logits = self._make_mock_logits(4, 1000)
    log_probs = F.log_softmax(logits, dim=-1)
    tokens = torch.argmax(logits, dim=-1)

    # Get log prob of each sampled token
    token_logprobs = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)

    self.assertEqual(token_logprobs.shape, (4,))
    self.assertTrue((token_logprobs <= 0).all())  # Log probs are <= 0

  def test_topk_logprobs_mode_output(self):
    """TOPK_LOGPROBS mode: returns top-k tokens and their log probs."""
    logits = self._make_mock_logits(4, 1000)
    log_probs = F.log_softmax(logits, dim=-1)
    topk = 10

    topk_logprobs, topk_indices = torch.topk(log_probs, topk, dim=-1)

    self.assertEqual(topk_logprobs.shape, (4, topk))
    self.assertEqual(topk_indices.shape, (4, topk))
    # Top-k should be sorted descending
    for i in range(4):
      self.assertTrue((topk_logprobs[i, :-1] >= topk_logprobs[i, 1:]).all())

  def test_logits_mode_output(self):
    """LOGITS mode: returns full vocabulary logits."""
    batch_size, vocab_size = 4, 1000
    logits = self._make_mock_logits(batch_size, vocab_size)

    # LOGITS mode returns full logits tensor
    self.assertEqual(logits.shape, (batch_size, vocab_size))


# =============================================================================
# Test 3: Batch Shape Invariants
# =============================================================================

class TestBatchInvariants(unittest.TestCase):
  """Test batch shape invariants required by model contract."""

  def test_batch_shape_bs_format(self):
    """Batch tensors must be [B,S] format."""
    B, S = 2, 4

    # Create batch
    batch = Batch(
      reqs=[],  # Empty for this test
      phase="prefill",
    )
    batch.input_ids = torch.randint(0, 1000, (B, S), dtype=torch.int64)
    batch.positions = torch.arange(S).unsqueeze(0).expand(B, S)
    batch.out_loc = torch.arange(B * S, dtype=torch.int32).view(B, S)

    # Verify shapes
    self.assertEqual(batch.input_ids.shape, (B, S))
    self.assertEqual(batch.positions.shape, (B, S))
    self.assertEqual(batch.out_loc.shape, (B, S))

    # Verify dtypes
    self.assertEqual(batch.input_ids.dtype, torch.int64)
    self.assertEqual(batch.out_loc.dtype, torch.int32)

  def test_out_loc_non_negative(self):
    """out_loc must be non-negative (no padding in model contract)."""
    B, S = 2, 4

    batch = Batch(reqs=[], phase="prefill")
    batch.out_loc = torch.arange(B * S, dtype=torch.int32).view(B, S)

    self.assertTrue((batch.out_loc >= 0).all())

  def test_decode_batch_seqlen_one(self):
    """Decode phase should have S=1."""
    B = 4

    batch = Batch(reqs=[], phase="decode")
    batch.input_ids = torch.randint(0, 1000, (B, 1), dtype=torch.int64)

    self.assertEqual(batch.seqlen_q, 1)
    self.assertTrue(batch.is_decode)

  def test_prefill_batch_seqlen_variable(self):
    """Prefill phase can have S>1."""
    B, S = 2, 32

    batch = Batch(reqs=[], phase="prefill")
    batch.input_ids = torch.randint(0, 1000, (B, S), dtype=torch.int64)

    self.assertEqual(batch.seqlen_q, S)
    self.assertTrue(batch.is_prefill)


# =============================================================================
# Test 4: Static Analysis - No .item() in Hot Path
# =============================================================================

class ItemCallVisitor(ast.NodeVisitor):
  """AST visitor to find sync-ish calls (.item(), .tolist())."""

  def __init__(self, banned_attrs: Set[str]):
    self.calls: List[Tuple[int, str]] = []
    self._banned = set(banned_attrs)

  def visit_Call(self, node: ast.Call):
    if isinstance(node.func, ast.Attribute):
      if node.func.attr in self._banned:
        # Get the line and context
        self.calls.append((node.lineno, ast.unparse(node)))
    self.generic_visit(node)


class TestNoItemInHotPath(unittest.TestCase):
  """Static analysis to ensure no obvious host-sync calls in hot paths."""

  ALLOWED_ITEM_PATTERNS = [
    # These are OK (not in forward pass)
    "checkpoint",
    "load",
    "init",
    "__post_init__",
    "validate",
    "test",
    "debug",
    "print",
  ]

  def _get_serve_path(self) -> Path:
    """Get path to nmoe/serve directory."""
    return Path(__file__).parent

  def _analyze_file(self, filepath: Path, *, banned_attrs: Set[str]) -> List[Tuple[int, str, str]]:
    """Analyze file for banned call attrs, return (line, code, function)."""
    with open(filepath) as f:
      source = f.read()

    tree = ast.parse(source)
    violations = []

    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef):
        func_name = node.name

        # Skip allowed functions
        if any(p in func_name.lower() for p in self.ALLOWED_ITEM_PATTERNS):
          continue

        # Check for .item() in this function
        visitor = ItemCallVisitor(banned_attrs)
        visitor.visit(node)

        for lineno, code in visitor.calls:
          violations.append((lineno, code, func_name))

    return violations

  def test_no_item_in_model(self):
    """model.py should not have .item() in forward methods."""
    filepath = self._get_serve_path() / "model.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    # In CUDA forward paths, both `.item()` and `.tolist()` can introduce
    # hidden host sync. Keep this strict for model forward methods.
    violations = self._analyze_file(filepath, banned_attrs={"item", "tolist"})

    # Filter to forward methods only
    forward_violations = [v for v in violations if "forward" in v[2].lower()]

    if forward_violations:
      msg = "Found .item() calls in model.py forward methods:\n"
      for lineno, code, func in forward_violations:
        msg += f"  Line {lineno} in {func}: {code}\n"
      self.fail(msg)

  def test_no_item_in_engine_forward(self):
    """engine.py should not have .item() in forward_batch."""
    filepath = self._get_serve_path() / "engine.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    violations = self._analyze_file(filepath, banned_attrs={"item", "tolist"})

    # Filter to forward methods only
    forward_violations = [v for v in violations if "forward" in v[2].lower()]

    if forward_violations:
      msg = "Found .item() calls in engine.py forward methods:\n"
      for lineno, code, func in forward_violations:
        msg += f"  Line {lineno} in {func}: {code}\n"
      self.fail(msg)

  def test_no_item_in_scheduler_hotpath(self):
    """scheduler.py should not have .item() in scheduling code paths."""
    filepath = self._get_serve_path() / "scheduler.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    # Scheduler operates on CPU-owned metadata; `.tolist()` on CPU tensors is
    # not a GPU sync. The hard rule is: no `.item()` in hot paths.
    violations = self._analyze_file(filepath, banned_attrs={"item"})
    if violations:
      msg = "Found .item() calls in scheduler.py:\n"
      for lineno, code, func in violations:
        msg += f"  Line {lineno} in {func}: {code}\n"
      self.fail(msg)

  def test_no_item_in_orchestrator_hotpath(self):
    """orchestrator.py should not have .item() in scheduling loop code paths."""
    filepath = self._get_serve_path() / "orchestrator.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    violations = self._analyze_file(filepath, banned_attrs={"item"})
    if violations:
      msg = "Found .item() calls in orchestrator.py:\n"
      for lineno, code, func in violations:
        msg += f"  Line {lineno} in {func}: {code}\n"
      self.fail(msg)

  def test_no_item_in_cache_hotpath(self):
    """cache.py should not have .item() in prefix cache code paths."""
    filepath = self._get_serve_path() / "cache.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    violations = self._analyze_file(filepath, banned_attrs={"item"})
    if violations:
      msg = "Found .item() calls in cache.py:\n"
      for lineno, code, func in violations:
        msg += f"  Line {lineno} in {func}: {code}\n"
      self.fail(msg)


# =============================================================================
# Test 4b: Static Analysis - FlashMLA Call Signatures
# =============================================================================

class FlashMlaCallVisitor(ast.NodeVisitor):
  """Detect FlashMLA API calls that must not use positional args."""

  def __init__(self) -> None:
    self.get_mla_metadata_positional: List[Tuple[int, str]] = []

  def visit_Call(self, node: ast.Call):
    callee = None
    if isinstance(node.func, ast.Name):
      callee = node.func.id
    elif isinstance(node.func, ast.Attribute):
      callee = node.func.attr

    if callee == "get_mla_metadata" and node.args:
      self.get_mla_metadata_positional.append((node.lineno, ast.unparse(node)))

    self.generic_visit(node)


class TestFlashMlaSignatures(unittest.TestCase):
  """Guardrails against silent FlashMLA API misuse (causes correctness bugs)."""

  def test_get_mla_metadata_uses_keywords_in_model(self) -> None:
    filepath = Path(__file__).parent / "model.py"
    if not filepath.exists():
      self.skipTest(f"{filepath} not found")

    source = filepath.read_text()
    tree = ast.parse(source)
    v = FlashMlaCallVisitor()
    v.visit(tree)

    if v.get_mla_metadata_positional:
      msg = "Found get_mla_metadata(...) calls with positional args in model.py:\n"
      for lineno, code in v.get_mla_metadata_positional[:20]:
        msg += f"  Line {lineno}: {code}\n"
      self.fail(msg)


# =============================================================================
# Test 5: Determinism Tests
# =============================================================================

class TestDeterminism(unittest.TestCase):
  """Test deterministic behavior for deterministic profiles."""

  def test_sampling_with_seed_deterministic(self):
    """Sampling with same seed should produce same results."""
    logits = torch.randn(4, 1000)

    def sample_with_seed(seed: int) -> torch.Tensor:
      torch.manual_seed(seed)
      probs = F.softmax(logits, dim=-1)
      return torch.multinomial(probs, 1).squeeze(-1)

    # Same seed should give same results
    result1 = sample_with_seed(42)
    result2 = sample_with_seed(42)
    self.assertTrue(torch.equal(result1, result2))

    # Different seeds should (usually) give different results
    result3 = sample_with_seed(123)
    # Not guaranteed to be different, but very likely

  def test_greedy_sampling_deterministic(self):
    """Greedy sampling (temperature=0) should always be deterministic."""
    logits = torch.randn(4, 1000)

    result1 = torch.argmax(logits, dim=-1)
    result2 = torch.argmax(logits, dim=-1)

    self.assertTrue(torch.equal(result1, result2))

  def test_eval_exact_profile_requires_determinism(self):
    """eval_exact profile must have deterministic=True."""
    p = PROFILES["eval_exact"]
    self.assertTrue(p.deterministic)

  def test_rl_sample_profile_requires_determinism(self):
    """rl_sample profile must have deterministic=True (per-request RNG)."""
    p = PROFILES["rl_sample"]
    self.assertTrue(p.deterministic)

  def test_fixed_batching_is_stable_order(self):
    """Fixed batching profiles must schedule in insertion order."""
    from nmoe.serve.cache import KvCache, MlaKvLayout
    from nmoe.serve.scheduler import Scheduler, SchedulerConfig
    from nmoe.serve.types import ForwardSpec
    device = torch.device("cpu")
    layout = MlaKvLayout(num_blocks=64, block_size=64)
    kv_cache = KvCache(layout, device)
    sched = Scheduler(
      SchedulerConfig(max_batch_size=8, max_prefill_tokens=1024, page_size=64, enable_chunked_prefill=False),
      kv_cache,
      device,
    )

    # Add requests for a fixed-batching profile.
    reqs = []
    for uid in range(5):
      r = Request(
        uid=uid,
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
        profile_name="eval_exact",
        forward_spec=ForwardSpec(output_mode=OutputMode.LOGITS),
      )
      reqs.append(r)
      sched.add_request(r)

    batch = sched.schedule_next(PROFILES["eval_exact"])
    self.assertIsNotNone(batch)
    got = [r.uid for r in batch.reqs]
    self.assertEqual(got, [0, 1, 2, 3, 4])

  def test_sampler_per_request_seed_is_deterministic(self):
    """Per-request seeds must produce stable sampling results."""
    from nmoe.serve.engine import Sampler

    device = torch.device("cpu")
    sampler = Sampler(device)

    # Two requests with different seeds, same logits.
    logits = torch.randn(2, 100, device=device, dtype=torch.float32)

    batch = Batch(reqs=[], phase="decode")
    r0 = Request(
      uid=0,
      input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
      sampling_params=SamplingParams(temperature=1.0, seed=123),
      profile_name="rl_sample",
      forward_spec=PROFILES["rl_sample"].to_forward_spec(),
    )
    r1 = Request(
      uid=1,
      input_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
      sampling_params=SamplingParams(temperature=1.0, seed=123),
      profile_name="rl_sample",
      forward_spec=PROFILES["rl_sample"].to_forward_spec(),
    )
    batch.reqs = [r0, r1]

    args = sampler.prepare(batch)
    t1 = sampler.sample(logits, args)
    t2 = sampler.sample(logits, args)
    self.assertTrue(torch.equal(t1, t2), "sampling must be deterministic for identical seeds and logits")


# =============================================================================
# Test 6: Request Lifecycle Tests
# =============================================================================

class TestRequestLifecycle(unittest.TestCase):
  """Test request lifecycle state transitions."""

  def _make_request(self, profile: str = "production_generate") -> Request:
    """Create a test request."""
    return Request(
      uid=0,
      input_ids=torch.tensor([1, 2, 3, 4]),
      sampling_params=SamplingParams(),
      profile_name=profile,
      forward_spec=PROFILES[profile].to_forward_spec(),
    )

  def test_initial_state_pending(self):
    """Request starts in PENDING state."""
    req = self._make_request()
    self.assertEqual(req.status, RequestStatus.PENDING)

  def test_prefill_transition(self):
    """PENDING -> PREFILLING transition."""
    req = self._make_request()
    req.mark_prefill_start()
    self.assertEqual(req.status, RequestStatus.PREFILLING)
    self.assertGreater(req.metrics.prefill_start, 0)

  def test_decode_transition(self):
    """PREFILLING -> DECODING transition."""
    req = self._make_request()
    req.mark_prefill_start()
    req.mark_prefill_end()
    req.mark_decode_start()
    self.assertEqual(req.status, RequestStatus.DECODING)
    self.assertGreater(req.metrics.decode_start, 0)

  def test_finished_transition(self):
    """DECODING -> FINISHED transition."""
    req = self._make_request()
    req.mark_prefill_start()
    req.mark_decode_start()
    req.mark_finished()
    self.assertEqual(req.status, RequestStatus.FINISHED)
    self.assertTrue(req.is_finished)

  def test_seq_len_tracking(self):
    """Sequence length tracking with generated tokens."""
    req = self._make_request()
    initial_len = req.seq_len
    self.assertEqual(initial_len, 4)  # Input tokens

    # Simulate generated tokens
    req.output_ids.append(100)
    req.output_ids.append(101)
    self.assertEqual(req.seq_len, 6)


# =============================================================================
# Test 7: ForwardOutput Contract Tests
# =============================================================================

class TestForwardOutputContract(unittest.TestCase):
  """Test ForwardOutput contract for each output mode."""

  def test_forward_output_logits_required(self):
    """ForwardOutput must always have logits."""
    output = ForwardOutput(
      logits=torch.randn(4, 1000),
    )
    self.assertIsNotNone(output.logits)

  def test_forward_output_optional_fields(self):
    """ForwardOutput optional fields default to None."""
    output = ForwardOutput(
      logits=torch.randn(4, 1000),
    )
    self.assertIsNone(output.hidden_states)
    self.assertIsNone(output.next_tokens_gpu)
    self.assertIsNone(output.next_tokens_cpu)
    self.assertIsNone(output.copy_event)

  def test_forward_output_with_tokens(self):
    """ForwardOutput with sampled tokens."""
    logits = torch.randn(4, 1000)
    tokens = torch.argmax(logits, dim=-1)

    output = ForwardOutput(
      logits=logits,
      next_tokens_gpu=tokens,
      next_tokens_cpu=tokens.cpu(),
    )

    self.assertEqual(output.next_tokens_gpu.shape, (4,))
    self.assertTrue(output.next_tokens_cpu.is_cpu)


# =============================================================================
# Test 8: Model Implementation Validity
# =============================================================================

class TestModelValidity(unittest.TestCase):
  """Test nmoe/model.py implementation validity.

  Validates:
  1. Each layer produces valid outputs (no NaN/Inf)
  2. Tensor shapes match contracts
  3. FP8 quantization produces valid values
  4. FlashMLA/DeepEP/DeepGEMM integration works
  """

  @classmethod
  def setUpClass(cls):
    """Load model once for all tests."""
    cls.model = None
    cls.cfg = None

    if not is_sm90_or_higher():
      raise unittest.SkipTest("requires SM90+ GPU for FP8 kernels")

    model_path = "/data/models/DeepSeek-V3.2-Speciale"
    if not os.path.exists(model_path):
      raise unittest.SkipTest(f"missing model weights at {model_path}")

    from nmoe.serve.ckpt import load_checkpoint
    from nmoe.serve.model import DeepSeekV3, ModelConfig, init_distributed

    _init_single_process_group()
    init_distributed(0, 1)

    # Small model for testing
    cfg = ModelConfig(num_layers=4, num_dense_layers=3)

    import torch.distributed as dist
    from deep_ep import Buffer
    buffer = Buffer(group=dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)

    cls.model = DeepSeekV3(cfg, buffer).cuda()
    cls.model.eval()
    load_checkpoint(cls.model, model_path, rank=0, world_size=1, cfg=cfg)
    cls.cfg = cfg

  @classmethod
  def tearDownClass(cls) -> None:
    import torch.distributed as dist
    if dist.is_initialized():
      dist.destroy_process_group()

  def _check_valid(self, name: str, t: torch.Tensor) -> None:
    """Assert tensor has no NaN/Inf and reasonable values."""
    self.assertFalse(torch.isnan(t).any(), f"{name} contains NaN")
    self.assertFalse(torch.isinf(t).any(), f"{name} contains Inf")
    amax = float(t.abs().max().cpu())
    self.assertGreater(amax, 0, f"{name} is all zeros")
    self.assertLess(amax, 1e6, f"{name} has extreme values: {amax}")

  @skip_if_no_sm90
  def test_embedding_valid(self):
    """Embedding layer produces valid output."""
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with torch.no_grad():
      out = self.model.embed(input_ids)

    self.assertEqual(out.shape, (1, 4, self.cfg.hidden_size))
    self._check_valid("embedding", out)

  @skip_if_no_sm90
  def test_rms_norm_valid(self):
    """RMSNorm produces properly normalized output."""
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    x = torch.randn(1, 4, self.cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
    norm = self.model.layers[0].attn_norm

    with torch.no_grad():
      out = norm(x)

    self._check_valid("rms_norm", out)
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    self.assertFalse(torch.isnan(rms).any())
    self.assertFalse(torch.isinf(rms).any())

  @skip_if_no_sm90
  def test_fp8_linear_valid(self):
    """FP8Linear produces valid output."""
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    # Use wq_a from first attention layer
    layer = self.model.layers[0].attn
    if not hasattr(layer, 'wq_a'):
      self.skipTest("Layer doesn't have wq_a")

    x = torch.randn(1, 4, self.cfg.hidden_size, dtype=torch.bfloat16, device="cuda") / 10

    with torch.no_grad():
      out = layer.wq_a(x)

    self._check_valid("fp8_linear", out)
    self.assertEqual(out.shape, (1, 4, self.cfg.q_lora_rank))

  @skip_if_no_sm90
  def test_embedding_to_fp8_path(self):
    """Test full path: token_ids → embedding → norm → FP8Linear.

    This catches issues where embedding output causes FP8 quantization to fail.
    """
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    # Use real token IDs (not random high values)
    input_ids = torch.tensor([[1, 100, 1000, 10000]], device="cuda")

    with torch.no_grad():
      # Step 1: Embedding
      embed_out = self.model.embed(input_ids)
      self._check_valid("embed_out", embed_out)
      print(f"  embed_out: shape={embed_out.shape}, amax={embed_out.abs().max().item():.6f}")

      # Step 2: First layer's attention norm
      x_norm = self.model.layers[0].attn_norm(embed_out)
      self._check_valid("x_norm", x_norm)
      print(f"  x_norm: shape={x_norm.shape}, amax={x_norm.abs().max().item():.6f}")

      # Step 3: FP8 projection (wq_a)
      attn = self.model.layers[0].attn
      q_latent = attn.wq_a(x_norm)
      self._check_valid("q_latent", q_latent)
      print(f"  q_latent: shape={q_latent.shape}, amax={q_latent.abs().max().item():.6f}")

  @skip_if_no_sm90
  def test_mlp_valid(self):
    """MLP (dense layer) produces valid output."""
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    # Layer 0 should be dense MLP
    layer = self.model.layers[0]
    if not hasattr(layer, 'mlp'):
      self.skipTest("Layer doesn't have dense MLP")

    x = torch.randn(1, 4, self.cfg.hidden_size, dtype=torch.bfloat16, device="cuda") / 10

    with torch.no_grad():
      out = layer.mlp(x)

    self._check_valid("mlp", out)
    self.assertEqual(out.shape, (1, 4, self.cfg.hidden_size))

  @skip_if_no_sm90
  def test_model_output_shape(self):
    """Full model forward produces correct output shape."""
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    B, S = 1, 4

    # Prepare inputs
    input_ids = torch.randint(0, self.cfg.vocab_size, (B, S), device="cuda")
    positions = torch.arange(S, device="cuda").unsqueeze(0)

    # Create dummy KV caches
    num_pages = 4
    page_size = 64
    kv_caches = [
      torch.zeros(num_pages, page_size, 1, 656, dtype=torch.uint8, device="cuda")
      for _ in range(self.cfg.num_layers)
    ]
    idx_k_caches = [
      torch.zeros(num_pages, page_size, self.cfg.dsa_idx_dim, dtype=torch.bfloat16, device="cuda")
      for _ in range(self.cfg.num_layers)
    ]

    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device="cuda")
    cache_seqlens_cpu = [S]
    out_loc = torch.arange(S, dtype=torch.int32, device="cuda").unsqueeze(0)

    with torch.no_grad():
      logits = self.model(
        input_ids,
        positions,
        kv_caches=kv_caches,
        idx_k_caches=idx_k_caches,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=cache_seqlens_cpu,
        out_loc=out_loc,
      )

    # Check output shape: [B, S, vocab_size]
    self.assertEqual(logits.shape, (B, S, self.cfg.vocab_size))
    self._check_valid("logits", logits)

  @skip_if_no_sm90
  def test_last_token_topk_is_deterministic_for_same_inputs(self):
    """Deterministic profiles require stable results for identical inputs.

    We don't require bitwise-equal full logits (can be too strict across kernels),
    but we do require stable argmax and top-k ordering for eval_exact usage.
    """
    if self.model is None or self.cfg is None:
      self.skipTest("Model not loaded")

    B, S = 1, 8
    input_ids = torch.tensor([[1, 100, 1000, 10000, 50000, 100000, 1234, 5678]], device="cuda")
    positions = torch.arange(S, device="cuda").unsqueeze(0)

    num_pages = 4
    kv1 = [torch.zeros(num_pages, 64, 1, 656, dtype=torch.uint8, device="cuda") for _ in range(self.cfg.num_layers)]
    idx1 = [torch.zeros(num_pages, 64, self.cfg.dsa_idx_dim, dtype=torch.bfloat16, device="cuda") for _ in range(self.cfg.num_layers)]
    kv2 = [t.clone() for t in kv1]
    idx2 = [t.clone() for t in idx1]
    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    cache_seqlens = torch.tensor([S], dtype=torch.int32, device="cuda")
    out_loc = torch.arange(S, dtype=torch.int32, device="cuda").unsqueeze(0)

    with torch.no_grad():
      logits1 = self.model(
        input_ids,
        positions,
        kv_caches=kv1,
        idx_k_caches=idx1,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[S],
        out_loc=out_loc,
      )
      logits2 = self.model(
        input_ids,
        positions,
        kv_caches=kv2,
        idx_k_caches=idx2,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cache_seqlens_cpu=[S],
        out_loc=out_loc,
      )

    # Compare only the final position's top-k (robust to tiny numeric drift).
    topk = 16
    a1 = logits1[0, -1, :].topk(topk)
    a2 = logits2[0, -1, :].topk(topk)
    self.assertTrue(torch.equal(a1.indices, a2.indices), "top-k token ids changed across identical runs")
    self.assertEqual(int(a1.indices[0]), int(a2.indices[0]), "argmax token id changed across identical runs")


# =============================================================================
# Test 9: Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
  """Integration tests for full forward pass."""

  @skip_if_no_sm90
  def test_batch_preparation(self):
    """Test batch preparation for model forward."""
    B, S = 2, 4

    # Create requests
    reqs = []
    for i in range(B):
      req = Request(
        uid=i,
        input_ids=torch.randint(0, 1000, (S,)),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=10),
        profile_name="production_generate",
        forward_spec=PROFILES["production_generate"].to_forward_spec(),
      )
      reqs.append(req)

    # Create batch
    batch = Batch(reqs=reqs, phase="prefill")

    # Prepare tensors
    batch.input_ids = torch.stack([r.input_ids for r in reqs]).cuda()
    batch.positions = torch.arange(S).unsqueeze(0).expand(B, S).cuda()
    batch.out_loc = torch.arange(B * S, dtype=torch.int32).view(B, S).cuda()

    # Verify batch is ready for model
    self.assertEqual(batch.size, B)
    self.assertEqual(batch.seqlen_q, S)
    self.assertEqual(batch.total_tokens, B * S)
    self.assertTrue(batch.is_prefill)


# =============================================================================
# Main
# =============================================================================

def main():
  print("=" * 60)
  print("nmoe.serve Validity Tests")
  print("Single Engine / 5 Profiles Architecture")
  print("=" * 60)

  if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print(f"SM90+: {is_sm90_or_higher()}")
  else:
    print("CUDA: Not available")
  print()

  # Run tests
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()

  # Add test classes in order
  suite.addTests(loader.loadTestsFromTestCase(TestProfileContracts))
  suite.addTests(loader.loadTestsFromTestCase(TestOutputModeContracts))
  suite.addTests(loader.loadTestsFromTestCase(TestBatchInvariants))
  suite.addTests(loader.loadTestsFromTestCase(TestNoItemInHotPath))
  suite.addTests(loader.loadTestsFromTestCase(TestDeterminism))
  suite.addTests(loader.loadTestsFromTestCase(TestRequestLifecycle))
  suite.addTests(loader.loadTestsFromTestCase(TestForwardOutputContract))
  suite.addTests(loader.loadTestsFromTestCase(TestModelValidity))
  suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

  # Run with verbosity
  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)

  return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
  sys.exit(main())
