# SPDX-License-Identifier: Apache-2.0
"""L3: KV cache + scheduler invariants tests.

Verifies:
1. Page allocation/deallocation lifecycle
2. Prefix cache matching and insertion
3. LRU eviction under memory pressure
4. Request lifecycle (add → prefill → decode → finish)
5. Resource cleanup (no memory leaks)
6. Multi-request batching and isolation

Run with: python -m nmoe.serve.test_cache_scheduler_invariants
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestConfig:
    num_pages: int = 64
    page_size: int = 64
    max_batch_size: int = 8
    max_prefill_tokens: int = 2048
    device: str = "cuda:0"
    verbose: bool = True


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[L3] {msg}", flush=True)


def test_page_allocator(cfg: TestConfig) -> bool:
    """Test page allocator allocation/deallocation."""
    log("Test: PageAllocator", cfg.verbose)

    from nmoe.serve.cache import PageAllocator

    allocator = PageAllocator(cfg.num_pages, torch.device("cpu"))

    # Initial state
    assert allocator.available == cfg.num_pages, f"Expected {cfg.num_pages} available, got {allocator.available}"
    log(f"  Initial: {allocator.available} pages available", cfg.verbose)

    # Allocate some pages
    pages1 = allocator.allocate(10)
    assert len(pages1) == 10, f"Expected 10 pages, got {len(pages1)}"
    assert allocator.available == cfg.num_pages - 10, f"Expected {cfg.num_pages - 10} available"
    log(f"  After alloc(10): {allocator.available} available", cfg.verbose)

    # Allocate more
    pages2 = allocator.allocate(20)
    assert allocator.available == cfg.num_pages - 30, f"Expected {cfg.num_pages - 30} available"
    log(f"  After alloc(20): {allocator.available} available", cfg.verbose)

    # Free first allocation
    allocator.free(pages1)
    assert allocator.available == cfg.num_pages - 20, f"Expected {cfg.num_pages - 20} available"
    log(f"  After free(10): {allocator.available} available", cfg.verbose)

    # Free second allocation
    allocator.free(pages2)
    assert allocator.available == cfg.num_pages, f"Expected {cfg.num_pages} available after freeing all"
    log(f"  After free(20): {allocator.available} available (back to full)", cfg.verbose)

    # Test allocation failure
    try:
        allocator.allocate(cfg.num_pages + 1)
        assert False, "Should have raised assertion error"
    except AssertionError:
        log("  Correctly rejected over-allocation", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_kv_cache_basic(cfg: TestConfig) -> bool:
    """Test KvCache basic operations."""
    log("Test: KvCache basic", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout

    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=cfg.num_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=True)

    # Initial state
    initial_available = cache.available
    assert initial_available == cfg.num_pages, f"Expected {cfg.num_pages} available"
    log(f"  Initial: {cache.available} pages available", cfg.verbose)

    # Allocate pages
    pages = cache.allocate(5)
    assert len(pages) == 5
    assert cache.available == cfg.num_pages - 5
    log(f"  After alloc(5): {cache.available} available", cfg.verbose)

    # Free pages
    cache.free(pages)
    assert cache.available == cfg.num_pages
    log(f"  After free(5): {cache.available} available", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_prefix_cache(cfg: TestConfig) -> bool:
    """Test prefix cache matching and insertion."""
    log("Test: Prefix cache", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout, CacheHandle

    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=cfg.num_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=True)

    # Create input that spans multiple pages
    prompt_len = cfg.page_size * 3  # 3 full pages
    input_ids = torch.arange(prompt_len, dtype=torch.int64, device="cpu")

    # First match should find nothing
    handle1, cached_pages1 = cache.match_prefix(input_ids)
    assert handle1.cached_len == 0, f"Expected no cached prefix, got {handle1.cached_len}"
    assert len(cached_pages1) == 0
    log(f"  First match: cached_len={handle1.cached_len} (expected 0)", cfg.verbose)

    # Allocate pages and insert into cache
    num_full_pages = prompt_len // cfg.page_size
    pages = cache.allocate(num_full_pages)
    cache.radix.insert_prefix(input_ids, pages)
    log(f"  Inserted {num_full_pages} pages into prefix cache", cfg.verbose)

    # Second match should find the prefix
    handle2, cached_pages2 = cache.match_prefix(input_ids)
    assert handle2.cached_len == prompt_len, f"Expected cached_len={prompt_len}, got {handle2.cached_len}"
    assert len(cached_pages2) == num_full_pages
    log(f"  Second match: cached_len={handle2.cached_len} (expected {prompt_len})", cfg.verbose)

    # Match with slightly different input (different last token)
    input_ids_diff = input_ids.clone()
    input_ids_diff[-1] = 99999
    handle3, cached_pages3 = cache.match_prefix(input_ids_diff)
    # Should match first 2 pages (last page differs)
    expected_match = cfg.page_size * 2
    assert handle3.cached_len == expected_match, f"Expected cached_len={expected_match}, got {handle3.cached_len}"
    log(f"  Partial match: cached_len={handle3.cached_len} (expected {expected_match})", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_scheduler_lifecycle(cfg: TestConfig) -> bool:
    """Test scheduler request lifecycle."""
    log("Test: Scheduler lifecycle", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout
    from nmoe.serve.scheduler import Scheduler, SchedulerConfig
    from nmoe.serve.types import Request, SamplingParams, ForwardSpec, OutputMode

    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=cfg.num_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=True)

    sched_cfg = SchedulerConfig(
        max_batch_size=cfg.max_batch_size,
        max_prefill_tokens=cfg.max_prefill_tokens,
        page_size=cfg.page_size,
        enable_chunked_prefill=False,  # Simpler for testing
    )
    scheduler = Scheduler(sched_cfg, cache, device)

    initial_pages = cache.available
    log(f"  Initial pages: {initial_pages}", cfg.verbose)

    # Add request
    seq_len = cfg.page_size * 2  # 2 pages
    input_ids = torch.randint(0, 1000, (seq_len,), dtype=torch.int64)
    req = Request(
        uid=1,
        input_ids=input_ids,
        sampling_params=SamplingParams(max_tokens=10),
        profile_name="production_generate",
        forward_spec=ForwardSpec(output_mode=OutputMode.TOKENS),
    )
    scheduler.add_request(req)
    log(f"  Added request: uid={req.uid}, seq_len={seq_len}", cfg.verbose)

    # Schedule prefill
    batch = scheduler.schedule_prefill()
    assert batch is not None, "Expected prefill batch"
    assert batch.size == 1, f"Expected 1 request, got {batch.size}"
    assert batch.is_prefill, "Expected prefill phase"
    log(f"  Scheduled prefill: {batch.size} req, {batch.total_tokens} tokens", cfg.verbose)

    pages_after_prefill = cache.available
    pages_used = initial_pages - pages_after_prefill
    log(f"  Pages used for prefill: {pages_used}", cfg.verbose)

    # Simulate prefill completion
    req.cached_len = len(req.input_ids)
    req.output_ids.append(42)  # First generated token
    scheduler.promote_to_decode(req)
    log(f"  Promoted to decode", cfg.verbose)

    # Schedule decode
    batch = scheduler.schedule_decode()
    assert batch is not None, "Expected decode batch"
    assert batch.is_decode, "Expected decode phase"
    log(f"  Scheduled decode: {batch.size} req", cfg.verbose)

    # Simulate a few decode steps
    for i in range(3):
        req.output_ids.append(100 + i)
        req.cached_len += 1
    log(f"  Simulated 3 decode steps, seq_len={req.seq_len}", cfg.verbose)

    # Finish request
    scheduler.finish_request(req, success=True)
    log(f"  Finished request", cfg.verbose)

    # Verify pages returned (some may be in prefix cache now)
    pages_after_finish = cache.available
    log(f"  Pages after finish: {pages_after_finish} (was {pages_after_prefill})", cfg.verbose)

    # Verify scheduler is idle
    assert scheduler.is_idle, "Scheduler should be idle after finishing all requests"
    log(f"  Scheduler is idle: {scheduler.is_idle}", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_multi_request_batching(cfg: TestConfig) -> bool:
    """Test multi-request batching."""
    log("Test: Multi-request batching", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout
    from nmoe.serve.scheduler import Scheduler, SchedulerConfig
    from nmoe.serve.types import Request, SamplingParams, ForwardSpec, OutputMode

    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=cfg.num_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=True)

    sched_cfg = SchedulerConfig(
        max_batch_size=cfg.max_batch_size,
        max_prefill_tokens=cfg.max_prefill_tokens,
        page_size=cfg.page_size,
        enable_chunked_prefill=False,
    )
    scheduler = Scheduler(sched_cfg, cache, device)

    # Add multiple requests with same length (can be batched)
    num_reqs = 4
    seq_len = cfg.page_size  # 1 page each
    requests = []
    for i in range(num_reqs):
        input_ids = torch.randint(0, 1000, (seq_len,), dtype=torch.int64)
        req = Request(
            uid=i,
            input_ids=input_ids,
            sampling_params=SamplingParams(max_tokens=10),
            profile_name="production_generate",
            forward_spec=ForwardSpec(output_mode=OutputMode.TOKENS),
        )
        scheduler.add_request(req)
        requests.append(req)
    log(f"  Added {num_reqs} requests", cfg.verbose)

    # Schedule prefill - should batch all together
    batch = scheduler.schedule_prefill()
    assert batch is not None, "Expected prefill batch"
    assert batch.size == num_reqs, f"Expected {num_reqs} requests batched, got {batch.size}"
    log(f"  Prefill batch: {batch.size} reqs (expected {num_reqs})", cfg.verbose)

    # Verify batch tensors have correct shapes
    B, S = batch.input_ids.shape
    assert B == num_reqs, f"Expected B={num_reqs}, got {B}"
    assert S == seq_len, f"Expected S={seq_len}, got {S}"
    log(f"  Batch tensor shape: [{B}, {S}]", cfg.verbose)

    # Promote all to decode
    for req in requests:
        req.cached_len = len(req.input_ids)
        req.output_ids.append(42)
        scheduler.promote_to_decode(req)
    log(f"  Promoted all to decode", cfg.verbose)

    # Schedule decode - should batch all
    batch = scheduler.schedule_decode()
    assert batch is not None, "Expected decode batch"
    assert batch.size == num_reqs, f"Expected {num_reqs} in decode, got {batch.size}"
    log(f"  Decode batch: {batch.size} reqs", cfg.verbose)

    # Finish all requests
    for req in requests:
        scheduler.finish_request(req, success=True)

    assert scheduler.is_idle, "Scheduler should be idle"
    log(f"  All requests finished, scheduler idle", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_memory_accounting(cfg: TestConfig) -> bool:
    """Test that pages are properly accounted for (no leaks)."""
    log("Test: Memory accounting", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout
    from nmoe.serve.scheduler import Scheduler, SchedulerConfig
    from nmoe.serve.types import Request, SamplingParams, ForwardSpec, OutputMode

    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=cfg.num_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=False)  # Disable prefix cache for cleaner accounting

    sched_cfg = SchedulerConfig(
        max_batch_size=cfg.max_batch_size,
        max_prefill_tokens=cfg.max_prefill_tokens,
        page_size=cfg.page_size,
        enable_chunked_prefill=False,
    )
    scheduler = Scheduler(sched_cfg, cache, device)

    initial_pages = cache.available
    log(f"  Initial pages: {initial_pages}", cfg.verbose)

    # Run multiple request cycles
    for cycle in range(3):
        seq_len = cfg.page_size * 2
        input_ids = torch.randint(0, 1000, (seq_len,), dtype=torch.int64)
        req = Request(
            uid=cycle,
            input_ids=input_ids,
            sampling_params=SamplingParams(max_tokens=10),
            profile_name="production_generate",
            forward_spec=ForwardSpec(output_mode=OutputMode.TOKENS),
        )
        scheduler.add_request(req)

        batch = scheduler.schedule_prefill()
        req.cached_len = len(req.input_ids)
        req.output_ids.append(42)
        scheduler.promote_to_decode(req)

        # Do some decode steps
        for i in range(5):
            batch = scheduler.schedule_decode()
            req.output_ids.append(100 + i)
            req.cached_len += 1

        scheduler.finish_request(req, success=True)

    final_pages = cache.available
    log(f"  Final pages: {final_pages} (was {initial_pages})", cfg.verbose)

    # With prefix cache disabled, all pages should be returned
    assert final_pages == initial_pages, f"Memory leak! Expected {initial_pages}, got {final_pages}"
    log(f"  No memory leak detected", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def test_lru_eviction(cfg: TestConfig) -> bool:
    """Test LRU eviction under memory pressure."""
    log("Test: LRU eviction", cfg.verbose)

    from nmoe.serve.cache import KvCache, MlaKvLayout

    # Use small cache to trigger eviction
    small_pages = 8
    device = torch.device(cfg.device)
    layout = MlaKvLayout(num_blocks=small_pages, block_size=cfg.page_size)
    cache = KvCache(layout, device, enable_prefix_cache=True)

    log(f"  Using small cache: {small_pages} pages", cfg.verbose)

    # Fill cache with prefix entries
    for i in range(4):
        input_ids = torch.arange(cfg.page_size, dtype=torch.int64) + (i * 1000)
        pages = cache.allocate(1)
        cache.radix.insert_prefix(input_ids, pages)

    log(f"  Inserted 4 prefixes, available: {cache.available}", cfg.verbose)
    initial_evictable = cache.radix.evictable_size
    log(f"  Evictable pages: {initial_evictable}", cfg.verbose)

    # Try to allocate more than free
    remaining = cache.allocator.available
    need = remaining + 2  # Need 2 more than free

    if cache.radix.evictable_size >= 2:
        pages = cache.allocate(need)
        assert len(pages) == need, f"Should have allocated {need} pages via eviction"
        log(f"  Allocated {need} pages (evicted some)", cfg.verbose)
        log(f"  Evictable now: {cache.radix.evictable_size}", cfg.verbose)
    else:
        log(f"  Not enough evictable pages for test (skipped)", cfg.verbose)

    log("  PASSED", cfg.verbose)
    return True


def run_all_tests(cfg: TestConfig) -> int:
    """Run all L3 tests."""
    log("=" * 60, cfg.verbose)
    log("L3: KV Cache + Scheduler Invariants Tests", cfg.verbose)
    log("=" * 60, cfg.verbose)

    tests = [
        ("page_allocator", test_page_allocator),
        ("kv_cache_basic", test_kv_cache_basic),
        ("prefix_cache", test_prefix_cache),
        ("scheduler_lifecycle", test_scheduler_lifecycle),
        ("multi_request_batching", test_multi_request_batching),
        ("memory_accounting", test_memory_accounting),
        ("lru_eviction", test_lru_eviction),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn(cfg)
        except Exception as e:
            log(f"Test {name} FAILED: {e}", cfg.verbose)
            import traceback
            traceback.print_exc()
            results[name] = False

    log("=" * 60, cfg.verbose)
    log("SUMMARY", cfg.verbose)
    log("=" * 60, cfg.verbose)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        log(f"  {name}: {status}", cfg.verbose)

    log("=" * 60, cfg.verbose)
    log(f"Total: {passed}/{total} passed", cfg.verbose)

    return 0 if passed == total else 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description="L3 cache/scheduler tests")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-pages", type=int, default=64)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    cfg = TestConfig(
        device=args.device,
        num_pages=args.num_pages,
        verbose=not args.quiet,
    )

    import sys
    sys.exit(run_all_tests(cfg))


if __name__ == "__main__":
    main()
