
  ---
  System Overview

  ┌─────────────────────────────────────────────────────────────────┐
  │                         USER REQUEST                            │
  └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      ADAPTER ROUTER                             │
  │  - Load user's consolidated adapter (if exists)                 │
  │  - Route to appropriate serving path                            │
  └─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │   ADAPTED INFERENCE      │        │   STANDARD INFERENCE     │
  │   (User has adapter)     │        │   (No adapter / new user)│
  │   - Base + LoRA          │        │   - Base model only      │
  │   - No context KV needed │        │   - Full context in KV   │
  └──────────────────────────┘        └──────────────────────────┘
                                                    │
                                                    ▼
                                      ┌──────────────────────────┐
                                      │   CONSOLIDATION TRIGGER  │
                                      │   (Async, after session) │
                                      └──────────────────────────┘
                                                    │
                                                    ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    CONSOLIDATION PIPELINE                       │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Component 1: Ingestion

  Purpose: Capture teacher activations from context

  ┌─────────────────────────────────────────────────────────────────┐
  │                        INGESTION                                │
  │                                                                 │
  │  Input: Raw context (conversation, docs, code)                  │
  │                                                                 │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
  │  │   Chunker   │───▶│   Teacher   │───▶│  Artifact   │         │
  │  │  (128K max) │    │   Forward   │    │   Writer    │         │
  │  └─────────────┘    └─────────────┘    └─────────────┘         │
  │                            │                   │                │
  │                            ▼                   ▼                │
  │                     ┌─────────────┐    ┌─────────────┐         │
  │                     │ Activations │    │  Artifact   │         │
  │                     │ + Routing   │    │   Store     │         │
  │                     └─────────────┘    │  (Blob+DB)  │         │
  │                                        └─────────────┘         │
  └─────────────────────────────────────────────────────────────────┘

  Artifact Schema (extending your existing distill schema):
  @dataclass
  class ConsolidationArtifact:
      # Identity
      user_id: str
      session_id: str
      chunk_id: int

      # Content
      context_bytes: bytes          # The context being consolidated
      query_bytes: bytes            # Representative queries

      # Teacher outputs
      hidden_targets: dict[int, torch.Tensor]  # layer -> activations
      routing_targets: dict[int, torch.Tensor] # layer -> router logits
      logit_targets: torch.Tensor              # final logits

      # Provenance
      base_model_id: str
      timestamp: datetime

  ---
  Component 2: Training

  Purpose: Train LoRA to match teacher activations

  ┌─────────────────────────────────────────────────────────────────┐
  │                        TRAINING                                 │
  │                                                                 │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
  │  │  Artifact   │───▶│   Student   │───▶│    Loss     │         │
  │  │   Loader    │    │  Forward    │    │  Compute    │         │
  │  └─────────────┘    │ (Base+LoRA) │    └─────────────┘         │
  │                     └─────────────┘           │                 │
  │                            ▲                  │                 │
  │                            │                  ▼                 │
  │                     ┌─────────────┐    ┌─────────────┐         │
  │                     │    LoRA     │◀───│  Optimizer  │         │
  │                     │   Adapter   │    │   Step      │         │
  │                     └─────────────┘    └─────────────┘         │
  │                                                                 │
  │  Loss = α·MSE(hidden) + β·CE(logits) + γ·KL(routing)           │
  └─────────────────────────────────────────────────────────────────┘

  Training Config:
  @dataclass
  class ConsolidationTrainingConfig:
      # Adapter
      lora_rank: int = 64
      lora_alpha: float = 16.0
      target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

      # Training
      learning_rate: float = 1e-4
      num_steps: int = 100
      batch_size: int = 1

      # Loss weights
      hidden_weight: float = 1.0
      logit_weight: float = 0.1
      routing_weight: float = 0.1

      # Matching layers
      match_layers: list[int] = [20, 30, 40, 50, 60]  # mid-band + terminal

  ---
  Component 3: Validation (Geometry Gate)

  Purpose: Ensure consolidation is safe before merge

  ┌─────────────────────────────────────────────────────────────────┐
  │                       VALIDATION                                │
  │                                                                 │
  │  Input: Candidate LoRA adapter                                  │
  │                                                                 │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │                    Geometry Checks                       │   │
  │  │                                                          │   │
  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │   │
  │  │  │   KL    │  │   V2    │  │ Calibr. │  │ Entropy │    │   │
  │  │  │  Check  │  │  Axes   │  │   Gap   │  │  Check  │    │   │
  │  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │   │
  │  │       │            │            │            │          │   │
  │  │       ▼            ▼            ▼            ▼          │   │
  │  │  ┌──────────────────────────────────────────────────┐  │   │
  │  │  │              Validation Decision                  │  │   │
  │  │  │  ALL pass → APPROVE    ANY fail → REJECT         │  │   │
  │  │  └──────────────────────────────────────────────────┘  │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  Output: {approved: bool, metrics: GeometryMetrics}             │
  └─────────────────────────────────────────────────────────────────┘

  Validation Thresholds:
  @dataclass
  class GeometryThresholds:
      max_kl_divergence: float = 0.1       # From base model
      min_v2_c1_delta: float = -0.005      # Boundary vs interior
      min_v2_c2_delta: float = -0.005      # Adjacency specificity
      max_calibration_gap: float = 0.2     # NLL - H
      max_entropy_drop: float = 0.5        # dH from base

  ---
  Component 4: Adapter Store

  Purpose: Manage per-user adapters with versioning

  ┌─────────────────────────────────────────────────────────────────┐
  │                      ADAPTER STORE                              │
  │                                                                 │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │                     DuckDB Index                         │   │
  │  │                                                          │   │
  │  │  user_id | version | created_at | geometry_ok | blob_ref │   │
  │  │  ────────┼─────────┼────────────┼─────────────┼───────── │   │
  │  │  user_1  │    3    │ 2026-01-12 │    true     │ s3://... │   │
  │  │  user_1  │    2    │ 2026-01-10 │    true     │ s3://... │   │
  │  │  user_2  │    1    │ 2026-01-11 │    true     │ s3://... │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  ┌─────────────────────────────────────────────────────────┐   │
  │  │                    Blob Storage                          │   │
  │  │                                                          │   │
  │  │  /adapters/{user_id}/v{version}/                         │   │
  │  │    ├── adapter_config.json                               │   │
  │  │    ├── adapter_weights.safetensors                       │   │
  │  │    └── geometry_report.json                              │   │
  │  └─────────────────────────────────────────────────────────┘   │
  │                                                                 │
  │  Operations:                                                    │
  │    - get_latest(user_id) -> Adapter                            │
  │    - save(user_id, adapter, metrics) -> version                │
  │    - rollback(user_id, version)                                │
  │    - merge(user_id, adapter) -> new_base                       │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Component 5: Serving

  Purpose: Inference with user-specific adapters

  ┌─────────────────────────────────────────────────────────────────┐
  │                        SERVING                                  │
  │                                                                 │
  │  ┌──────────────────────────────────────────────────────────┐  │
  │  │                    Request Handler                        │  │
  │  │                                                           │  │
  │  │  1. Extract user_id from request                          │  │
  │  │  2. Check adapter cache (hot) or store (cold)             │  │
  │  │  3. Route to appropriate serving path                     │  │
  │  └──────────────────────────────────────────────────────────┘  │
  │                           │                                     │
  │         ┌─────────────────┼─────────────────┐                  │
  │         ▼                 ▼                 ▼                  │
  │  ┌────────────┐   ┌────────────┐   ┌────────────┐              │
  │  │  Adapted   │   │  Fallback  │   │    New     │              │
  │  │  Serving   │   │  (KV/RAG)  │   │   User     │              │
  │  │            │   │            │   │            │              │
  │  │ Base+LoRA  │   │ Base+Cache │   │ Base only  │              │
  │  └────────────┘   └────────────┘   └────────────┘              │
  │                                                                 │
  │  Adapter Cache (per-GPU):                                       │
  │    - LRU eviction                                               │
  │    - Hot users stay loaded                                      │
  │    - Cold users loaded on demand (~100ms)                       │
  └─────────────────────────────────────────────────────────────────┘

  ---
  Component 6: Consolidation Trigger

  Purpose: Decide when to run consolidation

  @dataclass
  class ConsolidationPolicy:
      # Trigger conditions (any one)
      min_new_tokens: int = 50_000           # Tokens since last consolidation
      min_session_count: int = 10            # Sessions since last consolidation
      max_kv_cache_size: int = 10_000_000    # KV cache getting too big

      # Scheduling
      run_async: bool = True                  # Don't block user
      priority: str = "low"                   # Background job priority
      max_consolidation_time: int = 3600      # 1 hour max

      # Fallback
      on_validation_fail: str = "keep_in_retrieval"  # or "retry_lower_rank"

  ---
  Full Data Flow

  USER SESSION
      │
      ▼
  ┌─────────────────────┐
  │  Standard Inference │  ◀── First interaction / new context
  │  (Full KV cache)    │
  └─────────────────────┘
      │
      │ Session ends / trigger condition met
      ▼
  ┌─────────────────────┐
  │     Ingestion       │  ◀── Capture teacher activations
  │  (Async, batched)   │
  └─────────────────────┘
      │
      ▼
  ┌─────────────────────┐
  │     Training        │  ◀── Train LoRA on activation matching
  │  (Background job)   │
  └─────────────────────┘
      │
      ▼
  ┌─────────────────────┐
  │    Validation       │  ◀── Geometry gate
  └─────────────────────┘
      │
      ├── PASS ──▶ ┌─────────────────────┐
      │            │   Merge to Store    │
      │            └─────────────────────┘
      │                       │
      │                       ▼
      │            ┌─────────────────────┐
      │            │  Next Session:      │
      │            │  Adapted Inference  │  ◀── No KV cache needed
      │            └─────────────────────┘
      │
      └── FAIL ──▶ ┌─────────────────────┐
                   │  Keep in Retrieval  │  ◀── RAG fallback
                   └─────────────────────┘

  ---
  Key Design Decisions
  ┌─────────────────────┬────────────────────────────────┬──────────────────────────────────────────────┐
  │      Decision       │             Choice             │                  Rationale                   │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Adapter granularity │ Per-user                       │ Natural isolation, independent consolidation │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Merge strategy      │ Versioned, reversible          │ Can rollback if issues                       │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Validation          │ Hard gate                      │ Don't ship bad adapters                      │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Fallback            │ Retrieval/KV                   │ Graceful degradation                         │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Training            │ Async                          │ Don't block user sessions                    │
  ├─────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
  │ Serving             │ Adapter cache + on-demand load │ Balance memory vs latency                    │
  └─────────────────────┴────────────────────────────────┴──────────────────────────────────────────────┘
  ---
  Resource Estimates (per user)
  ┌──────────────────────┬───────────────────────────────┐
  │      Component       │           Size/Cost           │
  ├──────────────────────┼───────────────────────────────┤
  │ Adapter (rank 64)    │ ~450 MB                       │
  ├──────────────────────┼───────────────────────────────┤
  │ Adapter (rank 256)   │ ~1.8 GB                       │
  ├──────────────────────┼───────────────────────────────┤
  │ Training (100 steps) │ ~5 min on 1 GPU               │
  ├──────────────────────┼───────────────────────────────┤
  │ Validation           │ ~30 sec                       │
  ├──────────────────────┼───────────────────────────────┤
  │ Adapter load time    │ ~100 ms (cached) / ~2s (cold) │
  └──────────────────────┴───────────────────────────────┘
  ---
  Does this architecture match what you're envisioning? Want to drill into any component?

✻ Worked for 1m 7s

❯ from the remote team:   This matches the shape (async consolidation → per-user adapter → hard gate → serve with adapter + fallback), but a few parts need to change for it to be viable and to not conflate
  personalization vs global CL:

  - Personalization vs CL: per-user pipeline must not have merge(user_id, adapter) -> new_base (that’s a separate global-CL release pipeline, and would violate privacy unless opt-in + aggregation).
  - Artifact size: storing hidden_targets[layer] = activations for 128K is enormous (one BF16 hidden state tensor at DSv3 dim is ~128k * 7168 * 2B ≈ 1.75 GiB per layer per example). So “activation
  artifacts”
  need to be sparse/compressed (or just logits/top‑k), otherwise I/O dominates.
  - Use the existing distill/ primitive: what you wrote as logit_targets is exactly what DistillArtifact is designed to store sparsely (byte-offset keyed weighted continuations). Add router/routing
  targets only
  if needed, and keep them sparse too.
  - Adapter size estimates are off if LoRA is truly “on everything”: for DSv3 shape at EP=32, very rough storage is ~41 MiB * rank per GPU (so rank 64 ≈ 2.6 GiB/GPU; rank 256 ≈ 10.3 GiB/GPU). That makes
  “per-GPU
  LRU of many users” hard unless (a) rank/scope is much smaller, (b) adapters are quantized, or (c) you have a very different application mechanism than materializing weights.
  - Serving path: “standard inference = full context KV” can’t mean 10M tokens; the fallback has to be retrieval/paged KV anyway. The adapter is there to reduce how often you need that fallback, not
  eliminate it
  in the worst case.
  - Gate: geometry checks are necessary but not sufficient; the accept/reject must include an explicit recall suite (verbatim tests + paraphrases) plus base-regression tests, and thresholds must be
  calibrated
  empirically (not constants up front).

  If we drill into one component first, I’d pick artifact+loss, because it determines everything else. Do you want the primary objective to be:

  1. sparse logit distill (what distill/ already supports), or
  2. activation matching (but then we must define a compressed/sparse activation target)?

● The remote team is right on all points. Let me address the key corrections and the core question.

  ---
  Corrections Accepted
  ┌───────────────────────┬───────────────────────┬──────────────────────────────────────────────────────────────────┐
  │         Issue         │       My Design       │                            Correction                            │
  ├───────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Per-user merge → base │ Included              │ Remove. Per-user stays per-user. Global CL is separate pipeline. │
  ├───────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Hidden targets size   │ 1.75 GB/layer/example │ Impractical. Must be sparse/compressed.                          │
  ├───────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Adapter size          │ ~450 MB (rank 64)     │ ~2.6 GB/GPU at EP=32. Much harder to cache.                      │
  ├───────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Fallback              │ "Full KV cache"       │ Can't do 10M KV. Must be retrieval/paged anyway.                 │
  ├───────────────────────┼───────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Gate thresholds       │ Hardcoded constants   │ Must be empirically calibrated + recall tests.                   │
  └───────────────────────┴───────────────────────┴──────────────────────────────────────────────────────────────────┘
  ---
  The Core Question: What's the Training Target?

  Option 1: Sparse Logit Distillation

  Use existing distill/ infrastructure.

  # What you store
  DistillArtifact:
      x_bytes: bytes                    # Context
      targets: [PositionTarget, ...]    # Sparse (byte_offset → weighted continuations)

  # What you train on
  loss = sparse_first_token_distill_loss(student_logits, artifacts)

  Pros:
  - Infrastructure exists
  - Compact storage (~KB per position, not GB)
  - Proven to work for distillation

  Cons:
  - Only matches output distribution
  - Doesn't capture internal routing/computation
  - May not fully replicate "how" the model uses context

  ---
  Option 2: Activation Matching (Compressed)

  Define sparse activation targets.

  # What you store
  ActivationArtifact:
      x_bytes: bytes
      logit_targets: [PositionTarget, ...]  # Sparse, same as distill/
      routing_targets: [RoutingTarget, ...] # Sparse: which experts, what weights
      # NO dense hidden states

  # What you train on
  loss = (
      logit_loss(student_logits, logit_targets) +
      routing_loss(student_routing, routing_targets)
  )

  Pros:
  - Captures routing decisions (MoE-specific)
  - Still compact (routing is sparse by nature: top-k experts)
  - May better replicate ICL behavior

  Cons:
  - Need to define routing target format
  - More complex loss
  - Unclear if routing matching is necessary

  ---
  Option 3: Logits + Routing (Hybrid)

  Start with logits (proven), add routing only if needed.

  # Phase 1: Logit-only (baseline)
  loss = sparse_logit_distill_loss(...)

  # Phase 2: Add routing if logits aren't sufficient
  loss = sparse_logit_distill_loss(...) + λ * routing_match_loss(...)

  Pros:
  - Empirically driven
  - Don't add complexity until proven necessary
  - Builds on existing infra

  ---
  My Recommendation

  Start with Option 1 (sparse logit distillation) and measure.

  Why:
  1. You already have the infrastructure
  2. It's compact and proven
  3. The question "is output matching sufficient?" is empirical

  Add routing targets only if:
  - Logit-only consolidation fails on semantic recall
  - And the failure mode looks like "wrong computation path" not "wrong answer"

  ---
  Sparse Routing Target Format (If Needed)

  @dataclass
  class RoutingTarget:
      byte_offset: int
      layer: int
      expert_ids: tuple[int, ...]      # Top-k selected experts
      expert_weights: tuple[float, ...]  # Gating weights

  # Storage: ~10 bytes per position per MoE layer
  # For 128K positions × 30 MoE layers = ~38 MB (manageable)

  ---
  The Experimental Path

  1. Baseline: Sparse logit distillation only
     - Use existing distill/ schema
     - Measure recall on query types (salient, semantic, temporal)

  2. If baseline fails:
     - Diagnose: Is it capacity (need higher rank) or computation (need routing)?
     - If routing: Add sparse routing targets

  3. If routing helps:
     - Integrate into schema
     - Define routing loss weight empirically

  ---
  Revised Architecture (Artifact + Loss Only)

  # Artifact (extends existing DistillArtifact)
  @dataclass
  class ConsolidationArtifact:
      # Core (from distill/)
      example_id: str
      x_bytes: bytes
      targets: tuple[PositionTarget, ...]  # Sparse logit targets

      # Optional routing (add if needed)
      routing_targets: tuple[RoutingTarget, ...] | None = None

      # Provenance
      user_id: str
      session_id: str
      base_model_id: str

  # Loss
  def consolidation_loss(
      student_logits: torch.Tensor,
      student_routing: dict[int, torch.Tensor] | None,
      artifact: ConsolidationArtifact,
      routing_weight: float = 0.0,  # Start at 0, increase if needed
  ) -> torch.Tensor:

      loss = sparse_first_token_distill_loss(student_logits, artifact.targets)

      if artifact.routing_targets and routing_weight > 0:
          loss = loss + routing_weight * routing_match_loss(
              student_routing, artifact.routing_targets
          )

      return loss

  ---
  Does this framing work? Start with sparse logits, add routing empirically if needed?

✻ Cooked for 42s

❯ Yes: start with sparse logit distill, then add routing only if the failure mode demands it.

  Key adjustment so this works for “memory→weights” (not just vanilla distill):

  - Your DistillArtifact.x_bytes must be the student-visible text (typically query + teacher_answer in teacher-forcing form). The teacher can condition on [memory_chunk] + query, but when you write
  targets you
  should only distill the logits aligned to the query/answer region the student will actually see; otherwise the byte-offset mapping is distilling the wrong sequence.

  How I’d structure it:

  - Phase 0 (sanity): same tokenizer teacher/student, use existing DistillArtifact + sparse_first_token_distill_loss to learn “query-only → answer-as-if-memory-was-present”.
  - If recall fails: first try more supervision (more query types / paraphrases) and higher rank; only then add anchors.
  - If recall works but model drift/instability shows up: add constraints (base-regression suite + geometry/routing gates). Only add routing targets/loss if you see “right answers on train queries, wrong
  generalization” and diagnose it as path instability / expert reallocation.
  - If you add routing: don’t try to match “all layers, all tokens” first; start with a small set (e.g., a few mid/late MoE layers, only on answer tokens) and treat it as a regularizer to localize
  updates, not a
  primary objective.

  This framing works for both “per-user personalization” and “global CL” as a mechanism (distill → constrained update → gate), but the promotion target differs: per-user stays as an adapter; global CL
  promotes
  by merging to a new shared base.

● This is the complete design. Let me crystallize it.

  ---
  Memory → Weights via Constrained Distillation

  The Core Insight

  Teacher sees:   [memory_chunk] + [query]
  Student sees:   [query] only
  Distill:        logits aligned to [query/answer] region only

  The artifact stores what the student will see. The teacher's memory context is implicit in the targets, not stored.

  ---
  Phased Implementation

  Phase 0: Sanity (Baseline)

  Goal: Prove the mechanism works at all.

  Setup:
  - Same tokenizer for teacher/student
  - Existing DistillArtifact + sparse_first_token_distill_loss
  - Small context (32K), moderate rank (64)

  Artifact structure:
  # Teacher forward
  teacher_input = memory_chunk + query
  teacher_logits = model(teacher_input)

  # Distill only the query/answer region
  query_start = len(tokenize(memory_chunk))
  artifact = DistillArtifact(
      x_bytes=query.encode('utf-8'),  # Student-visible only
      targets=extract_targets(teacher_logits[query_start:]),  # Aligned to query
  )

  # Student training
  student_input = query  # No memory
  student_logits = model_with_lora(student_input)
  loss = sparse_first_token_distill_loss(student_logits, artifact)

  Success criteria:
  - Student reproduces teacher answers on trained queries
  - Basic recall works ("What was the original question?")

  ---
  Phase 1: If Recall Fails

  Diagnosis order:
  1. More supervision: More query types, paraphrases, diverse prompts
  2. Higher rank: 64 → 128 → 256
  3. Anchors: Add lightweight structure ("Turn N:", chunk IDs) only if needed

  Do NOT yet:
  - Add routing targets
  - Add geometry constraints
  - Change loss function

  ---
  Phase 2: If Drift/Instability

  Symptoms:
  - Recall works on trained queries
  - But base capabilities degrade
  - Or calibration shifts

  Add constraints:
  # Validation suite (run before accepting adapter)
  validation = {
      # Recall tests
      "verbatim_recall": test_exact_match(adapter, recall_queries),
      "paraphrase_recall": test_semantic_match(adapter, paraphrase_queries),

      # Base regression tests
      "base_capabilities": test_base_suite(adapter, mmlu_subset),
      "calibration": test_calibration_gap(adapter, held_out),

      # Geometry (your existing infrastructure)
      "kl_divergence": compute_kl_from_base(adapter),
      "v2_axes": compute_v2_health(adapter),
  }

  # Gate
  approved = (
      validation["verbatim_recall"] > 0.9 and
      validation["base_capabilities"] > baseline - 0.02 and
      validation["kl_divergence"] < threshold  # Calibrated empirically
  )

  Thresholds: Not hardcoded. Calibrate from Phase 0/1 runs.

  ---
  Phase 3: If Path Instability

  Symptoms:
  - Right answers on train queries
  - Wrong generalization to new queries
  - Diagnosis: Expert reallocation / routing instability

  Add routing targets as regularizer:
  @dataclass
  class RoutingTarget:
      byte_offset: int
      layer: int  # Only a few mid/late MoE layers
      expert_ids: tuple[int, ...]
      expert_weights: tuple[float, ...]

  # Loss (routing as regularizer, not primary)
  loss = (
      sparse_logit_distill_loss(student_logits, artifact.targets) +
      λ * routing_regularizer(student_routing, artifact.routing_targets)
  )

  # Start small
  routing_config = {
      "layers": [40, 50, 60],  # Few mid/late layers
      "tokens": "answer_only",  # Not all tokens
      "weight": 0.01,  # Small regularizer
  }

  ---
  Two Promotion Paths
  ┌───────────┬────────────────┬───────────────────────────────────────────────────┐
  │   Path    │     Target     │                  Merge Strategy                   │
  ├───────────┼────────────────┼───────────────────────────────────────────────────┤
  │ Per-user  │ User's adapter │ Never merge to global base. Stays isolated.       │
  ├───────────┼────────────────┼───────────────────────────────────────────────────┤
  │ Global CL │ Shared base    │ Aggregate, validate, release as new base version. │
  └───────────┴────────────────┴───────────────────────────────────────────────────┘
  Per-user (this design):
  User memory → Adapter → User's serving path
  (No global impact)

  Global CL (separate pipeline, future):
  Curated knowledge → Adapter → Validation → Merge to base → Release
  (Requires aggregation, privacy review, broader testing)

  ---
  Final Architecture (Minimal)

  ┌─────────────────────────────────────────────────────────┐
  │                    CONSOLIDATION                        │
  │                                                         │
  │  Input:                                                 │
  │    - memory_chunk: bytes (teacher context)              │
  │    - queries: list[str] (representative queries)        │
  │                                                         │
  │  Step 1: Teacher Forward                                │
  │    teacher_logits = model([memory_chunk] + query)       │
  │                                                         │
  │  Step 2: Extract Artifact                               │
  │    artifact = DistillArtifact(                          │
  │        x_bytes = query,  # Student-visible              │
  │        targets = sparse_topk(teacher_logits[query:])    │
  │    )                                                    │
  │                                                         │
  │  Step 3: Train Adapter                                  │
  │    student_logits = model_with_lora(query)              │
  │    loss = sparse_distill_loss(student_logits, artifact) │
  │                                                         │
  │  Step 4: Validate (Phase 2+)                            │
  │    if not passes_validation(adapter):                   │
  │        reject()                                         │
  │                                                         │
  │  Step 5: Store                                          │
  │    adapter_store.save(user_id, adapter)                 │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

  ---
  What's Built vs. What's Needed
  ┌────────────────────────────────────┬───────────────┬──────────────────────────────────────────┐
  │             Component              │    Status     │              Work Required               │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ DistillArtifact schema             │ ✓ Exists      │ None                                     │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ sparse_first_token_distill_loss    │ ✓ Exists      │ None                                     │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ Artifact producer (memory→targets) │ Partial       │ Adapt for memory chunk + query alignment │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ LoRA training loop                 │ Standard      │ Integrate with distill loss              │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ Validation suite                   │ Partial       │ Add recall tests, calibrate thresholds   │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ Geometry checks                    │ ✓ Exists      │ Wire into validation                     │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ Adapter store                      │ Design exists │ Implement                                │
  ├────────────────────────────────────┼───────────────┼──────────────────────────────────────────┤
  │ Serving with adapters              │ Standard      │ PEFT/similar                             │
  └────────────────────────────────────┴───────────────┴──────────────────────────────────────────┘
