# CLAUDE.md - Project Context & Behavioral Contract

---

# PROJECT OVERVIEW

We're building a state-of-the-art MoE training library for NVIDIA B200 GPUs.

**Goal**: Production-grade systems—model architecture, distributed training, custom kernels, optimizers, router controls, data loading, diagnostics—fully optimized for state-of-the-art MoE training on B200s.

**Ethos**: No fallbacks, no hacks, no shortcuts. Production-grade, Google-quality code with maniacal obsession with elegant minimalism.

---

# OPERATING MODES

## No-Edits Mode (Hard Gate)
Triggered by: "do not make any edits/changes", "review only", "planning/brainstorming mode"

In this mode:
- Do NOT modify tracked files
- Do NOT install deps
- Do NOT run destructive git operations
- Do NOT change cluster state
- ONLY read/inspect/analyze

Exit only when user explicitly authorizes: "proceed", "implement", "make the changes"

## Execution Mode
Default when user asks to implement/fix/build.

For wide-ranging changes (new public APIs, multi-file refactors, kernel/distributed rewrites):
- Propose approach + options first
- Wait for explicit confirmation before implementing

---

# COLLABORATION CONTRACT

## Scope / Approval
- Do exactly what was requested; ask before expanding scope
- After fixing a bug pattern, search for other occurrences (prefer `rg`) and fix them in-scope
- For ports/refactors, preserve semantics by default; call out intentional semantic deltas
- ONLY modify files explicitly requested - stable/working code is READ-ONLY unless told otherwise

## Git Safety
- Before edits, confirm branch (`git branch --show-current`) and working tree state (`git status -sb`)
- Do NOT run destructive git commands (`checkout`, `restore`, `reset`, `clean`) without explicit approval; explain what will be lost
- Never `git push` unless explicitly asked
- Only commit when explicitly asked

## Commands / Output
- Provide complete copy/paste-ready commands (include `cd`, env vars, and flags)
- If asked for "full diff/log output", do NOT truncate
- Avoid host side effects outside containers unless explicitly requested

## Approval Protocol
- Wait for explicit approval ("yes" / "proceed" / "ok" / "do it") before executing significant actions
- "What do you think?" = analyze, not implement
- When user says "just X" - execute X immediately without lengthy explanation

---

# PRINCIPLES

1. **One clear path per use-case**: Each supported mode has one explicit way to run. No multiple interchangeable stacks.
2. **Small, sharp surfaces**: Tiny modules with crisp responsibilities; few public knobs; declarative TOML config is source of truth.
3. **Explicit over magical**: No hidden background machinery or side effects; contracts and control flow are obvious.
4. **Hot paths first**: Inner loops and comm paths are lean, predictable, and measured. If it doesn't move tokens/s, stability, or correctness, it doesn't live there.
5. **Fail fast, fail loud**: Specific guardrails with actionable remedies. No silent downshifts.
6. **Minimal dependencies**: PyTorch + CuTeDSL + NVSHMEM. New layers must improve both clarity and performance.
7. **One source of truth**: One config format, one checkpoint format, one metrics schema. No duplicates to drift.
8. **Test what matters**: Deterministic resume, conservation, invariants. No scaffolding that mirrors system complexity.
9. **Container-first reproducibility**: Controlled build/runtime; off-target paths are explicit and opt-in.
10. **Documentation that guides, not overwhelms**: Precise runbooks and remedies; zero fluff.

---

# CRAFTSMANSHIP RUBRIC

For any change, ask:

| Question | Check |
|----------|-------|
| **Intent** | Does this improve tokens/s, stability, or correctness? |
| **Uniqueness** | Are we creating a second way to do something? If yes, why? |
| **Surface** | Did we add a new public knob? Could it be expressed via existing TOML? |
| **Hot path** | If step loop or comm changed, where is the 200-step NVTX + ms/step delta? |
| **Invariants** | Are B200, RDEP, determinism, and E%world enforced or clarified? |
| **Blast radius** | Did deps or coupling increase? |
| **Repro** | Is config/provenance captured to rerun months later? |
| **Elegance** | Is the code visibly simpler afterward? |

---

# SUPPORTED SCOPE (Intentionally Narrow)

- DeepSeek-shaped MoE architectures
- HuggingFace datasets (via `prep-mixture` CLI)
- B200 GPUs (sm_100a)
- RDEP expert parallelism (no tensor parallel, no NCCL all-to-all)

## Non-Goals
- **No tensor parallel (TP)** - ever in this library
- **No NCCL all-to-all** on the critical path for MoE; communications are RDEP-only

---

# TECHNICAL CONTRACTS

## Config
- TOML-only. CLI reads TOML with environment overrides; no YAML.

## Data
- No hardcoded locations; all paths provided via TOML
- Support robust dataset mixing (per-dataset weights/temperatures)
- Deterministic sharding and exact resume (RNG + shard/offset)

## Communications
- "No A2A for MoE" is the goal
- Must hold for forward path (RDEP dispatch + return only)
- Aim to hold for backward; if not feasible, never use NCCL all-to-all for MoE
- NCCL all-reduce is acceptable for data-parallel synchronization of dense/replicated parameters

## Platform
- Hard-target NVIDIA B200 (`sm_100a`)
- Build must error off-target (no silent downshift)

## Metrics/Tracking
- SQLite for experiments (`/data/experiments.db`)
- DuckDB for metrics (`/data/metrics/{run_id}/rank_{rank}.duckdb`)
- NVIZ reads from shared storage

---

# HARDWARE & PRECISION

| Aspect | Value |
|--------|-------|
| Primary target | NVIDIA B200 (`sm_100a`) |
| Supported dtypes | `bf16`, `fp8`, `nvfp4` (blockscaled FP4) |
| Default attention | MLA (Multi-headed Latent Attention) |
| Tokenizer | `o200k_harmony` (vocab_size=201088) |

## Optimizer & Schedule
- Separate LRs for dense, router, and expert params
- WSD scheduler (Warmup-Sustain-Decay) with token-based phases
- Optional Muon optimizer for attention layers

---

# RESEARCH WORKFLOW

1. **Architecture research** — Single GPU with small model (moonlet). Needs reasonable subset of training data.
2. **Ablations** — Compare specific changes (architecture, hyperparams, data). Single GPU (moonlet) or 8 GPU (moonlight). Hours, not days.
3. **Proxy runs** — Develop confidence around exact hyperparams and data mixtures. μP scaling, depth proxies, mixture tuning. Days.
4. **Production run** — Full training with high conviction. Weeks to months.

**Kernel work**: Must be fully optimized for NVFP4 and MXFP8 training with RDEP and stream sync for OUR workload.

**Distributed work**: Must be correct and optimal for our expert parallelism patterns.

---

# REFERENCES (Ideas, Not Dependencies)

| Reference | What We Borrow |
|-----------|----------------|
| Megatron-LM | Process-group design, sharded checkpoints |
| Torchtitan | Trainer ergonomics, metrics, schedules |
| Nanochat | Clear run flow, eval harness UX |
| TransformerEngine | FP8 scaling contracts and performance targets |

We borrow ideas; we do not depend on these at runtime.

---

# DEVELOPMENT WORKFLOW

## Session Start
- ALWAYS read AGENTS.md and AGENTS.md.local first
- If continuing from previous session, review conversation history/summary

## kubectl Sync Workflow
```bash
# Set pod name
export POD=$(kubectl get pods -l app=nmoe,stage=debug -o jsonpath='{.items[0].metadata.name}')

# Sync file to pod
kubectl cp nmoe/train.py $POD:/workspace/nmoe/nmoe/train.py

# Sync directory
kubectl cp nmoe/csrc $POD:/workspace/nmoe/nmoe/csrc

# Run on pod
kubectl exec $POD -- bash -c "source /workspace/nmoe/.venv/bin/activate && cd /workspace/nmoe && <command>"
```

**Rules**:
- NEVER run GPU-dependent code locally - always sync and run on pod
- Data is on cluster at /data - use kubectl exec to interact
- Always source venv: `source /workspace/nmoe/.venv/bin/activate`
- Set PYTHONPATH: `export PYTHONPATH=/workspace/nmoe`

## CUDA Build Protocol
- ONLY run `make` when CUDA source files (*.cu, *.cuh, *.cpp in csrc/) actually changed
- NEVER delete/overwrite compiled .so files without checking if rebuild needed
- Check timestamps: `kubectl exec $POD -- ls -la /workspace/nmoe/nmoe/csrc/*.so`
- NEVER use `make clean` unless absolutely necessary
- Full CUDA rebuild takes 45+ minutes - unnecessary rebuilds are unacceptable

## Config Management
- When training fails, FIRST check config, not code - stable code exists
- Avoid "config explosion" - update existing configs rather than creating new ones
- Track config changes and be able to revert

## Long-Running Jobs
- For downloads, training, data prep: periodically check status
- "Check in" = verify progress and report: status, progress, duration, errors, resource usage

---

# RESEARCH METHODOLOGY

## No Guessing
- NEVER guess at system state, file contents, or config values
- Before saying something doesn't exist: search harder, use multiple patterns, check different locations
- Before claiming something is correct: verify against reference implementation

## Hypothesis-Driven Research
Every change must have:
1. Hypothesis with reasoning
2. Testable experiment
3. Evidence from execution
4. Chain: hypothesis → experiment → observation → conclusion

NO "random thrashing at hyperparams" - that is not research.

## Ultrathink Protocol
When user says "ultrathink":
1. STOP and THINK DEEPLY - not quick analysis
2. GATHER ALL EVIDENCE - read files completely, check git diffs, review history
3. STRUCTURE THE ANALYSIS:
   - What do we KNOW (verified facts)?
   - What do we NOT KNOW (gaps)?
   - What are the HYPOTHESES (with reasoning)?
   - What EXPERIMENTS would test each hypothesis?
4. TRACE THROUGH CAREFULLY - execution paths, before/after diffs
5. PRESENT REASONED CONCLUSIONS with evidence

---

# CODE STYLE

- ALL imports at top of files - no inline/conditional imports
- Indentation: tab = 2 spaces
- Minimize if/else chains - avoid defensive clutter
- Production-grade, Google-quality code

---

# PROMPT TEMPLATES

## DEBUG Protocol
1. CHECK LOGS: `kubectl exec $POD -- tail -100 /tmp/<logfile>`
2. VERIFY STATE: `kubectl exec $POD -- ls -la <path>`
3. CHECK GIT DIFF: `git diff HEAD -- <relevant-files>`
4. FORM HYPOTHESIS based on evidence (not speculation)
5. DESIGN MINIMAL TEST
6. EXECUTE AND GATHER DATA
7. FIX PRECISELY - minimum change needed

## PROPOSE PLAN Format
```
## Summary
One sentence describing what and why.

## Approach
- Option A: <description> - Pros/Cons
- Option B: <description> - Pros/Cons

## Recommended: Option X
Reasoning.

## Implementation Steps
1. Step one (scope: which files)
2. Step two (scope: which files)

## Files to Modify
- file1.py - <what changes>

---
Awaiting approval to proceed.
```

---

# COMMON MISTAKES TO AVOID

1. **NOT READING AGENTS.md** - Always read at session start
2. **GUESSING SYSTEM STATE** - Read/check actual state, never speculate
3. **MODIFYING OUT-OF-SCOPE FILES** - Only modify explicitly requested files
4. **UNNECESSARY MAKE REBUILDS** - 45+ min cost, only when .cu files changed
5. **LAZY TROUBLESHOOTING** - Every diagnosis needs supporting data
6. **SCATTERED IMPORTS** - All imports at top, tab = 2 spaces
7. **CONFIG EXPLOSION** - Update existing configs, don't create new ones
8. **RUNNING LOCALLY** - Always sync to pod and run there
9. **VERBOSITY WHEN ACTION REQUESTED** - "just X" means do X immediately
10. **IGNORING CONVERSATION HISTORY** - Previous working configs may be in history
11. **IMPLEMENTING BEFORE APPROVAL** - Propose first, wait for explicit approval
12. **NOT CHECKING ON JOBS** - Long-running jobs need periodic status checks
13. **RANDOM HYPERPARAM THRASHING** - Every change needs hypothesis + test
14. **DELETING THEN COPYING FILES** - Preserve build artifacts when possible
15. **SURFACE-LEVEL ANALYSIS** - "Ultrathink" means deep systematic analysis
16. **CREATING SECOND WAYS** - One clear path per use-case
17. **SILENT DOWNSHIFTS** - Fail fast, fail loud
18. **ADDING PUBLIC KNOBS** - Could it be expressed via existing TOML?
