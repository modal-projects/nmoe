# Routing Geometry: why MoE routing doesn’t tear

Mixture-of-experts (MoE) models scale by making a small number of decisions: for each token, route it to a small subset of experts.

Those decisions are discrete. In principle they create hard switching surfaces: two nearby states can be sent through different subnetworks.

If the switches behave like hard switches, deep MoE transformers should be brittle. Small perturbations would flip experts and produce large jumps in the computation.

But modern MoE models are not brittle. They stack routed layers by the dozen and still train.

This post is about the simplest continuity mechanism we know that can make a hybrid system behave smooth, and how to test for it in a real top‑k router.

The mechanism is: in the region where routing is likely to change, the competing experts become functionally similar. Routing can flip without the computation changing much.

---

## A toy model (the whole story in one equation)

Consider a router that chooses between two experts:

```
f(x) = f1(x)   if g(x) > 0
     = f2(x)   if g(x) ≤ 0
```

The potential failure is not “the router is discontinuous.” That’s unavoidable.

The failure is the jump size near the boundary:

```
Δf(x) = f1(x) − f2(x)   (evaluated where g(x) ≈ 0)
```

If `Δf` is large where `g(x)` is near zero, then small perturbations flip experts and produce large computational jumps.

If `Δf` is small *in the same region*, routing can flip without changing the computation much. The hybrid system becomes continuous where it matters.

That is the mechanism we look for in top‑k MoE layers.

---

## Lifting the toy model to top‑k routing

To test the toy model’s mechanism in a top‑k MoE layer, we need three definitions. They are all “boring” on purpose: each one pins down a place where a story could otherwise sneak in.

### 1) The set boundary that flips top‑k membership

Fix a layer and its **exact** gate semantics (including grouping/masks/bias). Compute gate scores in **fp32** so margins are meaningful.

- `R_k(y) := TopK(Gate(y))` is the ordered top‑k expert set at gate input `y`.
- `m_set(y) := s_(k)(y) − s_(k+1)(y)` is the set margin.

If you care about top‑k routing, `m_set` is the boundary.

### 2) A routing sensitivity functional

In a pre‑norm transformer, the router sees a pre‑update gate input. It does not re‑route after applying its own update.

So we define a counterfactual:

- `y_pre`: the gate input the router actually sees.
- `y_post`: the gate input you would get after adding the layer’s routed update and re‑normalizing.

Then define routing self‑consistency:

- `ov_k^cf := |R_k(y_pre) ∩ R_k(y_post)| / k`

This is a single number per layer. High means “routing would be stable under counterfactual re‑gating.” Low means “it would change.”

### 3) The marginal swap routing actually makes

When `m_set(y)` is small, the likely top‑k change is a single swap: `e_k` trades with `e_(k+1)`.

So the continuity question becomes:

> Near low `m_set`, is swapping `e_k ↔ e_(k+1)` small?

We measure redundancy between the two competitors’ expert outputs, and include a hard control:

- **random‑e2 kill‑shot**: replace `e_(k+1)` with a random eligible competitor and confirm redundancy gets worse.

If that kill-shot doesn’t fire, we don’t trust the conclusion.

---

## What we measured (and why)

With those definitions in hand, there are three questions that matter:

1) Where does routing change happen?
2) When routing changes, does the computation jump?
3) Where does this picture break?

In a router, (1) is a statement about margins. In a model, (2) is a statement about expert functions. And (3) is a statement about depth.

---

## What we saw

### Routing change concentrates at the set boundary

Across domains and checkpoints, routing disagreement mass concentrates in the low `m_set` tail. This is the measurable version of “routing is likely to change near decision boundaries.”

### The boundary swap is benign (overlap-compatibility)

We then asked the toy-model question: when routing changes, does the computation jump?

V2 has two required criteria (two ``axes''), and they can decouple:

- **C1 (boundary vs interior)**: the adjacent competitors are *more redundant near the boundary* than in the interior.
- **C2 (adjacency specificity / random‑e2 kill-shot)**: the adjacent competitor is *more redundant than a random eligible competitor* at the boundary.

We call this *overlap-compatibility*: where the router is likely to switch, the adjacent “charts” overlap enough that the computation does not jump.

Reporting C1 and C2 separately matters in practice: across families and domains, we observe runs where V2 fails primarily on one axis (e.g. C1 erodes under tuning while C2 remains strong, or vice versa). Treating V2 as a single monolithic pass/fail would hide that structure.

### There is a terminal cliff in routing self-consistency

When we scan `ov_k^cf` across depth, the mid‑band is routing‑self‑consistent (high overlap), and then the terminal MoE layer falls off a cliff.

For one representative DeepSeek‑V3.2‑Speciale family run on FineWeb (k=8), `ov_k^cf` rises to about 0.80 through the mid‑band and collapses to about 0.276 at the terminal MoE layer.
The terminal adjacent jump is about −0.39 with a tight bootstrap CI.

One simple intuition for why the terminal layer can behave differently is that nothing routes on its output: there is no downstream MoE router that needs its representation to remain routing-self-consistent.

### The cliff follows the terminal position (cross-model)

A layer-index story is fragile. A terminal-position story is falsifiable.

So we bundled terminal-cliff scans from models with different terminal MoE depths and checked the same property in each: the dominant cliff localizes at that model’s terminal MoE layer, and there are genuinely distinct terminal depths across the bundle. In practice, this looks like “the cliff lands at the last routed layer,” even when the last routed layer is at a different depth.

### KL drift co-monotonically tracks base-domain regression (non-causal)

Post-training moves the model away from its base distribution. A simple way to quantify that movement is forward KL drift from the base model, measured on teacher-forced windows.

We checked whether larger KL drift tends to coincide with larger regression on base-domain loss across pre-registered edges in an open-weights family, on two canonical domains (`fineweb`, `code`). It does.

This is intentionally non‑causal: it’s a measured monotone relationship across real edges. The details are in the paper.

---

## A small baseline detail that mattered (and why we wrote a spec)

Our negative control initially failed in a way that looked like “behavioral change also concentrates at the boundary.”

It was an artifact.

If you define within‑window quantiles as order statistics with `k = ceil(q·T)`, the uniform null baseline is not `q`. It is:

```
q_eff = ceil(q·T) / T
```

Without that correction, a small offset can masquerade as a significant effect at small `q`. With the correction, the negative control passes.

This is why we wrote the paper in a spec-like style. Small details like baselines and bootstrap units belong in the contract, not in the interpretation.

---

## Reproducibility (briefly)

The paper expresses each claim as a property with explicit quantifiers (CI bounds, bootstrap protocol, required layer sets). The accompanying code produces JSON artifacts that store per-window arrays, and a verifier recomputes pass/fail decisions from the artifact alone.

If you want the full contract, see:

- `docs/routing_geometry_spec.tex`
- `docs/routing_geometry_artifacts.tex`

---

## Next

The natural next question is whether this geometry can be *preserved* under post-training by a structural intervention. The clean form of that question is an intervention test: does a recipe that constrains updates reduce KL drift and reduce base-domain regression relative to an unconstrained baseline, under controlled training?

## Open questions

- What happens to boundary redundancy as we push to ultra‑sparse MoE (more experts, thinner boundaries, smaller margins)?
- Can these measurements predict training instability early (before loss diverges)?
- What is the simplest intervention that preserves mid‑band overlap‑compatibility under post‑training?
