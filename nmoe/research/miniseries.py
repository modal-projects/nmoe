"""Karpathy-style miniseries runner for research scaling experiments.

One clear path:
  python -m nmoe.research.miniseries --base-config configs/speedrun/small_moe_ultra.toml --depths 8,10,12,14,16 --checkpoints 0.02,0.05,0.10,0.20

Behavior:
  - Generates a TOML config per dial value into an output directory.
  - Computes token-indexed checkpoint steps, writes them as cfg.eval_steps.
  - Runs `torchrun -m nmoe.train <generated_config>` sequentially.
  - Collects CORE summaries from {metrics_dir}/eval/{run_id}/step_XXXXXXX/core_summary.json.
  - Writes a joinable CSV (series/contract/miniseries/dial/checkpoint_frac).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any

from nmoe.research.manifold import (
  build_dense_depth_manifold,
  build_moe_depth_manifold,
  steps_for_token_budget,
  tokens_total_from_params,
)
from nmoe.research.results import MiniseriesRow, append_row_csv


def wsd_schedule_from_horizon_steps(
  *,
  horizon_steps: int,
  tokens_per_step: int,
  warmup_ratio: float,
  warmdown_ratio: float,
) -> tuple[int, int, int]:
  """Compute WSD schedule fields for a token-indexed run horizon.

  nmoe's WSD schedule uses:
    - warmup_steps (step-based)
    - hold_tokens, decay_tokens (token-based)

  This helper expresses warmup/warmdown as *fractions of horizon steps* and
  snaps hold/decay to exact step boundaries for reproducibility.

  Returns:
    (warmup_steps, hold_tokens, decay_tokens)
  """
  steps = int(horizon_steps)
  tps = int(tokens_per_step)
  if steps <= 0:
    raise ValueError(f"horizon_steps must be > 0 (got {steps})")
  if tps <= 0:
    raise ValueError(f"tokens_per_step must be > 0 (got {tps})")
  wu = float(warmup_ratio)
  wd = float(warmdown_ratio)
  if not (0.0 <= wu < 1.0):
    raise ValueError(f"warmup_ratio must be in [0,1) (got {wu})")
  if not (0.0 < wd < 1.0):
    raise ValueError(f"warmdown_ratio must be in (0,1) (got {wd})")
  if wu + wd >= 1.0:
    raise ValueError(f"warmup_ratio + warmdown_ratio must be < 1 (got {wu} + {wd})")

  warmup_steps = max(1, int(math.floor(float(steps) * wu)))
  warmdown_steps = max(1, int(math.floor(float(steps) * wd)))

  # In very short horizons, rounding can collapse hold to 0 steps; keep it valid.
  if warmup_steps + warmdown_steps >= steps:
    warmup_steps = max(1, steps - warmdown_steps)

  hold_steps = max(0, steps - warmdown_steps)
  hold_tokens = int(hold_steps) * tps
  decay_tokens = int(warmdown_steps) * tps
  return int(warmup_steps), int(hold_tokens), int(decay_tokens)


def _parse_csv_ints(s: str) -> list[int]:
  out: list[int] = []
  for part in str(s).split(","):
    part = part.strip()
    if not part:
      continue
    out.append(int(part))
  if not out:
    raise ValueError("expected a non-empty comma list of ints")
  return out


def _parse_csv_floats(s: str) -> list[float]:
  out: list[float] = []
  for part in str(s).split(","):
    part = part.strip()
    if not part:
      continue
    out.append(float(part))
  if not out:
    raise ValueError("expected a non-empty comma list of floats")
  return out


def _parse_csv_strs(s: str) -> list[str]:
  out: list[str] = []
  for part in str(s).split(","):
    part = part.strip()
    if not part:
      continue
    out.append(part)
  if not out:
    raise ValueError("expected a non-empty comma list of strings")
  return out


def _toml_quote(s: str) -> str:
  s = str(s)
  s = s.replace("\\", "\\\\").replace("\"", "\\\"")
  return f"\"{s}\""


def _toml_value(v: Any) -> str:
  if isinstance(v, bool):
    return "true" if v else "false"
  if isinstance(v, int):
    return str(v)
  if isinstance(v, float):
    if not math.isfinite(v):
      raise ValueError("refusing to write non-finite float")
    # Avoid scientific notation; keep it readable.
    s = f"{v:.12f}".rstrip("0").rstrip(".")
    return s if s else "0.0"
  if isinstance(v, str):
    return _toml_quote(v)
  if isinstance(v, list):
    return "[" + ", ".join(_toml_value(x) for x in v) + "]"
  raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _write_flat_toml(path: Path, d: dict[str, Any]) -> None:
  lines: list[str] = []
  for k in sorted(d.keys()):
    v = d[k]
    if v is None:
      continue
    lines.append(f"{k} = {_toml_value(v)}")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_toml(path: Path) -> dict[str, Any]:
  return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_core_summary(metrics_dir: Path, run_id: str, step: int) -> float | None:
  p = metrics_dir / "eval" / run_id / f"step_{step:07d}" / "core_summary.json"
  try:
    obj = json.loads(p.read_text(encoding="utf-8"))
    return float(obj.get("CORE"))
  except Exception:
    return None


def _read_training_metric(metrics_dir: Path, run_id: str, step: int, tag: str) -> float | None:
  try:
    import duckdb
  except Exception:
    return None
  db = metrics_dir / run_id / "rank_0.duckdb"
  if not db.exists():
    return None
  try:
    con = duckdb.connect(str(db), read_only=True)
    try:
      row = con.execute(
        "select value from metrics where step=? and tag=? limit 1",
        [int(step), str(tag)],
      ).fetchone()
    finally:
      con.close()
    if not row:
      return None
    return float(row[0])
  except Exception:
    return None


def _read_router_max_load(metrics_dir: Path, run_id: str, step: int) -> float | None:
  """Best-effort max over per-layer router max_load tags at a step."""
  try:
    import duckdb
  except Exception:
    return None
  db = metrics_dir / run_id / "rank_0.duckdb"
  if not db.exists():
    return None
  try:
    con = duckdb.connect(str(db), read_only=True)
    try:
      row = con.execute(
        "select max(value) from metrics where step=? and tag like ?",
        [int(step), "router/layer_%/max_load"],
      ).fetchone()
    finally:
      con.close()
    if not row or row[0] is None:
      return None
    return float(row[0])
  except Exception:
    return None


def _pick_free_port() -> int:
  # Keep it explicit and deterministic enough for a single host: bind ephemeral.
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.bind(("127.0.0.1", 0))
    return int(s.getsockname()[1])
  finally:
    try:
      s.close()
    except Exception:
      pass


def _git_sha(repo_root: Path) -> str | None:
  try:
    p = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=str(repo_root),
      check=False,
      capture_output=True,
      text=True,
    )
    if p.returncode != 0:
      return None
    s = (p.stdout or "").strip()
    return s or None
  except Exception:
    return None


def _dial_value_depth(d: int) -> str:
  return f"d{int(d):02d}"


def _dial_value_experts(E: int) -> str:
  return f"E{int(E)}"


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser("nmoe.research.miniseries")
  # Provenance / IDs (required for "series" mode; optional for quick runs).
  ap.add_argument("--series-id", default=None, help="series identifier (e.g. s20260112_moe420)")
  ap.add_argument("--contract-id", default=None, help="contract identifier (encodes load-bearing invariants)")
  ap.add_argument("--miniseries-id", default=None, help="miniseries identifier (encodes dial + variant)")

  ap.add_argument("--base-config", default="configs/speedrun/small_moe_ultra.toml")

  dial = ap.add_mutually_exclusive_group(required=True)
  dial.add_argument("--depths", default=None, help="comma list of depths, e.g. 8,10,12")
  dial.add_argument("--experts", default=None, help="comma list of routed expert counts E, e.g. 0,32,64,128,256")
  ap.add_argument("--depth", type=int, default=0, help="fixed depth when sweeping --experts (dial=E)")

  ap.add_argument("--checkpoints", required=True, help="comma list of fractions of horizon, e.g. 0.02,0.05,0.10,0.20")

  # Model variant / knobs (miniseries must keep these fixed unless explicitly dialed).
  ap.add_argument("--variant", choices=["moe", "dense"], default="moe")
  ap.add_argument("--k-total", type=int, default=8, help="total experts activated per token (routed+shared)")
  ap.add_argument("--shared-experts", type=int, default=0, help="number of shared experts (dense MLP capacity), <= K_total")
  ap.add_argument("--dense-prefix", type=int, default=0, help="number of leading dense layers before MoE layers")
  ap.add_argument("--fixed-routed-experts", type=int, default=None, help="fix E for all runs (depth dial only)")

  # Manifold shape (kept intentionally small).
  ap.add_argument("--dim-per-layer", type=int, default=64)
  ap.add_argument("--dim-round-to", type=int, default=128)
  ap.add_argument("--head-dim", type=int, default=128)
  ap.add_argument("--max-experts", type=int, default=4096)

  # Horizon policy (a contract property).
  ap.add_argument("--horizon-policy", choices=["totalDN", "activeDN", "isoTPE"], default="totalDN")
  ap.add_argument("--target-param-data-ratio", type=float, default=None, help="DN multiplier; interpretation depends on --horizon-policy (default: 8)")
  ap.add_argument("--coef-tokens-per-param", type=float, default=None, help="deprecated alias for --target-param-data-ratio")
  ap.add_argument("--iso-tpe-from-depth", type=int, default=12)
  ap.add_argument("--iso-tpe-from-E", type=int, default=64)
  ap.add_argument("--iso-tpe-from-DN", type=float, default=8.0, help="anchor DN used to derive tokens-per-expert target (activeDN)")

  ap.add_argument("--warmup-ratio", type=float, default=0.01)
  ap.add_argument("--warmdown-ratio", type=float, default=0.40)
  ap.add_argument("--valid-steps", type=int, default=4)
  ap.add_argument("--eval-max-per-task", type=int, default=200)
  ap.add_argument("--eval-max-time-s", type=float, default=120.0)
  ap.add_argument("--out-dir", default="/tmp/nmoe_miniseries", help="output directory (series mode: pass full /data/series/.../miniseries path)")
  ap.add_argument("--nproc-per-node", type=int, default=8)
  ap.add_argument("--stop-after-depth", type=int, default=0, help="stop after this depth (inclusive); 0 disables")
  ap.add_argument("--dry-run", action="store_true")
  args = ap.parse_args(argv)

  # Enforce "series mode" IDs when provided.
  if (args.series_id is not None) or (args.contract_id is not None) or (args.miniseries_id is not None):
    if not args.series_id or not args.contract_id or not args.miniseries_id:
      raise ValueError("--series-id/--contract-id/--miniseries-id must be provided together")

  fracs = _parse_csv_floats(args.checkpoints)
  for f in fracs:
    if f <= 0.0 or f > 1.0:
      raise ValueError(f"checkpoint fractions must be in (0,1] (got {f})")

  if args.depths is not None:
    dial_name = "depth"
    dial_values = _parse_csv_ints(args.depths)
    fixed_depth = None
  else:
    dial_name = "E"
    dial_values = _parse_csv_ints(args.experts)
    if int(args.depth) <= 0:
      raise ValueError("--depth must be set to a positive integer when sweeping --experts")
    fixed_depth = int(args.depth)

  target_ratio = args.target_param_data_ratio
  alias_ratio = args.coef_tokens_per_param
  if target_ratio is None and alias_ratio is None:
    coef_tokens_per_param = 8.0
  elif target_ratio is None:
    coef_tokens_per_param = float(alias_ratio)
  elif alias_ratio is None:
    coef_tokens_per_param = float(target_ratio)
  else:
    if float(alias_ratio) != float(target_ratio):
      raise ValueError("--coef-tokens-per-param is deprecated; use only --target-param-data-ratio")
    coef_tokens_per_param = float(target_ratio)
  warmup_ratio = float(args.warmup_ratio)
  warmdown_ratio = float(args.warmdown_ratio)

  base_path = Path(args.base_config)
  base = _load_toml(base_path)

  seq_len = int(base.get("seq_len", 0) or 0)
  batch_size = int(base.get("batch_size", 0) or 0)
  if seq_len <= 0 or batch_size <= 0:
    raise ValueError("base config must define seq_len and batch_size")
  tokens_per_step = int(seq_len * batch_size)
  vocab_size = int(base.get("vocab_size", 50304) or 50304)

  metrics_dir = Path(str(base.get("metrics_dir", "/data/metrics")))
  series_id = str(args.series_id) if args.series_id else time.strftime("ms_%Y%m%d_%H%M%S")
  contract_id = str(args.contract_id) if args.contract_id else "c_default"
  miniseries_id = str(args.miniseries_id) if args.miniseries_id else "m_default"

  out_dir = Path(args.out_dir)
  if args.series_id is None:
    out_dir = out_dir / series_id
  out_dir.mkdir(parents=True, exist_ok=True)

  results_csv = out_dir / "results.csv"

  # Record plan for reproducibility.
  repo_root = Path(__file__).resolve().parents[2]
  git_sha = _git_sha(repo_root)

  k_total = int(args.k_total)
  if k_total <= 0:
    raise ValueError(f"--k-total must be > 0 (got {k_total})")
  n_shared = int(args.shared_experts)
  if n_shared < 0 or n_shared > k_total:
    raise ValueError(f"--shared-experts must be in [0,{k_total}] (got {n_shared})")
  dense_prefix = int(args.dense_prefix)
  if dense_prefix < 0:
    raise ValueError(f"--dense-prefix must be >= 0 (got {dense_prefix})")

  if dial_name == "E" and args.variant == "dense":
    if any(int(v) != 0 for v in dial_values):
      raise ValueError("--variant=dense cannot be used with --experts containing E>0")

  horizon_policy = str(args.horizon_policy)

  iso_tpe_target: float | None = None
  iso_anchor: dict[str, Any] | None = None
  if horizon_policy == "isoTPE":
    anchor_depth = int(args.iso_tpe_from_depth)
    anchor_E = int(args.iso_tpe_from_E)
    anchor_DN = float(args.iso_tpe_from_DN)
    if anchor_depth <= 0:
      raise ValueError("--iso-tpe-from-depth must be > 0")
    if anchor_E <= 0:
      raise ValueError("--iso-tpe-from-E must be > 0")
    if anchor_DN <= 0:
      raise ValueError("--iso-tpe-from-DN must be > 0")
    anchor_spec = build_moe_depth_manifold(
      n_layers=anchor_depth,
      k_total=k_total,
      n_shared_experts=n_shared,
      n_dense_layers=min(dense_prefix, anchor_depth),
      dim_per_layer=int(args.dim_per_layer),
      dim_round_to=int(args.dim_round_to),
      head_dim=int(args.head_dim),
      max_experts=int(args.max_experts),
      vocab_size=vocab_size,
      n_routed_experts=anchor_E,
    )
    anchor_tokens_total = tokens_total_from_params(anchor_spec.params_active, coef_tokens_per_param=float(anchor_DN))
    k_routed = int(anchor_spec.n_activated_experts)
    if k_routed <= 0:
      raise ValueError("isoTPE anchor must have K_routed > 0")
    iso_tpe_target = float(anchor_tokens_total) * (float(k_routed) / float(anchor_E))
    iso_anchor = {
      "depth": int(anchor_depth),
      "E": int(anchor_E),
      "K_total": int(k_total),
      "shared": int(n_shared),
      "dense_prefix": int(min(dense_prefix, anchor_depth)),
      "contract_ref": "activeDN",
      "DN": float(anchor_DN),
      "tokens_total": int(anchor_tokens_total),
      "tokens_per_expert_target": float(iso_tpe_target),
    }

  plan = {
    "series_id": str(series_id),
    "created_at": float(time.time()),
    "git_sha": git_sha,
    "contracts": [
      {
        "contract_id": str(contract_id),
        "tokenizer": str(base.get("tokenizer", "unknown")),
        "tokens_per_step": int(tokens_per_step),
        "horizon_policy": (
          {
            "type": str(horizon_policy),
            "DN": float(coef_tokens_per_param),
            "anchor": iso_anchor,
          }
          if horizon_policy == "isoTPE"
          else {"type": str(horizon_policy), "DN": float(coef_tokens_per_param)}
        ),
        "schedule_policy": {"warmup_ratio": float(warmup_ratio), "warmdown_ratio": float(warmdown_ratio)},
        "eval": {
          "core_manifest": "configs/eval/core.toml",
          "core_science": {"max_examples_per_task": int(args.eval_max_per_task), "max_time_s": float(args.eval_max_time_s)},
        },
      }
    ],
    "miniseries": [
      {
        "miniseries_id": str(miniseries_id),
        "dial": {"name": str(dial_name), "values": dial_values if dial_name == "depth" else dial_values},
        "variant": {"type": str(args.variant)},
        "model_knobs": {
          "k_total": int(k_total),
          "shared": int(n_shared),
          "dense_prefix": int(dense_prefix),
          "fixed_routed_experts": (int(args.fixed_routed_experts) if args.fixed_routed_experts is not None else None),
          "dim_per_layer": int(args.dim_per_layer),
          "dim_round_to": int(args.dim_round_to),
          "head_dim": int(args.head_dim),
          "max_experts": int(args.max_experts),
        },
        "checkpoints_frac": fracs,
        "base_config": str(base_path),
        "out_dir": str(out_dir),
      }
    ],
  }
  (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  # Helper: build a spec for this run.
  def build_spec(*, depth: int, routed_experts: int | None) -> tuple[Any, str]:
    if args.variant == "dense":
      return (
        build_dense_depth_manifold(
          n_layers=depth,
          dim_per_layer=int(args.dim_per_layer),
          dim_round_to=int(args.dim_round_to),
          head_dim=int(args.head_dim),
          vocab_size=vocab_size,
        ),
        "dense",
      )
    # MoE variant: routed_experts==0 means "fully dense" for this run.
    if routed_experts is not None and int(routed_experts) == 0:
      return (
        build_dense_depth_manifold(
          n_layers=depth,
          dim_per_layer=int(args.dim_per_layer),
          dim_round_to=int(args.dim_round_to),
          head_dim=int(args.head_dim),
          vocab_size=vocab_size,
        ),
        "dense",
      )
    return (
      build_moe_depth_manifold(
        n_layers=depth,
        k_total=k_total,
        n_shared_experts=n_shared,
        n_dense_layers=min(dense_prefix, depth),
        dim_per_layer=int(args.dim_per_layer),
        dim_round_to=int(args.dim_round_to),
        head_dim=int(args.head_dim),
        max_experts=int(args.max_experts),
        vocab_size=vocab_size,
        n_routed_experts=routed_experts,
      ),
      "moe",
    )

  # Main loop (one run per dial value).
  for dv in dial_values:
    if dial_name == "depth":
      depth = int(dv)
      if depth <= 0:
        raise ValueError(f"depth must be > 0 (got {depth})")
      dial_value = _dial_value_depth(depth)
      routed_experts = int(args.fixed_routed_experts) if args.fixed_routed_experts is not None else None
    else:
      depth = int(fixed_depth)
      E = int(dv)
      if E < 0:
        raise ValueError(f"routed experts E must be >= 0 (got {E})")
      dial_value = _dial_value_experts(E)
      routed_experts = E

    spec, spec_variant = build_spec(depth=depth, routed_experts=routed_experts)

    run_id = f"{series_id}__{contract_id}__{miniseries_id}__{dial_value}"

    # Horizon tokens (token-indexed; schedule derived from horizon steps).
    if horizon_policy == "totalDN":
      total_tokens = tokens_total_from_params(spec.params_total, coef_tokens_per_param=float(coef_tokens_per_param))
    elif horizon_policy == "activeDN":
      total_tokens = tokens_total_from_params(spec.params_active, coef_tokens_per_param=float(coef_tokens_per_param))
    elif horizon_policy == "isoTPE":
      if iso_tpe_target is None or iso_anchor is None:
        raise RuntimeError("isoTPE requires a computed anchor target")
      if int(spec.n_routed_experts) <= 0:
        total_tokens = int(iso_anchor["tokens_total"])
      else:
        k_routed = int(spec.n_activated_experts)
        if k_routed <= 0:
          raise ValueError("isoTPE requires K_routed > 0 for MoE runs")
        total_tokens = int(math.ceil(float(iso_tpe_target) * (float(spec.n_routed_experts) / float(k_routed))))
    else:
      raise ValueError(f"unknown horizon policy: {horizon_policy}")

    checkpoint_specs: list[dict[str, Any]] = []
    for f in fracs:
      tok = int(math.ceil(float(total_tokens) * float(f)))
      step = steps_for_token_budget(tok, tokens_per_step=tokens_per_step)
      checkpoint_specs.append({"frac": float(f), "tokens": int(tok), "step": int(step)})

    eval_steps_unique = sorted({int(c["step"]) for c in checkpoint_specs})
    max_step = max(eval_steps_unique)

    warmup_steps, hold_tokens, decay_tokens = wsd_schedule_from_horizon_steps(
      horizon_steps=max_step,
      tokens_per_step=tokens_per_step,
      warmup_ratio=warmup_ratio,
      warmdown_ratio=warmdown_ratio,
    )

    cfg = dict(base)
    # Architecture manifold overrides.
    cfg["preset"] = str(miniseries_id)
    cfg["experiment_id"] = str(run_id)
    cfg["n_layers"] = int(spec.n_layers)
    cfg["dim"] = int(spec.dim)
    cfg["n_heads"] = int(spec.n_heads)
    cfg["inter_dim"] = int(spec.inter_dim)
    # Model-mode selection: has_moe = (n_layers > n_dense_layers).
    cfg["n_dense_layers"] = int(spec.n_dense_layers) if spec_variant == "moe" else int(spec.n_layers)
    cfg["n_routed_experts"] = int(spec.n_routed_experts) if spec_variant == "moe" else 0
    cfg["n_shared_experts"] = int(spec.n_shared_experts) if spec_variant == "moe" else 0
    cfg["n_activated_experts"] = int(spec.n_activated_experts) if spec_variant == "moe" else 0
    cfg["moe_inter_dim"] = int(spec.moe_inter_dim) if spec_variant == "moe" else 0

    # This runner is "Karpathy-style":
    #   - smooth signal: valid/bpb
    #   - capability overlay: CORE
    cfg["validation_enabled"] = True
    cfg["validation_log_bpb"] = True
    cfg["validation_steps"] = max(1, int(args.valid_steps))
    cfg["validation_every"] = 0
    cfg["validation_at_steps"] = [int(s) for s in eval_steps_unique]
    cfg["target_loss"] = None

    # Training horizon: run only up to the max requested checkpoint.
    cfg["steps"] = int(max_step)

    # Isolate checkpoints per run_id so keep-last purging and tracker files
    # cannot race across unrelated runs sharing /data/checkpoints.
    cfg["checkpoint_dir"] = str(out_dir / "checkpoints" / run_id)
    cfg["checkpoint_keep_last_n"] = 0

    # Scale-consistent LR schedule: express warmup/warmdown as horizon ratios.
    cfg["warmup_steps"] = int(warmup_steps)
    cfg["hold_tokens"] = int(hold_tokens)
    cfg["decay_tokens"] = int(decay_tokens)

    # CORE integration: explicit eval steps (token-indexed checkpoints).
    cfg["eval_enabled"] = True
    cfg["eval_tasks"] = "core"
    cfg["eval_mode"] = "inline"
    cfg["eval_tasks_file"] = "configs/eval/core.toml"
    cfg["eval_every"] = 0
    cfg["eval_steps"] = [int(s) for s in eval_steps_unique]
    cfg["eval_budget_max_examples"] = int(args.eval_max_per_task)
    cfg["eval_budget_max_time_s"] = float(args.eval_max_time_s)

    # Avoid intermediate checkpoint churn; training still writes the final step.
    cfg["checkpoint_every"] = int(max_step) + 1

    # Write config TOML for this depth.
    cfg_path = out_dir / f"{dial_value}.toml"
    _write_flat_toml(cfg_path, cfg)

    if args.dry_run:
      print(f"[miniseries] run_id={run_id}")
      print(f"[miniseries] dial={dial_name} value={dial_value} variant={spec_variant}")
      print(f"[miniseries] depth={spec.n_layers} dim={spec.dim} heads={spec.n_heads} E={spec.n_routed_experts} K={spec.n_activated_experts} shared={spec.n_shared_experts}")
      print(f"[miniseries] params_total={spec.params_total:,} params_active={spec.params_active:,}")
      print(f"[miniseries] total_tokens≈{total_tokens:,} tokens_per_step={tokens_per_step:,} steps={max_step:,}")
      print(f"[miniseries] schedule: warmup_steps={warmup_steps} hold_tokens={hold_tokens:,} decay_tokens={decay_tokens:,}")
      print(f"[miniseries] eval_steps={cfg['eval_steps']}")
      print(f"[miniseries] config={cfg_path}")
      continue

    master_port = _pick_free_port()
    cmd = [
      "torchrun",
      f"--nproc_per_node={int(args.nproc_per_node)}",
      f"--master_port={int(master_port)}",
      "-m",
      "nmoe.train",
      str(cfg_path),
    ]
    env = os.environ.copy()
    env["NMOE_RUN"] = str(run_id)
    print(f"[miniseries] launch: {' '.join(cmd)}")
    p = subprocess.run(cmd, env=env, check=False)
    if int(p.returncode) != 0:
      raise RuntimeError(f"train failed (depth={spec.n_layers}, run_id={run_id}, rc={p.returncode})")

    # Best-effort collection of CORE scores.
    for c in checkpoint_specs:
      step = int(c["step"])
      frac = float(c["frac"])
      core = _read_core_summary(metrics_dir, run_id, step)
      v_loss = _read_training_metric(metrics_dir, run_id, step, "valid/loss")
      v_bpb = _read_training_metric(metrics_dir, run_id, step, "valid/bpb")
      r_cv = _read_training_metric(metrics_dir, run_id, step, "router_agg/mean_cv")
      r_me = _read_training_metric(metrics_dir, run_id, step, "router_agg/mean_entropy")
      r_mine = _read_training_metric(metrics_dir, run_id, step, "router_agg/min_entropy")
      r_mx = _read_router_max_load(metrics_dir, run_id, step)
      tpe = None
      if int(spec.n_routed_experts) > 0 and int(spec.n_activated_experts) > 0:
        tpe = float(step * tokens_per_step) * (float(spec.n_activated_experts) / float(spec.n_routed_experts))
      row = MiniseriesRow(
        series_id=str(series_id),
        contract_id=str(contract_id),
        miniseries_id=str(miniseries_id),
        run_id=str(run_id),
        dial_name=str(dial_name),
        dial_value=str(dial_value),
        checkpoint_frac=float(frac),
        depth=int(spec.n_layers),
        checkpoint_step=int(step),
        tokens_per_step=int(tokens_per_step),
        tokens_seen=int(step) * int(tokens_per_step),
        dim=int(spec.dim),
        n_heads=int(spec.n_heads),
        n_routed_experts=int(spec.n_routed_experts),
        n_shared_experts=int(spec.n_shared_experts),
        n_activated_experts=int(spec.n_activated_experts),
        params_total=int(spec.params_total),
        params_active=int(spec.params_active),
        tokens_per_routed_expert_seen=tpe,
        core=core,
        valid_loss=v_loss,
        valid_bpb=v_bpb,
        router_mean_cv=r_cv,
        router_max_load=r_mx,
        router_mean_entropy=r_me,
        router_min_entropy=r_mine,
      )
      append_row_csv(results_csv, row)

    # Verification mode: for quick sanity checks, users may want to stop early.
    # If the requested depths are sorted ascending, `--stop-after-depth` stops
    # after collecting results for that depth.
    stop_after = int(getattr(args, "stop_after_depth", 0) or 0)
    if dial_name == "depth" and stop_after and int(spec.n_layers) >= int(stop_after):
      return


if __name__ == "__main__":
  main()
