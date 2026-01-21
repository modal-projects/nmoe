"""
Gate profile visualization (zero-dependency).

Reads `train.jsonl` logs produced by `arch_ablations.py` and writes
publication-friendly SVG plots of per-layer gate activation profiles.

Key visualizations:
  - Gate mean vs layer at a specific step or matched-quality threshold
  - Comparison across variants (overlaid lines)
  - Summary metrics: early_mean, late_mean, late/early ratio

Run:
  # At final checkpoint
  python -m nmoe.research.physics.viz_gate --runs /data/physics/exp3_whitepaper_* --step -1

  # At matched accuracy threshold (first step where acc >= threshold)
  python -m nmoe.research.physics.viz_gate --runs /data/physics/exp3_whitepaper_* --match-acc 0.65

  # At specific step
  python -m nmoe.research.physics.viz_gate --runs /data/physics/exp3_whitepaper_* --step 1000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _lerp(a: float, b: float, t: float) -> float:
  return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
  return lo if x < lo else hi if x > hi else x


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
  r, g, b = rgb
  return f"#{r:02x}{g:02x}{b:02x}"


def _lerp_rgb(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
  t = _clamp(float(t), 0.0, 1.0)
  return (
    int(round(_lerp(c0[0], c1[0], t))),
    int(round(_lerp(c0[1], c1[1], t))),
    int(round(_lerp(c0[2], c1[2], t))),
  )


def _escape(s: str) -> str:
  return (
    s.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
    .replace("'", "&apos;")
  )


def _safe_name(s: str) -> str:
  out: list[str] = []
  for ch in s:
    if ch.isalnum() or ch in ("-", "_", "."):
      out.append(ch)
    else:
      out.append("_")
  return "".join(out)


# Color palette for multiple variants
COLORS = [
  "#e41a1c",  # red
  "#377eb8",  # blue
  "#4daf4a",  # green
  "#984ea3",  # purple
  "#ff7f00",  # orange
  "#a65628",  # brown
  "#f781bf",  # pink
  "#999999",  # gray
]


def _extract_layer_num(key: str) -> int:
  """Extract layer number from stat key like 'mem.layer0.gate_mean'."""
  m = re.search(r"layer(\d+)", key)
  return int(m.group(1)) if m else -1


def _extract_gates(stats: dict) -> list[tuple[int, float]]:
  """Extract (layer_num, gate_mean) pairs from stats dict."""
  gates = []
  for k, v in stats.items():
    if "gate_mean" in k:
      layer = _extract_layer_num(k)
      if layer >= 0:
        gates.append((layer, float(v)))
  gates.sort(key=lambda x: x[0])
  return gates


def _find_entry_at_step(log_path: Path, step: int) -> dict | None:
  """Find log entry at specific step (-1 for last with gate stats)."""
  if not log_path.exists():
    return None

  last_entry_with_gates = None
  with open(log_path) as f:
    for line in f:
      try:
        entry = json.loads(line)
        stats = entry.get("stats", {})
        has_gates = any("gate_mean" in k for k in stats.keys())

        if step >= 0 and entry.get("step") == step:
          return entry
        if has_gates:
          last_entry_with_gates = entry
      except json.JSONDecodeError:
        continue

  if step == -1:
    return last_entry_with_gates
  return None


def _find_entry_at_acc(log_path: Path, acc_threshold: float) -> dict | None:
  """Find first log entry where answer_token_acc >= threshold."""
  if not log_path.exists():
    return None

  with open(log_path) as f:
    for line in f:
      try:
        entry = json.loads(line)
        acc = entry.get("answer_token_acc", 0)
        if acc >= acc_threshold:
          return entry
      except json.JSONDecodeError:
        continue
  return None


def _compute_gate_summary(gates: list[tuple[int, float]]) -> dict:
  """Compute early/mid/late gate means and ratio."""
  if not gates:
    return {}

  n = len(gates)
  third = max(1, n // 3)

  early = [v for _, v in gates[:third]]
  mid = [v for _, v in gates[third:2*third]]
  late = [v for _, v in gates[-third:]]

  # Also compute last-3-layers specifically (where close-late is concentrated)
  last3 = [v for _, v in gates[-3:]] if n >= 3 else late

  early_mean = sum(early) / len(early) if early else 0
  mid_mean = sum(mid) / len(mid) if mid else 0
  late_mean = sum(late) / len(late) if late else 0
  last3_mean = sum(last3) / len(last3) if last3 else 0

  # L0 gate for reference
  l0 = gates[0][1] if gates else 0

  return {
    "n_layers": n,
    "early_mean": early_mean,
    "mid_mean": mid_mean,
    "late_mean": late_mean,
    "late_early_ratio": late_mean / early_mean if early_mean > 0 else 0,
    "last3_mean": last3_mean,
    "last3_to_L0": last3_mean / l0 if l0 > 0 else 0,
    "L0": l0,
  }


def _write_svg_gate_profile(
  *,
  out_path: Path,
  title: str,
  subtitle: str,
  series: list[tuple[str, list[tuple[int, float]], str, dict]],  # (name, gates, color, meta)
) -> None:
  """Write an SVG line chart of gate profiles."""
  if not series:
    return

  # Find max layers
  max_layers = max(len(gates) for _, gates, _, _ in series)
  if max_layers == 0:
    return

  # Chart dimensions
  left = 80
  right = 220
  top = 70
  bottom = 50
  chart_w = 500
  chart_h = 300
  width = left + chart_w + right
  height = top + chart_h + bottom

  # Y range (gate values typically 0-0.5)
  y_min = 0.0
  y_max = 0.5

  parts: list[str] = []
  parts.append('<?xml version="1.0" encoding="UTF-8"?>')
  parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
  parts.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')

  # Title
  parts.append(f'<text x="{left}" y="26" font-size="16" font-family="sans-serif" font-weight="bold">{_escape(title)}</text>')
  parts.append(f'<text x="{left}" y="46" font-size="11" font-family="sans-serif" fill="#666666">{_escape(subtitle)}</text>')

  # Chart area background
  parts.append(f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#cccccc" stroke-width="1"/>')

  # Y-axis grid lines and labels
  n_y_ticks = 5
  for i in range(n_y_ticks + 1):
    y_val = y_min + (y_max - y_min) * i / n_y_ticks
    y_pos = top + chart_h - (chart_h * i / n_y_ticks)
    parts.append(f'<line x1="{left}" y1="{y_pos}" x2="{left + chart_w}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(f'<text x="{left - 8}" y="{y_pos + 4}" font-size="10" font-family="sans-serif" text-anchor="end">{y_val:.2f}</text>')

  # Y-axis label
  y_label_x = 20
  y_label_y = top + chart_h / 2
  parts.append(f'<text x="{y_label_x}" y="{y_label_y}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle" transform="rotate(-90 {y_label_x} {y_label_y})">gate_mean</text>')

  # X-axis labels (every 5 layers)
  for i in range(0, max_layers, 5):
    x_pos = left + (chart_w * i / max(1, max_layers - 1))
    parts.append(f'<text x="{x_pos}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">L{i}</text>')
  # Always show last layer
  x_pos = left + chart_w
  parts.append(f'<text x="{x_pos}" y="{top + chart_h + 20}" font-size="10" font-family="sans-serif" text-anchor="middle">L{max_layers-1}</text>')

  # X-axis label
  parts.append(f'<text x="{left + chart_w/2}" y="{top + chart_h + 40}" font-size="11" font-family="sans-serif" font-weight="bold" text-anchor="middle">Layer</text>')

  # Draw lines
  for name, gates, color, meta in series:
    if not gates:
      continue

    points = []
    for layer, v in gates:
      x = left + (chart_w * layer / max(1, max_layers - 1))
      y = top + chart_h - (chart_h * (v - y_min) / max(0.001, y_max - y_min))
      y = max(top, min(top + chart_h, y))
      points.append(f"{x:.1f},{y:.1f}")

    if len(points) >= 2:
      parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    # Draw markers at key points (first, last, and every 10)
    for layer, v in gates:
      if layer == 0 or layer == len(gates) - 1 or layer % 10 == 0:
        x = left + (chart_w * layer / max(1, max_layers - 1))
        y = top + chart_h - (chart_h * (v - y_min) / max(0.001, y_max - y_min))
        y = max(top, min(top + chart_h, y))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/>')

  # Legend with summary stats
  legend_x = left + chart_w + 15
  legend_y = top + 10
  line_height = 16

  for i, (name, gates, color, meta) in enumerate(series):
    y_base = legend_y + i * (line_height * 5)

    # Color swatch and name
    parts.append(f'<line x1="{legend_x}" y1="{y_base}" x2="{legend_x + 20}" y2="{y_base}" stroke="{color}" stroke-width="2.5"/>')
    parts.append(f'<circle cx="{legend_x + 10}" cy="{y_base}" r="3" fill="{color}"/>')

    # Truncate long names
    display_name = name if len(name) <= 20 else name[:17] + "..."
    parts.append(f'<text x="{legend_x + 28}" y="{y_base + 4}" font-size="9" font-family="sans-serif" font-weight="bold">{_escape(display_name)}</text>')

    # Summary stats
    step = meta.get("step", "?")
    acc = meta.get("acc", 0) * 100
    summary = _compute_gate_summary(gates)

    y = y_base + line_height
    parts.append(f'<text x="{legend_x}" y="{y + 4}" font-size="8" font-family="monospace" fill="#666666">step={step} acc={acc:.1f}%</text>')

    y += line_height
    l0 = summary.get("L0", 0)
    last3 = summary.get("last3_mean", 0)
    last3_ratio = summary.get("last3_to_L0", 0)
    parts.append(f'<text x="{legend_x}" y="{y + 4}" font-size="8" font-family="monospace" fill="#666666">L0={l0:.3f} last3={last3:.3f}</text>')

    y += line_height
    parts.append(f'<text x="{legend_x}" y="{y + 4}" font-size="8" font-family="monospace" fill="#666666">last3/L0={last3_ratio:.2f}</text>')

  parts.append("</svg>")
  out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _find_run_dirs(run_paths: list[Path]) -> list[tuple[str, Path]]:
  """Find all variant run directories from provided paths."""
  results = []
  for p in run_paths:
    if not p.exists():
      continue
    # Check if this is a run output dir (has runs/ subdir)
    runs_dir = p / "runs"
    if runs_dir.is_dir():
      for variant_dir in runs_dir.iterdir():
        if variant_dir.is_dir():
          log_path = variant_dir / "train.jsonl"
          if log_path.exists():
            # Use parent dir name + variant as label
            label = f"{p.name}/{variant_dir.name}"
            results.append((label, variant_dir))
    # Or maybe it's directly a variant dir
    elif (p / "train.jsonl").exists():
      results.append((p.name, p))
  return results


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description="Render gate profile plots as SVG.")
  p.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories (can use glob)")
  p.add_argument("--out", type=Path, default=None, help="Output directory for SVGs (default: first run's analysis/viz)")
  p.add_argument("--step", type=int, default=None, help="Step to plot (-1 for final)")
  p.add_argument("--match-acc", type=float, default=None, help="Plot at first step where acc >= threshold")
  p.add_argument("--summary", action="store_true", help="Print summary table to stdout")
  args = p.parse_args(argv)

  if args.step is None and args.match_acc is None:
    args.step = -1  # Default to final

  run_dirs = _find_run_dirs(args.runs)
  if not run_dirs:
    raise SystemExit(f"No valid run directories found in {args.runs}")

  out_dir = args.out
  if out_dir is None:
    # Use first run's parent analysis/viz dir
    first_parent = args.runs[0]
    if (first_parent / "analysis").exists():
      out_dir = first_parent / "analysis" / "viz"
    else:
      out_dir = first_parent.parent / "analysis" / "viz"
  out_dir.mkdir(parents=True, exist_ok=True)

  # Collect data for all variants
  series = []
  summaries = []

  for label, variant_dir in run_dirs:
    log_path = variant_dir / "train.jsonl"

    # Find the right entry
    if args.match_acc is not None:
      entry = _find_entry_at_acc(log_path, args.match_acc)
      mode = f"acc>={args.match_acc:.0%}"
    else:
      entry = _find_entry_at_step(log_path, args.step)
      mode = f"step={args.step}" if args.step >= 0 else "final"

    if entry is None:
      print(f"SKIP {label}: no entry for {mode}")
      continue

    step = entry.get("step", 0)
    acc = entry.get("answer_token_acc", 0)
    stats = entry.get("stats", {})
    gates = _extract_gates(stats)

    if not gates:
      print(f"SKIP {label}: no gate stats at step {step}")
      continue

    color = COLORS[len(series) % len(COLORS)]
    meta = {"step": step, "acc": acc}
    series.append((label, gates, color, meta))

    summary = _compute_gate_summary(gates)
    summary["label"] = label
    summary["step"] = step
    summary["acc"] = acc
    summaries.append(summary)

  if not series:
    raise SystemExit("No valid gate data found")

  # Generate SVG
  if args.match_acc is not None:
    title = f"Gate Profile at Matched Accuracy (>={args.match_acc:.0%})"
    subtitle = "Lower late/early ratio = stronger 'close-late' pattern"
    filename = f"gate_profile_acc{int(args.match_acc*100)}.svg"
  else:
    step_str = "final" if args.step == -1 else f"step{args.step}"
    title = f"Gate Profile ({step_str})"
    subtitle = "Lower late/early ratio = stronger 'close-late' pattern"
    filename = f"gate_profile_{step_str}.svg"

  out_path = out_dir / filename
  _write_svg_gate_profile(
    out_path=out_path,
    title=title,
    subtitle=subtitle,
    series=series,
  )
  print(f"Wrote {out_path}")

  # Print summary table
  if args.summary or True:  # Always print summary
    print("\n=== Gate Profile Summary ===")
    print(f"{'label':<50} {'step':>6} {'acc':>7} {'L0':>6} {'last3':>6} {'L3/L0':>6}")
    print("-" * 92)
    for s in summaries:
      print(f"{s['label']:<50} {s['step']:>6} {s['acc']*100:>6.1f}% {s['L0']:>6.3f} {s['last3_mean']:>6.3f} {s['last3_to_L0']:>6.2f}")
    print()

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
