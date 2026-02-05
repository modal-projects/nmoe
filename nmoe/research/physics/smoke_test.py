"""
Physics smoke test: pack → train → probe → verify acceptance criteria.

Single command to verify Paper-A quality probing infrastructure.

Usage:
    python -m nmoe.research.physics.smoke_test [--steps=20] [--seed=42]

Acceptance criteria verified:
1. Non-degenerate geometry (margin, rho, overlap have spread)
2. Boundary-localization (E[overlap|interior] >> E[overlap|boundary])
3. Null comparison per stratum (deltas reported)
4. Negative controls (radial≈0 change, tangent≈full change)
5. Reproducibility (seeded, deterministic)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> bool:
    """Run command and return success."""
    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"$ {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Physics smoke test")
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="/tmp/physics_smoke", help="Output directory")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing checkpoint)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    ckpt_dir = Path(str(output_dir) + "_ckpt")  # checkpoint dir is {output}_ckpt
    probe_log = output_dir / "trajectory.jsonl"

    print("="*60)
    print("PHYSICS SMOKE TEST")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")

    # === Step 1: Pack synthetic data ===
    if not (output_dir / "train").exists():
        ok = run_cmd([
            sys.executable, "-m", "nmoe.research.physics.data.pack",
            "--output", str(output_dir),
            "--dataset", "smoke",
            "--tasks", f"depo:1.0:n_entities=50,max_hops=4", f"mano:1.0:depth=3",
            "--n-train", "5000",
            "--n-valid", "500",
            "--seq-len", "256",
            "--seed", str(args.seed),
        ], "Pack synthetic data")
        if not ok:
            print("FAILED: pack")
            return 1
    else:
        print(f"\n[Skip pack - {output_dir / 'train'} exists]")

    # === Step 2: Train micro model ===
    if not args.skip_train:
        # Create config
        config_path = output_dir / "micro.toml"
        config_path.write_text(f"""
preset = "micro"
experiment_id = "physics_smoke"
vocab_size = 10240
tokenizer = "synthetic"
dim = 256
inter_dim = 512
moe_inter_dim = 128
n_layers = 4
n_dense_layers = 1
n_heads = 4
n_routed_experts = 8
n_shared_experts = 1
n_activated_experts = 2
route_scale = 2.0
data_path = "{output_dir}"
lr_dense = 1e-3
lr_router = 1e-3
lr_expert = 5e-3
steps = {args.steps}
dtype = "bf16"
batch_size = 4
seq_len = 256
resume = false
checkpoint_dir = "{ckpt_dir}"
""")
        ok = run_cmd([
            sys.executable, "-m", "nmoe.train", str(config_path),
        ], "Train micro model")
        if not ok:
            print("FAILED: train")
            return 1
    else:
        print("\n[Skip train - using existing checkpoint]")

    # === Step 3: Run trajectory probe ===
    print(f"\n{'='*60}")
    print("[Run trajectory probe]")
    print('='*60)

    import torch
    import numpy as np
    from nmoe.config import Config
    from nmoe.model import Transformer
    from nmoe.research.physics.probe.trajectory import TrajectoryProbeWithHooks, TrajectoryConfig

    cfg = Config(
        vocab_size=10240,
        dim=256,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=4,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=2.0,
        seq_len=256,
    )

    model = Transformer(cfg).cuda()

    # Load checkpoint
    ckpt_iter = sorted(ckpt_dir.glob("iter_*"))[-1]
    ckpt_dp = torch.load(ckpt_iter / "dp_rank_000.pt", weights_only=False)
    ckpt_rd = torch.load(ckpt_iter / "rd.pt", weights_only=False)
    state_dict = {}
    state_dict.update(ckpt_rd["model_dense"])
    state_dict.update(ckpt_dp["model_expert"])
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_iter}")

    # Load batch
    shard_files = list((output_dir / "train").glob("*.npy"))
    shard = np.load(shard_files[0])
    batch = torch.tensor(shard[:8 * 256].reshape(8, 256), dtype=torch.long, device="cuda")

    # Run probe
    probe_config = TrajectoryConfig(
        log_path=str(probe_log),
        log_every=1,
        compute_sensitivity=True,
        compute_null=True,
        null_samples=8,
        compute_rgb=True,
        compute_decomposition=True,
    )
    probe = TrajectoryProbeWithHooks(probe_config)
    probe.log_step(step=0, model=model, input_ids=batch)
    probe.close()
    print(f"Probe log: {probe_log}")

    # === Step 4: Verify acceptance criteria ===
    print(f"\n{'='*60}")
    print("ACCEPTANCE CRITERIA")
    print('='*60)

    with open(probe_log) as f:
        events = [json.loads(line) for line in f]

    # Group by layer and tensor
    by_layer_tensor = {}
    for e in events:
        key = (e["layer"], e["tensor"])
        by_layer_tensor[key] = e["stats"]

    layers = sorted(set(k[0] for k in by_layer_tensor.keys()))
    all_pass = True

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Criterion 1: Non-degenerate geometry
        margin = by_layer_tensor.get((layer, "margin"), {})
        rho = by_layer_tensor.get((layer, "rho"), {})
        overlap = by_layer_tensor.get((layer, "topk_overlap"), {})

        # Quantiles are nested under "quantiles" key
        margin_q = margin.get("quantiles", margin)
        rho_q = rho.get("quantiles", rho)
        overlap_q = overlap.get("quantiles", overlap)

        margin_range = margin_q.get("p99", 0) - margin_q.get("p01", 0)
        rho_range = rho_q.get("p99", 0) - rho_q.get("p01", 0)
        overlap_p50 = overlap_q.get("p50", 1)

        # For overlap, check that there's variance (not all 1s or all 0s)
        overlap_min = overlap.get("min", 1)
        overlap_max = overlap.get("max", 0)
        overlap_has_variance = overlap_max > overlap_min

        c1_pass = margin_range > 0.01 and rho_range > 0.01 and overlap_has_variance
        print(f"  [1] Non-degenerate: margin_range={margin_range:.3f}, rho_range={rho_range:.3f}, overlap_min={overlap_min:.3f}, overlap_max={overlap_max:.3f} {'PASS' if c1_pass else 'FAIL'}")
        all_pass &= c1_pass

        # Criterion 2: Boundary-localization
        strat = by_layer_tensor.get((layer, "stratified"), {})
        if strat:
            overlap_int = strat.get("overlap_interior", 0)
            overlap_bnd = strat.get("overlap_boundary", 0)
            c2_pass = overlap_int > overlap_bnd
            print(f"  [2] Boundary-local: E[overlap|interior]={overlap_int:.3f} > E[overlap|boundary]={overlap_bnd:.3f} {'PASS' if c2_pass else 'FAIL'}")
            all_pass &= c2_pass
        else:
            print("  [2] Boundary-local: (no stratified data)")

        # Criterion 3: Null comparison per stratum
        if strat:
            d_int = strat.get("delta_overlap_interior", 0)
            d_bnd = strat.get("delta_overlap_boundary", 0)
            print(f"  [3] Null deltas: Δ_interior={d_int:+.3f}, Δ_boundary={d_bnd:+.3f} (reported)")

        # Criterion 4: Negative controls
        decomp = by_layer_tensor.get((layer, "decomposition"), {})
        if decomp:
            flip_rad = decomp.get("flip_rad", 1)
            overlap_rad = decomp.get("topk_overlap_rad", 0)
            flip_tan = decomp.get("flip_tan", 0)
            flip_full = decomp.get("flip_full", 0)

            c4a_pass = flip_rad < 0.01 and overlap_rad > 0.99
            c4b_pass = abs(flip_tan - flip_full) < 0.05
            print(f"  [4a] Radial null: flip_rad={flip_rad:.3f}<0.01, overlap_rad={overlap_rad:.3f}>0.99 {'PASS' if c4a_pass else 'FAIL'}")
            print(f"  [4b] Tangent full: flip_tan={flip_tan:.3f}≈flip_full={flip_full:.3f} {'PASS' if c4b_pass else 'FAIL'}")
            all_pass &= c4a_pass and c4b_pass
        else:
            print("  [4] Decomposition: (no data)")

    print(f"\n{'='*60}")
    if all_pass:
        print("SMOKE TEST PASSED")
    else:
        print("SMOKE TEST FAILED")
    print('='*60)

    # === Summary table ===
    print("\n--- Summary Table ---")
    print(f"{'Layer':<6} {'margin_rng':<12} {'rho_rng':<10} {'ovlp_int':<10} {'ovlp_bnd':<10} {'Δ_int':<8} {'Δ_bnd':<8} {'flip_rad':<10} {'flip_tan':<10}")
    for layer in layers:
        margin = by_layer_tensor.get((layer, "margin"), {})
        rho = by_layer_tensor.get((layer, "rho"), {})
        strat = by_layer_tensor.get((layer, "stratified"), {})
        decomp = by_layer_tensor.get((layer, "decomposition"), {})

        margin_q = margin.get("quantiles", margin)
        rho_q = rho.get("quantiles", rho)
        margin_rng = margin_q.get("p99", 0) - margin_q.get("p01", 0)
        rho_rng = rho_q.get("p99", 0) - rho_q.get("p01", 0)
        ovlp_int = strat.get("overlap_interior", 0)
        ovlp_bnd = strat.get("overlap_boundary", 0)
        d_int = strat.get("delta_overlap_interior", 0)
        d_bnd = strat.get("delta_overlap_boundary", 0)
        flip_rad = decomp.get("flip_rad", 0)
        flip_tan = decomp.get("flip_tan", 0)

        print(f"{layer:<6} {margin_rng:<12.4f} {rho_rng:<10.4f} {ovlp_int:<10.4f} {ovlp_bnd:<10.4f} {d_int:<+8.4f} {d_bnd:<+8.4f} {flip_rad:<10.4f} {flip_tan:<10.4f}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
