"""n - nmoe training CLI."""

import os
import subprocess
import time
import shutil
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
  name="n",
  help="nmoe training CLI",
  add_completion=False,
  no_args_is_help=True,
)
console = Console()

NMOE_ROOT = Path(__file__).parent.parent.parent
NPROC = os.environ.get("NPROC", "8")


def _discover_data_dir() -> Path:
  """Discover data directory from env or common locations."""
  # Explicit env var takes precedence
  if "DATA_DIR" in os.environ:
    return Path(os.environ["DATA_DIR"])
  # Check common locations
  candidates = [
    Path("/data"),
    NMOE_ROOT / "data",
    Path.home() / "nmoe_data",
  ]
  for p in candidates:
    if p.exists():
      return p
  # Default to /data, will be created as needed
  return Path("/data")


DATA_DIR = _discover_data_dir()


def _get_port(name: str, default: int) -> int:
  """Get port from env, handling service discovery collisions."""
  val = os.environ.get(f"NMOE_{name}", os.environ.get(name, str(default)))
  try:
    return int(val)
  except ValueError:
    return default

JUPYTER_PORT = _get_port("JUPYTER_PORT", 8888)
NVIZ_PORT = _get_port("NVIZ_PORT", 3000)


def _with_nmoe_env(env: dict | None = None) -> dict:
  """Return env with required PYTHONPATH for nmoe's vendored deps.

  This makes `n speedrun` work in minimal environments (e.g. k8s debug pods)
  without requiring users to manually export PYTHONPATH.
  """
  out = (env or os.environ).copy()

  # Ensure quack and in-repo triton are importable.
  required = [
    str(NMOE_ROOT),
    str(NMOE_ROOT / "third_party" / "quack"),
    str(NMOE_ROOT / "triton" / "python"),
  ]
  # Only add entries that exist on disk (allows pip-installed fallbacks).
  required = [p for p in required if Path(p).exists()]

  parts = [p for p in str(out.get("PYTHONPATH", "")).split(":") if p]
  for p in required:
    if p not in parts:
      parts.append(p)
  if parts:
    out["PYTHONPATH"] = ":".join(parts)

  return out


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
  """Run command, exit on failure."""
  console.print(f"[blue]>[/blue] {' '.join(cmd)}")
  result = subprocess.run(cmd, cwd=cwd or NMOE_ROOT, env=_with_nmoe_env(env))
  if result.returncode != 0:
    raise typer.Exit(result.returncode)


def run_background(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.Popen:
  """Run command in background."""
  console.print(f"[blue]>[/blue] {' '.join(cmd)} &")
  return subprocess.Popen(
    cmd,
    cwd=cwd or NMOE_ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=_with_nmoe_env(env),
  )


def _has_npy_shards(dir_path: Path) -> bool:
  try:
    return dir_path.exists() and any(dir_path.rglob("*.npy"))
  except Exception:
    return False


def ensure_eval_bundle() -> Path:
  """Ensure CORE eval bundle exists. Fails if missing (should be installed by bootstrap.sh).

  Returns path to eval_bundle directory.
  """
  bundle_dir = DATA_DIR / "eval" / "eval_bundle"

  if bundle_dir.exists() and any(bundle_dir.rglob("*.jsonl")):
    return bundle_dir

  console.print(f"[red]CORE eval bundle not found at {bundle_dir}[/red]")
  console.print("[yellow]Run 'bash scripts/bootstrap.sh' to install it, or download manually:[/yellow]")
  console.print("  curl -L -o /tmp/eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip")
  console.print(f"  unzip /tmp/eval_bundle.zip -d {bundle_dir.parent}")
  raise typer.Exit(1)


def ensure_speedrun_data() -> Path:
  """Ensure speedrun train/val datasets exist.

  Canonical dataset: karpathy/fineweb-edu-100b-shuffle (tokenized to GPT-2 shards).
  Returns path to DATA_DIR/speedrun.
  """
  hf_dataset = os.environ.get("NMOE_SPEEDRUN_DATASET", "karpathy/fineweb-edu-100b-shuffle")
  val_data_file = os.environ.get("NMOE_SPEEDRUN_VAL_DATA_FILE", "shard_01822.parquet")
  train_tokens_budget = "10B"
  val_tokens_budget = "10485760"
  train_tokens_min = int(float(train_tokens_budget[:-1]) * 1_000_000_000) if train_tokens_budget.endswith("B") else int(train_tokens_budget)
  val_tokens_min = int(val_tokens_budget)

  train_dir = DATA_DIR / "speedrun" / "train"
  val_dir = DATA_DIR / "speedrun" / "val"

  def _manifest_ok(dir_path: Path, *, min_tokens: int) -> bool:
    m = dir_path / "manifest.json"
    if not m.exists():
      return False
    try:
      import json
      obj = json.loads(m.read_text())
    except Exception:
      return False
    src = str(obj.get("source_info", {}).get("source", ""))
    if not src.startswith(hf_dataset):
      return False
    if obj.get("tokenizer") != "gpt2":
      return False
    if int(obj.get("vocab_size", 0)) != 50304:
      return False
    if int(obj.get("eos_token_id", -1)) != 50256:
      return False
    if int(obj.get("total_tokens", 0)) < int(min_tokens):
      return False
    return True

  if _has_npy_shards(train_dir) and _has_npy_shards(val_dir):
    if not (_manifest_ok(train_dir, min_tokens=train_tokens_min) and _manifest_ok(val_dir, min_tokens=val_tokens_min)):
      raise typer.Exit(
        f"Non-canonical speedrun dataset detected at {DATA_DIR / 'speedrun'} (missing/mismatched manifest).\n"
        f"Expected: {hf_dataset} (gpt2 vocab=50304 eos=50256) with >= {train_tokens_budget} train tokens and >= {val_tokens_budget} val tokens.\n"
        f"To rebuild: rm -rf {DATA_DIR / 'speedrun'}"
      )
    console.print(f"[green]Data:[/green] {train_dir} (+ val)")
    return DATA_DIR / "speedrun"

  train_dir.mkdir(parents=True, exist_ok=True)
  val_dir.mkdir(parents=True, exist_ok=True)

  if not _has_npy_shards(train_dir):
    console.print(f"[yellow]Preparing speedrun train dataset → {train_dir}[/yellow]")
    run([
      "python", "-m", "nmoe.data.cli", "prep",
      "--source", "hub_parquet",
      "--dataset", hf_dataset,
      "--split", "train",
      "--output", str(train_dir),
      "--name", "speedrun_train",
      "--tokenizer", "gpt2",
      "--vocab-size", "50304",
      "--eos-token-id", "50256",
      "--max-tokens-total", train_tokens_budget,
      "--num-shards", "64",
      "--parallel",
    ])
    if not _manifest_ok(train_dir, min_tokens=train_tokens_min):
      raise typer.Exit(f"speedrun train dataset prepared but manifest is not canonical: {train_dir / 'manifest.json'}")

  if not _has_npy_shards(val_dir):
    console.print(f"[yellow]Preparing speedrun val dataset → {val_dir}[/yellow]")
    run([
      "python", "-m", "nmoe.data.cli", "prep",
      "--source", "hub_parquet",
      "--dataset", hf_dataset,
      "--split", "train",
      "--data-files", val_data_file,
      "--output", str(val_dir),
      "--name", "speedrun_val",
      "--tokenizer", "gpt2",
      "--vocab-size", "50304",
      "--eos-token-id", "50256",
      "--max-tokens-total", val_tokens_budget,
      "--num-shards", "8",
      "--parallel",
    ])
    if not _manifest_ok(val_dir, min_tokens=val_tokens_min):
      raise typer.Exit(f"speedrun val dataset prepared but manifest is not canonical: {val_dir / 'manifest.json'}")

  return DATA_DIR / "speedrun"


def ensure_data(name: str, tokens: str) -> Path:
  """Download data if not present, return path.

  Note: This is intentionally a small convenience helper. Golden-path training
  configs should still specify canonical data paths; this helper is for quick
  bring-up and smoke tests.
  """
  data_path = DATA_DIR / name
  if _has_npy_shards(data_path):
    console.print(f"[green]Data:[/green] {data_path}")
    return data_path

  data_path.mkdir(parents=True, exist_ok=True)
  console.print(f"[yellow]Downloading {tokens} tokens to {data_path}...[/yellow]")
  run([
    "python", "-m", "nmoe.data.cli", "prep",
    "--source", "hf",
    "--dataset", "HuggingFaceFW/fineweb-edu",
    "--split", "train",
    "--output", str(data_path),
    "--name", name,
    "--tokenizer", "gpt2",
    "--vocab-size", "50304",
    "--eos-token-id", "50256",
    "--max-tokens-total", tokens,
    "--num-shards", "32",
    "--parallel",
  ])
  return data_path


def start_training(config: str, *, data_path_override: Path | None, background: bool = False):
  """Start training, optionally in background."""
  cmd = [
    "torchrun", "--nproc_per_node", NPROC,
    "-m", "nmoe.train",
    config,
  ]
  if data_path_override is not None:
    cmd.append(f"--data_path={data_path_override}")
  if background:
    run_background(cmd)
    time.sleep(2)  # Let training initialize
  else:
    run(cmd)


def start_tunnel(port: int) -> str | None:
  """Start cloudflared tunnel, return public URL."""
  if not Path("/usr/local/bin/cloudflared").exists():
    console.print("[yellow]cloudflared not installed, skipping tunnel[/yellow]")
    return None

  console.print(f"[blue]Starting tunnel for port {port}...[/blue]")
  proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
  )

  # Wait for URL
  for _ in range(30):
    if proc.stdout is None:
      break
    line = proc.stdout.readline()
    if "trycloudflare.com" in line:
      import re
      match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
      if match:
        url = match.group(0)
        console.print(f"[green]Public URL:[/green] {url}")
        return url
    time.sleep(0.5)

  console.print("[yellow]Tunnel started but URL not captured[/yellow]")
  return None


def start_nviz_with_tunnel(dev: bool = False):
  """Start nviz and cloudflared tunnel."""
  nviz_path = NMOE_ROOT / "nviz"
  if not nviz_path.exists():
    console.print("[yellow]nviz not found[/yellow]")
    return
  if shutil.which("bun") is None:
    console.print("[red]bun not found. Run: bash scripts/bootstrap.sh[/red]")
    return

  env = os.environ.copy()
  env["NVIZ_METRICS_DIR"] = str(DATA_DIR / "metrics")
  env["PORT"] = str(NVIZ_PORT)

  if not (nviz_path / "node_modules").exists():
    console.print("[blue]Installing nviz deps (bun install)...[/blue]")
    run(["bun", "install"], cwd=nviz_path, env=env)

  if dev:
    # Dev mode: hot reload
    console.print("[blue]Starting nviz (dev mode)...[/blue]")
    run_background(["bun", "run", "dev"], cwd=nviz_path, env=env)
  else:
    # Production mode: build if needed, then start
    next_dir = nviz_path / ".next"
    if not next_dir.exists():
      console.print("[blue]Building nviz...[/blue]")
      run(["bun", "run", "build"], cwd=nviz_path, env=env)
    run_background(["bun", "run", "start"], cwd=nviz_path, env=env)

  time.sleep(2)
  start_tunnel(NVIZ_PORT)


def open_nmon():
  """Open nmon TUI (replaces current process)."""
  nmon_path = NMOE_ROOT / "tools" / "nmon" / "nmon"
  if not nmon_path.exists():
    console.print("[yellow]Building nmon...[/yellow]")
    run(["go", "build", "-o", "nmon", "./cmd/nmon"], cwd=NMOE_ROOT / "tools" / "nmon")
  console.print("[blue]Opening nmon...[/blue]")
  args = [str(nmon_path), f"--leaderboard={LEADERBOARD_PATH}"]
  os.execv(str(nmon_path), args)


def open_jupyter():
  """Start JupyterLab with cloudflared tunnel."""
  console.print(f"[blue]Starting JupyterLab on port {JUPYTER_PORT}...[/blue]")

  # Start JupyterLab in background
  run_background([
    "python", "-m", "jupyter", "lab",
    "--ip=0.0.0.0",
    f"--port={JUPYTER_PORT}",
    "--no-browser",
    "--allow-root",
    "--ServerApp.token=''",
    "--ServerApp.password=''",
    f"--notebook-dir={str(NMOE_ROOT)}",
  ])
  time.sleep(3)

  # Start tunnel
  start_tunnel(JUPYTER_PORT)

  console.print("[blue]JupyterLab running. Press Ctrl+C to stop.[/blue]")
  console.print("[dim]from nmoe.research import lab[/dim]")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass


# -----------------------------------------------------------------------------
# Commands: Start (new run by default, --attach for existing)
# -----------------------------------------------------------------------------

SPEEDRUN_CONFIGS = {
  "dense": "configs/speedrun/dense.toml",
  "moe": "configs/speedrun/moe.toml",         # 64 experts
  "super": "configs/speedrun/super.toml",     # 256 experts
  "ultra": "configs/speedrun/ultra.toml",     # 4096 experts
}

LEADERBOARD_PATH = NMOE_ROOT / "LEADERBOARD.json"


def _speedrun_leaderboard():
  """Print speedrun leaderboard from LEADERBOARD.json."""
  import json

  if not LEADERBOARD_PATH.exists():
    console.print("[yellow]No speedrun results yet. Run: n speedrun dense[/yellow]")
    return

  try:
    data = json.loads(LEADERBOARD_PATH.read_text())
    runs = data.get("runs", [])
  except Exception as e:
    console.print(f"[red]Error reading leaderboard: {e}[/red]")
    return

  if not runs:
    console.print("[yellow]No speedrun results yet. Run: n speedrun dense[/yellow]")
    return

  # For safety, sort here too (train.py also writes sorted).
  runs = list(runs)
  runs.sort(key=lambda r: (float(r.get("wall_time_s") or 1e18), int(r.get("steps") or 1e18)))

  console.print(f"\n{'═' * 80}")
  console.print("  nmoe Speedrun Leaderboard")
  console.print(f"{'═' * 80}")
  console.print(f"  {'Config':<12} {'HW':<6} {'Dtype':<8} {'Time':>8} {'Steps':>6} {'Loss':>8} {'CORE':>8} {'Tokens':>8} {'Date':<12}")
  console.print(f"  {'─' * 74}")

  for r in runs[:20]:
    config = r.get('config', '?')
    hw = r.get('hardware', '?')
    dtype = r.get('dtype', '?')
    loss = r.get('final_loss', 0)
    core = r.get('core_score', 0)
    tokens = r.get('tokens', 0)
    steps = r.get('steps', 0)
    wall_time = r.get('wall_time_s', 0)
    date = r.get('date', '?')[:10]

    tokens_str = f"{tokens/1e9:.1f}B" if tokens else "?"
    time_str = f"{wall_time/60:.1f}m" if wall_time else "?"
    steps_str = f"{int(steps)}" if steps else "?"
    loss_str = f"{loss:.4f}" if loss else "?"
    core_str = f"{core:.3f}" if core else "?"

    console.print(f"  {config:<12} {hw:<6} {dtype:<8} {time_str:>8} {steps_str:>6} {loss_str:>8} {core_str:>8} {tokens_str:>8} {date:<12}")

  console.print(f"{'═' * 80}\n")


@app.command()
def speedrun(
  config: str = typer.Argument("super", help="Config: dense, moe, super, ultra"),
  bf16: bool = typer.Option(False, "--bf16", help="Use bf16 instead of nvfp4"),
  fp8: bool = typer.Option(False, "--fp8", help="Use fp8 instead of nvfp4"),
  activation: str = typer.Option("", "--activation", "-a", help="Activation: swiglu, relu_squared, squared_reglu"),
  steps: int = typer.Option(0, "--steps", "-s", help="Override steps (0=use config default)"),
  attach: bool = typer.Option(False, "--attach", help="Attach to existing run"),
  leaderboard: bool = typer.Option(False, "--leaderboard", "-l", help="Show leaderboard"),
  no_nmon: bool = typer.Option(False, "--no-nmon", help="Don't open nmon TUI (run in foreground; headless/CI friendly)"),
):
  """Run speedrun benchmark. Opens nmon for monitoring by default.

  Examples:
    n speedrun dense          # Dense baseline (nvfp4)
    n speedrun moe            # MoE-64 (nvfp4)
    n speedrun moe --bf16     # MoE-64 (bf16)
    n speedrun super          # MoE-256 (nvfp4)
    n speedrun ultra          # MoE-4096 (nvfp4)
    n speedrun --leaderboard  # Show results
    n speedrun dense --activation=relu_squared  # Ablation
    n speedrun super --no-nmon  # Foreground run (no TUI)
  """
  if leaderboard:
    _speedrun_leaderboard()
    return

  if attach:
    open_nmon()
    return

  # Validate config
  if config not in SPEEDRUN_CONFIGS:
    console.print(f"[red]Unknown config: {config}[/red]")
    console.print(f"[yellow]Available: {', '.join(SPEEDRUN_CONFIGS.keys())}[/yellow]")
    raise typer.Exit(1)

  config_path = SPEEDRUN_CONFIGS[config]

  # Determine dtype
  if bf16 and fp8:
    console.print("[red]Cannot use both --bf16 and --fp8[/red]")
    raise typer.Exit(1)
  dtype_explicit = "bf16" if bf16 else ("fp8" if fp8 else "")
  dtype = dtype_explicit or "fp8"

  # Auto-default dtype for speedruns based on platform.
  # Contract: speedruns default to fp8 on B200/SM100; H100/SM90 is BF16-only bring-up.
  if not dtype_explicit:
    arch = os.environ.get("NMOE_CUDA_ARCH", "").strip().lower()
    if arch in ("90", "sm90", "9.0"):
      dtype = "bf16"
    else:
      try:
        import torch
        if torch.cuda.is_available():
          cap = tuple(torch.cuda.get_device_capability())
          if cap == (9, 0):
            dtype = "bf16"
      except Exception:
        pass

  # Ensure data exists
  speedrun_dir = ensure_speedrun_data()
  ensure_eval_bundle()

  # Build command with dynamic paths
  cmd = [
    "torchrun", "--nproc_per_node", NPROC,
    "-m", "nmoe.train",
    config_path,
    f"--dtype={dtype}",
    f"--data_root={DATA_DIR}",
    f"--data_path={speedrun_dir / 'train'}",
    f"--validation_data_path={speedrun_dir / 'val'}",
    f"--experiments_db={DATA_DIR / 'experiments.db'}",
    "--eval_enabled=true",
    "--eval_tasks=core",
  ]
  if steps > 0:
    cmd.append(f"--steps={steps}")
  if activation:
    if activation not in ("swiglu", "relu_squared", "squared_reglu"):
      console.print(f"[red]Unknown activation: {activation}[/red]")
      console.print("[yellow]Available: swiglu, relu_squared, squared_reglu[/yellow]")
      raise typer.Exit(1)
    cmd.append(f"--activation={activation}")

  console.print(f"\n[bold]Speedrun: {config} ({dtype})[/bold]")
  console.print(f"[dim]Config: {config_path}[/dim]\n")

  if no_nmon:
    # Headless/CI friendly: keep logs/exit code in the invoking shell.
    run(cmd)
    return

  # Default: start training in background and open nmon.
  run_background(cmd)
  time.sleep(2)
  open_nmon()


@app.command()
def research():
  """Open JupyterLab for research. Use: from nmoe.research import lab"""
  open_jupyter()


@app.command()
def train(
  config: str = typer.Argument("configs/moonlight.toml", help="Training config"),
  attach: bool = typer.Option(False, "--attach", "-a", help="Attach to existing run"),
):
  """Production training. Starts nviz dashboard."""
  start_nviz_with_tunnel()
  if not attach:
    data_path = ensure_data("fineweb_train", "10B")
    start_training(config, data_path_override=data_path, background=False)
  else:
    console.print("[blue]Attached to nviz. Press Ctrl+C to stop.[/blue]")
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
      pass


# -----------------------------------------------------------------------------
# Commands: List runs
# -----------------------------------------------------------------------------

@app.command(name="list")
def list_runs(limit: int = typer.Option(20, "--limit", "-n", help="Max runs to show")):
  """List available runs."""
  from datetime import datetime
  experiments_db = DATA_DIR / "experiments.db"
  metrics_dir = DATA_DIR / "metrics"

  run_info = []

  # Prefer experiments.db as source of truth
  if experiments_db.exists():
    try:
      import sqlite3
      conn = sqlite3.connect(str(experiments_db))
      cursor = conn.execute("""
        SELECT run, status, started_at, ended_at
        FROM runs
        ORDER BY started_at DESC
        LIMIT ?
      """, (limit * 2,))  # Fetch extra to account for filtering
      for row in cursor:
        run_id, status, started_at, ended_at = row
        try:
          ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        except Exception:
          ts = 0
        run_info.append((run_id, ts, status))
      conn.close()
    except Exception as e:
      console.print(f"[yellow]experiments.db error: {e}, falling back to metrics dir[/yellow]")
      run_info = []

  # Fallback: scan /data/metrics
  if not run_info and metrics_dir.exists():
    run_dirs = [d for d in metrics_dir.iterdir() if d.is_dir()]
    for run_dir in run_dirs:
      run_id = run_dir.name
      last_step = 0
      last_ts = 0.0

      # Authoritative live store: step_XXXXXXXX.parquet (rank 0 only).
      parquet = sorted(run_dir.glob("step_*.parquet"))
      if parquet:
        newest = max(parquet, key=lambda p: p.stat().st_mtime)
        last_ts = newest.stat().st_mtime
        try:
          # step_00001234.parquet
          stem = newest.name.split(".", 1)[0]
          last_step = int(stem.split("_", 1)[1])
        except Exception:
          last_step = 0
      else:
        # Backward-compat (older runs).
        db_path = run_dir / "rank_0.duckdb"
        if db_path.exists():
          last_ts = db_path.stat().st_mtime
          last_step = 0

      status = f"step {last_step:,}" if last_step else "no data"
      run_info.append((run_id, last_ts, status))

    run_info.sort(key=lambda x: x[1], reverse=True)

  if not run_info:
    console.print("[yellow]No runs found[/yellow]")
    return

  console.print("[bold]Available runs:[/bold]\n")
  for i, (run_id, ts, status) in enumerate(run_info[:limit]):
    time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"
    marker = "[green]← latest[/green]" if i == 0 else ""
    console.print(f"  {run_id:<30} {time_str}  {status:<12} {marker}")

  if len(run_info) > limit:
    console.print(f"\n  ... and {len(run_info) - limit} more (use --limit to show more)")


# -----------------------------------------------------------------------------
# Commands: Monitor (attach to latest run, --run <id> to specify)
# -----------------------------------------------------------------------------

@app.command(name="mon")
def mon(run_id: str = typer.Option(None, "--run", "-r", help="Run ID to monitor")):
  """TUI monitor. Attaches to latest run."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  open_nmon()


@app.command(name="viz")
def viz(
  run_id: str = typer.Option(None, "--run", "-r", help="Run ID to monitor"),
  dev: bool = typer.Option(False, "--dev", "-d", help="Dev mode with hot reload"),
):
  """Web dashboard. Attaches to latest run."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  start_nviz_with_tunnel(dev=dev)
  console.print("[blue]viz running. Press Ctrl+C to stop.[/blue]")
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass


@app.command(name="nb")
def nb(run_id: str = typer.Option(None, "--run", "-r", help="Run ID context")):
  """Jupyter notebook. Attaches to latest run context."""
  if run_id:
    os.environ["NMOE_RUN"] = run_id
  open_jupyter()


def main():
  app()


if __name__ == "__main__":
  main()
