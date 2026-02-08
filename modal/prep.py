"""Data preparation entrypoint for Modal.

Usage:
  # Single container:
  modal run modal/prep.py --source hub_parquet --dataset karpathy/fineweb-edu-100b-shuffle \
    --output /data/fineweb_edu --name fineweb_edu

  # Fan out across 8 containers for parallel downloads:
  modal run modal/prep.py --fan-out 8 --source hub_parquet \
    --dataset karpathy/fineweb-edu-100b-shuffle \
    --output /data/fineweb_edu --name fineweb_edu

TODO: Rethink the dataset processing lifecycle for Modal.
  - Mount a persistent volume at /root/.cache/huggingface so HF downloads survive
    across runs (avoids re-downloading ~100B-token datasets from scratch).
  - Separate the HF cache volume from the output data volume. Output can be
    indexed by run/config; the HF cache is shared and append-only.
  - With a persistent HF cache, re-running prep becomes cheap (cache hit on
    download, only tokenization + sharding work repeated).
"""
import json
import subprocess
from pathlib import Path

import modal

from image import nmoe_image, data_vol

app = modal.App("nmoe-prep")


@app.function(
  image=nmoe_image,
  volumes={"/data": data_vol},
  secrets=[modal.Secret.from_name("huggingface-secret")],
  timeout=86400,
)
def prep(args: list[str]):
  subprocess.run(
    ["python", "-m", "nmoe.data.cli", "prep"] + args,
    check=True,
  )


@app.function(
  image=nmoe_image,
  volumes={"/data": data_vol},
  secrets=[modal.Secret.from_name("huggingface-secret")],
  timeout=86400,
)
def prep_worker(worker_index: int, num_workers: int, args: list[str]):
  """Run prep for a single worker shard of the input files."""
  modified_args = list(args)

  # Redirect output to worker subdirectory
  for i, arg in enumerate(modified_args):
    if arg == "--output" and i + 1 < len(modified_args):
      modified_args[i + 1] = f"{modified_args[i + 1]}/worker_{worker_index:03d}"
      break

  # Add worker sharding args (appended last — overrides any existing values)
  modified_args.extend([
    "--num-workers", str(num_workers),
    "--worker-index", str(worker_index),
  ])

  subprocess.run(
    ["python", "-m", "nmoe.data.cli", "prep"] + modified_args,
    check=True,
  )


@app.function(
  image=nmoe_image,
  volumes={"/data": data_vol},
  timeout=3600,
)
def merge_manifests(output_dir: str, num_workers: int):
  """Merge per-worker manifests into a single combined manifest."""
  data_vol.reload()

  root = Path(output_dir)
  all_shards = []
  metadata = None
  total_tokens = 0
  total_docs = 0

  for i in range(num_workers):
    worker_dir = root / f"worker_{i:03d}"
    manifest_path = worker_dir / "manifest.json"
    if not manifest_path.exists():
      raise FileNotFoundError(f"Worker {i} manifest not found: {manifest_path}")

    with open(manifest_path) as f:
      m = json.load(f)

    if metadata is None:
      metadata = {
        k: v for k, v in m.items()
        if k not in ("shards", "total_tokens", "total_documents", "num_shards", "source_info")
      }

    total_tokens += m["total_tokens"]
    total_docs += m["total_documents"]

    for shard in m["shards"]:
      shard["path"] = f"worker_{i:03d}/{shard['path']}"
      shard["index_path"] = f"worker_{i:03d}/{shard['index_path']}"
      all_shards.append(shard)

  combined = {
    **metadata,
    "total_tokens": total_tokens,
    "total_documents": total_docs,
    "num_shards": len(all_shards),
    "shards": all_shards,
    "source_info": {"fan_out": num_workers},
  }

  manifest_path = root / "manifest.json"
  tmp_path = manifest_path.with_name("manifest.json.tmp")
  with open(tmp_path, "w") as f:
    json.dump(combined, f, indent=2)
  tmp_path.replace(manifest_path)

  data_vol.commit()
  print(f"Merged {num_workers} workers: {total_docs:,} docs, {total_tokens:,} tokens, {len(all_shards)} shards")


@app.local_entrypoint()
def main(*args: str):
  args_list = list(args)

  # Extract --fan-out N before forwarding to prep CLI
  fan_out = 1
  if "--fan-out" in args_list:
    idx = args_list.index("--fan-out")
    fan_out = int(args_list[idx + 1])
    args_list = args_list[:idx] + args_list[idx + 2:]

  if fan_out <= 1:
    prep.remote(args_list)
  else:
    # Fan out across multiple containers for parallel downloads
    list(prep_worker.starmap(
      [(i, fan_out, args_list) for i in range(fan_out)]
    ))
    # Merge per-worker manifests
    output_dir = None
    for i, arg in enumerate(args_list):
      if arg == "--output" and i + 1 < len(args_list):
        output_dir = args_list[i + 1]
        break
    if output_dir:
      merge_manifests.remote(output_dir, fan_out)
