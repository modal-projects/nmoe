"""Training entrypoint for Modal.

Usage:
  modal run modal/train.py --config configs/moonlet.toml
  modal run modal/train.py --config configs/moonlet.toml --data_path=/data/speedrun/train
  modal run modal/train.py --config configs/moonlight.toml --gpus 8
"""
import subprocess

import modal

from image import nmoe_image, data_vol, checkpoint_vol

app = modal.App("nmoe-train")


# Modal overrides:
# - experiments.db → /tmp: SQLite journal fsync deadlocks on 9p FUSE mounts
# - fsync=false: checkpoint fsync is redundant (volume background-commits handle durability)
_MODAL_OVERRIDES = ["--experiments_db=/tmp/experiments.db", "--fsync=false"]

# Checkpoint dir per volume version. v1 is default — v2 has severe write
# amplification with torch.save's pickle protocol (~10 MB/s vs ~350 MB/s).
_CKPT_DIR = {"v1": "/checkpoints", "v2": "/data/checkpoints"}

# /data (v2): training data, metrics — good for high file counts, concurrent reads
# /checkpoints (v1): checkpoint writes — v2 has severe write amplification with pickle
_VOLUMES = {"/data": data_vol, "/checkpoints": checkpoint_vol}


@app.function(image=nmoe_image, gpu="B200", volumes=_VOLUMES, timeout=86400)
def train_1gpu(config: str, overrides: list[str], ckpt_dir: str = "/checkpoints"):
  cmd = ["python", "-m", "nmoe.train", config] + _MODAL_OVERRIDES + [f"--checkpoint_dir={ckpt_dir}"] + overrides
  subprocess.run(cmd, check=True)


@app.function(image=nmoe_image, gpu="B200:8", volumes=_VOLUMES, timeout=86400)
def train_8gpu(config: str, overrides: list[str], ckpt_dir: str = "/checkpoints"):
  cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "nmoe.train", config] + _MODAL_OVERRIDES + [f"--checkpoint_dir={ckpt_dir}"] + overrides
  subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(*args: str):
  args_list = list(args)

  # Extract --config, --gpus, --ckpt-vol; everything else forwarded to nmoe.train
  config = "configs/moonlet.toml"
  gpus = 1
  ckpt_vol = "v1"
  overrides = []

  i = 0
  while i < len(args_list):
    if args_list[i] == "--config" and i + 1 < len(args_list):
      config = args_list[i + 1]
      i += 2
    elif args_list[i] == "--gpus" and i + 1 < len(args_list):
      gpus = int(args_list[i + 1])
      i += 2
    elif args_list[i] == "--ckpt-vol" and i + 1 < len(args_list):
      ckpt_vol = args_list[i + 1]
      i += 2
    else:
      overrides.append(args_list[i])
      i += 1

  # Normalize --key value → --key=value (nmoe.train expects --key=value format)
  normalized = []
  i = 0
  while i < len(overrides):
    arg = overrides[i]
    if arg.startswith("--") and "=" not in arg and i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
      normalized.append(f"{arg}={overrides[i + 1]}")
      i += 2
    else:
      normalized.append(arg)
      i += 1

  ckpt_dir = _CKPT_DIR[ckpt_vol]
  fn = train_1gpu if gpus == 1 else train_8gpu
  handle = fn.spawn(config, normalized, ckpt_dir=ckpt_dir)
  print(f"Spawned training: {handle.object_id}  (checkpoints → {ckpt_vol} volume at {ckpt_dir})")
