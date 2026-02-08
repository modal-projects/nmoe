"""Training entrypoint for Modal.

Usage:
  modal run modal/train.py --config configs/moonlet.toml
  modal run modal/train.py --config configs/moonlet.toml --data_path=/data/speedrun/train
  modal run modal/train.py --config configs/moonlight.toml --gpus 8
"""
import subprocess

import modal

from image import nmoe_image, data_vol

app = modal.App("nmoe-train")


# SQLite + 9p (Modal's volume filesystem) deadlocks on fsync during journal
# commits, blocking the entire mount. Keep experiments.db on local storage.
_MODAL_OVERRIDES = ["--experiments_db=/tmp/experiments.db"]


@app.function(image=nmoe_image, gpu="B200", volumes={"/data": data_vol}, timeout=86400)
def train_1gpu(config: str, overrides: list[str]):
  subprocess.run(["python", "-m", "nmoe.train", config] + _MODAL_OVERRIDES + overrides, check=True)


@app.function(image=nmoe_image, gpu="B200:8", volumes={"/data": data_vol}, timeout=86400)
def train_8gpu(config: str, overrides: list[str]):
  subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=8", "-m", "nmoe.train", config] + _MODAL_OVERRIDES + overrides,
    check=True,
  )


@app.local_entrypoint()
def main(*args: str):
  args_list = list(args)

  # Extract --config and --gpus; everything else is forwarded as nmoe.train overrides
  config = "configs/moonlet.toml"
  gpus = 1
  overrides = []

  i = 0
  while i < len(args_list):
    if args_list[i] == "--config" and i + 1 < len(args_list):
      config = args_list[i + 1]
      i += 2
    elif args_list[i] == "--gpus" and i + 1 < len(args_list):
      gpus = int(args_list[i + 1])
      i += 2
    else:
      overrides.append(args_list[i])
      i += 1

  fn = train_1gpu if gpus == 1 else train_8gpu
  handle = fn.spawn(config, overrides)
  print(f"Spawned training: {handle.object_id}")
