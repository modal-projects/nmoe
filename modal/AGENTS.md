# modal/

Modal-first execution for nmoe training, data prep, and debug.

## CLI

All Modal commands require `uv run` prefix (no active venv on host):

    uv run modal run [--detach] modal/train.py ...
    uv run modal run [--detach] modal/prep.py ...
    uv run modal shell modal/debug.py

## Image (`image.py`)

Two-stage image: `nmoe_build_image` (layers 1-5, all build steps) → `nmoe_image` (adds source mount).

Source mount (`add_local_dir` with `copy=False`) **must be the final operation** — no `pip_install`, `run_commands`, etc. can follow it. To extend the image with build steps (e.g. `debug.py` adding pytest), start from `nmoe_build_image` and add your own mount at the end.

Deps come from `pyproject.toml` (single source of truth). Build deps (layer 3, pre-kernel) vs runtime deps (layer 5, post-kernel) are split to avoid invalidating the ~45min kernel build cache.

## Entrypoints

- **train.py** — `--config` and `--gpus` are consumed; all other `--key=value` args forwarded to `nmoe.train`. Uses `.spawn()` for durable execution.
- **prep.py** — `--fan-out N` fans out across N containers for parallel downloads; all other args forwarded to `nmoe.data.cli prep`.
- **debug.py** — `modal shell` for interactive access with GPUs.

## Volume

`nmoe-data` mounted at `/data`. Writes are background-committed; final commit on container exit. Use `data_vol.reload()` when reading cross-container writes.
