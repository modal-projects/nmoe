r"""
nmoe.rl.selfplay: Math‑V2-style self-play data generation (generator→verifier→meta).

This is a capability path (not throughput). It runs:
  ProblemPool → generator → verifier → meta-verifier → accept/filter → emit JSONLs (+ replay bundles).

Usage:
  python -m nmoe.rl selfplay <config.toml> \
    --problem_source=prm800k \
    --output_dir=./mathv2_output \
    --max_problems=1000 \
    --replay_sample_rate=0.1

Outputs:
  <output_dir>/
    proof_verifier_train.jsonl
    proof_meta_verifier_train.jsonl
    counts.json
    replay_bundles/  (optional; sampled accepted trajectories)

Notes:
  - Tokenization is Harmony-only (tokenizer=o200k_harmony).
  - Sampling uses nucleus/top_p but scoring uses full-softmax logprobs (Option C).
"""

from __future__ import annotations

import os
import random
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

import torch

from nmoe.checkpoint import Checkpointer
from nmoe.config import Config
from nmoe.model import Transformer
from nmoe.rl.mathv2_selfplay import MathV2SelfPlayConfig, MathV2SelfPlayEmitConfig, MathV2SelfPlayRunner
from nmoe.rl.rollout_engine import LocalRolloutEngine, require_harmony_tokenizer
from nmoe.rl.tasks.problem_pool import ProblemPool


def _parse_flag(name: str, default: str | None = None) -> str | None:
    pref = f"--{name}="
    for a in sys.argv[2:]:
        if a.startswith(pref):
            return a[len(pref) :]
    return default


def _parse_int(name: str, default: int) -> int:
    v = _parse_flag(name)
    if v is None:
        return int(default)
    return int(v)


def _parse_float(name: str, default: float) -> float:
    v = _parse_flag(name)
    if v is None:
        return float(default)
    return float(v)


def _replay_sample_every_from_rate(rate: float) -> int:
    """Convert a rate in (0,1] to a deterministic sample_every integer."""
    r = float(rate)
    if r <= 0.0:
        return 0
    if r >= 1.0:
        return 1
    return max(1, int(round(1.0 / r)))


def _parse_hf_source(src: str) -> tuple[str, str]:
    # hf:<dataset>:<split> or hf:<dataset>:<subset>:<split> (subset ignored here)
    parts = src.split(":")
    if len(parts) < 3 or parts[0] != "hf":
        raise ValueError(f"unsupported problem_source (expected hf:*): {src}")
    if len(parts) == 3:
        _, dataset, split = parts
        return dataset, split
    if len(parts) == 4:
        _, dataset, _subset, split = parts
        return dataset, split
    raise ValueError(f"bad hf source spec: {src}")


def _build_problem_pool(problem_source: str, *, split: str, max_examples: int, seed: int) -> ProblemPool:
    src = str(problem_source)
    if src == "prm800k":
        return ProblemPool.from_hf(
            "trl-lib/prm800k",
            split=str(split),
            max_examples=int(max_examples),
            seed=int(seed),
            fields=("prompt", "problem", "question", "text"),
            streaming=False,
        )
    if src == "math_shepherd":
        return ProblemPool.from_hf(
            "trl-lib/math_shepherd",
            split=str(split),
            max_examples=int(max_examples),
            seed=int(seed),
            fields=("prompt", "problem", "question", "text"),
            streaming=False,
        )
    if src.startswith("hf:"):
        dataset, split_s = _parse_hf_source(src)
        return ProblemPool.from_hf(
            dataset,
            split=str(split_s),
            max_examples=int(max_examples),
            seed=int(seed),
            fields=("prompt", "problem", "question", "text"),
            streaming=False,
        )
    if src.startswith("deepseek_math_v2_inputs:"):
        path = src.split(":", 1)[1]
        return ProblemPool.from_deepseek_math_v2_inputs(path, max_examples=int(max_examples), seed=int(seed))
    raise ValueError(
        f"unknown --problem_source={problem_source!r} "
        "(expected prm800k, math_shepherd, hf:<dataset>:<split>, or deepseek_math_v2_inputs:<path>)"
    )


def _load_model_from_latest_checkpoint(model: torch.nn.Module, *, cfg: Config) -> int:
    ckpt_dir = getattr(cfg, "checkpoint_dir", None)
    if not isinstance(ckpt_dir, str) or not ckpt_dir:
        raise ValueError("config must set checkpoint_dir to load weights for self-play")

    checkpointer = Checkpointer(base=ckpt_dir, keep_last=0, async_io=False)
    step, dp_path = checkpointer.find_latest()
    if dp_path is None:
        raise RuntimeError(f"no checkpoint found in checkpoint_dir={ckpt_dir!r}")

    it_dir = Path(dp_path).parent
    rd_path = it_dir / "rd.pt"
    if not rd_path.exists():
        raise RuntimeError(f"missing dense checkpoint shard: {rd_path}")

    map_location = "cpu"
    if torch.cuda.is_available():
        map_location = f"cuda:{torch.cuda.current_device()}"

    rd = torch.load(str(rd_path), map_location=map_location, weights_only=False)
    # For self-play, we validate by *structure*, not training-time hyperparams.
    # Fingerprints can drift when config defaults change without affecting model weights.
    ri = rd.get("run_info", {}) if isinstance(rd.get("run_info", {}), dict) else {}
    if "H" in ri and ri["H"] is not None and int(getattr(cfg, "dim", 0)) != int(ri["H"]):
        raise RuntimeError(f"checkpoint/config mismatch: dim={getattr(cfg, 'dim', None)} saved_H={ri['H']}")
    if "L" in ri and ri["L"] is not None and int(getattr(cfg, "n_layers", 0)) != int(ri["L"]):
        raise RuntimeError(f"checkpoint/config mismatch: n_layers={getattr(cfg, 'n_layers', None)} saved_L={ri['L']}")
    if "E" in ri and ri["E"] is not None and int(getattr(cfg, "n_routed_experts", 0)) != int(ri["E"]):
        raise RuntimeError(
            f"checkpoint/config mismatch: n_routed_experts={getattr(cfg, 'n_routed_experts', None)} saved_E={ri['E']}"
        )
    if "K" in ri and ri["K"] is not None and int(getattr(cfg, "n_activated_experts", 0)) != int(ri["K"]):
        raise RuntimeError(
            f"checkpoint/config mismatch: n_activated_experts={getattr(cfg, 'n_activated_experts', None)} saved_K={ri['K']}"
        )
    emb = (rd.get("model_dense") or {}).get("embedding.weight")
    if hasattr(emb, "shape") and int(getattr(cfg, "vocab_size", 0)) != int(emb.shape[0]):
        raise RuntimeError(f"checkpoint/config mismatch: vocab_size={getattr(cfg, 'vocab_size', None)} saved={emb.shape[0]}")

    dp = torch.load(str(dp_path), map_location=map_location, weights_only=False)

    expected = set(model.state_dict().keys())
    dense_keys = set((rd.get("model_dense") or {}).keys())
    expert_keys = set((dp.get("model_expert") or {}).keys())
    provided = dense_keys | expert_keys
    unexpected = sorted(provided - expected)
    missing = sorted(expected - provided)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint shards (n={len(unexpected)}): {unexpected[:16]}")
    if missing:
        raise RuntimeError(f"missing keys in checkpoint shards (n={len(missing)}): {missing[:16]}")

    model.load_state_dict(rd["model_dense"], strict=False)
    model.load_state_dict(dp["model_expert"], strict=False)
    return int(step)


def _apply_config_overrides(cfg_dict: dict, argv: list[str]) -> dict:
    allowed = {f.name for f in fields(Config)}
    out = dict(cfg_dict)
    for arg in argv:
        if not arg.startswith("--") or "=" not in arg:
            continue
        key, val = arg[2:].split("=", 1)
        if key not in allowed:
            continue
        if val.lower() in ("true", "false"):
            out[key] = val.lower() == "true"
        elif val.lstrip("-").isdigit():
            out[key] = int(val)
        else:
            try:
                out[key] = float(val)
            except ValueError:
                out[key] = val
    return out


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    if len(sys.argv) < 2:
        print("Usage: python -m nmoe.rl selfplay <config.toml> [--key=value ...]", file=sys.stderr)
        raise SystemExit(1)

    with open(sys.argv[1], "rb") as f:
        base_cfg = tomllib.load(f)
    cfg_dict = _apply_config_overrides(base_cfg, sys.argv[2:])
    cfg = Config(**cfg_dict)

    if not torch.cuda.is_available():
        raise RuntimeError("selfplay requires CUDA (run in the container/GPU environment).")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("selfplay does not support torchrun/distributed yet (keep one clear path).")

    problem_source = str(_parse_flag("problem_source", "prm800k"))
    problem_split = str(_parse_flag("problem_split", "train"))
    output_dir_s = _parse_flag("output_dir", None)
    if output_dir_s is None:
        raise ValueError("--output_dir is required")
    output_dir = Path(output_dir_s)
    max_problems = _parse_int("max_problems", 1000)
    seed = _parse_int("seed", int(getattr(cfg, "seed", 0)))

    replay_rate = _parse_float("replay_sample_rate", 0.0)
    replay_every = _parse_int("replay_sample_every", _replay_sample_every_from_rate(replay_rate))

    max_new_tokens_proof = _parse_int("max_new_tokens_proof", 1024)
    max_new_tokens_verifier = _parse_int("max_new_tokens_verifier", 512)
    max_new_tokens_meta = _parse_int("max_new_tokens_meta", 512)

    temperature = float(getattr(cfg, "rl_temperature", 1.0))
    top_p = float(getattr(cfg, "rl_top_p", 1.0))

    import tiktoken

    enc = tiktoken.get_encoding(cfg.tokenizer)
    require_harmony_tokenizer(enc)

    model = Transformer(cfg).cuda().eval()
    step = _load_model_from_latest_checkpoint(model, cfg=cfg)

    engine = LocalRolloutEngine(model=model, enc=enc, device="cuda")
    runner = MathV2SelfPlayRunner(
        generator_engine=engine,
        verifier_engine=engine,
        meta_verifier_engine=engine,
        enc=enc,
        config=MathV2SelfPlayConfig(
            max_new_tokens_proof=max_new_tokens_proof,
            max_new_tokens_verifier=max_new_tokens_verifier,
            max_new_tokens_meta=max_new_tokens_meta,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=int(cfg.eos_token_id),
        ),
    )

    pool = _build_problem_pool(problem_source, split=problem_split, max_examples=max_problems, seed=seed)
    if len(pool) == 0:
        raise RuntimeError("loaded 0 problems (check dataset access and schema).")

    problems = list(pool.problems)
    rng = random.Random(int(seed))
    rng.shuffle(problems)
    problems = problems[: int(max_problems)]

    out: list = []
    for i, p in enumerate(problems, start=1):
        out.append(runner.run_one(p))
        if i % 10 == 0:
            ok = sum(1 for s in out if s.accepted)
            print(f"[selfplay] loaded_ckpt_step={step} problems={i}/{len(problems)} accepted={ok}")

    replay_dir = output_dir / "replay_bundles" if replay_every != 0 else None
    run_id = os.getenv("NMOE_RUN", "") or "mathv2_selfplay"
    artifacts = runner.emit(
        out,
        cfg=MathV2SelfPlayEmitConfig(
            out_dir=output_dir,
            run_id=run_id,
            replay_dir=replay_dir,
            replay_sample_every=int(replay_every),
            seed=int(seed),
            rank=0,
        ),
    )

    ok = sum(1 for s in out if s.accepted)
    print(
        f"[selfplay] done problems={len(out)} accepted={ok} "
        f"verifier_jsonl={artifacts['proof_verifier_train']} meta_jsonl={artifacts['proof_meta_verifier_train']}"
    )


if __name__ == "__main__":
    main()
