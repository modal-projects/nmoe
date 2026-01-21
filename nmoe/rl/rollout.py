from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Trajectory:
  prompt_len: int
  completion_len: int
  tokens: List[int]          # prompt + completion (includes EOS if produced)
  completion_text: str       # decoded completion (excludes prompt)


def _sample_top_p(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
  if temperature <= 0.0:
    raise ValueError(f"temperature must be > 0 (got {temperature})")
  if not (0.0 < top_p <= 1.0):
    raise ValueError(f"top_p must be in (0,1] (got {top_p})")

  logits = logits / float(temperature)
  if top_p >= 1.0:
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())

  probs = torch.softmax(logits, dim=-1)
  sorted_probs, sorted_indices = torch.sort(probs, descending=True)
  cumsum = torch.cumsum(sorted_probs, dim=-1)
  mask = cumsum - sorted_probs > top_p
  sorted_probs = sorted_probs.masked_fill(mask, 0.0)
  sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
  idx = torch.multinomial(sorted_probs, 1)
  return int(sorted_indices[idx].item())


def _model_supports_kv_cache(model) -> bool:
  """Check if model.forward accepts past_key_values and use_cache kwargs."""
  import inspect
  try:
    sig = inspect.signature(model.forward)
    params = sig.parameters
    return 'past_key_values' in params and 'use_cache' in params
  except (ValueError, TypeError):
    return False


def generate_one(
  model,
  *,
  enc,
  prompt_ids: Sequence[int],
  max_new_tokens: int,
  eos_token_id: int,
  temperature: float = 1.0,
  top_p: float = 1.0,
  use_cache: bool | None = None,
) -> Trajectory:
  """Generate a completion trajectory with optional KV caching.

  If the model supports KV cache (past_key_values, use_cache kwargs), uses
  cached generation for efficiency. Otherwise falls back to full-sequence
  forward passes.

  NOTE on Behavior Policy Mismatch:
    When top_p < 1.0, this uses nucleus sampling which changes the effective
    behavior distribution. However, completion_nll_mean computes logprobs
    under the full softmax model, not the nucleus-truncated distribution.

    Strictly, PPO should use behavior distribution logprobs. Most RLHF
    implementations accept this bias because:
    1. Nucleus sampling is close to full sampling for most tokens
    2. The bias is consistent across rollout and training
    3. Computing true nucleus logprobs requires knowing the truncation set

    For maximum rigor, set top_p=1.0 (though this may produce lower quality
    samples) or accept the small bias from nucleus sampling.
  """
  if max_new_tokens <= 0:
    raise ValueError(f"max_new_tokens must be > 0 (got {max_new_tokens})")
  if len(prompt_ids) <= 0:
    raise ValueError("prompt_ids must be non-empty")

  prompt_len = int(len(prompt_ids))
  generated_ids = list(prompt_ids)

  # Auto-detect KV cache support if not specified
  if use_cache is None:
    use_cache = _model_supports_kv_cache(model)

  model.eval()

  with torch.inference_mode():
    if use_cache:
      # KV-cached generation (HuggingFace-style models)
      input_ids = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)
      past_key_values = None

      for step in range(max_new_tokens):
        if step == 0:
          outputs = model(input_ids, past_key_values=None, use_cache=True)
        else:
          new_token = torch.tensor([[generated_ids[-1]]], device="cuda", dtype=torch.long)
          outputs = model(new_token, past_key_values=past_key_values, use_cache=True)

        # Handle output formats
        if isinstance(outputs, tuple):
          logits, past_key_values = outputs[0], outputs[1] if len(outputs) > 1 else None
        elif hasattr(outputs, 'logits'):
          logits = outputs.logits
          past_key_values = getattr(outputs, 'past_key_values', None)
        else:
          # Model returned raw logits, disable cache
          logits = outputs
          use_cache = False

        last_logits = logits[:, -1, :].squeeze(0)
        next_id = _sample_top_p(last_logits, temperature=temperature, top_p=top_p)
        generated_ids.append(next_id)

        if next_id == int(eos_token_id):
          break

        # If cache became None, fall back to non-cached
        if use_cache and past_key_values is None:
          use_cache = False

    if not use_cache:
      # Non-cached generation (nmoe Transformer and similar)
      # Re-run full sequence each step
      toks = torch.tensor([prompt_ids], device="cuda", dtype=torch.long)

      for _ in range(max_new_tokens):
        logits = model(toks)[:, -1, :].squeeze(0)
        next_id = _sample_top_p(logits, temperature=temperature, top_p=top_p)
        generated_ids.append(next_id)
        toks = torch.tensor([generated_ids], device="cuda", dtype=torch.long)

        if next_id == int(eos_token_id):
          break

  completion = generated_ids[prompt_len:]
  # Filter tokens outside tokenizer vocab (model vocab may be padded larger)
  tokenizer_vocab = enc.n_vocab if hasattr(enc, 'n_vocab') else None
  if tokenizer_vocab is not None:
    valid_completion = [t for t in completion if t < tokenizer_vocab]
  else:
    valid_completion = completion
  completion_text = enc.decode(valid_completion) if valid_completion else ""
  return Trajectory(
    prompt_len=prompt_len,
    completion_len=len(completion),
    tokens=generated_ids,
    completion_text=completion_text,
  )


def _pack_right(seqs: Sequence[Sequence[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
  lens = torch.tensor([len(s) for s in seqs], device="cpu", dtype=torch.long)
  tmax = int(lens.max().item()) if lens.numel() else 0
  out = torch.full((len(seqs), tmax), int(pad_id), dtype=torch.long)
  for i, s in enumerate(seqs):
    if len(s) == 0:
      continue
    out[i, :len(s)] = torch.tensor(list(s), dtype=torch.long)
  return out, lens


def completion_nll_mean(
  model,
  *,
  seqs: Sequence[Sequence[int]],
  prompt_lens: Sequence[int],
  completion_lens: Sequence[int],
  pad_id: int,
  device: torch.device,
  normalize_by_length: bool = True,
  max_length: int | None = None,
) -> torch.Tensor:
  """NLL over completion tokens for each sequence.

  Right-padding is safe here: logits for positions < true_length are invariant
  to tokens in positions > true_length under causal attention.

  Args:
    normalize_by_length: If True, return mean NLL (divide by actual length).
                         If False, return sum NLL (for Dr. GRPO constant normalization).
    max_length: If set and normalize_by_length=True, normalize by this constant
                instead of actual length. This is the Dr. GRPO fix for length bias.

  Dr. GRPO Note (arxiv 2503.20783):
    Using normalize_by_length=True with actual length causes "response length bias"
    where longer wrong answers are under-penalized. Two fixes:
    1. Set normalize_by_length=False and normalize in loss by constant
    2. Set max_length to the max completion length

  IMPORTANT: Scaled-Logprob PPO Behavior
    When max_length is set, this returns logp_scaled = log(π) / C where C = max_length.
    In PPO, this means the ratio becomes:
      ratio = exp(logp_scaled - logp_old_scaled)
            = exp((log π - log π_old) / C)
            = (π / π_old)^(1/C)

    This is NOT standard PPO where ratio = π/π_old. Consequences:
    - ε clipping is in "root-ratio" units, not ratio units
    - β KL coefficient operates in scaled-logprob space
    - OPSM delta is in scaled-logprob units
    - With C=512 and ε=10, the effective trust region is very wide

    This is intentional per Dr. GRPO to fix length bias, but hyperparameters
    (ε, β, δ) should be understood as operating in this scaled space.
  """
  if not (len(seqs) == len(prompt_lens) == len(completion_lens)):
    raise ValueError("seqs/prompt_lens/completion_lens must have same length")
  if len(seqs) == 0:
    return torch.empty((0,), device=device, dtype=torch.float32)

  tok_cpu, lens_cpu = _pack_right(seqs, pad_id=pad_id)
  tokens = tok_cpu.to(device=device, non_blocking=True)

  prompt_lens_t = torch.tensor(list(prompt_lens), device=device, dtype=torch.long)
  completion_lens_t = torch.tensor(list(completion_lens), device=device, dtype=torch.long)

  if (prompt_lens_t <= 0).any():
    raise ValueError("prompt_lens must be >= 1 for all samples")
  if (completion_lens_t < 0).any():
    raise ValueError("completion_lens must be >= 0 for all samples")

  # logits at position t predict token at t+1
  logits = model(tokens)[:, :-1, :]
  targets = tokens[:, 1:]
  n, t = targets.shape
  nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none").reshape(n, t)

  mask = torch.zeros((n, t), device=device, dtype=torch.float32)
  for i in range(n):
    p = int(prompt_lens_t[i].item())
    c = int(completion_lens_t[i].item())
    if c <= 0:
      continue
    start = p - 1
    end = p + c - 2
    if start < 0 or start >= t:
      continue
    end = min(end, t - 1)
    mask[i, start:end + 1] = 1.0

  nll_sum = (nll * mask).sum(dim=1)

  if not normalize_by_length:
    return nll_sum

  if max_length is not None:
    # Dr. GRPO: normalize by constant to fix length bias
    return nll_sum / float(max_length)
  else:
    # Original: normalize by actual length (causes length bias)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return nll_sum / denom

