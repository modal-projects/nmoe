"""RolloutEngine: token-in / token-out generation contract for RL.

SOTA RL stacks converge on the same correctness primitive:
- prompts are provided as token IDs (no retokenization drift)
- rollouts return token IDs + behavior logprobs (for PPO/GRPO ratios)

This module defines that contract and provides a local model implementation.
Server-backed engines can implement the same interface later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import torch

from nmoe.rl.rewards_harmony import HARMONY_TOKENS, harmony_encode


class StopReason(str, Enum):
    EOS = "eos"
    MAX_TOKENS = "max_tokens"
    TOOL_CALL = "tool_call"


@dataclass(frozen=True)
class RolloutRequest:
    prompt_tokens: list[int]
    n: int = 1
    max_new_tokens: int = 512
    eos_token_id: int = 0
    temperature: float = 1.0
    top_p: float = 1.0


@dataclass(frozen=True)
class RolloutSample:
    """One sampled continuation.

    logprobs are per-token behavior logprobs for completion tokens only:
      len(logprobs) == completion_len == len(tokens) - prompt_len

    Semantics:
      When sampling uses nucleus/top_p, logprobs are still computed under the
      full softmax distribution (no nucleus filtering). This avoids comparing
      logprobs from different distributions in PPO/GRPO ratios.
    """

    tokens: list[int]  # prompt + completion (includes EOS if generated)
    prompt_len: int
    completion_len: int
    completion_text: str
    logprobs: list[float]  # behavior logprobs for completion tokens
    stop_reason: StopReason


def logp_mean_from_logprobs(
    logprobs: Sequence[float],
    *,
    completion_len: int,
    max_length: int | None = None,
) -> float:
    """Aggregate per-token logprobs into the trainer's mean-logp convention.

    In nmoe RL training, logp_mean is represented as:
      logp_mean = (sum logp over completion tokens) / C
    where C is:
      - max_length if provided (Dr.GRPO constant length normalization), else
      - completion_len (mean per token)
    """
    if completion_len < 0:
        raise ValueError(f"completion_len must be >= 0 (got {completion_len})")
    if len(logprobs) != int(completion_len):
        raise ValueError(f"logprobs length mismatch: len={len(logprobs)} completion_len={completion_len}")
    if completion_len == 0:
        return 0.0

    denom = int(max_length) if max_length is not None else int(completion_len)
    if denom <= 0:
        raise ValueError(f"invalid denom (got {denom})")
    s = 0.0
    for lp in logprobs:
        f = float(lp)
        if math.isnan(f):
            raise ValueError("logprobs must not contain NaN")
        s += f
    return float(s / float(denom))


class RolloutEngine(Protocol):
    def generate(self, req: RolloutRequest) -> list[RolloutSample]:
        """Generate n samples for a single prompt."""


def require_harmony_tokenizer(enc) -> None:
    """Fail fast if tokenizer cannot encode Harmony specials as single tokens."""
    for name in ("start", "end", "message", "channel"):
        ids = harmony_encode(enc, HARMONY_TOKENS[name])
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError(
                "RolloutEngine requires a Harmony-capable tokenizer (single-token Harmony specials). "
                f"Token {HARMONY_TOKENS[name]!r} encoded as {ids}; use tokenizer=o200k_harmony."
            )


def _sample_top_p_with_logprob(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 (got {temperature})")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0,1] (got {top_p})")

    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_p >= 1.0:
        return int(torch.multinomial(probs, 1).item())

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > float(top_p)
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    denom = sorted_probs.sum().clamp_min(1e-12)
    sorted_probs = sorted_probs / denom

    sidx = int(torch.multinomial(sorted_probs, 1).item())
    tid = int(sorted_indices[sidx].item())
    return int(tid)


def _model_supports_kv_cache(model) -> bool:
    import inspect

    try:
        sig = inspect.signature(model.forward)
        params = sig.parameters
        return "past_key_values" in params and "use_cache" in params
    except (ValueError, TypeError):
        return False


class LocalRolloutEngine:
    """Local (in-process) rollout engine.

    This is the correctness reference: token-in, token-out, and explicit
    behavior logprobs matching the sampling distribution (temperature + top_p).
    """

    def __init__(self, *, model, enc, device: torch.device | str = "cuda"):
        self.model = model
        self.enc = enc
        self.device = torch.device(device)
        require_harmony_tokenizer(enc)

    @torch.inference_mode()
    def generate(self, req: RolloutRequest) -> list[RolloutSample]:
        if int(req.n) <= 0:
            return []
        if int(req.max_new_tokens) <= 0:
            raise ValueError(f"max_new_tokens must be > 0 (got {req.max_new_tokens})")
        if not req.prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")

        use_cache = _model_supports_kv_cache(self.model)
        out: list[RolloutSample] = []
        for _ in range(int(req.n)):
            out.append(self._generate_one(req, use_cache=use_cache))
        return out

    def _generate_one(self, req: RolloutRequest, *, use_cache: bool) -> RolloutSample:
        prompt_tokens = list(req.prompt_tokens)
        prompt_len = len(prompt_tokens)
        toks = list(prompt_tokens)
        logps: list[float] = []
        stop = StopReason.MAX_TOKENS

        self.model.eval()

        if use_cache:
            input_ids = torch.tensor([prompt_tokens], device=self.device, dtype=torch.long)
            past_key_values = None

            for step in range(int(req.max_new_tokens)):
                if step == 0:
                    outputs = self.model(input_ids, past_key_values=None, use_cache=True)
                else:
                    new_token = torch.tensor([[toks[-1]]], device=self.device, dtype=torch.long)
                    outputs = self.model(new_token, past_key_values=past_key_values, use_cache=True)

                if isinstance(outputs, tuple):
                    logits, past_key_values = outputs[0], outputs[1] if len(outputs) > 1 else None
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                    past_key_values = getattr(outputs, "past_key_values", None)
                else:
                    logits = outputs
                    use_cache = False
                    break

                last_logits = logits[:, -1, :].squeeze(0)
                next_id = _sample_top_p_with_logprob(
                    last_logits, temperature=float(req.temperature), top_p=float(req.top_p)
                )
                # Option C: sample with nucleus (top_p), but score with full softmax.
                # This matches common RLHF practice: sampling strategy is separate from
                # the distribution used for PPO/GRPO ratios.
                logp = float(torch.log_softmax(last_logits, dim=-1)[int(next_id)].detach().item())
                toks.append(int(next_id))
                logps.append(float(logp))
                if int(next_id) == int(req.eos_token_id):
                    stop = StopReason.EOS
                    break

                if use_cache and past_key_values is None:
                    use_cache = False
                    break

        if not use_cache:
            # Re-run full sequence each step.
            for _ in range(int(req.max_new_tokens)):
                t = torch.tensor([toks], device=self.device, dtype=torch.long)
                last_logits = self.model(t)[:, -1, :].squeeze(0)
                next_id = _sample_top_p_with_logprob(
                    last_logits, temperature=float(req.temperature), top_p=float(req.top_p)
                )
                logp = float(torch.log_softmax(last_logits, dim=-1)[int(next_id)].detach().item())
                toks.append(int(next_id))
                logps.append(float(logp))
                if int(next_id) == int(req.eos_token_id):
                    stop = StopReason.EOS
                    break

        completion = toks[prompt_len:]
        tokenizer_vocab = self.enc.n_vocab if hasattr(self.enc, "n_vocab") else None
        if tokenizer_vocab is not None:
            valid_completion = [t for t in completion if int(t) < int(tokenizer_vocab)]
        else:
            valid_completion = completion
        completion_text = self.enc.decode(list(valid_completion)) if valid_completion else ""

        return RolloutSample(
            tokens=toks,
            prompt_len=prompt_len,
            completion_len=len(completion),
            completion_text=completion_text,
            logprobs=logps,
            stop_reason=stop,
        )
