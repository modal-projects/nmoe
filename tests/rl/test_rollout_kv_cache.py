"""Behavioral tests for KV-cache generation.

These are "kill tests": they exercise the actual fast-path semantics, not
string/AST checks. They must fail if we regress back to full-sequence replay
when a model supports KV caching.
"""

from __future__ import annotations

import pytest


class TestRolloutKVCache:
    def test_generate_one_uses_kv_cache_incremental_inputs(self):
        """KV-cache path must pass full prompt once, then 1-token increments."""
        import torch
        from nmoe.rl.rollout import generate_one, _model_supports_kv_cache

        if not torch.cuda.is_available():
            pytest.skip("generate_one uses CUDA tensors; requires a GPU")

        vocab_size = 16
        eos = 0
        prompt = [5, 6, 7]
        completion = [8, 9, eos]

        class _Enc:
            n_vocab = vocab_size

            def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(f"<{t}>" for t in token_ids)

        class _CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._next = list(completion)
                self.calls = []

            def forward(self, input_ids, past_key_values=None, use_cache: bool = False):
                self.calls.append(
                    {
                        "shape": tuple(input_ids.shape),
                        "past_is_none": past_key_values is None,
                        "use_cache": use_cache,
                    }
                )

                next_id = self._next.pop(0) if self._next else eos
                logits = torch.full(
                    (input_ids.shape[0], input_ids.shape[1], vocab_size),
                    -1e9,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                logits[:, -1, next_id] = 0.0

                if use_cache:
                    out = type("Out", (), {})()
                    out.logits = logits
                    out.past_key_values = ("pkv",)
                    return out
                return logits

        model = _CacheModel()
        assert _model_supports_kv_cache(model)
        traj = generate_one(
            model,
            enc=_Enc(),
            prompt_ids=prompt,
            max_new_tokens=8,
            eos_token_id=eos,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
        )

        assert traj.tokens[: len(prompt)] == prompt
        assert traj.tokens[len(prompt) :] == completion

        # First call consumes the full prompt; subsequent calls consume 1 token.
        assert model.calls[0]["shape"] == (1, len(prompt))
        assert all(c["shape"] == (1, 1) for c in model.calls[1:])
        assert model.calls[0]["past_is_none"] is True
        assert all((not c["past_is_none"]) for c in model.calls[1:])
        assert all(c["use_cache"] for c in model.calls)

    def test_generate_one_force_disable_cache_uses_full_sequence(self):
        """If use_cache=False, we must not pass past_key_values and must replay full context."""
        import torch
        from nmoe.rl.rollout import generate_one, _model_supports_kv_cache

        if not torch.cuda.is_available():
            pytest.skip("generate_one uses CUDA tensors; requires a GPU")

        vocab_size = 16
        eos = 0
        prompt = [5, 6, 7]
        completion = [8, 9, eos]

        class _Enc:
            n_vocab = vocab_size

            def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(f"<{t}>" for t in token_ids)

        class _CacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._next = list(completion)
                self.calls = []

            def forward(self, input_ids, past_key_values=None, use_cache: bool = False):
                self.calls.append({"shape": tuple(input_ids.shape), "use_cache": use_cache})
                next_id = self._next.pop(0) if self._next else eos
                logits = torch.full(
                    (input_ids.shape[0], input_ids.shape[1], vocab_size),
                    -1e9,
                    device=input_ids.device,
                    dtype=torch.float32,
                )
                logits[:, -1, next_id] = 0.0
                if use_cache:
                    out = type("Out", (), {})()
                    out.logits = logits
                    out.past_key_values = ("pkv",)
                    return out
                return logits

        model = _CacheModel()
        assert _model_supports_kv_cache(model)
        traj = generate_one(
            model,
            enc=_Enc(),
            prompt_ids=prompt,
            max_new_tokens=8,
            eos_token_id=eos,
            temperature=1.0,
            top_p=1.0,
            use_cache=False,
        )
        assert traj.tokens[: len(prompt)] == prompt
        assert traj.tokens[len(prompt) :] == completion

        # Full-sequence replay: shapes grow by 1 each generated token.
        assert model.calls[0]["shape"] == (1, len(prompt))
        assert model.calls[1]["shape"] == (1, len(prompt) + 1)
        assert model.calls[2]["shape"] == (1, len(prompt) + 2)
        assert not any(c["use_cache"] for c in model.calls)

    def test_generate_one_without_kv_cache_replays_full_sequence(self):
        """If the model doesn't support KV cache, the implementation must replay full context."""
        import torch
        from nmoe.rl.rollout import generate_one, _model_supports_kv_cache

        if not torch.cuda.is_available():
            pytest.skip("generate_one uses CUDA tensors; requires a GPU")

        vocab_size = 16
        eos = 0
        prompt = [5, 6, 7]
        completion = [8, 9, eos]

        class _Enc:
            n_vocab = vocab_size

            def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
                if isinstance(token_ids, int):
                    token_ids = [token_ids]
                return "".join(f"<{t}>" for t in token_ids)

        class _NoCacheModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._next = list(completion)
                self.calls = []

            def forward(self, tokens):
                self.calls.append({"shape": tuple(tokens.shape)})
                next_id = self._next.pop(0) if self._next else eos
                logits = torch.full(
                    (tokens.shape[0], tokens.shape[1], vocab_size),
                    -1e9,
                    device=tokens.device,
                    dtype=torch.float32,
                )
                logits[:, -1, next_id] = 0.0
                return logits

        model = _NoCacheModel()
        assert not _model_supports_kv_cache(model)
        traj = generate_one(
            model,
            enc=_Enc(),
            prompt_ids=prompt,
            max_new_tokens=8,
            eos_token_id=eos,
            temperature=1.0,
            top_p=1.0,
            use_cache=None,
        )
        assert traj.tokens[: len(prompt)] == prompt
        assert traj.tokens[len(prompt) :] == completion

        assert model.calls[0]["shape"] == (1, len(prompt))
        assert model.calls[1]["shape"] == (1, len(prompt) + 1)
        assert model.calls[2]["shape"] == (1, len(prompt) + 2)

