"""Math-V2-style self-play loop: generator ↔ verifier ↔ meta-verifier.

This is a capability path (not throughput):
- Generator produces a proof attempt for a problem.
- Verifier judges the proof with a discrete score in {0, 0.5, 1}.
- Meta-verifier judges the verifier's analysis with a discrete score in {0, 0.5, 1}.

Key contracts:
- Every stage records token-exact transcripts (TrajectoryRecord).
- Parse failures are explicit (failure_category="parse_error"), never silent.

Training set emission:
- Verifier training: (problem, proof) → verifier gold_score from meta-verified samples.
- Meta-verifier training: (problem, proof, verifier_response) → meta gold_score.
- Generator rewards: proof gets reward signal from accepted samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from nmoe.rl.failures import FailureCategory
from nmoe.rl.replay_bundle import ReplayBundleWriter
from nmoe.rl.rollout_engine import RolloutEngine, RolloutRequest, RolloutSample
from nmoe.rl.rewards_harmony import CHANNELS, HARMONY_TOKENS, harmony_encode, harmony_message
from nmoe.rl.tasks.proof import ProofMetaVerifierTask, ProofVerifierTask
from nmoe.rl.trajectory_record import TrajectoryRecord


@dataclass(frozen=True)
class MathV2StageResult:
    """One stage output with token-exact transcript and parsed score (if any)."""

    prompt: str
    sample: RolloutSample
    record: TrajectoryRecord
    text: str
    score: str | None = None
    failure_category: str = FailureCategory.OK.value


@dataclass(frozen=True)
class MathV2SelfPlaySample:
    """One full generator→verifier→meta-verifier chain."""

    problem: str
    proof: MathV2StageResult
    verifier: MathV2StageResult
    meta: MathV2StageResult
    accepted: bool


@dataclass(frozen=True)
class MathV2SelfPlayConfig:
    """Generation parameters for the self-play loop."""

    max_new_tokens_proof: int = 1024
    max_new_tokens_verifier: int = 512
    max_new_tokens_meta: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    eos_token_id: int | None = None


@dataclass(frozen=True)
class MathV2SelfPlayEmitConfig:
    """Emission config for training samples + replay bundles."""

    out_dir: str | Path
    run_id: str = "mathv2_selfplay"
    replay_dir: str | Path | None = None
    replay_sample_every: int = 0  # 0 disables
    seed: int = 0
    rank: int = 0


def _sample_to_record(sample: RolloutSample) -> TrajectoryRecord:
    toks = list(sample.tokens)
    return TrajectoryRecord(prompt_tokens=toks[: int(sample.prompt_len)], tokens=toks, tool_events=[])


def _extract_verifier_score(text: str) -> str | None:
    task = ProofVerifierTask(problem="P", proof="Y", gold_score="0")
    return task.extract_answer(text)


def _extract_meta_score(text: str) -> str | None:
    task = ProofMetaVerifierTask(problem="P", proof="Y", verifier_response="V", gold_meta_score="0")
    return task.extract_answer(text)


def _encode(enc, text: str) -> list[int]:
    if not hasattr(enc, "encode"):
        raise TypeError("tokenizer must implement encode()")
    return harmony_encode(enc, text)


def _eos(enc, cfg: MathV2SelfPlayConfig) -> int:
    if cfg.eos_token_id is not None:
        return int(cfg.eos_token_id)
    return int(harmony_encode(enc, HARMONY_TOKENS["end"])[0])


def _generate_completion(
    engine: RolloutEngine,
    *,
    enc,
    prompt: str,
    max_new_tokens: int,
    cfg: MathV2SelfPlayConfig,
) -> MathV2StageResult:
    prompt_ids = _encode(enc, prompt)
    eos_id = _eos(enc, cfg)
    samples = engine.generate(
        RolloutRequest(
            prompt_tokens=prompt_ids,
            n=1,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=int(eos_id),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
        )
    )
    if len(samples) != 1:
        raise RuntimeError(f"RolloutEngine returned {len(samples)} samples for n=1")
    s = samples[0]
    record = _sample_to_record(s)
    text = s.completion_text
    return MathV2StageResult(
        prompt=prompt,
        sample=s,
        record=record,
        text=text,
    )


class MathV2SelfPlayRunner:
    """Runs a minimal Math-V2 self-play loop with token-exact transcripts."""

    def __init__(
        self,
        *,
        generator_engine: RolloutEngine,
        verifier_engine: RolloutEngine,
        meta_verifier_engine: RolloutEngine,
        enc,
        config: MathV2SelfPlayConfig | None = None,
    ):
        self.generator_engine = generator_engine
        self.verifier_engine = verifier_engine
        self.meta_verifier_engine = meta_verifier_engine
        self.enc = enc
        self.cfg = config or MathV2SelfPlayConfig()

    def run_one(self, problem: str) -> MathV2SelfPlaySample:
        # 1) Generator produces a proof attempt (free-form).
        gen_prompt = harmony_message(
            role="user",
            channel=CHANNELS["commentary"],
            content=(
                "You are a theorem prover. Produce a proof attempt.\n\n"
                "Use the analysis channel for reasoning, and put the proof attempt in the final channel.\n\n"
                f"Problem:\n{problem}\n"
            ),
        )
        proof = _generate_completion(
            self.generator_engine,
            enc=self.enc,
            prompt=gen_prompt,
            max_new_tokens=self.cfg.max_new_tokens_proof,
            cfg=self.cfg,
        )

        # 2) Verifier judges the proof.
        v_task = ProofVerifierTask(problem=problem, proof=proof.text, gold_score="0")
        verifier = _generate_completion(
            self.verifier_engine,
            enc=self.enc,
            prompt=v_task.to_prompt(),
            max_new_tokens=self.cfg.max_new_tokens_verifier,
            cfg=self.cfg,
        )
        v_score = _extract_verifier_score(verifier.text)
        if v_score is None:
            verifier = field_replace(verifier, score=None, failure_category=FailureCategory.PARSE_ERROR.value)
        else:
            verifier = field_replace(verifier, score=v_score)

        # 3) Meta-verifier judges the verifier response.
        m_task = ProofMetaVerifierTask(
            problem=problem,
            proof=proof.text,
            verifier_response=verifier.text,
            gold_meta_score="0",
        )
        meta = _generate_completion(
            self.meta_verifier_engine,
            enc=self.enc,
            prompt=m_task.to_prompt(),
            max_new_tokens=self.cfg.max_new_tokens_meta,
            cfg=self.cfg,
        )
        m_score = _extract_meta_score(meta.text)
        if m_score is None:
            meta = field_replace(meta, score=None, failure_category=FailureCategory.PARSE_ERROR.value)
        else:
            meta = field_replace(meta, score=m_score)

        accepted = (verifier.score is not None) and (meta.score == "1")
        return MathV2SelfPlaySample(problem=problem, proof=proof, verifier=verifier, meta=meta, accepted=accepted)

    def run_batch(self, problems: Sequence[str]) -> list[MathV2SelfPlaySample]:
        out: list[MathV2SelfPlaySample] = []
        for p in problems:
            out.append(self.run_one(str(p)))
        return out

    def emit(self, samples: Sequence[MathV2SelfPlaySample], *, cfg: MathV2SelfPlayEmitConfig) -> dict[str, Path]:
        """Emit accepted training samples and (optionally) replay bundles."""
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        verifier_path = out_dir / "proof_verifier_train.jsonl"
        meta_path = out_dir / "proof_meta_verifier_train.jsonl"
        counts_path = out_dir / "counts.json"

        n_ver = 0
        n_meta = 0
        with verifier_path.open("w", encoding="utf-8") as f_ver, meta_path.open("w", encoding="utf-8") as f_meta:
            for s in samples:
                if not s.accepted:
                    continue
                if s.verifier.score is None or s.meta.score is None:
                    continue
                f_ver.write(
                    json.dumps(
                        {"problem": s.problem, "proof": s.proof.text, "score": float(s.verifier.score)},
                        sort_keys=True,
                    )
                    + "\n"
                )
                n_ver += 1
                f_meta.write(
                    json.dumps(
                        {
                            "problem": s.problem,
                            "proof": s.proof.text,
                            "verifier_response": s.verifier.text,
                            "meta_score": float(s.meta.score),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                n_meta += 1
        counts_path.write_text(
            json.dumps({"proof_verifier_train": n_ver, "proof_meta_verifier_train": n_meta}, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

        if cfg.replay_dir is not None and int(cfg.replay_sample_every) != 0:
            writer = ReplayBundleWriter(
                base_dir=cfg.replay_dir,
                run_id=cfg.run_id,
                sample_every=int(cfg.replay_sample_every),
                seed=int(cfg.seed),
                rank=int(cfg.rank),
            )
            for i, s in enumerate(samples):
                if not s.accepted:
                    continue
                # One bundle per stage, keyed by stable stage IDs.
                writer.maybe_write(step=i, task_id="mathv2/proof", sample_idx=0, record=s.proof.record)
                writer.maybe_write(step=i, task_id="mathv2/verifier", sample_idx=0, record=s.verifier.record)
                writer.maybe_write(step=i, task_id="mathv2/meta", sample_idx=0, record=s.meta.record)

        return {
            "proof_verifier_train": verifier_path,
            "proof_meta_verifier_train": meta_path,
            "counts": counts_path,
        }


def field_replace(stage: MathV2StageResult, **kwargs: Any) -> MathV2StageResult:
    """Small helper to keep dataclasses frozen and patches minimal."""
    from dataclasses import replace

    return replace(stage, **kwargs)
