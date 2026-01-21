"""Agentic RL training loop with multi-turn generation and tool use.

Supports:
- R1-Zero style single-turn (format emergence)
- Multi-turn agentic with tool execution
- GDPO multi-reward training
- WSD curriculum scheduling
- Both Harmony and R1-Zero output formats
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from nmoe.rl.grpo import gdpo_decoupled_advantages, grpo_loss
from nmoe.rl.rollout import (
    generate_one,
    completion_nll_mean,
    Trajectory,
)
from nmoe.rl.turns import (
    AgentTurn,
    generate_turn_sync,
    turn_completion_nll_mean,
)
from nmoe.rl.rewards_gdpo import (
    RewardSignals,
    TrajectoryContext,
    compute_all_rewards,
    batch_rewards_to_tensors,
    reshape_for_gdpo,
)
from nmoe.rl.rewards_harmony import HARMONY_TOKENS, harmony_encode
from nmoe.rl.curriculum import (
    WSDCurriculum,
    CurriculumStage,
    ThinkingBudgetConfig,
)
from nmoe.rl.tasks import Task, TaskPool
from nmoe.rl.tools import AsyncToolExecutor, ToolConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AgenticTrainConfig:
    """Configuration for agentic RL training."""

    # Training mode
    mode: str = "single_turn"  # "single_turn" or "multi_turn"
    format_type: str = "harmony"  # "harmony" or "r1zero"

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    group_size: int = 4  # Samples per prompt for GRPO

    # Tool execution (multi-turn only)
    max_tool_rounds: int = 10
    tool_timeout_ms: int = 30000
    enable_tools: bool = False

    # Optimizer
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100

    # Training loop
    total_examples: int = 100_000
    batch_size: int = 8  # Prompts per step (total samples = batch_size * group_size)
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500

    # GRPO
    kl_coef: float = 0.01
    clip_eps: float = 0.2  # PPO clipping epsilon
    normalize_rewards: bool = True

    # Multi-reward (GDPO)
    use_multi_reward: bool = True
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        # Structure (high priority early)
        "struct_has_start": 0.5,
        "struct_has_end": 0.5,
        "struct_proper_nesting": 0.5,
        "chan_has_analysis": 0.3,
        "chan_has_final": 0.3,
        # Tools (medium priority)
        "tool_syntax_valid": 0.2,
        "tool_executed": 0.2,
        "tool_no_error": 0.1,
        # Task (increases over time via curriculum)
        "answer_correct": 1.0,
        # Efficiency (late curriculum)
        "thinking_efficiency": 0.0,
        "tool_efficiency": 0.0,
    })

    # Thinking budget
    thinking_budget_initial: int = 2048
    thinking_budget_final: int = 512
    thinking_efficiency_weight: float = 0.1

    # Paths
    output_dir: str = "./outputs/agentic_rl"
    checkpoint_dir: str = "./checkpoints/agentic_rl"

    # Replay bundle persistence (debugging / correctness)
    # Disabled by default; when enabled, any write failure is fatal (fail-closed).
    replay_bundle_dir: str = ""  # e.g. "/data/replay_bundles"
    replay_run_id: str = ""      # stable identifier for the run
    replay_sample_every: int = 0  # 0=disabled, 1=all, N=~1/N sampled
    replay_seed: int = 0          # deterministic sampling seed


# =============================================================================
# Training State
# =============================================================================

@dataclass
class TrainState:
    """Mutable training state."""
    examples_seen: int = 0
    steps: int = 0
    epoch: int = 0

    # Rolling metrics for adaptive curriculum
    recent_accuracy: list[float] = field(default_factory=list)
    recent_format_rate: list[float] = field(default_factory=list)

    # Best checkpoint tracking
    best_accuracy: float = 0.0
    best_step: int = 0

    def update_metrics(self, accuracy: float, format_rate: float, window: int = 100):
        """Update rolling metrics."""
        self.recent_accuracy.append(accuracy)
        self.recent_format_rate.append(format_rate)
        if len(self.recent_accuracy) > window:
            self.recent_accuracy = self.recent_accuracy[-window:]
            self.recent_format_rate = self.recent_format_rate[-window:]

    @property
    def rolling_accuracy(self) -> float:
        if not self.recent_accuracy:
            return 0.0
        return sum(self.recent_accuracy) / len(self.recent_accuracy)

    @property
    def rolling_format_rate(self) -> float:
        if not self.recent_format_rate:
            return 0.0
        return sum(self.recent_format_rate) / len(self.recent_format_rate)


# =============================================================================
# Batch Generation
# =============================================================================

def generate_batch_single_turn(
    model,
    enc,
    tasks: list[Task],
    config: AgenticTrainConfig,
    device: torch.device,
) -> tuple[list[Trajectory], list[RewardSignals]]:
    """Generate trajectories for single-turn (no tools).

    Args:
        model: Language model
        enc: Tokenizer
        tasks: List of tasks to generate for
        config: Training configuration
        device: Target device

    Returns:
        Tuple of (trajectories, reward_signals)
    """
    trajectories = []
    rewards = []

    eos_token_id = (
        int(harmony_encode(enc, HARMONY_TOKENS["end"])[0])
        if config.format_type == "harmony"
        else int(getattr(enc, "eos_token_id", 0))
    )

    for task in tasks:
        prompt = task.to_prompt()
        prompt_ids = harmony_encode(enc, prompt) if config.format_type == "harmony" else enc.encode(prompt)

        # Generate G samples for this prompt
        for _ in range(config.group_size):
            traj = generate_one(
                model,
                enc=enc,
                prompt_ids=prompt_ids,
                max_new_tokens=config.max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            trajectories.append(traj)

            # Compute rewards
            ctx = TrajectoryContext(
                response_text=traj.completion_text,
                tokens=traj.tokens,
                task=task,
                reasoning_tokens=len(traj.tokens) - traj.prompt_len,
                total_tokens=len(traj.tokens),
                format_type=config.format_type,
                tokenizer=enc,
            )
            reward = compute_all_rewards(ctx)
            rewards.append(reward)

    return trajectories, rewards


def generate_batch_multi_turn(
    model,
    enc,
    tasks: list[Task],
    config: AgenticTrainConfig,
    tool_executor: AsyncToolExecutor,
    device: torch.device,
) -> tuple[list[AgentTurn], list[RewardSignals]]:
    """Generate agent turns for multi-turn (with tools).

    Args:
        model: Language model
        enc: Tokenizer
        tasks: List of tasks to generate for
        config: Training configuration
        tool_executor: Async tool executor
        device: Target device

    Returns:
        Tuple of (agent_turns, reward_signals)
    """
    turns = []
    rewards = []

    eos_token_id = (
        int(harmony_encode(enc, HARMONY_TOKENS["end"])[0])
        if config.format_type == "harmony"
        else int(getattr(enc, "eos_token_id", 0))
    )

    def _task_workspace_dir(task: Task) -> str | None:
        repo_path = getattr(task, "repo_path", None)
        if isinstance(repo_path, str) and repo_path:
            return repo_path

        inputs_path = getattr(task, "inputs_path", None)
        if isinstance(inputs_path, str) and inputs_path:
            try:
                return str(Path(inputs_path).parent)
            except Exception:
                return None

        return None

    for task in tasks:
        prompt = task.to_prompt()
        prompt_ids = harmony_encode(enc, prompt) if config.format_type == "harmony" else enc.encode(prompt)

        scoped_executor = tool_executor
        created_scoped = False
        if config.enable_tools:
            ws_dir = _task_workspace_dir(task)
            if ws_dir is not None:
                ws = Path(ws_dir)
                if ws.is_dir():
                    # Per-task sandbox: execute tools in the task workspace and only allow writes there.
                    scoped_cfg = replace(
                        tool_executor.config,
                        cwd=str(ws),
                        allowed_paths=[str(ws)],
                    )
                    scoped_executor = AsyncToolExecutor(scoped_cfg)
                    created_scoped = True

        try:
            # Generate G samples for this prompt
            for _ in range(config.group_size):
                turn = generate_turn_sync(
                    model,
                    enc=enc,
                    prompt_ids=prompt_ids,
                    tool_executor=scoped_executor if config.enable_tools else None,
                    max_new_tokens=config.max_new_tokens,
                    max_tool_rounds=config.max_tool_rounds,
                    eos_token_id=eos_token_id,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                turns.append(turn)

                # Compute rewards
                ctx = TrajectoryContext(
                    response_text=turn.full_text,
                    tokens=turn.tokens,
                    tool_sites=turn.to_tool_sites(),
                    task=task,
                    reasoning_tokens=turn.reasoning_tokens,
                    total_tokens=len(turn.tokens),
                    format_type=config.format_type,
                    tokenizer=enc,
                )
                reward = compute_all_rewards(ctx)
                rewards.append(reward)
        finally:
            if created_scoped:
                scoped_executor.close()

    return turns, rewards


# =============================================================================
# Loss Computation
# =============================================================================

def compute_grpo_loss(
    model,
    ref_model,
    trajectories: list[Trajectory] | list[AgentTurn],
    rewards: list[RewardSignals],
    config: AgenticTrainConfig,
    weights: dict[str, float],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO loss with multi-reward aggregation.

    Args:
        model: Policy model
        ref_model: Reference model for KL
        trajectories: Generated trajectories or turns
        rewards: Reward signals for each trajectory
        config: Training configuration
        weights: Current curriculum weights
        pad_id: Padding token ID
        device: Target device

    Returns:
        Tuple of (loss, metrics_dict)
    """
    batch_size = len(trajectories)
    group_size = config.group_size
    n_prompts = batch_size // group_size

    # Compute aggregated rewards
    if config.use_multi_reward:
        # GDPO: decoupled normalization per reward, then weighted aggregation
        rewards_dict = batch_rewards_to_tensors(rewards, device=device)
        rewards_dict = reshape_for_gdpo(rewards_dict, group_size)  # [B, G]

        weights_full = {k: float(weights.get(k, 0.0)) for k in rewards_dict.keys()}
        advantages_grouped = gdpo_decoupled_advantages(
            rewards_dict,
            weights=weights_full,
            normalize_mean=True,
            normalize_std=False,
        )

        # Optional per-group std normalization (kept for backwards compatibility with config.normalize_rewards)
        if config.normalize_rewards:
            std = advantages_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
            advantages_grouped = advantages_grouped / std

        advantages = advantages_grouped.view(-1).detach()  # [B*G]

        # For logging only: raw weighted sum of reward signals (not normalized)
        raw_weighted = torch.zeros(n_prompts, group_size, device=device)
        for name, tensor in rewards_dict.items():
            w = weights_full.get(name, 0.0)
            if w != 0.0:
                raw_weighted += w * tensor
        reward_tensor = raw_weighted.view(-1)  # [B*G]
    else:
        # Single reward: just correctness
        reward_tensor = torch.tensor(
            [r.answer_correct for r in rewards],
            device=device,
            dtype=torch.float32,
        )
        if config.normalize_rewards:
            reward_tensor = reward_tensor.view(n_prompts, group_size)
            mean = reward_tensor.mean(dim=1, keepdim=True)
            std = reward_tensor.std(dim=1, keepdim=True) + 1e-8
            reward_tensor = ((reward_tensor - mean) / std).view(-1)

        # Standard group-normalized advantages
        advantages = reward_tensor.view(n_prompts, group_size)
        mean = advantages.mean(dim=1, keepdim=True)
        std = advantages.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-8)
        advantages = ((advantages - mean) / std).view(-1).detach()

    # Compute NLLs
    if isinstance(trajectories[0], AgentTurn):
        # Multi-turn
        policy_nll = turn_completion_nll_mean(
            model,
            turns=trajectories,
            pad_id=pad_id,
            device=device,
        )
        with torch.no_grad():
            ref_nll = turn_completion_nll_mean(
                ref_model,
                turns=trajectories,
                pad_id=pad_id,
                device=device,
            )
    else:
        # Single-turn
        seqs = [t.tokens for t in trajectories]
        prompt_lens = [t.prompt_len for t in trajectories]
        completion_lens = [len(t.tokens) - t.prompt_len for t in trajectories]

        policy_nll = completion_nll_mean(
            model,
            seqs=seqs,
            prompt_lens=prompt_lens,
            completion_lens=completion_lens,
            pad_id=pad_id,
            device=device,
        )
        with torch.no_grad():
            ref_nll = completion_nll_mean(
                ref_model,
                seqs=seqs,
                prompt_lens=prompt_lens,
                completion_lens=completion_lens,
                pad_id=pad_id,
                device=device,
            )

    # Convert NLL to log-probs for GRPO
    logp_mean = -policy_nll
    logp_mean_ref = -ref_nll
    # For online RL without separate behavior policy, use current as "old"
    logp_mean_old = logp_mean.detach()

    # GRPO loss with PPO clipping
    loss, grpo_metrics = grpo_loss(
        logp_mean=logp_mean,
        logp_mean_old=logp_mean_old,
        logp_mean_ref=logp_mean_ref,
        advantages=advantages,
        clip_eps=config.clip_eps,
        kl_coef=config.kl_coef,
    )

    # Metrics
    metrics = {
        "loss": loss.item(),
        "pg_loss": grpo_metrics.pg_loss,
        "kl_loss": grpo_metrics.kl_loss,
        "mean_reward": reward_tensor.mean().item(),
        "mean_nll": policy_nll.mean().item(),
        "kl_mean": grpo_metrics.kl_mean,
        "clip_frac": grpo_metrics.clip_frac,
        "advantage_mean": grpo_metrics.advantage_mean,
    }

    # Per-reward metrics
    if config.use_multi_reward:
        for name in ["answer_correct", "struct_proper_nesting", "tool_executed"]:
            if name in rewards_dict:
                metrics[f"reward_{name}"] = rewards_dict[name].mean().item()

    return loss, metrics


# =============================================================================
# Training Loop
# =============================================================================

def train_step(
    model,
    ref_model,
    optimizer: Optimizer,
    enc,
    task_pool: TaskPool,
    config: AgenticTrainConfig,
    curriculum: WSDCurriculum | None,
    state: TrainState,
    tool_executor: AsyncToolExecutor | None,
    device: torch.device,
) -> dict[str, float]:
    """Execute one training step.

    Args:
        model: Policy model
        ref_model: Reference model
        optimizer: Optimizer
        enc: Tokenizer
        task_pool: Pool of tasks
        config: Training configuration
        curriculum: Optional curriculum scheduler
        state: Training state
        tool_executor: Optional tool executor
        device: Target device

    Returns:
        Dict of metrics
    """
    model.train()

    # Get current curriculum weights
    if curriculum is not None:
        weights = curriculum.get_weights()
        # Advance curriculum
        batch_metrics = {
            "accuracy": state.rolling_accuracy,
            "format_rate": state.rolling_format_rate,
        }
        stage_name = curriculum.step(batch_metrics)
        if stage_name:
            logger.info(f"Curriculum transition to stage: {stage_name}")
    else:
        weights = config.reward_weights

    # Sample tasks
    tasks = task_pool.sample(config.batch_size)

    # Generate trajectories
    if config.mode == "multi_turn" and tool_executor is not None:
        trajectories, rewards = generate_batch_multi_turn(
            model, enc, tasks, config, tool_executor, device
        )
    else:
        trajectories, rewards = generate_batch_single_turn(
            model, enc, tasks, config, device
        )

    # Persist replay bundles (debugging kernel: token-exact transcript + provenance).
    if config.replay_sample_every > 0 and config.replay_bundle_dir and config.replay_run_id:
        from nmoe.rl.replay_bundle import ReplayBundleWriter
        from nmoe.rl.trajectory_record import TrajectoryRecord

        rank = int(dist.get_rank()) if dist.is_initialized() else 0
        if rank == 0:
            writer = ReplayBundleWriter(
                base_dir=config.replay_bundle_dir,
                run_id=config.replay_run_id,
                sample_every=int(config.replay_sample_every),
                seed=int(config.replay_seed),
                rank=rank,
            )
            for i, (traj, r) in enumerate(zip(trajectories, rewards)):
                task_idx = i // int(config.group_size)
                task = tasks[task_idx]
                task_id = getattr(task, "task_id", "") or f"task_{task_idx}"

                prov_path = None
                repo_path = getattr(task, "repo_path", None)
                if isinstance(repo_path, str) and repo_path:
                    p = Path(repo_path) / ".nmoe_workspace.json"
                    if p.exists():
                        prov_path = p

                if hasattr(traj, "record") and traj.record is not None:
                    record = traj.record
                else:
                    # Single-turn trajectory: prompt is a prefix of tokens.
                    prompt_len = int(getattr(traj, "prompt_len", 0))
                    toks = list(getattr(traj, "tokens", []))
                    record = TrajectoryRecord(
                        prompt_tokens=toks[:prompt_len],
                        tokens=toks,
                        tool_events=[],
                    )

                try:
                    writer.maybe_write(
                        step=int(state.steps),
                        task_id=str(task_id),
                        sample_idx=int(i % int(config.group_size)),
                        record=record,
                        rewards=r,
                        provenance_path=prov_path,
                    )
                except Exception as e:
                    raise RuntimeError(f"replay bundle write failed: {e}") from e

    # Compute loss
    # Right padding token id: for causal NLL, any id is safe beyond true length.
    # Prefer EOS to avoid depending on a "<|pad|>" special token.
    pad_id = (
        int(harmony_encode(enc, HARMONY_TOKENS["end"])[0])
        if config.format_type == "harmony"
        else int(getattr(enc, "eos_token_id", 0))
    )
    loss, metrics = compute_grpo_loss(
        model, ref_model, trajectories, rewards,
        config, weights, pad_id, device
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer.step()

    # Update state
    state.steps += 1
    state.examples_seen += config.batch_size * config.group_size

    # Update rolling metrics
    accuracy = sum(1 for r in rewards if r.answer_correct > 0) / len(rewards)
    format_rate = sum(1 for r in rewards if r.struct_proper_nesting > 0) / len(rewards)
    state.update_metrics(accuracy, format_rate)

    metrics["accuracy"] = accuracy
    format_rate = format_rate
    metrics["format_rate"] = format_rate
    metrics["examples_seen"] = state.examples_seen
    metrics["step"] = state.steps

    return metrics


def train_loop(
    model,
    ref_model,
    optimizer: Optimizer,
    enc,
    task_pool: TaskPool,
    config: AgenticTrainConfig,
    curriculum: WSDCurriculum | None = None,
    tool_executor: AsyncToolExecutor | None = None,
    device: torch.device = torch.device("cuda"),
    resume_from: str | None = None,
) -> TrainState:
    """Main training loop.

    Args:
        model: Policy model
        ref_model: Reference model (frozen)
        optimizer: Optimizer
        enc: Tokenizer
        task_pool: Pool of tasks
        config: Training configuration
        curriculum: Optional curriculum scheduler
        tool_executor: Optional tool executor
        device: Target device
        resume_from: Optional checkpoint to resume from

    Returns:
        Final training state
    """
    state = TrainState()

    # Resume if specified
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        state.examples_seen = checkpoint.get("examples_seen", 0)
        state.steps = checkpoint.get("steps", 0)
        if curriculum and "curriculum_examples" in checkpoint:
            curriculum.examples_seen = checkpoint["curriculum_examples"]
        logger.info(f"Resumed from {resume_from} at step {state.steps}")

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Create output directories
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training: mode={config.mode}, format={config.format_type}")
    logger.info(f"Total examples: {config.total_examples}, batch_size: {config.batch_size}")

    start_time = time.time()

    while state.examples_seen < config.total_examples:
        metrics = train_step(
            model, ref_model, optimizer, enc, task_pool, config,
            curriculum, state, tool_executor, device
        )

        # Logging
        if state.steps % config.log_interval == 0:
            elapsed = time.time() - start_time
            examples_per_sec = state.examples_seen / elapsed
            logger.info(
                f"Step {state.steps} | "
                f"Examples {state.examples_seen}/{config.total_examples} | "
                f"Loss {metrics['loss']:.4f} | "
                f"Accuracy {metrics['accuracy']:.2%} | "
                f"Format {metrics['format_rate']:.2%} | "
                f"{examples_per_sec:.1f} ex/s"
            )

        # Checkpointing
        if state.steps % config.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{state.steps}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "examples_seen": state.examples_seen,
                "steps": state.steps,
                "curriculum_examples": curriculum.examples_seen if curriculum else 0,
                "config": config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Track best
            if state.rolling_accuracy > state.best_accuracy:
                state.best_accuracy = state.rolling_accuracy
                state.best_step = state.steps
                best_path = checkpoint_dir / "best.pt"
                torch.save({
                    "model": model.state_dict(),
                    "accuracy": state.best_accuracy,
                    "step": state.best_step,
                }, best_path)
                logger.info(f"New best accuracy: {state.best_accuracy:.2%}")

    # Final checkpoint
    final_path = checkpoint_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "examples_seen": state.examples_seen,
        "steps": state.steps,
        "final_accuracy": state.rolling_accuracy,
    }, final_path)
    logger.info(f"Training complete. Final checkpoint: {final_path}")

    return state


# =============================================================================
# Convenience: Build Default Curriculum
# =============================================================================

def build_default_curriculum(config: AgenticTrainConfig) -> WSDCurriculum:
    """Build a default 3-stage curriculum.

    Stages:
    1. Warmup (10%): Focus on format + basic structure
    2. Sustain (70%): Balance format, tools, correctness
    3. Decay (20%): Focus on correctness + efficiency

    Args:
        config: Training configuration

    Returns:
        WSDCurriculum instance
    """
    total = config.total_examples

    warmup = CurriculumStage(
        name="warmup",
        examples_count=int(total * 0.1),
        weights={
            "struct_has_start": 1.0,
            "struct_has_end": 1.0,
            "struct_proper_nesting": 1.0,
            "chan_has_analysis": 0.5,
            "chan_has_final": 0.5,
            "answer_correct": 0.2,
            "tool_syntax_valid": 0.1,
        },
        transitions_to="sustain",
    )

    sustain = CurriculumStage(
        name="sustain",
        examples_count=int(total * 0.7),
        weights={
            "struct_has_start": 0.3,
            "struct_has_end": 0.3,
            "struct_proper_nesting": 0.5,
            "chan_has_analysis": 0.2,
            "chan_has_final": 0.2,
            "answer_correct": 1.0,
            "tool_syntax_valid": 0.3,
            "tool_executed": 0.3,
            "tool_no_error": 0.2,
        },
        transitions_to="decay",
    )

    decay = CurriculumStage(
        name="decay",
        examples_count=int(total * 0.2),
        weights={
            "struct_proper_nesting": 0.2,
            "answer_correct": 1.0,
            "thinking_efficiency": 0.3,
            "tool_efficiency": 0.2,
        },
        weight_decay_rate=0.95,
    )

    return WSDCurriculum(stages=[warmup, sustain, decay])


# =============================================================================
# Self-Play Task Pool Builders
# =============================================================================

def build_selfplay_pool(
    task_type: str,
    repo_paths: list[str] | None = None,
    proof_dataset: str | None = None,
    workspaces_dir: str = "/tmp/nmoe_workspaces",
    seed: int = 42,
) -> TaskPool:
    """Build a self-play task pool.

    Args:
        task_type: One of "git", "agents", "proof", "proof_meta"
        repo_paths: For "git" type, list of repository paths to mine
        proof_dataset: For "proof*" types, path to JSONL dataset
        workspaces_dir: Directory for isolated workspaces
        seed: Random seed

    Returns:
        TaskPool instance
    """
    if task_type == "git":
        from nmoe.rl.tasks.git_episodes import GitCommitTaskPool
        from nmoe.rl.tools.codex import CodexConfig, CodexExecutor

        if not repo_paths:
            raise ValueError("git task type requires --repo-paths")

        repo = str(Path(repo_paths[0]).resolve())
        workspaces = str(Path(workspaces_dir).resolve())

        executor = CodexExecutor(CodexConfig(
            timeout_ms=120_000,
            read_write_paths=[repo, workspaces],
        ))

        # For multiple repos, we'd need a composite pool
        # For now, use the first repo
        pool = GitCommitTaskPool(
            repo_path=Path(repo),
            executor=executor,
            workspaces_dir=Path(workspaces_dir),
            verify_option_b=True,
            seed=seed,
        )
        stats = pool.scan()
        logger.info(f"GitCommitTaskPool: {stats}")
        return pool

    elif task_type == "agents":
        from nmoe.rl.tasks.agents import AgentSelfPlayTaskPool

        pool = AgentSelfPlayTaskPool(
            root_dir=Path(workspaces_dir) / "agent_env",
            seed=seed,
            digits=120,  # Genuinely tools-first
        )
        return pool

    elif task_type == "proof":
        from nmoe.rl.tasks.proof import ProofTaskPool

        if not proof_dataset:
            raise ValueError("proof task type requires --proof-dataset")

        pool = ProofTaskPool.from_jsonl(
            Path(proof_dataset),
            task_type="verifier",
        )
        return pool

    elif task_type == "proof_meta":
        from nmoe.rl.tasks.proof import ProofTaskPool

        if not proof_dataset:
            raise ValueError("proof_meta task type requires --proof-dataset")

        pool = ProofTaskPool.from_jsonl(
            Path(proof_dataset),
            task_type="meta_verifier",
        )
        return pool

    else:
        raise ValueError(f"Unknown task type: {task_type}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Agentic RL training with self-play task pools."""
    import argparse

    parser = argparse.ArgumentParser(description="Agentic RL Training")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--mode", type=str, default="single_turn", choices=["single_turn", "multi_turn"])
    parser.add_argument("--format", type=str, default="harmony", choices=["harmony", "r1zero"])
    parser.add_argument("--total-examples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default="./outputs/agentic_rl")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--enable-tools", action="store_true")

    # Self-play task pool args
    parser.add_argument(
        "--task-type", type=str, default="agents",
        choices=["git", "agents", "proof", "proof_meta"],
        help="Self-play task type"
    )
    parser.add_argument(
        "--repo-paths", type=str, nargs="+",
        help="Repository paths for git task type"
    )
    parser.add_argument(
        "--proof-dataset", type=str,
        help="Path to JSONL dataset for proof task types"
    )
    parser.add_argument(
        "--workspaces-dir", type=str, default="/tmp/nmoe_workspaces",
        help="Directory for isolated workspaces"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Load model (placeholder - integrate with actual model loading)
    logger.info(f"Loading model: {args.model}")
    # model = load_model(args.model)
    # ref_model = copy.deepcopy(model)
    # enc = load_tokenizer(args.model)

    # Build config
    config = AgenticTrainConfig(
        mode=args.mode,
        format_type=args.format,
        total_examples=args.total_examples,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        enable_tools=args.enable_tools,
    )

    # Build self-play task pool
    task_pool = build_selfplay_pool(
        task_type=args.task_type,
        repo_paths=args.repo_paths,
        proof_dataset=args.proof_dataset,
        workspaces_dir=args.workspaces_dir,
        seed=args.seed,
    )
    logger.info(f"Built task pool: {type(task_pool).__name__}")

    # Build curriculum
    curriculum = build_default_curriculum(config)

    # Build tool executor (if needed)
    tool_executor = None
    if config.mode == "multi_turn" and config.enable_tools:
        tool_config = ToolConfig(
            executor_type="codex_python",
            timeout_default_ms=config.tool_timeout_ms,
            allowed_paths=[args.workspaces_dir],
            cwd=args.workspaces_dir,
        )
        tool_executor = AsyncToolExecutor(tool_config)

    logger.info("Training configuration:")
    logger.info(f"  Mode: {config.mode}")
    logger.info(f"  Format: {config.format_type}")
    logger.info(f"  Task type: {args.task_type}")
    logger.info(f"  Tools: {config.enable_tools}")
    logger.info(f"  Total examples: {config.total_examples}")

    # Note: Actual training requires model loading
    # state = train_loop(
    #     model, ref_model, optimizer, enc, task_pool, config,
    #     curriculum=curriculum,
    #     tool_executor=tool_executor,
    #     resume_from=args.resume,
    # )


if __name__ == "__main__":
    main()
