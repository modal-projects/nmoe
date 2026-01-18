"""R1-Zero style reinforcement learning with GRPO.

Pure RL from base model without SFT warmup. Uses Group Relative Policy
Optimization (GRPO) which eliminates the critic model by using group-relative
baselines. Reasoning capabilities emerge naturally through RL without explicit
chain-of-thought supervision.

Features:
- Composable Rubric-based rewards (format + accuracy + tool use)
- OPSM (Off-Policy Sequence Masking) - prevents reward hacking
- TIS (Truncated Importance Sampling) - reduces off-policy variance
- Dr.GRPO STD normalization - reduces variance
- Dual-clip PPO - caps penalty for negative advantages
- Asymmetric advantage scaling - penalizes bad more than rewarding good
- Multiple KL variants (k1, k2, k3/reverse KL)
- Reward categories for debugging

Extended Features (Agentic RL):
- GDPO multi-reward aggregation with curriculum
- Multi-turn generation with tool execution
- Harmony format (OpenAI channel-based) support
- Async scatter/gather tool execution
- WSD-style curriculum scheduling
- Task pools (math, code, agentic)

Reference:
- DeepSeek-R1 (https://arxiv.org/abs/2501.12948)
- primeintellect/verifiers (Rubric pattern)
- THUDM/slime (OPSM, TIS, Dr.GRPO, dual-clip)
"""
# Core GRPO / GDPO
from nmoe.rl.grpo import (
    GRPOMetrics,
    grpo_loss,
    group_relative_advantages,
    gdpo_decoupled_advantages,
    compute_kl,
    compute_opsm_mask,
    compute_tis_weights,
    filter_zero_std_groups,
)

# Original rewards (single-turn)
from nmoe.rl.rewards import (
    RewardResult,
    RewardFunc,
    Rubric,
    format_reward,
    gsm8k_accuracy_reward,
    gsm8k_gold_from_answer_field,
    python_tests_reward,
    reward_math_gsm8k,
    reward_code_unittest,
    math_rubric,
    code_rubric,
    extract_tag,
    async_reward_with_timeout,
    condition_reward,
    conditioned_length_reward,
)

# Rollout / Generation
from nmoe.rl.rollout import Trajectory, generate_one, completion_nll_mean

# Curriculum scheduling
from nmoe.rl.curriculum import (
    StageType,
    CurriculumStage,
    WSDCurriculum,
    ThinkingBudgetConfig,
    thinking_budget_reward,
    get_thinking_budget,
)

# Harmony format rewards
from nmoe.rl.rewards_harmony import (
    HarmonyMessage,
    ParsedHarmonyResponse,
    HarmonyTokenizer,
    parse_harmony_text,
    compute_harmony_rewards,
    compute_r1zero_rewards,
)

# Tool-specific rewards
from nmoe.rl.rewards_tools import (
    ToolRewardSignals,
    ToolCallSite,
    compute_tool_call_rewards,
    compute_all_tool_rewards,
    check_python_syntax,
    verify_python_with_tests,
)

# GDPO multi-reward aggregation
from nmoe.rl.rewards_gdpo import (
    RewardSignals,
    TrajectoryContext,
    compute_all_rewards as compute_gdpo_rewards,
    compute_trajectory_rewards,
    batch_rewards_to_tensors,
    reshape_for_gdpo,
)

# Multi-turn generation
from nmoe.rl.turns import (
    AgentMessage,
    AgentTurn,
    generate_turn_async,
    generate_turn_sync,
    generate_batch_async,
    generate_batch_sync,
    turn_completion_nll_mean,
)

# Tool execution
from nmoe.rl.tools import (
    ToolType,
    ToolCall,
    ToolResult,
    ToolConfig,
)
from nmoe.rl.tools.executor import AsyncToolExecutor

# Tasks
from nmoe.rl.tasks import Task, TaskPool

# Agentic training
from nmoe.rl.train_agentic import (
    AgenticTrainConfig,
    TrainState,
    train_step,
    train_loop,
    build_default_curriculum,
)

__all__ = [
    # GRPO / GDPO
    "GRPOMetrics",
    "grpo_loss",
    "group_relative_advantages",
    "gdpo_decoupled_advantages",
    "compute_kl",
    "compute_opsm_mask",
    "compute_tis_weights",
    "filter_zero_std_groups",
    # Original Rewards
    "RewardResult",
    "RewardFunc",
    "Rubric",
    "format_reward",
    "gsm8k_accuracy_reward",
    "gsm8k_gold_from_answer_field",
    "python_tests_reward",
    "reward_math_gsm8k",
    "reward_code_unittest",
    "math_rubric",
    "code_rubric",
    "extract_tag",
    "async_reward_with_timeout",
    "condition_reward",
    "conditioned_length_reward",
    # Rollout
    "Trajectory",
    "generate_one",
    "completion_nll_mean",
    # Curriculum
    "StageType",
    "CurriculumStage",
    "WSDCurriculum",
    "ThinkingBudgetConfig",
    "thinking_budget_reward",
    "get_thinking_budget",
    # Harmony Rewards
    "HarmonyMessage",
    "ParsedHarmonyResponse",
    "HarmonyTokenizer",
    "parse_harmony_text",
    "compute_harmony_rewards",
    "compute_r1zero_rewards",
    # Tool Rewards
    "ToolRewardSignals",
    "ToolCallSite",
    "compute_tool_call_rewards",
    "compute_all_tool_rewards",
    "check_python_syntax",
    "verify_python_with_tests",
    # GDPO Aggregation
    "RewardSignals",
    "TrajectoryContext",
    "compute_gdpo_rewards",
    "compute_trajectory_rewards",
    "batch_rewards_to_tensors",
    "reshape_for_gdpo",
    # Multi-turn
    "AgentMessage",
    "AgentTurn",
    "generate_turn_async",
    "generate_turn_sync",
    "generate_batch_async",
    "generate_batch_sync",
    "turn_completion_nll_mean",
    # Tools
    "ToolType",
    "ToolCall",
    "ToolResult",
    "ToolConfig",
    "AsyncToolExecutor",
    # Tasks
    "Task",
    "TaskPool",
    # Agentic Training
    "AgenticTrainConfig",
    "TrainState",
    "train_step",
    "train_loop",
    "build_default_curriculum",
]
