"""Tests for nmoe.rl.curriculum.

These are spec-style tests aligned with RL_DESIGN.md + TODO.md claims:
- WSD curriculum stage transitions and decay
- Adaptive weight adjustment
- Thinking budget schedule and efficiency reward
"""

from __future__ import annotations

import math

import pytest


class TestStageType:
    def test_stage_type_string_values(self):
        from nmoe.rl.curriculum import StageType

        assert str(StageType.WARMUP) == "warmup"
        assert str(StageType.SUSTAIN) == "sustain"
        assert str(StageType.DECAY) == "decay"

    def test_curriculum_stage_normalizes_names(self):
        from nmoe.rl.curriculum import CurriculumStage, StageType

        stage = CurriculumStage(
            name=StageType.WARMUP,
            examples_count=1,
            weights={"x": 1.0},
            transitions_to=StageType.SUSTAIN,
        )
        assert stage.name == "warmup"
        assert stage.transitions_to == "sustain"


class TestWSDCurriculum:
    def test_stage_transition_on_examples_count(self):
        from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum

        stages = [
            CurriculumStage(
                name="warmup",
                examples_count=2,
                weights={"answer_correct": 1.0},
                transitions_to="sustain",
            ),
            CurriculumStage(
                name="sustain",
                examples_count=3,
                weights={"answer_correct": 2.0},
                transitions_to=None,
            ),
        ]
        cur = WSDCurriculum(stages=stages, adaptive=False)

        assert cur.current_stage == "warmup"
        assert cur.step() is None
        assert cur.step() == "sustain"
        assert cur.current_stage == "sustain"
        assert cur.examples_in_stage == 0

        assert cur.step() is None
        assert cur.step() is None
        assert cur.step() is None  # sustain is terminal
        assert cur.current_stage == "sustain"

    def test_get_weights_applies_positive_only_decay(self):
        from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum

        stages = [
            CurriculumStage(
                name="only",
                examples_count=1000,
                weights={"pos": 1.0, "neg": -1.0, "zero": 0.0},
                weight_decay_rate=0.1,
                transitions_to=None,
            )
        ]
        cur = WSDCurriculum(stages=stages, adaptive=False)

        # Advance within stage
        cur.step()
        cur.step()
        w = cur.get_weights()

        expected_pos = 1.0 * (1.0 - 0.1) ** 2
        assert math.isclose(w["pos"], expected_pos, rel_tol=1e-6)
        assert w["neg"] == -1.0
        assert w["zero"] == 0.0

    def test_adapt_weights_respects_adaptive_flag(self):
        from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum

        stages = [
            CurriculumStage(
                name="only",
                examples_count=10,
                weights={"struct_has_start": 1.0, "answer_correct": 3.0, "tool_executed": 0.5, "tool_syntax_valid": 0.5},
                transitions_to=None,
            )
        ]
        cur = WSDCurriculum(stages=stages, adaptive=False)
        cur.step({"struct_success": 1.0, "accuracy": 0.2, "tool_executed": 0.0})

        w = cur.get_weights()
        w2 = cur.adapt_weights(w)
        assert w2 == w

    def test_adapt_weights_adjusts_struct_accuracy_tools(self):
        from nmoe.rl.curriculum import CurriculumStage, WSDCurriculum

        stages = [
            CurriculumStage(
                name="only",
                examples_count=10,
                weights={
                    "struct_has_start": 1.0,
                    "has_reasoning": 1.0,
                    "answer_correct": 3.0,
                    "tool_executed": 0.5,
                    "tool_syntax_valid": 0.5,
                },
                transitions_to=None,
            )
        ]
        cur = WSDCurriculum(stages=stages, adaptive=True)
        # Push metrics into rolling window
        cur.step({"struct_success": 0.99, "accuracy": 0.2, "tool_executed": 0.0})

        w = cur.get_weights()
        w2 = cur.adapt_weights(w)

        # struct/has_ weights decay by 0.9 when struct_success>0.95
        assert math.isclose(w2["struct_has_start"], w["struct_has_start"] * 0.9, rel_tol=1e-6)
        assert math.isclose(w2["has_reasoning"], w["has_reasoning"] * 0.9, rel_tol=1e-6)

        # answer_correct boosted when accuracy<0.30 (capped at 4.0)
        assert math.isclose(w2["answer_correct"], min(4.0, w["answer_correct"] * 1.1), rel_tol=1e-6)

        # tool weights boosted when tool_executed<0.50 (capped at 2.0)
        assert math.isclose(w2["tool_syntax_valid"], min(2.0, w["tool_syntax_valid"] * 1.2), rel_tol=1e-6)
        assert math.isclose(w2["tool_executed"], min(2.0, w["tool_executed"] * 1.2), rel_tol=1e-6)


class TestThinkingBudget:
    def test_budget_schedule_applies_last_threshold(self):
        from nmoe.rl.curriculum import ThinkingBudgetConfig, get_thinking_budget

        cfg = ThinkingBudgetConfig()
        assert get_thinking_budget(cfg, total_examples=0) == 4096
        assert get_thinking_budget(cfg, total_examples=500) == 2048
        assert get_thinking_budget(cfg, total_examples=6000) == 1024

    def test_thinking_budget_reward(self):
        from nmoe.rl.curriculum import ThinkingBudgetConfig, thinking_budget_reward

        cfg = ThinkingBudgetConfig(max_reasoning_tokens=100, penalty_per_token=0.01, efficiency_bonus=0.2, budget_schedule=[(0, 10)])

        # Wrong answer => no efficiency reward
        assert thinking_budget_reward(5, False, cfg, total_examples=0) == 0.0

        # Correct under budget => positive bonus
        under = thinking_budget_reward(5, True, cfg, total_examples=0)
        assert under > 0.0

        # Correct over budget => negative penalty
        over = thinking_budget_reward(15, True, cfg, total_examples=0)
        assert over < 0.0
        assert math.isclose(over, -0.01 * 5, rel_tol=1e-6)

