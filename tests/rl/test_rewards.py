"""Tests for nmoe.rl.rewards module."""
import pytest


class TestFormatReward:
    """Test format_reward function."""

    def test_valid_format_gets_1(self):
        """Valid format should get reward of 1.0."""
        from nmoe.rl.rewards import format_reward

        result = format_reward("<think>reasoning</think><answer>42</answer>")
        assert result.value == 1.0
        assert result.category == "valid_format"

    def test_missing_tags_gets_0(self):
        """Missing tags should get lower reward."""
        from nmoe.rl.rewards import format_reward

        result = format_reward("Just an answer without tags")
        assert result.value < 1.0

    def test_returns_reward_result(self):
        """Should return RewardResult with value and category."""
        from nmoe.rl.rewards import format_reward, RewardResult

        result = format_reward("<think>x</think><answer>y</answer>")

        assert isinstance(result, RewardResult)
        assert hasattr(result, "value")
        assert hasattr(result, "category")
        assert 0.0 <= result.value <= 1.0


class TestPythonTestsReward:
    """Test python_tests_reward function."""

    def test_passing_tests_get_1(self):
        """Passing tests should get reward of 1.0."""
        from nmoe.rl.rewards import python_tests_reward

        code = "<think>add function</think><answer>def add(a, b): return a + b</answer>"
        tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0"

        result = python_tests_reward(code, tests=tests)
        assert result.value == 1.0 or result == 1.0

    def test_failing_tests_get_0(self):
        """Failing tests should get reward of 0.0."""
        from nmoe.rl.rewards import python_tests_reward

        code = "<think>wrong</think><answer>def add(a, b): return a - b</answer>"
        tests = "assert add(1, 2) == 3"

        result = python_tests_reward(code, tests=tests)
        result_value = result.value if hasattr(result, 'value') else result
        assert result_value == 0.0

    def test_syntax_error_gets_0(self):
        """Syntax errors should get reward of 0.0."""
        from nmoe.rl.rewards import python_tests_reward

        code = "<think>broken</think><answer>def add(a, b) return a + b</answer>"
        tests = "assert add(1, 2) == 3"

        result = python_tests_reward(code, tests=tests)
        result_value = result.value if hasattr(result, 'value') else result
        assert result_value == 0.0


class TestJudgeRubric:
    """Test JudgeRubric class."""

    def test_initialization(self):
        """Should initialize without errors."""
        from nmoe.rl.rewards import JudgeRubric

        rubric = JudgeRubric(client=None, model="gpt-4o-mini")

        assert rubric.model == "gpt-4o-mini"
        assert rubric.prompt is not None

    def test_has_required_methods(self):
        """Should have judge and score methods."""
        from nmoe.rl.rewards import JudgeRubric

        rubric = JudgeRubric(client=None)

        assert hasattr(rubric, "judge")
        assert hasattr(rubric, "score_async")
        assert callable(rubric.judge)


class TestRewardComposition:
    """Test reward composition and aggregation."""

    def test_good_format_better_than_bad(self):
        """Good format should score higher than bad format."""
        from nmoe.rl.rewards import format_reward

        good = format_reward("<think>thought</think><answer>answer</answer>")
        bad = format_reward("just raw text no tags")

        assert good.value > bad.value

    def test_reward_result_fields(self):
        """RewardResult should have expected fields."""
        from nmoe.rl.rewards import RewardResult

        result = RewardResult(value=0.5, category="test")

        assert result.value == 0.5
        assert result.category == "test"
