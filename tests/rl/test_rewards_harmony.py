"""Tests for nmoe.rl.rewards_harmony module."""
import pytest


class TestParseHarmonyText:
    """Test parse_harmony_text function."""

    def test_valid_harmony_format(self):
        """Should parse valid Harmony format."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        text = """<|start|>assistant<|channel|>analysis<|message|>Let me think about this.<|end|>
<|start|>assistant<|channel|>final<|message|>The answer is 42.<|end|>"""

        result = parse_harmony_text(text)

        assert result.has_start
        assert result.has_end
        assert result.has_message
        assert result.has_channel
        assert result.properly_nested
        assert len(result.messages) == 2
        assert "analysis" in result.channels_present
        assert "final" in result.channels_present

    def test_missing_start_token(self):
        """Should detect missing start token."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        text = "Just plain text without any tokens"
        result = parse_harmony_text(text)

        assert not result.has_start
        assert len(result.parse_errors) > 0

    def test_extracts_channel_content(self):
        """Should extract content from channels."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        text = """<|start|>assistant<|channel|>analysis<|message|>Reasoning here<|end|>
<|start|>assistant<|channel|>final<|message|>Final answer<|end|>"""

        result = parse_harmony_text(text)

        assert result.analysis_content == "Reasoning here"
        assert result.final_content == "Final answer"

    def test_get_channel_method(self):
        """Should get channel content by name."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        text = "<|start|>assistant<|channel|>analysis<|message|>Test content<|end|>"
        result = parse_harmony_text(text)

        assert result.get_channel("analysis") == "Test content"
        assert result.get_channel("nonexistent") is None

    def test_has_analysis_property(self):
        """Should detect analysis channel presence."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        with_analysis = "<|start|>a<|channel|>analysis<|message|>x<|end|>"
        without_analysis = "<|start|>a<|channel|>final<|message|>x<|end|>"

        assert parse_harmony_text(with_analysis).has_analysis
        assert not parse_harmony_text(without_analysis).has_analysis

    def test_has_final_property(self):
        """Should detect final channel presence."""
        from nmoe.rl.rewards_harmony import parse_harmony_text

        with_final = "<|start|>a<|channel|>final<|message|>x<|end|>"
        without_final = "<|start|>a<|channel|>analysis<|message|>x<|end|>"

        assert parse_harmony_text(with_final).has_final
        assert not parse_harmony_text(without_final).has_final


class TestCheckNesting:
    """Test nesting validation."""

    def test_proper_nesting(self):
        """Should detect proper nesting."""
        from nmoe.rl.rewards_harmony import _check_nesting

        good = "<|start|>x<|end|><|start|>y<|end|>"
        assert _check_nesting(good)

    def test_nested_starts_invalid(self):
        """Should reject nested starts."""
        from nmoe.rl.rewards_harmony import _check_nesting

        bad = "<|start|><|start|>x<|end|><|end|>"
        assert not _check_nesting(bad)

    def test_end_before_start_invalid(self):
        """Should reject end before start."""
        from nmoe.rl.rewards_harmony import _check_nesting

        bad = "<|end|><|start|>x<|end|>"
        assert not _check_nesting(bad)

    def test_unclosed_start_invalid(self):
        """Should reject unclosed start."""
        from nmoe.rl.rewards_harmony import _check_nesting

        bad = "<|start|>x"
        assert not _check_nesting(bad)


class TestHarmonyStructureRewards:
    """Test harmony_structure_rewards function."""

    def test_all_present_gets_all_ones(self):
        """Should return 1.0 for all present tokens."""
        from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_structure_rewards

        text = "<|start|>a<|channel|>final<|message|>x<|end|>"
        parsed = parse_harmony_text(text)
        rewards = harmony_structure_rewards(parsed)

        assert rewards["struct_has_start"] == 1.0
        assert rewards["struct_has_end"] == 1.0
        assert rewards["struct_has_message"] == 1.0
        assert rewards["struct_has_channel"] == 1.0
        assert rewards["struct_proper_nesting"] == 1.0

    def test_missing_tokens_get_zero(self):
        """Should return 0.0 for missing tokens."""
        from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_structure_rewards

        text = "plain text"
        parsed = parse_harmony_text(text)
        rewards = harmony_structure_rewards(parsed)

        assert rewards["struct_has_start"] == 0.0
        assert rewards["struct_has_end"] == 0.0


class TestHarmonyChannelRewards:
    """Test harmony_channel_rewards function."""

    def test_channels_present(self):
        """Should detect channel presence."""
        from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_channel_rewards

        text = """<|start|>a<|channel|>analysis<|message|>This is reasoning content here<|end|>
<|start|>a<|channel|>final<|message|>Answer<|end|>"""

        parsed = parse_harmony_text(text)
        rewards = harmony_channel_rewards(parsed)

        assert rewards["chan_has_analysis"] == 1.0
        assert rewards["chan_has_final"] == 1.0
        assert rewards["chan_analysis_nonempty"] == 1.0
        assert rewards["chan_final_nonempty"] == 1.0

    def test_empty_content_detected(self):
        """Should detect empty channel content."""
        from nmoe.rl.rewards_harmony import parse_harmony_text, harmony_channel_rewards

        text = "<|start|>a<|channel|>analysis<|message|>short<|end|>"
        parsed = parse_harmony_text(text)
        rewards = harmony_channel_rewards(parsed)

        # Content < 10 chars counts as empty for analysis
        assert rewards["chan_analysis_nonempty"] == 0.0


class TestComputeHarmonyRewards:
    """Test compute_harmony_rewards unified interface."""

    def test_combines_all_rewards(self):
        """Should combine structure and channel rewards."""
        from nmoe.rl.rewards_harmony import compute_harmony_rewards

        text = """<|start|>a<|channel|>analysis<|message|>This is my reasoning process<|end|>
<|start|>a<|channel|>final<|message|>42<|end|>"""

        rewards = compute_harmony_rewards(text)

        # Structure rewards
        assert "struct_has_start" in rewards
        assert "struct_has_end" in rewards
        assert "struct_proper_nesting" in rewards

        # Channel rewards
        assert "chan_has_analysis" in rewards
        assert "chan_has_final" in rewards


class TestParseR1ZeroFormat:
    """Test R1-Zero format parsing."""

    def test_parse_think_answer(self):
        """Should parse <think>/<answer> format."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        text = "<think>Let me reason through this.</think><answer>42</answer>"
        result = parse_r1zero_format(text)

        assert result.has_analysis
        assert result.has_final
        assert result.analysis_content == "Let me reason through this."
        assert result.final_content == "42"

    def test_maps_to_harmony_channels(self):
        """Should map think->analysis, answer->final."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        text = "<think>reasoning</think><answer>result</answer>"
        result = parse_r1zero_format(text)

        assert "analysis" in result.channels_present
        assert "final" in result.channels_present

    def test_think_only(self):
        """Should handle think tag only."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        text = "<think>Just thinking</think>"
        result = parse_r1zero_format(text)

        assert result.has_analysis
        assert not result.has_final

    def test_answer_only(self):
        """Should handle answer tag only."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        text = "<answer>Direct answer</answer>"
        result = parse_r1zero_format(text)

        assert not result.has_analysis
        assert result.has_final

    def test_no_tags(self):
        """Should handle missing tags."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        text = "Just plain text"
        result = parse_r1zero_format(text)

        assert not result.has_analysis
        assert not result.has_final

    def test_proper_nesting_order(self):
        """Should check think comes before answer."""
        from nmoe.rl.rewards_harmony import parse_r1zero_format

        good = "<think>first</think><answer>second</answer>"
        bad = "<answer>first</answer><think>second</think>"

        assert parse_r1zero_format(good).properly_nested
        # Note: parse_r1zero_format checks think.end <= answer.start
        # so reversed order should fail nesting check
        assert not parse_r1zero_format(bad).properly_nested


class TestComputeR1ZeroRewards:
    """Test compute_r1zero_rewards function."""

    def test_full_format_rewards(self):
        """Should give full rewards for proper format."""
        from nmoe.rl.rewards_harmony import compute_r1zero_rewards

        text = "<think>My reasoning process here</think><answer>42</answer>"
        rewards = compute_r1zero_rewards(text)

        assert rewards["chan_has_analysis"] == 1.0
        assert rewards["chan_has_final"] == 1.0
        assert rewards["chan_analysis_nonempty"] == 1.0
        assert rewards["chan_final_nonempty"] == 1.0

    def test_missing_tags_penalized(self):
        """Should penalize missing tags."""
        from nmoe.rl.rewards_harmony import compute_r1zero_rewards

        text = "Just a plain answer"
        rewards = compute_r1zero_rewards(text)

        assert rewards["chan_has_analysis"] == 0.0
        assert rewards["chan_has_final"] == 0.0


class TestHarmonyMessage:
    """Test HarmonyMessage dataclass."""

    def test_creation(self):
        """Should create message with all fields."""
        from nmoe.rl.rewards_harmony import HarmonyMessage

        msg = HarmonyMessage(
            role="assistant",
            channel="analysis",
            content="Test content",
        )

        assert msg.role == "assistant"
        assert msg.channel == "analysis"
        assert msg.content == "Test content"

    def test_default_values(self):
        """Should have empty defaults."""
        from nmoe.rl.rewards_harmony import HarmonyMessage

        msg = HarmonyMessage()

        assert msg.role == ""
        assert msg.channel == ""
        assert msg.content == ""


class TestParsedHarmonyResponse:
    """Test ParsedHarmonyResponse dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        from nmoe.rl.rewards_harmony import ParsedHarmonyResponse

        result = ParsedHarmonyResponse()

        assert result.messages == []
        assert result.raw_text == ""
        assert not result.has_start
        assert not result.has_end
        assert not result.properly_nested
        assert len(result.channels_present) == 0

    def test_get_channel_tokens(self):
        """Should return empty list for missing channel."""
        from nmoe.rl.rewards_harmony import ParsedHarmonyResponse

        result = ParsedHarmonyResponse()
        tokens = result.get_channel_tokens("analysis")

        assert tokens == []
