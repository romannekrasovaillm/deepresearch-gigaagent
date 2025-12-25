"""Tests for reward functions."""

import pytest

from src.rewards.rewards import (
    FactualReward,
    ProcessReward,
    CitationReward,
    CodeReward,
    CompositeRewardFunction,
    create_default_reward_function,
)


class TestProcessReward:
    """Tests for process efficiency reward."""

    def test_efficient_trajectory(self):
        reward = ProcessReward(max_steps=15)

        # Short trajectory
        trajectory = [
            {"action": "grep_search", "action_input": {"pattern": "test"}},
            {"action": "read_file_chunk", "action_input": {"path": "test.txt"}},
            {"action": "final_answer", "action_input": {"answer": "done"}},
        ]

        result = reward.compute(trajectory, "answer")

        assert result.value > 0.5  # Should be good score for short trajectory

    def test_inefficient_trajectory(self):
        reward = ProcessReward(max_steps=10)

        # Long trajectory with repeats
        trajectory = [
            {"action": "grep_search", "action_input": {"pattern": "test"}},
            {"action": "grep_search", "action_input": {"pattern": "test"}},  # Repeat
            {"action": "grep_search", "action_input": {"pattern": "test"}},  # Repeat
        ] * 5

        result = reward.compute(trajectory, "answer")

        assert result.value < 0.5  # Should be low score


class TestCitationReward:
    """Tests for citation accuracy reward."""

    def test_valid_citations(self):
        reward = CitationReward()

        # Trajectory that reads paper_0001
        trajectory = [
            {"action": "grep_search",
             "action_input": {},
             "observation": "paper_0001/methods.txt:10: some text"},
            {"action": "read_file_chunk",
             "action_input": {"path": "paper_0001/methods.txt"}},
        ]

        # Answer cites paper_0001
        answer = "According to paper_0001, the method works well."

        result = reward.compute(trajectory, answer)

        assert result.value == 1.0  # All citations valid

    def test_invalid_citations(self):
        reward = CitationReward()

        # Trajectory that reads nothing
        trajectory = []

        # Answer cites papers that weren't read
        answer = "According to paper_0042 and paper_0099, ..."

        result = reward.compute(trajectory, answer)

        assert result.value == 0.0  # No valid citations


class TestCodeReward:
    """Tests for code reasoning reward."""

    def test_successful_code(self):
        reward = CodeReward()

        trajectory = [
            {"action": "execute_python",
             "action_input": {"code": "print(2+2)"},
             "observation": "4"},
        ]

        answer = "The result is 4."

        result = reward.compute(trajectory, answer)

        assert result.value > 0.5  # Code executed and result used

    def test_no_code_when_required(self):
        reward = CodeReward()

        trajectory = [
            {"action": "grep_search", "action_input": {}},
        ]

        answer = "Some answer"
        metadata = {"requires_code": True}

        result = reward.compute(trajectory, answer, metadata=metadata)

        assert result.value == 0.0  # Code was required but not used


class TestCompositeReward:
    """Tests for combined reward function."""

    def test_combined_reward(self):
        reward_fn = create_default_reward_function()

        trajectory = [
            {"action": "grep_search",
             "action_input": {"pattern": "test"},
             "observation": "paper_0001/test.txt:1: test"},
            {"action": "read_file_chunk",
             "action_input": {"path": "paper_0001/test.txt"},
             "observation": "test content"},
        ]

        result = reward_fn.compute(
            trajectory=trajectory,
            final_answer="The answer is test, according to paper_0001.",
            golden_answer="The answer is test.",
        )

        assert 0.0 <= result.total <= 1.0
        assert len(result.components) == 4  # 4 reward functions

    def test_weights_normalize(self):
        reward_fn = create_default_reward_function(
            weights={"R_fact": 1.0, "R_process": 1.0, "R_citation": 1.0, "R_code": 1.0}
        )

        # Check weights sum to 1
        total_weight = sum(rf.weight for rf in reward_fn.reward_functions)
        assert abs(total_weight - 1.0) < 0.01
