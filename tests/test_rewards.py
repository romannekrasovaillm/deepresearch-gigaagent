#!/usr/bin/env python3
"""Tests for reward system."""

import pytest
from src.rewards.reward_functions import (
    RewardConfig,
    RewardResult,
    R_fact,
    R_process,
    R_citation,
    R_code,
    compute_rewards,
)
from src.tools.tool_env import Episode


class TestRewardFunctions:
    """Test individual reward functions."""

    def test_r_fact_with_judge_score(self):
        """Test R_fact uses judge score when provided."""
        episode = Episode(question="test", final_answer="answer")
        reward = R_fact(episode, "golden", judge_score=0.8)
        assert reward == 0.8

    def test_r_fact_fallback_heuristic(self):
        """Test R_fact fallback when no judge score."""
        episode = Episode(
            question="test",
            final_answer="The model achieved 95% accuracy on benchmark"
        )
        reward = R_fact(episode, "The model achieved 95% accuracy on benchmark")
        assert reward > 0.5  # High overlap should score well

    def test_r_process_optimal(self):
        """Test R_process for optimal trajectory."""
        episode = Episode(
            question="test",
            turns=[
                {"action": "grep_search", "action_input": {"pattern": "test"}, "status": "success"},
                {"action": "read_file_chunk", "action_input": {"path": "test.txt"}, "status": "success"},
            ]
        )
        reward = R_process(episode)
        assert reward == 1.0  # No penalties

    def test_r_process_penalizes_extra_turns(self):
        """Test R_process penalizes too many turns."""
        turns = [
            {"action": f"tool_{i}", "action_input": {}, "status": "success"}
            for i in range(15)
        ]
        episode = Episode(question="test", turns=turns)
        config = RewardConfig(max_turns_before_penalty=10, penalty_per_extra_turn=0.05)

        reward = R_process(episode, config)
        assert reward < 1.0  # Should have penalty

    def test_r_process_penalizes_repeats(self):
        """Test R_process penalizes repeated actions."""
        episode = Episode(
            question="test",
            turns=[
                {"action": "grep_search", "action_input": {"pattern": "same"}, "status": "success"},
                {"action": "grep_search", "action_input": {"pattern": "same"}, "status": "success"},
            ]
        )
        reward = R_process(episode)
        assert reward < 1.0  # Should have repeat penalty

    def test_r_citation_verified(self):
        """Test R_citation for verified citations."""
        episode = Episode(
            question="test",
            turns=[
                {
                    "action": "read_file_chunk",
                    "action_input": {"path": "/mnt/library/by_id/paper_0001/text.txt"},
                    "status": "success",
                    "observation": "paper_0001 content",
                },
            ],
            final_answer="According to paper_0001, the method works."
        )
        reward = R_citation(episode)
        assert reward > 0.5  # Paper was accessed before citing

    def test_r_citation_unverified(self):
        """Test R_citation for unverified citations."""
        episode = Episode(
            question="test",
            turns=[],  # No tool usage
            final_answer="According to paper_0099, this is true."  # Paper never accessed
        )
        reward = R_citation(episode)
        assert reward < 0.5  # Citation not verified

    def test_r_code_successful(self):
        """Test R_code for successful code execution."""
        episode = Episode(
            question="test",
            turns=[
                {
                    "action": "execute_python",
                    "action_input": {"code": "1 + 1"},
                    "status": "success",
                    "observation": "[RESULT]\n2",
                },
            ]
        )
        reward = R_code(episode)
        assert reward > 0.5  # Successful code should score well

    def test_r_code_failed(self):
        """Test R_code for failed code execution."""
        episode = Episode(
            question="test",
            turns=[
                {
                    "action": "execute_python",
                    "action_input": {"code": "raise Exception()"},
                    "status": "error",
                    "observation": "[ERROR] Exception",
                },
            ]
        )
        reward = R_code(episode)
        assert reward < 0.5  # Failed code should score poorly

    def test_r_code_no_code_neutral(self):
        """Test R_code is neutral when no code used."""
        episode = Episode(
            question="test",
            turns=[
                {"action": "grep_search", "action_input": {}, "status": "success"},
            ]
        )
        reward = R_code(episode)
        assert reward == 0.5  # Neutral for non-code tasks


class TestComputeRewards:
    """Test combined reward computation."""

    def test_compute_rewards_all_components(self):
        """Test all reward components are computed."""
        episode = Episode(
            question="test",
            turns=[
                {
                    "action": "grep_search",
                    "action_input": {"pattern": "test"},
                    "status": "success",
                    "observation": "paper_0001/text.txt: found",
                },
            ],
            final_answer="Based on paper_0001, the answer is yes."
        )

        result = compute_rewards(
            episode,
            golden_answer="The answer is yes",
            judge_score=0.9,
        )

        assert isinstance(result, RewardResult)
        assert 0 <= result.r_fact <= 1
        assert 0 <= result.r_process <= 1
        assert 0 <= result.r_citation <= 1
        assert 0 <= result.r_code <= 1
        assert 0 <= result.total <= 1

    def test_compute_rewards_weights(self):
        """Test custom weights are applied."""
        episode = Episode(question="test", turns=[], final_answer="answer")

        config = RewardConfig(
            w_fact=1.0,
            w_process=0.0,
            w_citation=0.0,
            w_code=0.0,
        )

        result = compute_rewards(episode, "answer", judge_score=0.8, config=config)

        # Total should equal R_fact since other weights are 0
        assert abs(result.total - 0.8) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
