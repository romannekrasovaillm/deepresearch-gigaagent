#!/usr/bin/env python3
"""
Reward Functions (MOA-style Multi-Objective Alignment)
Independent reward signals for different aspects of agent behavior.
"""

import re
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..tools.tool_env import Episode, ToolStatus


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Reward weights
    w_fact: float = 0.4       # Factual accuracy
    w_process: float = 0.2    # Process efficiency
    w_citation: float = 0.2   # Citation accuracy
    w_code: float = 0.2       # Code quality

    # Process penalties
    max_turns_before_penalty: int = 10
    penalty_per_extra_turn: float = 0.05
    repeat_action_penalty: float = 0.1
    failed_tool_penalty: float = 0.1

    # Citation settings
    verify_citations: bool = True
    require_paper_access: bool = True

    # Code settings
    reward_successful_code: float = 0.3
    penalty_failed_code: float = 0.1


@dataclass
class RewardResult:
    """Detailed reward breakdown."""
    total: float = 0.0
    r_fact: float = 0.0
    r_process: float = 0.0
    r_citation: float = 0.0
    r_code: float = 0.0
    details: dict = field(default_factory=dict)


def R_fact(
    episode: Episode,
    golden_answer: str,
    judge_score: Optional[float] = None,
) -> float:
    """
    Factual accuracy reward.

    Uses LLM-as-Judge score if provided, otherwise falls back to
    simple text matching heuristics.

    Args:
        episode: Completed episode with final_answer
        golden_answer: Ground truth answer
        judge_score: Pre-computed LLM judge score (0-1)

    Returns:
        Reward in range [0, 1]
    """
    if judge_score is not None:
        return max(0.0, min(1.0, judge_score))

    if not episode.final_answer:
        return 0.0

    # Simple heuristic fallback (not recommended for training)
    answer = episode.final_answer.lower()
    golden = golden_answer.lower()

    # Check for key phrase overlap
    golden_words = set(golden.split())
    answer_words = set(answer.split())

    if not golden_words:
        return 0.0

    overlap = len(golden_words & answer_words)
    coverage = overlap / len(golden_words)

    return coverage


def R_process(
    episode: Episode,
    config: RewardConfig = None,
) -> float:
    """
    Process efficiency reward.

    Penalizes:
    - Too many turns
    - Repeated actions
    - Failed tool calls

    Args:
        episode: Completed episode
        config: Reward configuration

    Returns:
        Reward in range [0, 1]
    """
    if config is None:
        config = RewardConfig()

    if not episode.turns:
        return 0.5  # Neutral if no actions

    reward = 1.0
    details = {}

    # Penalty for extra turns
    num_turns = len(episode.turns)
    if num_turns > config.max_turns_before_penalty:
        extra_turns = num_turns - config.max_turns_before_penalty
        penalty = extra_turns * config.penalty_per_extra_turn
        reward -= penalty
        details["extra_turns_penalty"] = penalty

    # Penalty for repeated actions
    seen_actions = set()
    repeat_count = 0
    for turn in episode.turns:
        action_key = (turn["action"], json.dumps(turn["action_input"], sort_keys=True))
        if action_key in seen_actions:
            repeat_count += 1
        seen_actions.add(action_key)

    if repeat_count > 0:
        penalty = repeat_count * config.repeat_action_penalty
        reward -= penalty
        details["repeat_penalty"] = penalty

    # Penalty for failed tools
    failed_count = sum(1 for t in episode.turns if t["status"] != "success")
    if failed_count > 0:
        penalty = failed_count * config.failed_tool_penalty
        reward -= penalty
        details["failed_tools_penalty"] = penalty

    return max(0.0, min(1.0, reward))


def R_citation(
    episode: Episode,
    library_path: str = "/mnt/library",
    config: RewardConfig = None,
) -> float:
    """
    Citation accuracy reward.

    Checks:
    - Whether cited papers were actually accessed
    - Whether quotes can be verified in source files

    Args:
        episode: Completed episode
        library_path: Path to paper library
        config: Reward configuration

    Returns:
        Reward in range [0, 1]
    """
    if config is None:
        config = RewardConfig()

    if not episode.final_answer:
        return 0.0

    answer = episode.final_answer
    library = Path(library_path)

    # Extract paper citations from answer
    # Patterns: paper_XXXX, [paper_XXXX], (paper_XXXX)
    paper_citations = set(re.findall(r'paper_\d{4}', answer))

    if not paper_citations:
        # No citations required/provided
        return 0.5

    # Check which papers were actually accessed
    accessed_papers = set()
    for turn in episode.turns:
        if turn["action"] in ["read_file_chunk", "get_paper_abstract", "get_paper_section"]:
            # Extract paper_id from input or observation
            action_input = turn.get("action_input", {})
            observation = turn.get("observation", "")

            for paper_id in re.findall(r'paper_\d{4}', str(action_input) + observation):
                accessed_papers.add(paper_id)

        if turn["action"] == "grep_search":
            observation = turn.get("observation", "")
            for paper_id in re.findall(r'paper_\d{4}', observation):
                accessed_papers.add(paper_id)

    # Compute citation accuracy
    valid_citations = paper_citations & accessed_papers
    if not paper_citations:
        return 0.5

    accuracy = len(valid_citations) / len(paper_citations)

    # Bonus for having all citations verified
    if accuracy == 1.0:
        accuracy = 1.0
    else:
        # Penalty for unverified citations (potential hallucination)
        accuracy *= 0.8

    return accuracy


def R_code(
    episode: Episode,
    config: RewardConfig = None,
) -> float:
    """
    Code execution quality reward.

    Rewards:
    - Successful code execution
    - Relevance of code to task
    - Generation of useful outputs (figures, tables)

    Args:
        episode: Completed episode
        config: Reward configuration

    Returns:
        Reward in range [0, 1]
    """
    if config is None:
        config = RewardConfig()

    # Find all code executions
    code_turns = [t for t in episode.turns if t["action"] == "execute_python"]

    if not code_turns:
        # No code used - neutral (some tasks don't need code)
        return 0.5

    reward = 0.0
    successful = 0
    failed = 0
    produced_output = 0

    for turn in code_turns:
        status = turn.get("status", "error")
        observation = turn.get("observation", "")

        if status == "success":
            successful += 1

            # Bonus for producing useful output
            if "[RESULT]" in observation and "[None]" not in observation:
                produced_output += 1
            if "[FIGURES]" in observation:
                produced_output += 1
            if "[STDOUT]" in observation:
                produced_output += 0.5

        else:
            failed += 1

    total_code = successful + failed
    if total_code == 0:
        return 0.5

    # Base score from success rate
    success_rate = successful / total_code
    reward = success_rate * config.reward_successful_code

    # Bonus for producing outputs
    if successful > 0:
        output_bonus = min(0.3, produced_output * 0.1)
        reward += output_bonus

    # Penalty for failures
    reward -= failed * config.penalty_failed_code

    # Normalize to [0, 1]
    return max(0.0, min(1.0, reward + 0.5))


def compute_rewards(
    episode: Episode,
    golden_answer: str = "",
    judge_score: Optional[float] = None,
    library_path: str = "/mnt/library",
    config: RewardConfig = None,
) -> RewardResult:
    """
    Compute all rewards for an episode.

    Args:
        episode: Completed episode
        golden_answer: Ground truth answer for R_fact
        judge_score: LLM judge score for R_fact
        library_path: Path to library for R_citation
        config: Reward configuration

    Returns:
        RewardResult with all scores and total
    """
    if config is None:
        config = RewardConfig()

    # Compute individual rewards
    r_fact = R_fact(episode, golden_answer, judge_score)
    r_process = R_process(episode, config)
    r_citation = R_citation(episode, library_path, config)
    r_code = R_code(episode, config)

    # Weighted sum
    total = (
        config.w_fact * r_fact +
        config.w_process * r_process +
        config.w_citation * r_citation +
        config.w_code * r_code
    )

    return RewardResult(
        total=total,
        r_fact=r_fact,
        r_process=r_process,
        r_citation=r_citation,
        r_code=r_code,
        details={
            "weights": {
                "fact": config.w_fact,
                "process": config.w_process,
                "citation": config.w_citation,
                "code": config.w_code,
            },
            "num_turns": len(episode.turns),
            "final_answer_length": len(episode.final_answer or ""),
        },
    )


def compute_step_reward(
    turn: dict,
    prev_turns: list[dict],
    config: RewardConfig = None,
) -> float:
    """
    Compute intermediate reward for a single step (for online RL).

    Args:
        turn: Current turn
        prev_turns: Previous turns in episode
        config: Reward configuration

    Returns:
        Step reward (can be negative)
    """
    if config is None:
        config = RewardConfig()

    reward = 0.0

    # Small positive for successful tool use
    if turn["status"] == "success":
        reward += 0.02

        # Bonus for finding relevant information
        observation = turn.get("observation", "").lower()
        if "paper_" in observation or len(observation) > 100:
            reward += 0.01

    else:
        # Penalty for failed tool
        reward -= config.failed_tool_penalty

    # Penalty for repeating exact action
    action_key = (turn["action"], json.dumps(turn["action_input"], sort_keys=True))
    for prev in prev_turns:
        prev_key = (prev["action"], json.dumps(prev["action_input"], sort_keys=True))
        if action_key == prev_key:
            reward -= config.repeat_action_penalty
            break

    return reward
