"""Combined reward computation (MOA-style)."""

from dataclasses import dataclass
from typing import Optional

from .fact_reward import FactReward, FactScore
from .process_reward import ProcessReward, ProcessScore
from .citation_reward import CitationReward, CitationScore
from .code_reward import CodeReward, CodeScore


@dataclass
class RewardWeights:
    """Weights for combining rewards."""

    R_fact: float = 0.40
    R_process: float = 0.20
    R_citation: float = 0.20
    R_code: float = 0.20


@dataclass
class CombinedScore:
    """Combined reward with all components."""

    fact_score: FactScore
    process_score: ProcessScore
    citation_score: CitationScore
    code_score: CodeScore
    weights: RewardWeights
    total: float
    normalized: float
    clipped: float


class CombinedReward:
    """
    MOA-style multi-objective reward computation.

    Combines:
    - R_fact: Factual accuracy (LLM-Judge)
    - R_process: Search efficiency
    - R_citation: Citation accuracy
    - R_code: Code reasoning quality
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        judge_model: str = "gpt-4o",
        library_path: Optional[str] = None,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        success_bonus: float = 0.2,
        failure_penalty: float = -0.3,
        timeout_penalty: float = -0.5,
    ):
        self.weights = weights or RewardWeights()
        self.clip_range = clip_range
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.timeout_penalty = timeout_penalty

        # Initialize reward functions
        self.fact_reward = FactReward(judge_model=judge_model)
        self.process_reward = ProcessReward()
        self.citation_reward = CitationReward(library_path=library_path)
        self.code_reward = CodeReward(judge_model=judge_model)

    def compute(
        self,
        question: str,
        golden_answer: str,
        agent_answer: str,
        env_metrics: dict,
        required_papers: Optional[list[str]] = None,
        timeout: bool = False,
    ) -> CombinedScore:
        """
        Compute combined reward.

        Args:
            question: Original research question
            golden_answer: Reference answer
            agent_answer: Agent's answer
            env_metrics: Metrics from environment (from get_metrics())
            required_papers: Papers that should be cited
            timeout: Whether episode ended due to timeout

        Returns:
            CombinedScore with all components
        """
        # Compute individual rewards
        fact_score = self.fact_reward.compute(question, golden_answer, agent_answer)

        process_score = self.process_reward.compute(
            total_steps=env_metrics.get("total_turns", 0),
            search_queries=env_metrics.get("search_queries", []),
            opened_files=env_metrics.get("opened_files", []),
            required_files=required_papers,
            found_answer=bool(agent_answer),
        )

        citation_score = self.citation_reward.compute(
            agent_answer=agent_answer,
            opened_files=env_metrics.get("opened_files", []),
            required_papers=required_papers,
        )

        code_score = self.code_reward.compute(
            question=question,
            code_executions=env_metrics.get("code_executions", []),
            agent_answer=agent_answer,
        )

        # Weighted combination
        weighted_sum = (
            self.weights.R_fact * fact_score.total
            + self.weights.R_process * process_score.total
            + self.weights.R_citation * citation_score.total
            + self.weights.R_code * code_score.total
        )

        # Outcome adjustments
        if timeout:
            weighted_sum += self.timeout_penalty
        elif fact_score.accuracy >= 0.9:
            weighted_sum += self.success_bonus
        elif fact_score.accuracy <= 0.2:
            weighted_sum += self.failure_penalty

        # Normalize to [0, 1]
        raw_min = -2.0  # Theoretical minimum
        raw_max = 2.0  # Theoretical maximum
        normalized = (weighted_sum - raw_min) / (raw_max - raw_min)
        normalized = max(0, min(1, normalized))

        # Clip to range
        clipped = max(self.clip_range[0], min(self.clip_range[1], weighted_sum))

        return CombinedScore(
            fact_score=fact_score,
            process_score=process_score,
            citation_score=citation_score,
            code_score=code_score,
            weights=self.weights,
            total=weighted_sum,
            normalized=normalized,
            clipped=clipped,
        )

    def get_reward_breakdown(self, score: CombinedScore) -> dict:
        """Get detailed breakdown of reward components."""
        return {
            "R_fact": {
                "weight": self.weights.R_fact,
                "value": score.fact_score.total,
                "weighted": self.weights.R_fact * score.fact_score.total,
                "accuracy": score.fact_score.accuracy,
                "completeness": score.fact_score.completeness,
                "relevance": score.fact_score.relevance,
            },
            "R_process": {
                "weight": self.weights.R_process,
                "value": score.process_score.total,
                "weighted": self.weights.R_process * score.process_score.total,
                "details": score.process_score.details,
            },
            "R_citation": {
                "weight": self.weights.R_citation,
                "value": score.citation_score.total,
                "weighted": self.weights.R_citation * score.citation_score.total,
                "correct": score.citation_score.correct_citations,
                "incorrect": score.citation_score.incorrect_citations,
            },
            "R_code": {
                "weight": self.weights.R_code,
                "value": score.code_score.total,
                "weighted": self.weights.R_code * score.code_score.total,
                "success": score.code_score.execution_success,
                "failure": score.code_score.execution_failure,
            },
            "total": score.total,
            "normalized": score.normalized,
            "clipped": score.clipped,
        }
