"""
MOA-style multi-objective reward functions.

Implements independent reward signals for different aspects
of agent performance: factual accuracy, process efficiency,
citation correctness, and code reasoning quality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import re
import json


@dataclass
class RewardScore:
    """Single reward component score."""
    name: str
    value: float  # 0.0 to 1.0
    weight: float
    explanation: str = ""


@dataclass
class CombinedReward:
    """Combined reward from multiple components."""
    total: float
    components: list[RewardScore]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "components": [
                {
                    "name": c.name,
                    "value": c.value,
                    "weight": c.weight,
                    "weighted": c.value * c.weight,
                    "explanation": c.explanation,
                }
                for c in self.components
            ]
        }


class RewardFunction(ABC):
    """Base class for reward functions."""

    name: str
    weight: float = 1.0

    @abstractmethod
    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> RewardScore:
        """
        Compute reward for a trajectory.

        Args:
            trajectory: List of (action, observation) pairs
            final_answer: Agent's final answer
            golden_answer: Expected correct answer (if available)
            metadata: Additional context (required papers, etc.)

        Returns:
            RewardScore with value and explanation
        """
        pass


class FactualReward(RewardFunction):
    """
    R_fact: Factual accuracy reward.

    Measures how well the agent's answer matches the expected answer.
    Uses LLM-as-Judge or string matching depending on configuration.
    """

    name = "R_fact"

    def __init__(
        self,
        weight: float = 0.4,
        use_llm_judge: bool = True,
        judge_model: Optional[str] = None
    ):
        self.weight = weight
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self._judge = None

    def set_judge(self, judge):
        """Set LLM judge instance."""
        self._judge = judge

    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> RewardScore:
        if not golden_answer:
            # No golden answer, can't compute factual accuracy
            return RewardScore(
                name=self.name,
                value=0.5,  # Neutral
                weight=self.weight,
                explanation="No golden answer provided"
            )

        if self.use_llm_judge and self._judge:
            score, explanation = self._judge.evaluate_factual(
                final_answer, golden_answer, metadata
            )
            return RewardScore(
                name=self.name,
                value=score,
                weight=self.weight,
                explanation=explanation
            )
        else:
            # Fallback to simple string matching
            score = self._string_similarity(final_answer, golden_answer)
            return RewardScore(
                name=self.name,
                value=score,
                weight=self.weight,
                explanation=f"String similarity: {score:.2f}"
            )

    def _string_similarity(self, answer: str, golden: str) -> float:
        """Simple word overlap similarity."""
        answer_words = set(answer.lower().split())
        golden_words = set(golden.lower().split())

        if not golden_words:
            return 0.0

        overlap = len(answer_words & golden_words)
        precision = overlap / len(answer_words) if answer_words else 0
        recall = overlap / len(golden_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return min(f1, 1.0)


class ProcessReward(RewardFunction):
    """
    R_process: Process efficiency reward.

    Penalizes inefficient search patterns:
    - Excessive tool calls
    - Repeated searches
    - Circular exploration
    """

    name = "R_process"

    def __init__(
        self,
        weight: float = 0.2,
        max_steps: int = 15,
        repeat_penalty: float = 0.1
    ):
        self.weight = weight
        self.max_steps = max_steps
        self.repeat_penalty = repeat_penalty

    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> RewardScore:
        n_steps = len(trajectory)

        # Base score from step efficiency
        if n_steps == 0:
            step_score = 0.0
        elif n_steps <= self.max_steps:
            # Reward finishing in fewer steps
            step_score = 1.0 - (n_steps / self.max_steps) * 0.5
        else:
            # Penalty for exceeding max steps
            step_score = 0.5 * (self.max_steps / n_steps)

        # Penalty for repeated actions
        action_hashes = []
        for step in trajectory:
            action = step.get("action", "")
            params = json.dumps(step.get("action_input", {}), sort_keys=True)
            action_hashes.append(f"{action}:{params}")

        unique_actions = len(set(action_hashes))
        if n_steps > 0:
            repeat_ratio = unique_actions / n_steps
        else:
            repeat_ratio = 1.0

        repeat_score = repeat_ratio

        # Combined score
        final_score = (step_score * 0.6 + repeat_score * 0.4)

        explanation_parts = [
            f"Steps: {n_steps}/{self.max_steps}",
            f"Unique actions: {unique_actions}/{n_steps}",
        ]

        return RewardScore(
            name=self.name,
            value=final_score,
            weight=self.weight,
            explanation="; ".join(explanation_parts)
        )


class CitationReward(RewardFunction):
    """
    R_citation: Citation accuracy reward.

    Verifies that the agent only cites files it actually read.
    Penalizes hallucinated citations.
    """

    name = "R_citation"

    def __init__(self, weight: float = 0.2):
        self.weight = weight

    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> RewardScore:
        # Extract files that were actually read
        read_files = set()
        for step in trajectory:
            action = step.get("action", "")
            action_input = step.get("action_input", {})

            if action in ["read_file_chunk", "verify_quote"]:
                path = action_input.get("path", "")
                if path:
                    read_files.add(self._normalize_path(path))

            # Also count grep results as "seen"
            if action == "grep_search":
                observation = step.get("observation", "")
                for line in observation.split('\n'):
                    if ':' in line:
                        file_part = line.split(':')[0]
                        read_files.add(self._normalize_path(file_part))

        # Extract paper_ids mentioned in final answer
        cited_papers = self._extract_citations(final_answer)

        if not cited_papers:
            # No citations to verify
            return RewardScore(
                name=self.name,
                value=1.0,
                weight=self.weight,
                explanation="No citations in answer"
            )

        # Check which citations are valid
        valid_citations = 0
        for paper_id in cited_papers:
            # Check if any read file contains this paper_id
            if any(paper_id in f for f in read_files):
                valid_citations += 1

        citation_accuracy = valid_citations / len(cited_papers)

        return RewardScore(
            name=self.name,
            value=citation_accuracy,
            weight=self.weight,
            explanation=f"Valid: {valid_citations}/{len(cited_papers)} citations"
        )

    def _normalize_path(self, path: str) -> str:
        """Normalize file path for comparison."""
        return path.strip().lower().replace('\\', '/')

    def _extract_citations(self, text: str) -> list[str]:
        """Extract paper_id patterns from text."""
        # Match patterns like paper_001, paper_0042, etc.
        pattern = r'paper_\d{3,4}'
        return list(set(re.findall(pattern, text.lower())))


class CodeReward(RewardFunction):
    """
    R_code: Code reasoning quality reward.

    Evaluates the quality and relevance of code execution:
    - Did the code execute successfully?
    - Was the code relevant to the task?
    - Did it produce meaningful results?
    """

    name = "R_code"

    def __init__(self, weight: float = 0.2):
        self.weight = weight

    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> RewardScore:
        # Find code execution steps
        code_steps = [
            step for step in trajectory
            if step.get("action") == "execute_python"
        ]

        if not code_steps:
            # Check if code was expected
            requires_code = metadata.get("requires_code", False) if metadata else False
            if requires_code:
                return RewardScore(
                    name=self.name,
                    value=0.0,
                    weight=self.weight,
                    explanation="Code was required but not used"
                )
            return RewardScore(
                name=self.name,
                value=0.5,  # Neutral if code wasn't needed
                weight=self.weight,
                explanation="No code execution (not required)"
            )

        # Evaluate code execution quality
        successful_executions = 0
        total_executions = len(code_steps)

        for step in code_steps:
            observation = step.get("observation", "")
            # Check for error indicators
            if "Error:" in observation or "Traceback" in observation:
                continue
            if "success" not in observation.lower() and "(no output)" in observation:
                continue
            successful_executions += 1

        execution_rate = successful_executions / total_executions if total_executions > 0 else 0

        # Check if code results were used in final answer
        code_outputs = []
        for step in code_steps:
            obs = step.get("observation", "")
            # Extract numbers or results from output
            numbers = re.findall(r'\d+\.?\d*', obs)
            code_outputs.extend(numbers)

        results_used = 0
        for output in code_outputs[:10]:  # Check first 10 outputs
            if output in final_answer:
                results_used += 1

        usage_score = min(results_used / max(len(code_outputs[:10]), 1), 1.0)

        # Combined score
        final_score = execution_rate * 0.6 + usage_score * 0.4

        return RewardScore(
            name=self.name,
            value=final_score,
            weight=self.weight,
            explanation=f"Executed: {successful_executions}/{total_executions}; Results used: {results_used}"
        )


class CompositeRewardFunction:
    """
    Combines multiple reward functions with weights.

    Implements MOA-style multi-objective optimization.
    """

    def __init__(self, reward_functions: list[RewardFunction]):
        self.reward_functions = reward_functions
        self._normalize_weights()

    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total_weight = sum(rf.weight for rf in self.reward_functions)
        if total_weight > 0:
            for rf in self.reward_functions:
                rf.weight = rf.weight / total_weight

    def compute(
        self,
        trajectory: list[dict],
        final_answer: str,
        golden_answer: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> CombinedReward:
        """Compute combined reward from all components."""
        scores = []

        for rf in self.reward_functions:
            score = rf.compute(trajectory, final_answer, golden_answer, metadata)
            scores.append(score)

        total = sum(s.value * s.weight for s in scores)

        return CombinedReward(total=total, components=scores)

    def add_function(self, reward_function: RewardFunction):
        """Add a new reward function."""
        self.reward_functions.append(reward_function)
        self._normalize_weights()


def create_default_reward_function(
    judge_model: Optional[str] = None,
    weights: Optional[dict] = None
) -> CompositeRewardFunction:
    """
    Create default MOA-style reward function.

    Args:
        judge_model: Model for LLM-as-Judge (e.g., "deepseek-v3")
        weights: Custom weights dict {R_fact: 0.4, R_process: 0.2, ...}

    Returns:
        CompositeRewardFunction with all components
    """
    default_weights = {
        "R_fact": 0.4,
        "R_process": 0.2,
        "R_citation": 0.2,
        "R_code": 0.2,
    }

    if weights:
        default_weights.update(weights)

    reward_functions = [
        FactualReward(
            weight=default_weights["R_fact"],
            use_llm_judge=judge_model is not None,
            judge_model=judge_model
        ),
        ProcessReward(weight=default_weights["R_process"]),
        CitationReward(weight=default_weights["R_citation"]),
        CodeReward(weight=default_weights["R_code"]),
    ]

    return CompositeRewardFunction(reward_functions)
