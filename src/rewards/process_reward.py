"""R_process: Search efficiency reward."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessScore:
    """Process efficiency scores."""

    step_penalty: float
    repeat_penalty: float
    irrelevant_penalty: float
    efficiency_bonus: float
    total: float
    details: str = ""


class ProcessReward:
    """Reward for search efficiency."""

    def __init__(
        self,
        penalty_per_step: float = 0.02,
        penalty_repeated_search: float = 0.1,
        penalty_irrelevant_file: float = 0.05,
        bonus_direct_hit: float = 0.1,
        bonus_efficient_path: float = 0.15,
        max_steps_before_penalty: int = 10,
        optimal_steps_threshold: int = 5,
    ):
        self.penalty_per_step = penalty_per_step
        self.penalty_repeated_search = penalty_repeated_search
        self.penalty_irrelevant_file = penalty_irrelevant_file
        self.bonus_direct_hit = bonus_direct_hit
        self.bonus_efficient_path = bonus_efficient_path
        self.max_steps_before_penalty = max_steps_before_penalty
        self.optimal_steps_threshold = optimal_steps_threshold

    def compute(
        self,
        total_steps: int,
        search_queries: list[str],
        opened_files: list[str],
        required_files: Optional[list[str]] = None,
        found_answer: bool = True,
    ) -> ProcessScore:
        """
        Compute process efficiency reward.

        Args:
            total_steps: Total number of actions taken
            search_queries: List of search queries made
            opened_files: List of files that were read
            required_files: Files that should have been read
            found_answer: Whether agent provided an answer

        Returns:
            ProcessScore with breakdown
        """
        details = []

        # Step penalty (only after threshold)
        step_penalty = 0.0
        if total_steps > self.max_steps_before_penalty:
            excess_steps = total_steps - self.max_steps_before_penalty
            step_penalty = excess_steps * self.penalty_per_step
            details.append(f"Step penalty: -{step_penalty:.3f} ({excess_steps} excess steps)")

        # Repeated search penalty
        unique_queries = set(search_queries)
        repeated = len(search_queries) - len(unique_queries)
        repeat_penalty = repeated * self.penalty_repeated_search
        if repeated > 0:
            details.append(f"Repeat penalty: -{repeat_penalty:.3f} ({repeated} repeated)")

        # Irrelevant file penalty
        irrelevant_penalty = 0.0
        if required_files:
            required_set = set(required_files)
            opened_set = set(opened_files)
            irrelevant = opened_set - required_set
            irrelevant_penalty = len(irrelevant) * self.penalty_irrelevant_file
            if irrelevant:
                details.append(
                    f"Irrelevant penalty: -{irrelevant_penalty:.3f} ({len(irrelevant)} files)"
                )

        # Efficiency bonuses
        efficiency_bonus = 0.0

        # Direct hit bonus (found answer quickly)
        if found_answer and total_steps <= self.optimal_steps_threshold:
            efficiency_bonus += self.bonus_direct_hit
            details.append(f"Direct hit bonus: +{self.bonus_direct_hit:.3f}")

        # Efficient path bonus (read only required files)
        if required_files:
            required_set = set(required_files)
            opened_set = set(opened_files)
            if required_set <= opened_set and len(opened_set) == len(required_set):
                efficiency_bonus += self.bonus_efficient_path
                details.append(f"Efficient path bonus: +{self.bonus_efficient_path:.3f}")

        # Total (positive is good)
        total = efficiency_bonus - step_penalty - repeat_penalty - irrelevant_penalty

        return ProcessScore(
            step_penalty=step_penalty,
            repeat_penalty=repeat_penalty,
            irrelevant_penalty=irrelevant_penalty,
            efficiency_bonus=efficiency_bonus,
            total=total,
            details="; ".join(details) if details else "No adjustments",
        )


def compute_process_reward(
    total_steps: int,
    search_queries: list[str],
    opened_files: list[str],
    required_files: Optional[list[str]] = None,
) -> float:
    """Convenience function for process reward."""
    reward = ProcessReward()
    score = reward.compute(total_steps, search_queries, opened_files, required_files)
    return score.total
