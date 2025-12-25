"""Data generation module for synthetic training data."""

from .prompts import (
    RETRIEVAL_PROMPT,
    MULTIHOP_PROMPT,
    COMPUTATION_PROMPT,
    SYNTHESIS_PROMPT,
    SFT_DEMO_PROMPT,
)
from .synthetic_generator import (
    SyntheticGenerator,
    TaskType,
    GeneratedTask,
    generate_dataset,
)

__all__ = [
    "RETRIEVAL_PROMPT",
    "MULTIHOP_PROMPT",
    "COMPUTATION_PROMPT",
    "SYNTHESIS_PROMPT",
    "SFT_DEMO_PROMPT",
    "SyntheticGenerator",
    "TaskType",
    "GeneratedTask",
    "generate_dataset",
]
