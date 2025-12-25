"""Synthetic dataset generation module."""

from .synthetic import (
    DatasetGenerator,
    TaskType,
    GeneratedSample,
    generate_dataset,
)

from .prompts import (
    RETRIEVAL_PROMPT,
    MULTIHOP_PROMPT,
    COMPUTATION_PROMPT,
    SYNTHESIS_PROMPT,
    SFT_DEMO_PROMPT,
)

__all__ = [
    "DatasetGenerator",
    "TaskType",
    "GeneratedSample",
    "generate_dataset",
    "RETRIEVAL_PROMPT",
    "MULTIHOP_PROMPT",
    "COMPUTATION_PROMPT",
    "SYNTHESIS_PROMPT",
    "SFT_DEMO_PROMPT",
]
