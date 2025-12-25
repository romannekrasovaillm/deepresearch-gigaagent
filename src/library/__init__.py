"""Library processing module - DOCX to Markdown conversion."""

from .convert import (
    extract_metadata,
    convert_to_markdown,
    split_by_sections,
    process_paper,
    build_library,
)
from .add_paper import add_paper, generate_paper_id

__all__ = [
    "extract_metadata",
    "convert_to_markdown",
    "split_by_sections",
    "process_paper",
    "build_library",
    "add_paper",
    "generate_paper_id",
]
