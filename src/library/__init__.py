"""Library management module - DOCX conversion and indexing."""

from .converter import DocxConverter, convert_to_markdown, extract_metadata
from .indexer import LibraryIndexer, build_library
from .add_paper import add_paper, generate_paper_id

__all__ = [
    "DocxConverter",
    "convert_to_markdown",
    "extract_metadata",
    "LibraryIndexer",
    "build_library",
    "add_paper",
    "generate_paper_id",
]
