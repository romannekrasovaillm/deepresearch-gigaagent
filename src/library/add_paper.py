"""Add new papers to existing library."""

import json
from pathlib import Path
from typing import Optional

from .converter import PaperMetadata
from .indexer import LibraryIndexer


def generate_paper_id(library_path: Path) -> str:
    """Generate next available paper ID."""
    indexer = LibraryIndexer(library_path)
    return indexer.get_next_paper_id()


def add_paper(
    docx_path: str | Path,
    library_path: str | Path,
    paper_id: Optional[str] = None,
) -> PaperMetadata:
    """
    Add a new paper to the library.

    Args:
        docx_path: Path to DOCX file
        library_path: Path to library root
        paper_id: Optional custom paper ID

    Returns:
        PaperMetadata for the added paper
    """
    docx_path = Path(docx_path)
    library_path = Path(library_path)

    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    indexer = LibraryIndexer(library_path)
    metadata = indexer.process_single_paper(docx_path, paper_id)

    return metadata


def add_papers_batch(
    docx_paths: list[Path],
    library_path: Path,
) -> list[PaperMetadata]:
    """Add multiple papers at once."""
    results = []
    for docx_path in docx_paths:
        try:
            metadata = add_paper(docx_path, library_path)
            results.append(metadata)
        except Exception as e:
            print(f"Failed to add {docx_path}: {e}")
    return results


def update_index_metadata(
    library_path: Path,
    paper_id: str,
    updates: dict,
) -> None:
    """Update metadata for existing paper in index."""
    index_path = library_path / "index.json"

    if not index_path.exists():
        raise FileNotFoundError("Library index not found")

    index = json.loads(index_path.read_text(encoding="utf-8"))

    for paper in index:
        if paper.get("paper_id") == paper_id:
            paper.update(updates)
            break
    else:
        raise ValueError(f"Paper not found: {paper_id}")

    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python add_paper.py <docx_path> <library_path>")
        sys.exit(1)

    docx_path = Path(sys.argv[1])
    library_path = Path(sys.argv[2])

    metadata = add_paper(docx_path, library_path)
    print(f"Added: {metadata.paper_id} - {metadata.title}")
