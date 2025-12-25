"""
Add Paper Module - Incrementally add new papers to existing library.

Allows adding new papers without reprocessing the entire library.
New papers are immediately available to the agent via grep/find.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .convert import process_paper

console = Console()
app = typer.Typer()


def generate_paper_id(library_dir: Path) -> str:
    """
    Generate next available paper ID.

    Args:
        library_dir: Path to library directory

    Returns:
        Next paper ID in format paper_XXXX
    """
    index_path = library_dir / "index.json"

    if not index_path.exists():
        return "paper_0001"

    index = json.loads(index_path.read_text())

    # Find highest existing ID
    max_id = 0
    for paper in index:
        pid = paper.get("paper_id", "")
        if pid.startswith("paper_"):
            try:
                num = int(pid.split("_")[1])
                max_id = max(max_id, num)
            except (ValueError, IndexError):
                continue

    return f"paper_{max_id + 1:04d}"


def add_paper(
    docx_path: Path,
    library_dir: Path,
    paper_id: Optional[str] = None,
    topic: Optional[str] = None,
) -> dict:
    """
    Add a single new paper to the library.

    Args:
        docx_path: Path to the DOCX file
        library_dir: Path to existing library
        paper_id: Custom paper ID (auto-generated if None)
        topic: Topic category for the paper

    Returns:
        Metadata dictionary for the new paper
    """
    docx_path = Path(docx_path)
    library_dir = Path(library_dir)

    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    # Ensure library exists
    by_id_dir = library_dir / "by_id"
    by_id_dir.mkdir(parents=True, exist_ok=True)

    # Generate or validate paper ID
    if paper_id is None:
        paper_id = generate_paper_id(library_dir)
    elif (by_id_dir / paper_id).exists():
        raise ValueError(f"Paper ID already exists: {paper_id}")

    console.print(f"[blue]Adding paper: {docx_path.name} -> {paper_id}[/blue]")

    # Process the paper
    metadata = process_paper(docx_path, by_id_dir, paper_id)
    metadata["added_at"] = datetime.now().isoformat()

    if topic:
        metadata["topic"] = topic
        # Create symlink in topic directory
        topic_dir = library_dir / "by_topic" / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        link_path = topic_dir / paper_id
        if not link_path.exists():
            link_path.symlink_to(by_id_dir / paper_id)

    # Update index
    index_path = library_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []

    index.append(metadata)
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    console.print(f"[green]Successfully added: {paper_id}[/green]")
    console.print(f"  Title: {metadata.get('title', 'Unknown')[:60]}")
    console.print(f"  Sections: {len(metadata.get('sections', []))}")

    return metadata


def add_papers_batch(
    input_dir: Path,
    library_dir: Path,
    topic: Optional[str] = None,
    pattern: str = "*.docx",
) -> list[dict]:
    """
    Add multiple new papers from a directory.

    Args:
        input_dir: Directory with new DOCX files
        library_dir: Path to existing library
        topic: Topic category for all papers
        pattern: Glob pattern for files

    Returns:
        List of metadata dictionaries for added papers
    """
    input_dir = Path(input_dir)
    library_dir = Path(library_dir)

    docx_files = list(input_dir.glob(pattern))

    if not docx_files:
        console.print(f"[yellow]No files matching {pattern} in {input_dir}[/yellow]")
        return []

    console.print(f"[blue]Adding {len(docx_files)} papers...[/blue]")

    results = []
    for docx_path in docx_files:
        try:
            metadata = add_paper(docx_path, library_dir, topic=topic)
            results.append(metadata)
        except Exception as e:
            console.print(f"[red]Error adding {docx_path.name}: {e}[/red]")

    console.print(f"[green]Added {len(results)}/{len(docx_files)} papers[/green]")
    return results


@app.command("add")
def cli_add(
    docx_path: Path = typer.Argument(..., help="Path to DOCX file"),
    library_dir: Path = typer.Argument(..., help="Library directory"),
    paper_id: Optional[str] = typer.Option(None, help="Custom paper ID"),
    topic: Optional[str] = typer.Option(None, help="Topic category"),
):
    """Add a single paper to the library."""
    add_paper(docx_path, library_dir, paper_id, topic)


@app.command("batch")
def cli_batch(
    input_dir: Path = typer.Argument(..., help="Directory with DOCX files"),
    library_dir: Path = typer.Argument(..., help="Library directory"),
    topic: Optional[str] = typer.Option(None, help="Topic category for all"),
    pattern: str = typer.Option("*.docx", help="File pattern"),
):
    """Add multiple papers from a directory."""
    add_papers_batch(input_dir, library_dir, topic, pattern)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
