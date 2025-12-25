#!/usr/bin/env python3
"""CLI script for converting DOCX papers to structured library."""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from library import build_library, add_paper

app = typer.Typer(help="Convert DOCX papers to structured library")
console = Console()


@app.command("build")
def build(
    input_dir: Path = typer.Argument(..., help="Directory with DOCX files"),
    output_dir: Path = typer.Argument(..., help="Output library directory"),
    pattern: str = typer.Option("*.docx", "--pattern", "-p", help="File pattern"),
):
    """Build library from directory of DOCX files."""
    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Building library from {input_dir}[/bold]")
    results = build_library(input_dir, output_dir)
    console.print(f"[green]Done! Processed {len(results)} papers[/green]")


@app.command("add")
def add_single(
    docx_path: Path = typer.Argument(..., help="Path to DOCX file"),
    library_path: Path = typer.Argument(..., help="Library directory"),
    paper_id: str = typer.Option(None, "--id", help="Custom paper ID"),
):
    """Add a single paper to existing library."""
    if not docx_path.exists():
        console.print(f"[red]File not found: {docx_path}[/red]")
        raise typer.Exit(1)

    metadata = add_paper(docx_path, library_path, paper_id)
    console.print(f"[green]Added: {metadata.paper_id} - {metadata.title}[/green]")


@app.command("info")
def info(
    library_path: Path = typer.Argument(..., help="Library directory"),
):
    """Show library information."""
    import json

    index_path = library_path / "index.json"
    if not index_path.exists():
        console.print(f"[red]Library not found: {library_path}[/red]")
        raise typer.Exit(1)

    index = json.loads(index_path.read_text(encoding="utf-8"))

    console.print(f"[bold]Library: {library_path}[/bold]")
    console.print(f"Total papers: {len(index)}")
    console.print("\nRecent papers:")
    for paper in index[-5:]:
        console.print(f"  {paper['paper_id']}: {paper['title'][:60]}")


def main():
    app()


if __name__ == "__main__":
    main()
