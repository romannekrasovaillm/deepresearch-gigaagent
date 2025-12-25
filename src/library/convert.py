"""
Library Conversion Module - DOCX to Markdown with metadata extraction.

Converts scientific papers from DOCX format to structured Markdown library
optimized for grep/bash-based retrieval by Deep Research agent.
"""

import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from docx import Document

console = Console()
app = typer.Typer()


def extract_metadata(docx_path: Path) -> dict:
    """
    Extract metadata from DOCX file using python-docx.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        Dictionary with title, author, created date, and source file
    """
    try:
        doc = Document(docx_path)
        props = doc.core_properties

        # First paragraph is usually the title
        title = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                title = text
                break

        return {
            "title": title or props.title or docx_path.stem,
            "author": props.author or "Unknown",
            "created": str(props.created) if props.created else None,
            "modified": str(props.modified) if props.modified else None,
            "source_file": docx_path.name,
            "keywords": props.keywords or "",
            "subject": props.subject or "",
        }
    except Exception as e:
        console.print(f"[yellow]Warning: Could not extract metadata from {docx_path}: {e}[/yellow]")
        return {
            "title": docx_path.stem,
            "author": "Unknown",
            "source_file": docx_path.name,
        }


def convert_to_markdown(docx_path: Path, extract_media_dir: Optional[Path] = None) -> str:
    """
    Convert DOCX to Markdown using pandoc with optimal settings.

    Args:
        docx_path: Path to the DOCX file
        extract_media_dir: Directory to extract images (optional)

    Returns:
        Markdown content as string
    """
    cmd = [
        "pandoc",
        str(docx_path),
        "-t", "markdown",
        "--wrap=none",  # No line wrapping for grep compatibility
        "--standalone",
    ]

    if extract_media_dir:
        extract_media_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--extract-media", str(extract_media_dir)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        console.print(f"[red]Timeout converting {docx_path}[/red]")
        return fallback_conversion(docx_path)
    except subprocess.CalledProcessError as e:
        console.print(f"[yellow]Pandoc failed for {docx_path}, using fallback: {e}[/yellow]")
        return fallback_conversion(docx_path)
    except FileNotFoundError:
        console.print("[red]Pandoc not found! Please install: sudo apt install pandoc[/red]")
        return fallback_conversion(docx_path)


def fallback_conversion(docx_path: Path) -> str:
    """
    Fallback conversion using python-docx when pandoc fails.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        Plain text extracted from document
    """
    try:
        doc = Document(docx_path)
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Detect headings by style
                if para.style and "heading" in para.style.name.lower():
                    level = 1
                    if "2" in para.style.name:
                        level = 2
                    elif "3" in para.style.name:
                        level = 3
                    lines.append(f"{'#' * level} {text}")
                else:
                    lines.append(text)
        return "\n\n".join(lines)
    except Exception as e:
        console.print(f"[red]Fallback conversion failed for {docx_path}: {e}[/red]")
        return ""


def split_by_sections(markdown: str) -> dict[str, str]:
    """
    Split Markdown content into sections by ## headers.

    Args:
        markdown: Full markdown content

    Returns:
        Dictionary mapping section names to content
    """
    sections = {}
    current = "preamble"
    sections[current] = ""

    for line in markdown.split('\n'):
        # Match ## headings (level 2)
        if line.startswith('## '):
            current = line[3:].strip().lower()
            # Normalize section name for filesystem
            current = re.sub(r'[^a-zа-яё0-9]+', '_', current)
            current = current.strip('_')[:50]  # Limit length
            sections[current] = ""
        else:
            sections[current] += line + '\n'

    # Clean up empty sections
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def extract_abstract(markdown: str, sections: dict[str, str]) -> str:
    """
    Extract abstract from markdown content.

    Args:
        markdown: Full markdown content
        sections: Pre-split sections dictionary

    Returns:
        Abstract text or empty string
    """
    # Try common section names
    for key in ['abstract', 'аннотация', 'summary', 'краткое_содержание']:
        if key in sections:
            return sections[key]

    # Try to find abstract pattern in text
    abstract_match = re.search(
        r'(?:^|\n)(?:abstract|аннотация)[:\s]*\n?(.*?)(?=\n#|\n\n\n|\Z)',
        markdown,
        re.IGNORECASE | re.DOTALL
    )
    if abstract_match:
        return abstract_match.group(1).strip()

    return ""


def process_paper(docx_path: Path, output_dir: Path, paper_id: str) -> dict:
    """
    Full processing pipeline for a single paper.

    Args:
        docx_path: Path to the source DOCX file
        output_dir: Base output directory (by_id folder)
        paper_id: Unique identifier for this paper

    Returns:
        Metadata dictionary for this paper
    """
    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract metadata
    metadata = extract_metadata(docx_path)
    metadata["paper_id"] = paper_id

    # 2. Convert to Markdown
    media_dir = paper_dir / "media"
    markdown = convert_to_markdown(docx_path, media_dir)

    if not markdown:
        metadata["error"] = "Conversion failed"
        return metadata

    # Save full text
    (paper_dir / "full_text.md").write_text(markdown, encoding="utf-8")

    # 3. Split into sections
    sections = split_by_sections(markdown)
    sections_dir = paper_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    for name, content in sections.items():
        section_file = sections_dir / f"{name}.txt"
        section_file.write_text(content, encoding="utf-8")

    metadata["sections"] = list(sections.keys())

    # 4. Extract and save abstract separately (for quick lookup)
    abstract = extract_abstract(markdown, sections)
    (paper_dir / "abstract.txt").write_text(abstract, encoding="utf-8")
    metadata["has_abstract"] = bool(abstract)

    # 5. Save metadata
    (paper_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 6. Calculate stats
    metadata["word_count"] = len(markdown.split())
    metadata["char_count"] = len(markdown)

    return metadata


def build_library(
    input_dir: Path,
    output_dir: Path,
    max_workers: int = 4,
    file_pattern: str = "*.docx"
) -> list[dict]:
    """
    Convert all DOCX files to structured library.

    Args:
        input_dir: Directory with source DOCX files
        output_dir: Output library directory
        max_workers: Number of parallel workers
        file_pattern: Glob pattern for input files

    Returns:
        List of metadata dictionaries for all processed papers
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        return []

    # Find all DOCX files
    docx_files = sorted(input_dir.glob(file_pattern))

    if not docx_files:
        console.print(f"[yellow]No {file_pattern} files found in {input_dir}[/yellow]")
        return []

    console.print(f"[green]Found {len(docx_files)} files to process[/green]")

    # Create output structure
    by_id_dir = output_dir / "by_id"
    by_id_dir.mkdir(parents=True, exist_ok=True)

    index = []
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting papers...", total=len(docx_files))

        # Process files (can be parallelized, but keeping sequential for stability)
        for i, docx_path in enumerate(docx_files, 1):
            paper_id = f"paper_{i:04d}"
            progress.update(task, description=f"Processing {docx_path.name[:30]}...")

            try:
                metadata = process_paper(docx_path, by_id_dir, paper_id)
                index.append(metadata)

                if "error" in metadata:
                    errors.append((paper_id, metadata["error"]))

            except Exception as e:
                console.print(f"[red]Error processing {docx_path}: {e}[/red]")
                errors.append((paper_id, str(e)))
                index.append({
                    "paper_id": paper_id,
                    "source_file": docx_path.name,
                    "error": str(e),
                })

            progress.advance(task)

    # Save master index
    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Create topic placeholder directories
    for topic in ["rl", "alignment", "scaling", "reasoning", "agents", "other"]:
        (output_dir / "by_topic" / topic).mkdir(parents=True, exist_ok=True)

    # Summary
    console.print(f"\n[green]Done! Processed {len(index)} papers[/green]")
    if errors:
        console.print(f"[yellow]Errors: {len(errors)}[/yellow]")
        for pid, err in errors[:5]:
            console.print(f"  - {pid}: {err[:50]}")

    console.print(f"[blue]Library saved to: {output_dir}[/blue]")
    console.print(f"[blue]Index file: {output_dir / 'index.json'}[/blue]")

    return index


@app.command()
def convert(
    input_dir: Path = typer.Argument(..., help="Directory with DOCX files"),
    output_dir: Path = typer.Argument(..., help="Output library directory"),
    workers: int = typer.Option(4, help="Number of parallel workers"),
    pattern: str = typer.Option("*.docx", help="File pattern to match"),
):
    """Convert DOCX papers to structured Markdown library."""
    build_library(input_dir, output_dir, workers, pattern)


@app.command()
def verify(library_dir: Path = typer.Argument(..., help="Library directory to verify")):
    """Verify library structure and integrity."""
    library_dir = Path(library_dir)

    if not (library_dir / "index.json").exists():
        console.print("[red]No index.json found![/red]")
        return

    index = json.loads((library_dir / "index.json").read_text())
    console.print(f"[green]Found {len(index)} papers in index[/green]")

    # Verify each paper
    missing = []
    for paper in index:
        paper_dir = library_dir / "by_id" / paper["paper_id"]
        if not paper_dir.exists():
            missing.append(paper["paper_id"])
        elif not (paper_dir / "full_text.md").exists():
            missing.append(f"{paper['paper_id']} (no full_text.md)")

    if missing:
        console.print(f"[yellow]Missing papers: {len(missing)}[/yellow]")
        for m in missing[:10]:
            console.print(f"  - {m}")
    else:
        console.print("[green]All papers verified![/green]")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
