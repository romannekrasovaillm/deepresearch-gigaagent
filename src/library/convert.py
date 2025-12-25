#!/usr/bin/env python3
"""
Library Conversion Module
Converts DOCX papers to structured Markdown library.
"""

import json
import re
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from docx import Document

console = Console()
app = typer.Typer(help="Convert DOCX papers to structured library")


@dataclass
class PaperMetadata:
    """Metadata for a single paper."""
    paper_id: str
    title: str
    author: Optional[str] = None
    created: Optional[str] = None
    source_file: str = ""
    sections: list[str] = None
    word_count: int = 0
    hash: str = ""

    def __post_init__(self):
        if self.sections is None:
            self.sections = []


def extract_metadata(docx_path: Path) -> PaperMetadata:
    """Extract metadata from DOCX file using python-docx."""
    doc = Document(docx_path)
    props = doc.core_properties

    # First paragraph is usually the title
    title = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            title = text
            break

    # Count words
    full_text = "\n".join(p.text for p in doc.paragraphs)
    word_count = len(full_text.split())

    # Create hash for deduplication
    content_hash = hashlib.md5(full_text.encode()).hexdigest()[:12]

    return PaperMetadata(
        paper_id="",  # Will be set later
        title=title or props.title or docx_path.stem,
        author=props.author,
        created=str(props.created) if props.created else None,
        source_file=docx_path.name,
        word_count=word_count,
        hash=content_hash,
    )


def convert_to_markdown(docx_path: Path, extract_media_dir: Optional[Path] = None) -> str:
    """Convert DOCX to Markdown using pandoc with optimal settings."""
    cmd = [
        "pandoc",
        str(docx_path),
        "-t", "markdown",
        "--wrap=none",
        "--standalone",
    ]

    if extract_media_dir:
        cmd.extend(["--extract-media", str(extract_media_dir)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            console.print(f"[yellow]Warning: pandoc returned {result.returncode}[/yellow]")
            console.print(f"[dim]{result.stderr}[/dim]")
        return result.stdout
    except subprocess.TimeoutExpired:
        console.print(f"[red]Timeout converting {docx_path.name}[/red]")
        return ""
    except FileNotFoundError:
        # Fallback to mammoth if pandoc not installed
        console.print("[yellow]pandoc not found, using mammoth fallback[/yellow]")
        return convert_with_mammoth(docx_path)


def convert_with_mammoth(docx_path: Path) -> str:
    """Fallback conversion using mammoth."""
    try:
        import mammoth
        with open(docx_path, "rb") as f:
            result = mammoth.convert_to_markdown(f)
            if result.messages:
                for msg in result.messages:
                    console.print(f"[dim]mammoth: {msg}[/dim]")
            return result.value
    except Exception as e:
        console.print(f"[red]mammoth conversion failed: {e}[/red]")
        return ""


def split_by_sections(markdown: str) -> dict[str, str]:
    """Split Markdown content by ## headers into sections."""
    sections = {}
    current = "preamble"
    sections[current] = ""

    for line in markdown.split('\n'):
        # Detect headers (## Title or # Title)
        if line.startswith('## ') or line.startswith('# '):
            header_level = 2 if line.startswith('## ') else 1
            current = line[header_level + 1:].strip().lower()
            # Normalize section name
            current = re.sub(r'[^a-zа-яё0-9]+', '_', current)
            current = current.strip('_')[:50]  # Limit length
            if not current:
                current = "section"
            # Handle duplicate names
            base_name = current
            counter = 1
            while current in sections:
                current = f"{base_name}_{counter}"
                counter += 1
            sections[current] = ""
        else:
            sections[current] += line + '\n'

    # Clean up and filter empty sections
    return {k: v.strip() for k, v in sections.items() if v.strip()}


def extract_abstract(sections: dict[str, str]) -> str:
    """Extract abstract from sections with fallback strategies."""
    # Try common abstract section names
    for key in ['abstract', 'аннотация', 'annotation', 'summary', 'overview']:
        if key in sections:
            return sections[key]

    # Fallback: first substantial paragraph from preamble
    if 'preamble' in sections:
        paragraphs = sections['preamble'].split('\n\n')
        for para in paragraphs:
            if len(para.split()) > 30:  # At least 30 words
                return para

    return ""


def process_paper(
    docx_path: Path,
    output_dir: Path,
    paper_id: str,
    extract_media: bool = True,
) -> PaperMetadata:
    """Process a single paper: convert and save structured output."""
    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    metadata = extract_metadata(docx_path)
    metadata.paper_id = paper_id

    # Convert to markdown
    media_dir = paper_dir / "media" if extract_media else None
    if media_dir:
        media_dir.mkdir(exist_ok=True)

    markdown = convert_to_markdown(docx_path, media_dir)
    if not markdown:
        console.print(f"[red]Failed to convert {docx_path.name}[/red]")
        return metadata

    # Save full text
    (paper_dir / "full_text.md").write_text(markdown, encoding='utf-8')

    # Split into sections
    sections = split_by_sections(markdown)
    metadata.sections = list(sections.keys())

    sections_dir = paper_dir / 'sections'
    sections_dir.mkdir(exist_ok=True)

    for name, content in sections.items():
        section_file = sections_dir / f'{name}.txt'
        section_file.write_text(content, encoding='utf-8')

    # Extract and save abstract separately (for quick lookup)
    abstract = extract_abstract(sections)
    (paper_dir / "abstract.txt").write_text(abstract, encoding='utf-8')

    # Save metadata
    (paper_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    return metadata


def generate_paper_id(index: int, title: str = "") -> str:
    """Generate unique paper ID."""
    return f"paper_{index:04d}"


def build_library(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "*.docx",
    skip_existing: bool = True,
) -> list[PaperMetadata]:
    """Convert all DOCX files in input_dir to structured library."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all DOCX files
    docx_files = sorted(input_dir.glob(file_pattern))
    if not docx_files:
        console.print(f"[yellow]No {file_pattern} files found in {input_dir}[/yellow]")
        return []

    console.print(f"[green]Found {len(docx_files)} documents to process[/green]")

    # Create output structure
    by_id_dir = output_dir / 'by_id'
    by_id_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index if exists
    index_path = output_dir / "index.json"
    existing_index = {}
    next_id = 1

    if index_path.exists() and skip_existing:
        try:
            existing = json.loads(index_path.read_text())
            existing_index = {m['source_file']: m for m in existing}
            # Find next available ID
            existing_ids = [int(m['paper_id'].split('_')[1]) for m in existing]
            next_id = max(existing_ids) + 1 if existing_ids else 1
            console.print(f"[dim]Loaded existing index with {len(existing)} papers[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not load existing index: {e}[/yellow]")

    # Process papers
    index = list(existing_index.values())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting papers...", total=len(docx_files))

        for docx_path in docx_files:
            # Skip if already processed
            if skip_existing and docx_path.name in existing_index:
                progress.update(task, advance=1, description=f"Skipping {docx_path.name}")
                continue

            progress.update(task, description=f"Processing {docx_path.name}")

            paper_id = generate_paper_id(next_id)
            next_id += 1

            try:
                metadata = process_paper(docx_path, by_id_dir, paper_id)
                index.append(asdict(metadata))
            except Exception as e:
                console.print(f"[red]Error processing {docx_path.name}: {e}[/red]")

            progress.update(task, advance=1)

    # Save updated index
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    # Create topic index (by analyzing titles/abstracts)
    create_topic_index(output_dir, index)

    console.print(f"[green]Done! Processed {len(index)} papers total[/green]")
    console.print(f"[dim]Library saved to {output_dir}[/dim]")

    return [PaperMetadata(**m) for m in index]


def create_topic_index(output_dir: Path, index: list[dict]) -> None:
    """Create a simple topic-based index for faster search."""
    # Common ML/AI topics
    topics = {
        'rl': ['reinforcement learning', 'rl', 'ppo', 'grpo', 'reward', 'policy'],
        'alignment': ['alignment', 'rlhf', 'dpo', 'safety', 'preference'],
        'llm': ['language model', 'llm', 'transformer', 'gpt', 'bert', 'attention'],
        'training': ['training', 'fine-tuning', 'pretraining', 'optimization'],
        'evaluation': ['benchmark', 'evaluation', 'metric', 'gsm8k', 'mmlu'],
        'reasoning': ['reasoning', 'chain-of-thought', 'cot', 'thinking'],
        'agents': ['agent', 'tool use', 'function calling', 'planning'],
        'scaling': ['scaling', 'emergent', 'capability', 'compute'],
    }

    topic_index = {topic: [] for topic in topics}

    for paper in index:
        title_lower = paper.get('title', '').lower()

        for topic, keywords in topics.items():
            for keyword in keywords:
                if keyword in title_lower:
                    topic_index[topic].append(paper['paper_id'])
                    break

    # Save topic index
    topic_index_path = output_dir / "topic_index.json"
    topic_index_path.write_text(
        json.dumps(topic_index, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def add_paper(
    docx_path: str | Path,
    library_dir: str | Path = "/mnt/library",
) -> PaperMetadata:
    """Add a single new paper to existing library."""
    docx_path = Path(docx_path)
    library_dir = Path(library_dir)

    if not docx_path.exists():
        raise FileNotFoundError(f"File not found: {docx_path}")

    # Load existing index
    index_path = library_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []

    # Check for duplicates
    source_files = {m['source_file'] for m in index}
    if docx_path.name in source_files:
        console.print(f"[yellow]Paper already in library: {docx_path.name}[/yellow]")
        for m in index:
            if m['source_file'] == docx_path.name:
                return PaperMetadata(**m)

    # Generate new ID
    existing_ids = [int(m['paper_id'].split('_')[1]) for m in index]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    paper_id = generate_paper_id(next_id)

    # Process paper
    by_id_dir = library_dir / 'by_id'
    by_id_dir.mkdir(parents=True, exist_ok=True)

    metadata = process_paper(docx_path, by_id_dir, paper_id)

    # Update index
    index.append(asdict(metadata))
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    console.print(f"[green]Added paper: {metadata.title} ({paper_id})[/green]")
    return metadata


# CLI Commands
@app.command()
def convert(
    input_dir: Path = typer.Argument(..., help="Directory with DOCX files"),
    output_dir: Path = typer.Option(
        Path("/mnt/library"),
        "--output", "-o",
        help="Output library directory"
    ),
    pattern: str = typer.Option("*.docx", "--pattern", "-p", help="File glob pattern"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--no-skip-existing"),
):
    """Convert DOCX papers to structured library."""
    build_library(input_dir, output_dir, pattern, skip_existing)


@app.command()
def add(
    docx_path: Path = typer.Argument(..., help="Path to DOCX file"),
    library_dir: Path = typer.Option(
        Path("/mnt/library"),
        "--library", "-l",
        help="Library directory"
    ),
):
    """Add a single paper to existing library."""
    add_paper(docx_path, library_dir)


@app.command()
def info(
    library_dir: Path = typer.Option(
        Path("/mnt/library"),
        "--library", "-l",
        help="Library directory"
    ),
):
    """Show library statistics."""
    index_path = library_dir / "index.json"
    if not index_path.exists():
        console.print(f"[red]Library not found at {library_dir}[/red]")
        return

    index = json.loads(index_path.read_text())
    total_words = sum(m.get('word_count', 0) for m in index)

    console.print(f"\n[bold]Library Statistics[/bold]")
    console.print(f"  Papers: {len(index)}")
    console.print(f"  Total words: {total_words:,}")
    console.print(f"  Average words/paper: {total_words // len(index) if index else 0:,}")

    # Show topics
    topic_index_path = library_dir / "topic_index.json"
    if topic_index_path.exists():
        topics = json.loads(topic_index_path.read_text())
        console.print(f"\n[bold]Topics:[/bold]")
        for topic, papers in sorted(topics.items(), key=lambda x: -len(x[1])):
            if papers:
                console.print(f"  {topic}: {len(papers)} papers")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
