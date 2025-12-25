"""Library indexer - builds and maintains paper index."""

import json
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .converter import DocxConverter, PaperMetadata

console = Console()


class LibraryIndexer:
    """Builds and maintains the paper library index."""

    def __init__(self, library_path: Path):
        self.library_path = Path(library_path)
        self.by_id_path = self.library_path / "by_id"
        self.index_path = self.library_path / "index.json"
        self.converter = DocxConverter()

        # Create directories
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.by_id_path.mkdir(exist_ok=True)

    def load_index(self) -> list[dict]:
        """Load existing index or return empty list."""
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return []

    def save_index(self, index: list[dict]) -> None:
        """Save index to file."""
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_next_paper_id(self) -> str:
        """Generate next paper ID."""
        index = self.load_index()
        if not index:
            return "paper_0001"

        # Find max ID
        max_id = 0
        for paper in index:
            paper_id = paper.get("paper_id", "paper_0000")
            try:
                num = int(paper_id.split("_")[1])
                max_id = max(max_id, num)
            except (IndexError, ValueError):
                continue

        return f"paper_{max_id + 1:04d}"

    def process_single_paper(
        self,
        docx_path: Path,
        paper_id: Optional[str] = None,
    ) -> PaperMetadata:
        """Process a single paper and add to library."""
        if paper_id is None:
            paper_id = self.get_next_paper_id()

        metadata = self.converter.process_paper(
            docx_path,
            self.by_id_path,
            paper_id,
        )

        # Update index
        index = self.load_index()
        # Remove if already exists
        index = [p for p in index if p.get("paper_id") != paper_id]
        index.append(metadata.to_dict())
        self.save_index(index)

        return metadata

    def build_from_directory(
        self,
        input_dir: Path,
        pattern: str = "*.docx",
    ) -> list[PaperMetadata]:
        """Build library from directory of DOCX files."""
        input_dir = Path(input_dir)
        docx_files = list(input_dir.glob(pattern))

        if not docx_files:
            console.print(f"[yellow]No DOCX files found in {input_dir}[/yellow]")
            return []

        results: list[PaperMetadata] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing papers...", total=len(docx_files))

            for docx_path in docx_files:
                paper_id = self.get_next_paper_id()
                progress.update(
                    task,
                    description=f"Processing {docx_path.name} -> {paper_id}",
                )

                try:
                    metadata = self.process_single_paper(docx_path, paper_id)
                    results.append(metadata)
                    console.print(f"[green]✓[/green] {paper_id}: {metadata.title[:50]}")
                except Exception as e:
                    console.print(f"[red]✗[/red] {docx_path.name}: {e}")

                progress.advance(task)

        console.print(f"\n[bold green]Done![/bold green] Processed {len(results)} papers")
        return results

    def search_index(
        self,
        query: str,
        field: str = "title",
        limit: int = 10,
    ) -> list[dict]:
        """Search papers in index by field."""
        index = self.load_index()
        query_lower = query.lower()

        results = []
        for paper in index:
            field_value = str(paper.get(field, "")).lower()
            if query_lower in field_value:
                results.append(paper)
                if len(results) >= limit:
                    break

        return results

    def get_paper_path(self, paper_id: str) -> Optional[Path]:
        """Get path to paper directory."""
        paper_path = self.by_id_path / paper_id
        if paper_path.exists():
            return paper_path
        return None

    def list_papers(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> Iterator[dict]:
        """List papers, optionally filtered by topic/keyword."""
        index = self.load_index()

        for paper in index[:limit]:
            if topic is None:
                yield paper
            else:
                # Search in title and abstract
                topic_lower = topic.lower()
                title = paper.get("title", "").lower()
                abstract = paper.get("abstract", "").lower()
                if topic_lower in title or topic_lower in abstract:
                    yield paper


def build_library(input_dir: Path, output_dir: Path) -> list[PaperMetadata]:
    """Convenience function to build library from directory."""
    indexer = LibraryIndexer(output_dir)
    return indexer.build_from_directory(input_dir)
