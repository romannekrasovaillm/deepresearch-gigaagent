"""
Library indexer for fast paper lookup and search.

Provides metadata-based search without full-text indexing,
enabling the agent to quickly find relevant papers.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """Search result for a paper."""
    paper_id: str
    title: str
    score: float
    abstract_preview: str = ""
    section_match: Optional[str] = None


class LibraryIndexer:
    """Index and search library metadata."""

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)
        self.index_path = self.library_path / "index.json"
        self._papers: list[dict] = []
        self._load_index()

    def _load_index(self):
        """Load index from disk."""
        if self.index_path.exists():
            try:
                self._papers = json.loads(
                    self.index_path.read_text(encoding='utf-8')
                )
            except Exception as e:
                print(f"Error loading index: {e}")
                self._papers = []

    def reload(self):
        """Reload index from disk."""
        self._load_index()

    @property
    def paper_count(self) -> int:
        """Number of papers in library."""
        return len(self._papers)

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Get paper metadata by ID."""
        for paper in self._papers:
            if paper.get('paper_id') == paper_id:
                return paper
        return None

    def list_papers(
        self,
        topic: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        List papers, optionally filtered by topic keyword.

        Args:
            topic: Optional keyword to filter by (searches title and abstract)
            limit: Maximum number of results

        Returns:
            List of paper metadata dictionaries
        """
        if not topic:
            return self._papers[:limit]

        topic_lower = topic.lower()
        results = []

        for paper in self._papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract_preview', '').lower()

            if topic_lower in title or topic_lower in abstract:
                results.append(paper)
                if len(results) >= limit:
                    break

        return results

    def search_by_title(
        self,
        query: str,
        limit: int = 10
    ) -> list[SearchResult]:
        """
        Search papers by title using simple keyword matching.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        query_words = set(query.lower().split())
        results = []

        for paper in self._papers:
            title = paper.get('title', '')
            title_lower = title.lower()
            title_words = set(title_lower.split())

            # Score by word overlap
            overlap = len(query_words & title_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                results.append(SearchResult(
                    paper_id=paper.get('paper_id', ''),
                    title=title,
                    score=score,
                    abstract_preview=paper.get('abstract_preview', ''),
                ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def get_paper_sections(self, paper_id: str) -> list[str]:
        """Get list of available sections for a paper."""
        paper_dir = self.library_path / "by_id" / paper_id / "sections"
        if not paper_dir.exists():
            return []

        return [
            f.stem for f in paper_dir.glob("*.txt")
        ]

    def get_section_path(
        self,
        paper_id: str,
        section: str
    ) -> Optional[Path]:
        """Get path to a specific section file."""
        section_file = (
            self.library_path / "by_id" / paper_id / "sections" / f"{section}.txt"
        )
        if section_file.exists():
            return section_file
        return None

    def get_full_text_path(self, paper_id: str) -> Optional[Path]:
        """Get path to full text markdown."""
        full_text = self.library_path / "by_id" / paper_id / "full_text.md"
        if full_text.exists():
            return full_text
        return None

    def get_abstract_path(self, paper_id: str) -> Optional[Path]:
        """Get path to abstract file."""
        abstract = self.library_path / "by_id" / paper_id / "abstract.txt"
        if abstract.exists():
            return abstract
        return None

    def get_papers_by_author(
        self,
        author: str,
        limit: int = 20
    ) -> list[dict]:
        """Find papers by author name."""
        author_lower = author.lower()
        results = []

        for paper in self._papers:
            paper_author = paper.get('author', '') or ''
            if author_lower in paper_author.lower():
                results.append(paper)
                if len(results) >= limit:
                    break

        return results

    def get_statistics(self) -> dict:
        """Get library statistics."""
        total_words = sum(p.get('word_count', 0) for p in self._papers)
        total_sections = sum(p.get('section_count', 0) for p in self._papers)

        return {
            "total_papers": len(self._papers),
            "total_words": total_words,
            "total_sections": total_sections,
            "avg_words_per_paper": total_words // max(len(self._papers), 1),
        }

    def export_titles_list(self, output_path: Path):
        """Export list of all titles for quick reference."""
        lines = []
        for paper in self._papers:
            lines.append(f"{paper.get('paper_id')}: {paper.get('title')}")

        output_path.write_text('\n'.join(lines), encoding='utf-8')

    def find_related_papers(
        self,
        paper_id: str,
        limit: int = 5
    ) -> list[SearchResult]:
        """
        Find papers related to given paper by keyword overlap.

        Uses title words from source paper to find similar papers.
        """
        source = self.get_paper(paper_id)
        if not source:
            return []

        # Extract keywords from title
        title = source.get('title', '')
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'и', 'в', 'на', 'с', 'к', 'по', 'за', 'из', 'от', 'до',
        }
        words = set(re.findall(r'\w+', title.lower())) - stop_words

        results = []
        for paper in self._papers:
            if paper.get('paper_id') == paper_id:
                continue

            other_title = paper.get('title', '')
            other_words = set(re.findall(r'\w+', other_title.lower())) - stop_words

            overlap = len(words & other_words)
            if overlap >= 2:  # At least 2 common keywords
                score = overlap / max(len(words), 1)
                results.append(SearchResult(
                    paper_id=paper.get('paper_id', ''),
                    title=other_title,
                    score=score,
                    abstract_preview=paper.get('abstract_preview', ''),
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Library indexer CLI")
    parser.add_argument("library_path", help="Path to library directory")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--search", type=str, help="Search by title")
    parser.add_argument("--list", type=str, help="List papers by topic")

    args = parser.parse_args()

    indexer = LibraryIndexer(args.library_path)

    if args.stats:
        stats = indexer.get_statistics()
        for k, v in stats.items():
            print(f"{k}: {v}")

    if args.search:
        results = indexer.search_by_title(args.search)
        for r in results:
            print(f"[{r.score:.2f}] {r.paper_id}: {r.title}")

    if args.list:
        papers = indexer.list_papers(topic=args.list)
        for p in papers:
            print(f"{p.get('paper_id')}: {p.get('title')}")
