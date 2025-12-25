"""Bash-based tools for file navigation and search."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    error: Optional[str] = None


class BashTools:
    """Collection of bash-based tools for paper navigation."""

    def __init__(
        self,
        library_path: str | Path,
        use_ripgrep: bool = True,
        max_results: int = 20,
    ):
        self.library_path = Path(library_path)
        self.use_ripgrep = use_ripgrep and self._check_ripgrep()
        self.max_results = max_results

        if not self.library_path.exists():
            raise ValueError(f"Library path does not exist: {library_path}")

    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def grep_search(
        self,
        pattern: str,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        max_results: Optional[int] = None,
    ) -> ToolResult:
        """
        Search for pattern in library files.

        Args:
            pattern: Search pattern (regex supported with ripgrep)
            path: Subpath within library (default: entire library)
            case_sensitive: Case-sensitive search
            max_results: Maximum number of results

        Returns:
            ToolResult with matching lines
        """
        search_path = self.library_path
        if path:
            search_path = self.library_path / path

        if not search_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"Path not found: {path}",
            )

        max_results = max_results or self.max_results

        if self.use_ripgrep:
            cmd = ["rg", "--max-count", str(max_results)]
            if not case_sensitive:
                cmd.append("-i")
            cmd.extend([pattern, str(search_path)])
        else:
            cmd = ["grep", "-r"]
            if not case_sensitive:
                cmd.append("-i")
            cmd.extend([pattern, str(search_path)])
            cmd.extend(["|", "head", "-n", str(max_results)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout.strip()
            if not output:
                return ToolResult(
                    success=True,
                    output="No matches found.",
                )

            # Limit output lines
            lines = output.split("\n")[:max_results]
            return ToolResult(
                success=True,
                output="\n".join(lines),
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Search timed out after 30 seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def read_file_chunk(
        self,
        path: str,
        start: int = 1,
        num_lines: int = 50,
    ) -> ToolResult:
        """
        Read a chunk of a file.

        Args:
            path: Relative path within library
            start: Starting line number (1-indexed)
            num_lines: Number of lines to read

        Returns:
            ToolResult with file content
        """
        file_path = self.library_path / path

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {path}",
            )

        try:
            # Use sed for efficient line extraction
            end = start + num_lines - 1
            cmd = ["sed", "-n", f"{start},{end}p", str(file_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            return ToolResult(
                success=True,
                output=result.stdout,
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Read timed out",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def find_files(
        self,
        name_pattern: str,
        file_type: Optional[str] = None,
    ) -> ToolResult:
        """
        Find files by name pattern.

        Args:
            name_pattern: Glob pattern for filename
            file_type: Filter by extension (e.g., "txt", "md")

        Returns:
            ToolResult with list of matching files
        """
        try:
            pattern = f"*{name_pattern}*"
            if file_type:
                pattern = f"*{name_pattern}*.{file_type}"

            cmd = [
                "find",
                str(self.library_path),
                "-type",
                "f",
                "-name",
                pattern,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if not result.stdout.strip():
                return ToolResult(
                    success=True,
                    output="No files found.",
                )

            # Make paths relative to library
            lines = result.stdout.strip().split("\n")
            relative_paths = []
            for line in lines[: self.max_results]:
                try:
                    rel_path = Path(line).relative_to(self.library_path)
                    relative_paths.append(str(rel_path))
                except ValueError:
                    relative_paths.append(line)

            return ToolResult(
                success=True,
                output="\n".join(relative_paths),
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Search timed out",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def list_papers(
        self,
        topic: Optional[str] = None,
        limit: int = 20,
    ) -> ToolResult:
        """
        List papers from index, optionally filtered by topic.

        Args:
            topic: Filter papers containing this keyword
            limit: Maximum number of papers to return

        Returns:
            ToolResult with paper list
        """
        index_path = self.library_path / "index.json"

        if not index_path.exists():
            return ToolResult(
                success=False,
                output="",
                error="Library index not found. Run indexing first.",
            )

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)

            results = []
            for paper in index:
                if topic:
                    topic_lower = topic.lower()
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()
                    if topic_lower not in title and topic_lower not in abstract:
                        continue

                paper_id = paper.get("paper_id", "unknown")
                title = paper.get("title", "Untitled")[:80]
                results.append(f"{paper_id}: {title}")

                if len(results) >= limit:
                    break

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No papers found for topic: {topic}" if topic else "No papers in library.",
                )

            return ToolResult(
                success=True,
                output="\n".join(results),
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )

    def verify_quote(
        self,
        path: str,
        snippet: str,
        fuzzy_threshold: float = 0.85,
    ) -> ToolResult:
        """
        Verify that a quote exists in a file.

        Args:
            path: Relative path to file
            snippet: Text snippet to verify
            fuzzy_threshold: Similarity threshold for fuzzy matching

        Returns:
            ToolResult with verification status
        """
        file_path = self.library_path / path

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        try:
            content = file_path.read_text(encoding="utf-8")

            # Exact match
            if snippet in content:
                return ToolResult(
                    success=True,
                    output="VERIFIED: Exact match found.",
                )

            # Normalized match (ignore whitespace)
            normalized_content = " ".join(content.split())
            normalized_snippet = " ".join(snippet.split())

            if normalized_snippet in normalized_content:
                return ToolResult(
                    success=True,
                    output="VERIFIED: Match found (normalized whitespace).",
                )

            # Fuzzy match using simple ratio
            snippet_words = set(normalized_snippet.lower().split())
            content_words = set(normalized_content.lower().split())

            if not snippet_words:
                return ToolResult(
                    success=False,
                    output="NOT VERIFIED: Empty snippet.",
                )

            overlap = len(snippet_words & content_words) / len(snippet_words)

            if overlap >= fuzzy_threshold:
                return ToolResult(
                    success=True,
                    output=f"VERIFIED: Fuzzy match ({overlap:.1%} word overlap).",
                )

            return ToolResult(
                success=True,
                output="NOT VERIFIED: Quote not found in file.",
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
            )


# Global instance for functional API
_tools_instance: Optional[BashTools] = None


def _get_tools() -> BashTools:
    """Get or create global tools instance."""
    global _tools_instance
    if _tools_instance is None:
        # Default library path - should be configured
        _tools_instance = BashTools("./data/library")
    return _tools_instance


def init_tools(library_path: str | Path) -> None:
    """Initialize global tools with library path."""
    global _tools_instance
    _tools_instance = BashTools(library_path)


# Functional API
def grep_search(
    pattern: str,
    path: Optional[str] = None,
    case_sensitive: bool = False,
) -> str:
    """Search for pattern in library."""
    result = _get_tools().grep_search(pattern, path, case_sensitive)
    if not result.success:
        return f"Error: {result.error}"
    return result.output


def read_file_chunk(
    path: str,
    start: int = 1,
    num_lines: int = 50,
) -> str:
    """Read chunk of file."""
    result = _get_tools().read_file_chunk(path, start, num_lines)
    if not result.success:
        return f"Error: {result.error}"
    return result.output


def find_files(
    name_pattern: str,
    file_type: Optional[str] = None,
) -> str:
    """Find files by name pattern."""
    result = _get_tools().find_files(name_pattern, file_type)
    if not result.success:
        return f"Error: {result.error}"
    return result.output


def list_papers(
    topic: Optional[str] = None,
    limit: int = 20,
) -> str:
    """List papers in library."""
    result = _get_tools().list_papers(topic, limit)
    if not result.success:
        return f"Error: {result.error}"
    return result.output


def verify_quote(
    path: str,
    snippet: str,
) -> str:
    """Verify quote exists in file."""
    result = _get_tools().verify_quote(path, snippet)
    if not result.success:
        return f"Error: {result.error}"
    return result.output
