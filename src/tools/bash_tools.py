"""
Bash-based tools for deterministic file navigation and search.

These tools wrap bash commands (grep, sed, find) to provide
reliable, reproducible search operations for the agent.
"""

import subprocess
import shlex
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    truncated: bool = False


class BaseTool(ABC):
    """Base class for all tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def to_schema(self) -> dict:
        """Return JSON schema for tool parameters."""
        pass


class GrepSearch(BaseTool):
    """
    Search for patterns in library files using ripgrep (rg) or grep.

    Provides deterministic, reproducible search results.
    """

    name = "grep_search"
    description = (
        "Search for a pattern in the library files. "
        "Returns matching lines with file paths and line numbers. "
        "Use for finding specific terms, concepts, or citations."
    )

    def __init__(
        self,
        library_path: str | Path,
        max_results: int = 20,
        use_ripgrep: bool = True
    ):
        self.library_path = Path(library_path)
        self.max_results = max_results
        self.use_ripgrep = use_ripgrep and self._check_ripgrep()

    def _check_ripgrep(self) -> bool:
        """Check if ripgrep is available."""
        try:
            subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        case_insensitive: bool = True,
        context_lines: int = 0
    ) -> ToolResult:
        """
        Search for pattern in files.

        Args:
            pattern: Search pattern (regex supported)
            path: Subdirectory to search in (relative to library)
            case_insensitive: Ignore case in search
            context_lines: Number of context lines before/after match

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
                error=f"Path not found: {search_path}"
            )

        try:
            if self.use_ripgrep:
                cmd = self._build_rg_command(
                    pattern, search_path, case_insensitive, context_lines
                )
            else:
                cmd = self._build_grep_command(
                    pattern, search_path, case_insensitive, context_lines
                )

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.library_path)
            )

            output = result.stdout.strip()

            # Truncate if too many results
            lines = output.split('\n')
            truncated = len(lines) > self.max_results
            if truncated:
                lines = lines[:self.max_results]
                lines.append(f"... (truncated, showing first {self.max_results} results)")
                output = '\n'.join(lines)

            return ToolResult(
                success=True,
                output=output if output else "No matches found.",
                truncated=truncated
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Search timed out after 30 seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def _build_rg_command(
        self,
        pattern: str,
        path: Path,
        case_insensitive: bool,
        context_lines: int
    ) -> list[str]:
        """Build ripgrep command."""
        cmd = ["rg", "--line-number", "--no-heading"]

        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        cmd.extend(["--max-count", "50"])  # Limit matches per file
        cmd.append(pattern)
        cmd.append(str(path))

        return cmd

    def _build_grep_command(
        self,
        pattern: str,
        path: Path,
        case_insensitive: bool,
        context_lines: int
    ) -> list[str]:
        """Build grep command."""
        cmd = ["grep", "-r", "-n"]

        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])

        cmd.append(pattern)
        cmd.append(str(path))

        return cmd

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (supports regex)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search in (optional)"
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Ignore case (default: true)"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around matches (default: 0)"
                    }
                },
                "required": ["pattern"]
            }
        }


class ReadFileChunk(BaseTool):
    """
    Read a chunk of a file using sed for streaming.

    Enables efficient reading of large files by loading
    only the needed portion.
    """

    name = "read_file_chunk"
    description = (
        "Read a portion of a file by line numbers. "
        "Use for reading specific sections without loading entire file. "
        "Supports reading abstracts, sections, or context around a match."
    )

    def __init__(
        self,
        library_path: str | Path,
        max_lines: int = 100
    ):
        self.library_path = Path(library_path)
        self.max_lines = max_lines

    def execute(
        self,
        path: str,
        start: int = 1,
        num_lines: int = 50
    ) -> ToolResult:
        """
        Read lines from a file.

        Args:
            path: File path (relative to library or absolute)
            start: Starting line number (1-indexed)
            num_lines: Number of lines to read

        Returns:
            ToolResult with file content
        """
        # Resolve path
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.library_path / path

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}"
            )

        if not file_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a file: {path}"
            )

        # Limit lines
        num_lines = min(num_lines, self.max_lines)
        end = start + num_lines - 1

        try:
            # Use sed for efficient line extraction
            cmd = ["sed", "-n", f"{start},{end}p", str(file_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            output = result.stdout
            if not output.strip():
                return ToolResult(
                    success=True,
                    output=f"(No content in lines {start}-{end})"
                )

            # Add line numbers for context
            lines = output.split('\n')
            numbered_lines = [
                f"{start + i:4d}: {line}"
                for i, line in enumerate(lines)
            ]

            return ToolResult(
                success=True,
                output='\n'.join(numbered_lines)
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Read operation timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path (relative to library or absolute)"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, default: 1)"
                    },
                    "num_lines": {
                        "type": "integer",
                        "description": f"Number of lines to read (default: 50, max: {self.max_lines})"
                    }
                },
                "required": ["path"]
            }
        }


class FindFiles(BaseTool):
    """
    Find files by name pattern using find command.

    Useful for locating papers by partial name or section files.
    """

    name = "find_files"
    description = (
        "Find files matching a name pattern. "
        "Use for locating papers by keyword in filename "
        "or finding section files across papers."
    )

    def __init__(
        self,
        library_path: str | Path,
        max_results: int = 30
    ):
        self.library_path = Path(library_path)
        self.max_results = max_results

    def execute(
        self,
        name_pattern: str,
        file_type: Optional[str] = None
    ) -> ToolResult:
        """
        Find files matching pattern.

        Args:
            name_pattern: Glob pattern for filename (e.g., "*methods*")
            file_type: File extension filter (e.g., "txt", "md")

        Returns:
            ToolResult with list of matching files
        """
        try:
            cmd = ["find", str(self.library_path), "-type", "f"]

            # Add name pattern
            if file_type:
                cmd.extend(["-name", f"*{name_pattern}*.{file_type}"])
            else:
                cmd.extend(["-name", f"*{name_pattern}*"])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )

            output = result.stdout.strip()
            if not output:
                return ToolResult(
                    success=True,
                    output="No files found matching pattern."
                )

            # Make paths relative to library
            lines = output.split('\n')
            relative_paths = []
            for line in lines:
                try:
                    rel_path = Path(line).relative_to(self.library_path)
                    relative_paths.append(str(rel_path))
                except ValueError:
                    relative_paths.append(line)

            # Truncate if needed
            truncated = len(relative_paths) > self.max_results
            if truncated:
                relative_paths = relative_paths[:self.max_results]
                relative_paths.append(f"... ({len(lines)} total matches)")

            return ToolResult(
                success=True,
                output='\n'.join(relative_paths),
                truncated=truncated
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Search timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": "Pattern to match in filename"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "File extension filter (e.g., 'txt', 'md')"
                    }
                },
                "required": ["name_pattern"]
            }
        }


class VerifyQuote(BaseTool):
    """
    Verify that a quote exists in a file.

    Protects against hallucinations by checking if
    claimed citations actually exist in the source.
    """

    name = "verify_quote"
    description = (
        "Verify that a quote or text snippet exists in a file. "
        "Use before including citations to prevent hallucinations. "
        "Returns the matching line if found."
    )

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)

    def execute(
        self,
        path: str,
        snippet: str,
        fuzzy: bool = True
    ) -> ToolResult:
        """
        Verify quote exists in file.

        Args:
            path: File path (relative to library)
            snippet: Text snippet to verify
            fuzzy: Allow partial matches (default: True)

        Returns:
            ToolResult with verification status and matching line
        """
        # Resolve path
        if Path(path).is_absolute():
            file_path = Path(path)
        else:
            file_path = self.library_path / path

        if not file_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}"
            )

        try:
            content = file_path.read_text(encoding='utf-8')

            # Normalize whitespace for matching
            normalized_content = ' '.join(content.split())
            normalized_snippet = ' '.join(snippet.split())

            if fuzzy:
                # Check if key words are present in same order
                words = normalized_snippet.split()[:10]  # First 10 words
                pattern = r'.*?'.join(map(re.escape, words))

                import re
                match = re.search(pattern, normalized_content, re.IGNORECASE)
                if match:
                    # Find the full line
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_content), match.end() + 50)
                    context = normalized_content[start:end]

                    return ToolResult(
                        success=True,
                        output=f"VERIFIED (fuzzy match): ...{context}..."
                    )
            else:
                if normalized_snippet.lower() in normalized_content.lower():
                    return ToolResult(
                        success=True,
                        output="VERIFIED: Exact match found."
                    )

            return ToolResult(
                success=True,
                output="NOT FOUND: Quote could not be verified in the file."
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to check"
                    },
                    "snippet": {
                        "type": "string",
                        "description": "Text snippet to verify"
                    },
                    "fuzzy": {
                        "type": "boolean",
                        "description": "Allow partial/fuzzy matching (default: true)"
                    }
                },
                "required": ["path", "snippet"]
            }
        }


class ListPapers(BaseTool):
    """
    List papers from the library index.

    Provides quick access to paper metadata without
    needing to search through files.
    """

    name = "list_papers"
    description = (
        "List papers in the library, optionally filtered by topic. "
        "Returns paper IDs and titles for quick reference. "
        "Use before deep searching to identify relevant papers."
    )

    def __init__(self, library_path: str | Path):
        self.library_path = Path(library_path)

    def execute(
        self,
        topic: Optional[str] = None,
        limit: int = 20
    ) -> ToolResult:
        """
        List papers from index.

        Args:
            topic: Optional keyword filter for title/abstract
            limit: Maximum number of results

        Returns:
            ToolResult with paper list
        """
        index_path = self.library_path / "index.json"

        if not index_path.exists():
            return ToolResult(
                success=False,
                output="",
                error="Library index not found. Run library build first."
            )

        try:
            import json
            papers = json.loads(index_path.read_text(encoding='utf-8'))

            if topic:
                topic_lower = topic.lower()
                papers = [
                    p for p in papers
                    if topic_lower in p.get('title', '').lower()
                    or topic_lower in p.get('abstract_preview', '').lower()
                ]

            papers = papers[:limit]

            if not papers:
                return ToolResult(
                    success=True,
                    output="No papers found matching criteria."
                )

            lines = []
            for p in papers:
                lines.append(f"{p.get('paper_id')}: {p.get('title', 'Untitled')}")

            return ToolResult(
                success=True,
                output='\n'.join(lines)
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional topic keyword filter"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)"
                    }
                }
            }
        }


# Import re at module level for VerifyQuote
import re


def create_tools(library_path: str | Path) -> dict[str, BaseTool]:
    """Create all bash tools for a library."""
    library_path = Path(library_path)

    return {
        "grep_search": GrepSearch(library_path),
        "read_file_chunk": ReadFileChunk(library_path),
        "find_files": FindFiles(library_path),
        "verify_quote": VerifyQuote(library_path),
        "list_papers": ListPapers(library_path),
    }
