#!/usr/bin/env python3
"""
Bash Tools for Library Navigation
Deterministic bash commands for file navigation and search.
"""

import json
import subprocess
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Default library path
DEFAULT_LIBRARY_PATH = "/mnt/library"


@dataclass
class SearchResult:
    """Result from grep/find operations."""
    file_path: str
    line_number: Optional[int] = None
    content: str = ""
    paper_id: Optional[str] = None


def grep_search(
    pattern: str,
    path: str = DEFAULT_LIBRARY_PATH,
    max_results: int = 20,
    case_insensitive: bool = True,
    context_lines: int = 0,
) -> list[SearchResult]:
    """
    Deterministic grep search using ripgrep (rg) for speed.

    Args:
        pattern: Regex pattern to search for
        path: Directory or file to search in
        max_results: Maximum number of results to return
        case_insensitive: Whether to ignore case
        context_lines: Number of context lines before/after match

    Returns:
        List of SearchResult objects
    """
    # Use ripgrep if available, fallback to grep
    try:
        cmd = ["rg", "--line-number", "--no-heading"]
        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        cmd.extend([pattern, path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        # Fallback to grep
        cmd = ["grep", "-r", "-n"]
        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        cmd.extend([pattern, path])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

    results = []
    for line in result.stdout.split('\n')[:max_results]:
        if not line.strip():
            continue

        # Parse line format: file:line_num:content
        match = re.match(r'^([^:]+):(\d+):(.*)$', line)
        if match:
            file_path = match.group(1)
            line_num = int(match.group(2))
            content = match.group(3)

            # Extract paper_id from path
            paper_id = None
            if '/by_id/' in file_path:
                parts = file_path.split('/by_id/')
                if len(parts) > 1:
                    paper_id = parts[1].split('/')[0]

            results.append(SearchResult(
                file_path=file_path,
                line_number=line_num,
                content=content.strip(),
                paper_id=paper_id,
            ))

    return results


def read_file_chunk(
    path: str,
    start: int = 1,
    num_lines: int = 50,
) -> str:
    """
    Read a chunk of a file using sed for streaming.

    Args:
        path: Path to file
        start: Starting line number (1-indexed)
        num_lines: Number of lines to read

    Returns:
        File content as string
    """
    end = start + num_lines - 1

    try:
        result = subprocess.run(
            ["sed", "-n", f"{start},{end}p", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"[Error: Timeout reading {path}]"
    except Exception as e:
        return f"[Error reading file: {e}]"


def find_files(
    name_pattern: str,
    path: str = DEFAULT_LIBRARY_PATH,
    file_type: str = "f",
    max_results: int = 50,
) -> list[str]:
    """
    Find files by name pattern.

    Args:
        name_pattern: Glob pattern for file name (e.g., "*.md", "*attention*")
        path: Directory to search in
        file_type: 'f' for files, 'd' for directories
        max_results: Maximum number of results

    Returns:
        List of matching file paths
    """
    try:
        result = subprocess.run(
            ["find", path, "-type", file_type, "-name", name_pattern],
            capture_output=True,
            text=True,
            timeout=30,
        )
        files = [f for f in result.stdout.strip().split('\n') if f]
        return files[:max_results]
    except Exception as e:
        return [f"[Error: {e}]"]


def verify_quote(
    path: str,
    snippet: str,
    fuzzy_match: bool = True,
) -> dict:
    """
    Verify that a quote/snippet actually exists in the file.
    Protection against hallucinations.

    Args:
        path: Path to file
        snippet: Text snippet to verify
        fuzzy_match: Allow minor differences (whitespace, punctuation)

    Returns:
        Dict with 'verified' boolean and 'context' if found
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Normalize for comparison
        def normalize(text: str) -> str:
            if fuzzy_match:
                # Remove extra whitespace, normalize quotes
                text = re.sub(r'\s+', ' ', text)
                text = text.replace('"', '"').replace('"', '"')
                text = text.replace("'", "'").replace("'", "'")
                text = text.lower().strip()
            return text

        normalized_snippet = normalize(snippet)
        normalized_content = normalize(content)

        if normalized_snippet in normalized_content:
            # Find actual position for context
            idx = normalized_content.find(normalized_snippet)

            # Get surrounding context (50 chars each side)
            start = max(0, idx - 50)
            end = min(len(normalized_content), idx + len(normalized_snippet) + 50)
            context = normalized_content[start:end]

            return {
                "verified": True,
                "exact_match": snippet in content,
                "context": f"...{context}...",
            }
        else:
            return {
                "verified": False,
                "exact_match": False,
                "context": None,
                "suggestion": "Quote not found in file. Check for typos or wrong file.",
            }

    except FileNotFoundError:
        return {
            "verified": False,
            "error": f"File not found: {path}",
        }
    except Exception as e:
        return {
            "verified": False,
            "error": str(e),
        }


def list_papers(
    topic: Optional[str] = None,
    library_path: str = DEFAULT_LIBRARY_PATH,
    limit: int = 100,
) -> list[dict]:
    """
    List papers from index, optionally filtered by topic.

    Args:
        topic: Topic to filter by (from topic_index.json)
        library_path: Path to library
        limit: Maximum papers to return

    Returns:
        List of paper metadata dicts
    """
    library_path = Path(library_path)
    index_path = library_path / "index.json"

    if not index_path.exists():
        return []

    try:
        index = json.loads(index_path.read_text())
    except Exception:
        return []

    if topic:
        topic_index_path = library_path / "topic_index.json"
        if topic_index_path.exists():
            try:
                topics = json.loads(topic_index_path.read_text())
                topic_lower = topic.lower()

                if topic_lower in topics:
                    paper_ids = set(topics[topic_lower])
                    index = [p for p in index if p['paper_id'] in paper_ids]
            except Exception:
                pass

    return index[:limit]


def get_paper_metadata(
    paper_id: str,
    library_path: str = DEFAULT_LIBRARY_PATH,
) -> Optional[dict]:
    """
    Get metadata for a specific paper.

    Args:
        paper_id: Paper ID (e.g., "paper_0042")
        library_path: Path to library

    Returns:
        Metadata dict or None if not found
    """
    library_path = Path(library_path)
    metadata_path = library_path / "by_id" / paper_id / "metadata.json"

    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return None


def get_paper_abstract(
    paper_id: str,
    library_path: str = DEFAULT_LIBRARY_PATH,
) -> str:
    """
    Get abstract for a specific paper.

    Args:
        paper_id: Paper ID
        library_path: Path to library

    Returns:
        Abstract text or empty string
    """
    library_path = Path(library_path)
    abstract_path = library_path / "by_id" / paper_id / "abstract.txt"

    if not abstract_path.exists():
        return ""

    try:
        return abstract_path.read_text(encoding='utf-8')
    except Exception:
        return ""


def get_paper_section(
    paper_id: str,
    section: str,
    library_path: str = DEFAULT_LIBRARY_PATH,
) -> str:
    """
    Get a specific section from a paper.

    Args:
        paper_id: Paper ID
        section: Section name (e.g., "methods", "results")
        library_path: Path to library

    Returns:
        Section content or empty string
    """
    library_path = Path(library_path)
    section_path = library_path / "by_id" / paper_id / "sections" / f"{section}.txt"

    if not section_path.exists():
        # Try to find similar section name
        sections_dir = library_path / "by_id" / paper_id / "sections"
        if sections_dir.exists():
            for f in sections_dir.iterdir():
                if section.lower() in f.stem.lower():
                    return f.read_text(encoding='utf-8')
        return ""

    try:
        return section_path.read_text(encoding='utf-8')
    except Exception:
        return ""


# Tool definitions for ToolEnv
BASH_TOOLS = {
    "grep_search": {
        "function": grep_search,
        "description": "Search for pattern in library files. Returns file paths, line numbers, and matching content.",
        "parameters": {
            "pattern": {"type": "string", "description": "Regex pattern to search for", "required": True},
            "path": {"type": "string", "description": "Path to search in", "default": DEFAULT_LIBRARY_PATH},
            "max_results": {"type": "integer", "description": "Maximum results", "default": 20},
            "case_insensitive": {"type": "boolean", "description": "Ignore case", "default": True},
        },
    },
    "read_file_chunk": {
        "function": read_file_chunk,
        "description": "Read lines from a file. Use for streaming large files.",
        "parameters": {
            "path": {"type": "string", "description": "File path", "required": True},
            "start": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
            "num_lines": {"type": "integer", "description": "Number of lines to read", "default": 50},
        },
    },
    "find_files": {
        "function": find_files,
        "description": "Find files by name pattern using glob.",
        "parameters": {
            "name_pattern": {"type": "string", "description": "Glob pattern (e.g., '*.md')", "required": True},
            "path": {"type": "string", "description": "Directory to search", "default": DEFAULT_LIBRARY_PATH},
            "max_results": {"type": "integer", "description": "Maximum results", "default": 50},
        },
    },
    "verify_quote": {
        "function": verify_quote,
        "description": "Verify a quote exists in a file. Protection against hallucinations.",
        "parameters": {
            "path": {"type": "string", "description": "File path", "required": True},
            "snippet": {"type": "string", "description": "Text to verify", "required": True},
            "fuzzy_match": {"type": "boolean", "description": "Allow minor differences", "default": True},
        },
    },
    "list_papers": {
        "function": list_papers,
        "description": "List papers from library index, optionally by topic.",
        "parameters": {
            "topic": {"type": "string", "description": "Topic filter (rl, alignment, llm, etc.)"},
            "limit": {"type": "integer", "description": "Maximum papers", "default": 100},
        },
    },
    "get_paper_metadata": {
        "function": get_paper_metadata,
        "description": "Get metadata for a specific paper by ID.",
        "parameters": {
            "paper_id": {"type": "string", "description": "Paper ID (e.g., paper_0042)", "required": True},
        },
    },
    "get_paper_abstract": {
        "function": get_paper_abstract,
        "description": "Get abstract for a specific paper.",
        "parameters": {
            "paper_id": {"type": "string", "description": "Paper ID", "required": True},
        },
    },
    "get_paper_section": {
        "function": get_paper_section,
        "description": "Get a specific section from a paper (methods, results, etc.).",
        "parameters": {
            "paper_id": {"type": "string", "description": "Paper ID", "required": True},
            "section": {"type": "string", "description": "Section name", "required": True},
        },
    },
}
