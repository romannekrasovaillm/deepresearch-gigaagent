"""
Bash Tools - Deterministic navigation and search tools for the agent.

These tools use bash commands (grep, sed, find) for reliable, reproducible
operations on the paper library. They are deterministic - same input always
produces same output.
"""

import json
import subprocess
import shlex
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# Default library path - configurable via environment
DEFAULT_LIBRARY_PATH = "/mnt/library"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


def _run_command(cmd: list[str], timeout: int = 30) -> ToolResult:
    """
    Execute a shell command safely.

    Args:
        cmd: Command as list of arguments
        timeout: Maximum execution time in seconds

    Returns:
        ToolResult with output or error
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=DEFAULT_LIBRARY_PATH,
        )

        if result.returncode == 0:
            return ToolResult(success=True, output=result.stdout.strip())
        else:
            # grep returns 1 when no matches found - not an error
            if result.returncode == 1 and not result.stderr:
                return ToolResult(success=True, output="No matches found.")
            return ToolResult(
                success=False,
                output=result.stdout.strip(),
                error=result.stderr.strip() or f"Exit code: {result.returncode}"
            )

    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"Command timed out after {timeout}s")
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def grep_search(
    pattern: str,
    path: str = "",
    max_results: int = 20,
    case_insensitive: bool = True,
    context_lines: int = 0,
) -> str:
    """
    Search for a pattern in the library using ripgrep (or grep fallback).

    Deterministic search - same pattern always returns same results.

    Args:
        pattern: Search pattern (regex supported)
        path: Subdirectory to search in (relative to library root)
        max_results: Maximum number of results to return
        case_insensitive: Whether to ignore case
        context_lines: Number of context lines around matches

    Returns:
        Matching lines with file paths, or "No matches found"

    Example:
        >>> grep_search("PPO-Clip", "by_id/paper_042")
        "paper_042/methods.txt:15: PPO-Clip uses a clipped surrogate objective..."
    """
    search_path = Path(DEFAULT_LIBRARY_PATH)
    if path:
        search_path = search_path / path

    # Prefer ripgrep (rg) for speed, fallback to grep
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        use_rg = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        use_rg = False

    if use_rg:
        cmd = ["rg", "--max-count", str(max_results)]
        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        cmd.extend([pattern, str(search_path)])
    else:
        cmd = ["grep", "-r", f"--max-count={max_results}"]
        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        cmd.extend([pattern, str(search_path)])

    result = _run_command(cmd)

    if not result.success:
        return f"Error: {result.error}"

    # Truncate very long outputs
    output = result.output
    if len(output) > 5000:
        lines = output.split('\n')[:max_results]
        output = '\n'.join(lines) + f"\n... (truncated, {len(lines)} matches shown)"

    return output if output else "No matches found."


def read_file_chunk(
    path: str,
    start: int = 1,
    num_lines: int = 50,
) -> str:
    """
    Read a specific chunk of a file using sed for streaming.

    Efficient for large files - only reads requested portion.

    Args:
        path: File path relative to library root
        start: Starting line number (1-indexed)
        num_lines: Number of lines to read

    Returns:
        File content with line numbers

    Example:
        >>> read_file_chunk("by_id/paper_042/methods.txt", start=10, num_lines=30)
        "10: The training procedure consists of...\n11: ..."
    """
    file_path = Path(DEFAULT_LIBRARY_PATH) / path

    if not file_path.exists():
        return f"Error: File not found: {path}"

    if not file_path.is_file():
        return f"Error: Not a file: {path}"

    end = start + num_lines - 1

    # Use sed for efficient chunk reading
    cmd = ["sed", "-n", f"{start},{end}p", str(file_path)]
    result = _run_command(cmd)

    if not result.success:
        return f"Error: {result.error}"

    # Add line numbers
    lines = result.output.split('\n')
    numbered = [f"{start + i}: {line}" for i, line in enumerate(lines)]

    return '\n'.join(numbered)


def find_files(
    name_pattern: str,
    file_type: str = "f",
    max_results: int = 50,
) -> str:
    """
    Find files by name pattern in the library.

    Args:
        name_pattern: Glob pattern for file names (e.g., "*methods*", "*.txt")
        file_type: "f" for files, "d" for directories
        max_results: Maximum number of results

    Returns:
        List of matching file paths

    Example:
        >>> find_files("*reward*")
        "by_id/paper_012/sections/reward_modeling.txt\n..."
    """
    cmd = [
        "find",
        DEFAULT_LIBRARY_PATH,
        "-type", file_type,
        "-name", name_pattern,
    ]

    result = _run_command(cmd, timeout=60)

    if not result.success:
        return f"Error: {result.error}"

    # Make paths relative to library root
    lines = result.output.strip().split('\n')
    relative = []
    for line in lines[:max_results]:
        if line:
            try:
                rel = Path(line).relative_to(DEFAULT_LIBRARY_PATH)
                relative.append(str(rel))
            except ValueError:
                relative.append(line)

    return '\n'.join(relative) if relative else "No files found."


def verify_quote(path: str, snippet: str) -> str:
    """
    Verify that a quote exists in the specified file.

    Anti-hallucination tool - checks if the agent's citation is accurate.

    Args:
        path: File path relative to library root
        snippet: Text snippet to verify (at least 20 chars recommended)

    Returns:
        "VERIFIED: Quote found at line X" or "NOT FOUND: Quote does not exist"

    Example:
        >>> verify_quote("by_id/paper_042/methods.txt", "clipped surrogate objective")
        "VERIFIED: Quote found at line 15"
    """
    file_path = Path(DEFAULT_LIBRARY_PATH) / path

    if not file_path.exists():
        return f"Error: File not found: {path}"

    # Use grep to find the snippet
    cmd = ["grep", "-n", "-F", snippet, str(file_path)]
    result = _run_command(cmd)

    if result.success and result.output:
        # Extract line number from first match
        first_line = result.output.split('\n')[0]
        line_num = first_line.split(':')[0]
        return f"VERIFIED: Quote found at line {line_num}"
    else:
        return "NOT FOUND: Quote does not exist in the specified file."


def list_papers(
    topic: Optional[str] = None,
    keyword: Optional[str] = None,
    max_results: int = 100,
) -> str:
    """
    List papers from the library index.

    Fast metadata-based lookup without reading full files.

    Args:
        topic: Filter by topic category (rl, alignment, scaling, etc.)
        keyword: Filter by keyword in title/abstract
        max_results: Maximum number of results

    Returns:
        JSON-formatted list of papers with IDs and titles

    Example:
        >>> list_papers(keyword="reward")
        '[{"paper_id": "paper_012", "title": "Reward Modeling..."}, ...]'
    """
    index_path = Path(DEFAULT_LIBRARY_PATH) / "index.json"

    if not index_path.exists():
        return "Error: Library index not found. Run conversion first."

    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return "Error: Invalid index.json format."

    results = []

    for paper in index:
        # Filter by topic
        if topic and paper.get("topic", "").lower() != topic.lower():
            continue

        # Filter by keyword
        if keyword:
            title = paper.get("title", "").lower()
            subject = paper.get("subject", "").lower()
            if keyword.lower() not in title and keyword.lower() not in subject:
                continue

        results.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title", "Unknown")[:100],
            "author": paper.get("author", "Unknown"),
            "topic": paper.get("topic", "unclassified"),
        })

        if len(results) >= max_results:
            break

    return json.dumps(results, ensure_ascii=False, indent=2)


def read_abstract(paper_id: str) -> str:
    """
    Read the abstract of a specific paper.

    Quick lookup for deciding whether to read the full paper.

    Args:
        paper_id: Paper identifier (e.g., "paper_042")

    Returns:
        Abstract text or error message

    Example:
        >>> read_abstract("paper_042")
        "We present a novel approach to reinforcement learning..."
    """
    abstract_path = Path(DEFAULT_LIBRARY_PATH) / "by_id" / paper_id / "abstract.txt"

    if not abstract_path.exists():
        return f"Error: Abstract not found for {paper_id}"

    content = abstract_path.read_text(encoding="utf-8").strip()

    if not content:
        return f"No abstract available for {paper_id}"

    return content


def get_paper_metadata(paper_id: str) -> str:
    """
    Get full metadata for a specific paper.

    Args:
        paper_id: Paper identifier

    Returns:
        JSON-formatted metadata

    Example:
        >>> get_paper_metadata("paper_042")
        '{"paper_id": "paper_042", "title": "...", "sections": [...], ...}'
    """
    metadata_path = Path(DEFAULT_LIBRARY_PATH) / "by_id" / paper_id / "metadata.json"

    if not metadata_path.exists():
        return f"Error: Metadata not found for {paper_id}"

    return metadata_path.read_text(encoding="utf-8")


# Tool schemas for verifiers integration
BASH_TOOL_SCHEMAS = {
    "grep_search": {
        "name": "grep_search",
        "description": "Search for a pattern in the paper library using grep/ripgrep. Returns matching lines with file paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex supported)"
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (relative to library root)",
                    "default": ""
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 20
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case",
                    "default": True
                }
            },
            "required": ["pattern"]
        }
    },
    "read_file_chunk": {
        "name": "read_file_chunk",
        "description": "Read a specific portion of a file. Efficient for large files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to library root"
                },
                "start": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed)",
                    "default": 1
                },
                "num_lines": {
                    "type": "integer",
                    "description": "Number of lines to read",
                    "default": 50
                }
            },
            "required": ["path"]
        }
    },
    "find_files": {
        "name": "find_files",
        "description": "Find files by name pattern in the library.",
        "parameters": {
            "type": "object",
            "properties": {
                "name_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (e.g., '*methods*', '*.txt')"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 50
                }
            },
            "required": ["name_pattern"]
        }
    },
    "verify_quote": {
        "name": "verify_quote",
        "description": "Verify that a quote exists in a file. Use to check citation accuracy.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to library root"
                },
                "snippet": {
                    "type": "string",
                    "description": "Text snippet to verify"
                }
            },
            "required": ["path", "snippet"]
        }
    },
    "list_papers": {
        "name": "list_papers",
        "description": "List papers from the library index with optional filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Filter by topic (rl, alignment, scaling, etc.)"
                },
                "keyword": {
                    "type": "string",
                    "description": "Filter by keyword in title"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 100
                }
            }
        }
    },
    "read_abstract": {
        "name": "read_abstract",
        "description": "Read the abstract of a specific paper for quick preview.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper identifier (e.g., 'paper_042')"
                }
            },
            "required": ["paper_id"]
        }
    },
    "get_paper_metadata": {
        "name": "get_paper_metadata",
        "description": "Get full metadata for a specific paper.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper identifier"
                }
            },
            "required": ["paper_id"]
        }
    }
}
