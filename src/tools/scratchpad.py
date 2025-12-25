"""
Scratchpad - Agent's working memory for notes and findings.

Allows the agent to persist findings between tool calls,
avoiding "forgetting" important information during long research sessions.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default workspace path
DEFAULT_WORKSPACE = "/workspace"


def _get_notes_path(workspace_dir: str = DEFAULT_WORKSPACE) -> Path:
    """Get path to notes file."""
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / "notes.md"


def _get_data_path(workspace_dir: str = DEFAULT_WORKSPACE) -> Path:
    """Get path to structured data file."""
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace / "data.json"


def add_to_notes(
    text: str,
    section: Optional[str] = None,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Add text to the agent's working notes.

    Use to save important findings, quotes, or intermediate results
    that will be needed for final synthesis.

    Args:
        text: Text to add to notes
        section: Optional section header (e.g., "Key Findings", "Quotes")
        workspace_dir: Workspace directory path

    Returns:
        Confirmation message

    Example:
        >>> add_to_notes("Paper X shows 12% improvement over baseline", section="Results")
        "Added to notes under 'Results'"
    """
    notes_path = _get_notes_path(workspace_dir)

    timestamp = datetime.now().strftime("%H:%M:%S")

    # Build note entry
    entry_parts = []
    if section:
        entry_parts.append(f"\n## {section}\n")
    entry_parts.append(f"[{timestamp}] {text}\n")
    entry = ''.join(entry_parts)

    # Append to notes file
    with open(notes_path, 'a', encoding='utf-8') as f:
        f.write(entry)

    return f"Added to notes{f' under \"{section}\"' if section else ''}"


def read_notes(workspace_dir: str = DEFAULT_WORKSPACE) -> str:
    """
    Read all saved notes.

    Use before final synthesis to recall all findings.

    Args:
        workspace_dir: Workspace directory path

    Returns:
        All notes content or message if empty

    Example:
        >>> read_notes()
        "## Key Findings\\n[10:23:45] Paper X shows 12% improvement..."
    """
    notes_path = _get_notes_path(workspace_dir)

    if not notes_path.exists():
        return "No notes yet. Use add_to_notes() to save findings."

    content = notes_path.read_text(encoding='utf-8').strip()

    if not content:
        return "Notes are empty. Use add_to_notes() to save findings."

    return content


def clear_notes(workspace_dir: str = DEFAULT_WORKSPACE) -> str:
    """
    Clear all notes (start fresh session).

    Args:
        workspace_dir: Workspace directory path

    Returns:
        Confirmation message
    """
    notes_path = _get_notes_path(workspace_dir)

    if notes_path.exists():
        # Backup old notes
        backup_path = notes_path.with_suffix('.md.bak')
        shutil.copy(notes_path, backup_path)
        notes_path.unlink()

    return "Notes cleared (backup saved as notes.md.bak)"


def save_figure(
    source_path: str,
    name: str,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Save a figure from code execution with a descriptive name.

    Args:
        source_path: Path to the generated figure
        name: Descriptive name for the figure
        workspace_dir: Workspace directory path

    Returns:
        Path to saved figure

    Example:
        >>> save_figure("/tmp/figure_123.png", "accuracy_comparison")
        "Figure saved: /workspace/figures/accuracy_comparison.png"
    """
    workspace = Path(workspace_dir)
    figures_dir = workspace / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source_path)
    if not source.exists():
        return f"Error: Source file not found: {source_path}"

    # Sanitize name
    safe_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
    dest_path = figures_dir / f"{safe_name}{source.suffix}"

    shutil.copy(source, dest_path)

    return f"Figure saved: {dest_path}"


def save_data(
    key: str,
    value: any,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Save structured data for later retrieval.

    Use for storing extracted metrics, paper comparisons, etc.

    Args:
        key: Data key (e.g., "paper_metrics", "comparison_results")
        value: Data to save (must be JSON-serializable)
        workspace_dir: Workspace directory path

    Returns:
        Confirmation message

    Example:
        >>> save_data("paper_scores", {"paper_042": 0.89, "paper_043": 0.92})
        "Data saved under key 'paper_scores'"
    """
    data_path = _get_data_path(workspace_dir)

    # Load existing data
    if data_path.exists():
        data = json.loads(data_path.read_text())
    else:
        data = {}

    data[key] = value
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    return f"Data saved under key '{key}'"


def load_data(
    key: Optional[str] = None,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Load saved structured data.

    Args:
        key: Specific key to load (None = all data)
        workspace_dir: Workspace directory path

    Returns:
        JSON-formatted data

    Example:
        >>> load_data("paper_scores")
        '{"paper_042": 0.89, "paper_043": 0.92}'
    """
    data_path = _get_data_path(workspace_dir)

    if not data_path.exists():
        return "No data saved yet. Use save_data() first."

    data = json.loads(data_path.read_text())

    if key:
        if key not in data:
            return f"Key '{key}' not found. Available: {list(data.keys())}"
        return json.dumps(data[key], ensure_ascii=False, indent=2)

    return json.dumps(data, ensure_ascii=False, indent=2)


# Tool schemas for verifiers integration
SCRATCHPAD_SCHEMAS = {
    "add_to_notes": {
        "name": "add_to_notes",
        "description": "Save important findings to working notes. Use to persist information between tool calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to add to notes"
                },
                "section": {
                    "type": "string",
                    "description": "Optional section header (e.g., 'Key Findings', 'Quotes')"
                }
            },
            "required": ["text"]
        }
    },
    "read_notes": {
        "name": "read_notes",
        "description": "Read all saved notes. Use before final synthesis to recall findings.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "save_figure": {
        "name": "save_figure",
        "description": "Save a generated figure with a descriptive name.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to the generated figure file"
                },
                "name": {
                    "type": "string",
                    "description": "Descriptive name for the figure"
                }
            },
            "required": ["source_path", "name"]
        }
    },
    "save_data": {
        "name": "save_data",
        "description": "Save structured data (metrics, comparisons) for later use.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Data key identifier"
                },
                "value": {
                    "description": "Data to save (JSON-serializable)"
                }
            },
            "required": ["key", "value"]
        }
    },
    "load_data": {
        "name": "load_data",
        "description": "Load previously saved structured data.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Specific key to load (omit for all data)"
                }
            }
        }
    }
}
