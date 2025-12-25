#!/usr/bin/env python3
"""
Scratchpad Module
Agent's working memory for notes and findings.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field
import shutil


@dataclass
class Note:
    """A single note entry."""
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: Optional[str] = None  # paper_id or file path
    category: str = "general"


@dataclass
class Scratchpad:
    """Agent's working memory."""
    notes: list[Note] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    summary: str = ""


# Default workspace path
DEFAULT_WORKSPACE = "/workspace"

# In-memory scratchpad (per session)
_scratchpad: Optional[Scratchpad] = None


def get_scratchpad(workspace_dir: str = DEFAULT_WORKSPACE) -> Scratchpad:
    """Get or create scratchpad instance."""
    global _scratchpad
    if _scratchpad is None:
        _scratchpad = Scratchpad()

        # Try to load from file if exists
        workspace = Path(workspace_dir)
        notes_file = workspace / "notes.json"
        if notes_file.exists():
            try:
                data = json.loads(notes_file.read_text())
                _scratchpad = Scratchpad(
                    notes=[Note(**n) for n in data.get('notes', [])],
                    figures=data.get('figures', []),
                    summary=data.get('summary', ''),
                )
            except Exception:
                pass

    return _scratchpad


def save_scratchpad(workspace_dir: str = DEFAULT_WORKSPACE):
    """Persist scratchpad to disk."""
    scratchpad = get_scratchpad(workspace_dir)
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    notes_json = workspace / "notes.json"
    data = {
        'notes': [asdict(n) for n in scratchpad.notes],
        'figures': scratchpad.figures,
        'summary': scratchpad.summary,
    }
    notes_json.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # Also save as readable Markdown
    notes_md = workspace / "notes.md"
    md_content = format_notes_markdown(scratchpad)
    notes_md.write_text(md_content)


def format_notes_markdown(scratchpad: Scratchpad) -> str:
    """Format scratchpad as readable Markdown."""
    lines = ["# Research Notes\n"]

    if scratchpad.summary:
        lines.append(f"## Summary\n\n{scratchpad.summary}\n")

    if scratchpad.notes:
        lines.append("## Notes\n")

        # Group by category
        by_category = {}
        for note in scratchpad.notes:
            cat = note.category or "general"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(note)

        for category, notes in by_category.items():
            lines.append(f"### {category.title()}\n")
            for note in notes:
                source_info = f" (from {note.source})" if note.source else ""
                lines.append(f"- {note.content}{source_info}")
            lines.append("")

    if scratchpad.figures:
        lines.append("## Figures\n")
        for fig in scratchpad.figures:
            lines.append(f"- {fig}")
        lines.append("")

    return "\n".join(lines)


async def add_to_notes(
    text: str,
    source: Optional[str] = None,
    category: str = "general",
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Add a note to the scratchpad.

    Args:
        text: Content of the note
        source: Source paper_id or file path
        category: Category (findings, quotes, questions, general)
        workspace_dir: Workspace directory

    Returns:
        Confirmation message
    """
    scratchpad = get_scratchpad(workspace_dir)

    note = Note(
        content=text.strip(),
        source=source,
        category=category,
    )
    scratchpad.notes.append(note)

    # Auto-save
    save_scratchpad(workspace_dir)

    return f"[Note added] Category: {category}, Source: {source or 'none'}"


async def read_notes(
    category: Optional[str] = None,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Read all notes from scratchpad.

    Args:
        category: Filter by category (optional)
        workspace_dir: Workspace directory

    Returns:
        Formatted notes as string
    """
    scratchpad = get_scratchpad(workspace_dir)

    if not scratchpad.notes:
        return "[No notes yet]"

    notes = scratchpad.notes
    if category:
        notes = [n for n in notes if n.category == category]

    if not notes:
        return f"[No notes in category: {category}]"

    lines = []
    for i, note in enumerate(notes, 1):
        source_info = f" [from {note.source}]" if note.source else ""
        lines.append(f"{i}. [{note.category}] {note.content}{source_info}")

    return "\n".join(lines)


async def clear_notes(
    category: Optional[str] = None,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Clear notes from scratchpad.

    Args:
        category: Clear only specific category (optional)
        workspace_dir: Workspace directory

    Returns:
        Confirmation message
    """
    scratchpad = get_scratchpad(workspace_dir)

    if category:
        original_count = len(scratchpad.notes)
        scratchpad.notes = [n for n in scratchpad.notes if n.category != category]
        removed = original_count - len(scratchpad.notes)
        save_scratchpad(workspace_dir)
        return f"[Cleared {removed} notes from category: {category}]"
    else:
        count = len(scratchpad.notes)
        scratchpad.notes = []
        save_scratchpad(workspace_dir)
        return f"[Cleared all {count} notes]"


async def save_figure(
    source_path: str,
    description: str = "",
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Register a figure in the scratchpad.

    Args:
        source_path: Path to the figure file
        description: Description of the figure
        workspace_dir: Workspace directory

    Returns:
        Confirmation message
    """
    scratchpad = get_scratchpad(workspace_dir)
    workspace = Path(workspace_dir)

    source = Path(source_path)
    if not source.exists():
        return f"[Error] Figure not found: {source_path}"

    # Copy to workspace if not already there
    if not str(source).startswith(str(workspace)):
        dest = workspace / "figures" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)
        final_path = str(dest)
    else:
        final_path = source_path

    entry = f"{final_path}: {description}" if description else final_path
    scratchpad.figures.append(entry)

    save_scratchpad(workspace_dir)
    return f"[Figure saved] {final_path}"


async def set_summary(
    summary: str,
    workspace_dir: str = DEFAULT_WORKSPACE,
) -> str:
    """
    Set the research summary.

    Args:
        summary: Summary text
        workspace_dir: Workspace directory

    Returns:
        Confirmation message
    """
    scratchpad = get_scratchpad(workspace_dir)
    scratchpad.summary = summary.strip()
    save_scratchpad(workspace_dir)
    return "[Summary updated]"


def reset_scratchpad():
    """Reset scratchpad (for testing)."""
    global _scratchpad
    _scratchpad = None


# Tool definitions for ToolEnv
SCRATCHPAD_TOOLS = {
    "add_to_notes": {
        "function": add_to_notes,
        "description": "Add important findings to your research notes. Notes persist across tool calls.",
        "parameters": {
            "text": {"type": "string", "description": "Content of the note", "required": True},
            "source": {"type": "string", "description": "Source paper_id or file path"},
            "category": {
                "type": "string",
                "description": "Category: findings, quotes, questions, general",
                "default": "general",
            },
        },
    },
    "read_notes": {
        "function": read_notes,
        "description": "Read your research notes. Use before writing final answer.",
        "parameters": {
            "category": {"type": "string", "description": "Filter by category (optional)"},
        },
    },
    "clear_notes": {
        "function": clear_notes,
        "description": "Clear notes from scratchpad.",
        "parameters": {
            "category": {"type": "string", "description": "Clear only this category (optional)"},
        },
    },
    "save_figure": {
        "function": save_figure,
        "description": "Register a generated figure in your notes.",
        "parameters": {
            "source_path": {"type": "string", "description": "Path to figure file", "required": True},
            "description": {"type": "string", "description": "Description of figure"},
        },
    },
    "set_summary": {
        "function": set_summary,
        "description": "Set the overall research summary.",
        "parameters": {
            "summary": {"type": "string", "description": "Summary text", "required": True},
        },
    },
}
