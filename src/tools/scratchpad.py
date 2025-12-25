"""
Scratchpad for agent's working memory.

Allows the agent to take notes, save findings, and maintain
context across long research sessions.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

from .bash_tools import BaseTool, ToolResult


class Scratchpad(BaseTool):
    """
    Agent's working memory / notebook.

    Provides persistent note-taking capabilities so the agent
    can record findings and reference them later during synthesis.
    """

    name = "scratchpad"
    description = (
        "Agent's working notebook for recording findings. "
        "Use 'add' to save important discoveries, quotes, or analysis. "
        "Use 'read' to review notes before final synthesis. "
        "Use 'clear' to start fresh for a new task."
    )

    def __init__(self, workspace_path: str | Path):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.notes_file = self.workspace_path / "notes.md"
        self.figures_dir = self.workspace_path / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def execute(
        self,
        action: str,
        text: Optional[str] = None,
        tag: Optional[str] = None
    ) -> ToolResult:
        """
        Interact with scratchpad.

        Args:
            action: 'add', 'read', 'clear', or 'search'
            text: Text to add (for 'add' action)
            tag: Optional tag for organizing notes (e.g., 'finding', 'quote', 'idea')

        Returns:
            ToolResult with operation result
        """
        action = action.lower().strip()

        if action == "add":
            return self._add_note(text, tag)
        elif action == "read":
            return self._read_notes()
        elif action == "clear":
            return self._clear_notes()
        elif action == "search":
            return self._search_notes(text)
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown action: {action}. Use 'add', 'read', 'clear', or 'search'."
            )

    def _add_note(self, text: Optional[str], tag: Optional[str]) -> ToolResult:
        """Add a note to the scratchpad."""
        if not text:
            return ToolResult(
                success=False,
                output="",
                error="No text provided to add"
            )

        timestamp = datetime.now().strftime("%H:%M:%S")
        tag_str = f"[{tag}] " if tag else ""

        entry = f"\n### {tag_str}{timestamp}\n{text}\n"

        # Append to notes file
        with open(self.notes_file, 'a', encoding='utf-8') as f:
            if f.tell() == 0:  # New file
                f.write("# Research Notes\n")
            f.write(entry)

        # Count total notes
        note_count = self._count_notes()

        return ToolResult(
            success=True,
            output=f"Note added ({note_count} total notes)"
        )

    def _read_notes(self) -> ToolResult:
        """Read all notes from scratchpad."""
        if not self.notes_file.exists():
            return ToolResult(
                success=True,
                output="(No notes yet)"
            )

        content = self.notes_file.read_text(encoding='utf-8')
        if not content.strip():
            return ToolResult(
                success=True,
                output="(No notes yet)"
            )

        return ToolResult(
            success=True,
            output=content
        )

    def _clear_notes(self) -> ToolResult:
        """Clear all notes."""
        if self.notes_file.exists():
            # Archive old notes
            archive_name = f"notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            archive_path = self.workspace_path / "archive" / archive_name
            archive_path.parent.mkdir(exist_ok=True)

            self.notes_file.rename(archive_path)

            return ToolResult(
                success=True,
                output=f"Notes cleared (archived to {archive_name})"
            )

        return ToolResult(
            success=True,
            output="Notes already empty"
        )

    def _search_notes(self, query: Optional[str]) -> ToolResult:
        """Search notes for a query string."""
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="No search query provided"
            )

        if not self.notes_file.exists():
            return ToolResult(
                success=True,
                output="No notes to search"
            )

        content = self.notes_file.read_text(encoding='utf-8')
        query_lower = query.lower()

        # Find matching sections
        sections = content.split('\n### ')
        matches = []

        for section in sections[1:]:  # Skip header
            if query_lower in section.lower():
                matches.append(f"### {section.split(chr(10))[0]}")

        if not matches:
            return ToolResult(
                success=True,
                output=f"No matches for '{query}'"
            )

        return ToolResult(
            success=True,
            output=f"Found {len(matches)} matches:\n" + '\n'.join(matches)
        )

    def _count_notes(self) -> int:
        """Count number of note entries."""
        if not self.notes_file.exists():
            return 0

        content = self.notes_file.read_text(encoding='utf-8')
        return content.count('\n### ')

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "read", "clear", "search"],
                        "description": "Action to perform"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to add (for 'add') or search query (for 'search')"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag: 'finding', 'quote', 'idea', 'question'"
                    }
                },
                "required": ["action"]
            }
        }


class AddToNotes(BaseTool):
    """Convenience wrapper for adding notes."""

    name = "add_to_notes"
    description = (
        "Add a finding or note to the research notebook. "
        "Use to record important discoveries for later synthesis."
    )

    def __init__(self, workspace_path: str | Path):
        self._scratchpad = Scratchpad(workspace_path)

    def execute(self, text: str, tag: Optional[str] = None) -> ToolResult:
        """Add note to scratchpad."""
        return self._scratchpad.execute("add", text, tag)

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Note text to save"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional category tag"
                    }
                },
                "required": ["text"]
            }
        }


class ReadNotes(BaseTool):
    """Convenience wrapper for reading notes."""

    name = "read_notes"
    description = (
        "Read all notes from the research notebook. "
        "Use before final synthesis to review findings."
    )

    def __init__(self, workspace_path: str | Path):
        self._scratchpad = Scratchpad(workspace_path)

    def execute(self) -> ToolResult:
        """Read all notes."""
        return self._scratchpad.execute("read")

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }


class SaveFigure(BaseTool):
    """Save figure with metadata."""

    name = "save_figure"
    description = (
        "Save a generated figure with description. "
        "Call after execute_python that generates plots."
    )

    def __init__(self, workspace_path: str | Path):
        self.workspace_path = Path(workspace_path)
        self.figures_dir = self.workspace_path / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.figures_dir / "manifest.json"

    def execute(
        self,
        source_path: str,
        description: str,
        new_name: Optional[str] = None
    ) -> ToolResult:
        """
        Save and catalog a figure.

        Args:
            source_path: Path to the figure file
            description: Description of what the figure shows
            new_name: Optional new filename

        Returns:
            ToolResult with save confirmation
        """
        import json
        import shutil

        source = Path(source_path)
        if not source.exists():
            # Try relative to workspace
            source = self.workspace_path / source_path
            if not source.exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Figure not found: {source_path}"
                )

        # Generate name if not provided
        if new_name:
            dest_name = new_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest_name = f"figure_{timestamp}{source.suffix}"

        dest_path = self.figures_dir / dest_name

        # Copy file
        shutil.copy2(source, dest_path)

        # Update manifest
        manifest = []
        if self.manifest_file.exists():
            try:
                manifest = json.loads(self.manifest_file.read_text())
            except json.JSONDecodeError:
                pass

        manifest.append({
            "filename": dest_name,
            "description": description,
            "created": datetime.now().isoformat(),
            "source": source_path,
        })

        self.manifest_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )

        return ToolResult(
            success=True,
            output=f"Figure saved: {dest_path}"
        )

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to figure file"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the figure shows"
                    },
                    "new_name": {
                        "type": "string",
                        "description": "Optional new filename"
                    }
                },
                "required": ["source_path", "description"]
            }
        }
