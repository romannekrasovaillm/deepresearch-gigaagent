"""Scratchpad tools for agent's working memory."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


class Scratchpad:
    """Agent's working memory for notes and artifacts."""

    def __init__(self, workspace_path: str | Path):
        self.workspace = Path(workspace_path)
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.notes_path = self.workspace / "notes.md"
        self.figures_dir = self.workspace / "figures"
        self.figures_dir.mkdir(exist_ok=True)

    def add_to_notes(self, text: str, section: Optional[str] = None) -> str:
        """
        Add text to notes file.

        Args:
            text: Text to add
            section: Optional section header

        Returns:
            Confirmation message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.notes_path, "a", encoding="utf-8") as f:
            if section:
                f.write(f"\n## {section}\n")
            f.write(f"*[{timestamp}]*\n\n")
            f.write(text.strip())
            f.write("\n\n---\n")

        return f"Added to notes: {len(text)} characters"

    def read_notes(self) -> str:
        """
        Read all notes.

        Returns:
            Content of notes file
        """
        if not self.notes_path.exists():
            return "Notes are empty."

        content = self.notes_path.read_text(encoding="utf-8")
        if not content.strip():
            return "Notes are empty."

        return content

    def clear_notes(self) -> str:
        """Clear all notes."""
        if self.notes_path.exists():
            self.notes_path.unlink()
        return "Notes cleared."

    def save_figure(
        self,
        source_path: str | Path,
        name: Optional[str] = None,
    ) -> str:
        """
        Save a figure to workspace.

        Args:
            source_path: Path to figure file
            name: Optional custom name

        Returns:
            Path to saved figure
        """
        source = Path(source_path)
        if not source.exists():
            return f"Error: Source file not found: {source_path}"

        if name is None:
            name = source.name

        dest = self.figures_dir / name

        # Handle name conflicts
        counter = 1
        while dest.exists():
            stem = source.stem
            suffix = source.suffix
            dest = self.figures_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        shutil.copy2(source, dest)
        return f"Figure saved: {dest}"

    def list_figures(self) -> list[str]:
        """List all saved figures."""
        return [str(f) for f in self.figures_dir.glob("*") if f.is_file()]

    def get_figure_path(self, name: str) -> Optional[Path]:
        """Get path to a figure by name."""
        path = self.figures_dir / name
        if path.exists():
            return path
        return None

    def reset(self) -> str:
        """Reset workspace (clear notes and figures)."""
        self.clear_notes()
        for f in self.figures_dir.glob("*"):
            f.unlink()
        return "Workspace reset."


# Global instance
_scratchpad: Optional[Scratchpad] = None


def _get_scratchpad() -> Scratchpad:
    """Get or create global scratchpad."""
    global _scratchpad
    if _scratchpad is None:
        _scratchpad = Scratchpad("./data/workspace")
    return _scratchpad


def init_scratchpad(workspace_path: str | Path) -> None:
    """Initialize global scratchpad."""
    global _scratchpad
    _scratchpad = Scratchpad(workspace_path)


# Functional API
def add_to_notes(text: str, section: Optional[str] = None) -> str:
    """Add text to notes."""
    return _get_scratchpad().add_to_notes(text, section)


def read_notes() -> str:
    """Read all notes."""
    return _get_scratchpad().read_notes()


def clear_notes() -> str:
    """Clear all notes."""
    return _get_scratchpad().clear_notes()


def save_figure(source_path: str, name: Optional[str] = None) -> str:
    """Save figure to workspace."""
    return _get_scratchpad().save_figure(source_path, name)


def list_figures() -> str:
    """List all saved figures."""
    figures = _get_scratchpad().list_figures()
    if not figures:
        return "No figures saved."
    return "\n".join(figures)
