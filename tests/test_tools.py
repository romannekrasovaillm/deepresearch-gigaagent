#!/usr/bin/env python3
"""Tests for agent tools."""

import pytest
import asyncio
import tempfile
from pathlib import Path

from src.tools.bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    verify_quote,
)
from src.tools.code_interpreter import CodeInterpreter, execute_python
from src.tools.scratchpad import add_to_notes, read_notes, clear_notes, reset_scratchpad
from src.tools.tool_env import ToolEnv, ToolStatus


class TestBashTools:
    """Test bash-based tools."""

    def setup_method(self):
        """Create temp library for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.library = Path(self.temp_dir) / "library"
        self.library.mkdir()

        # Create test papers
        paper_dir = self.library / "by_id" / "paper_0001"
        paper_dir.mkdir(parents=True)

        (paper_dir / "full_text.md").write_text(
            "# Test Paper\n\nThis is about reinforcement learning.\n"
            "The batch size was 256.\n"
        )
        (paper_dir / "abstract.txt").write_text(
            "This paper studies RL methods."
        )

    def test_grep_search_finds_pattern(self):
        """Test grep finds patterns in files."""
        results = grep_search("reinforcement", str(self.library))
        assert len(results) > 0
        assert any("reinforcement" in r.content.lower() for r in results)

    def test_read_file_chunk(self):
        """Test reading file chunks."""
        file_path = self.library / "by_id" / "paper_0001" / "full_text.md"
        content = read_file_chunk(str(file_path), start=1, num_lines=5)
        assert "Test Paper" in content

    def test_find_files(self):
        """Test finding files by pattern."""
        files = find_files("*.md", str(self.library))
        assert len(files) > 0
        assert any("full_text.md" in f for f in files)

    def test_verify_quote_exists(self):
        """Test quote verification for existing quote."""
        file_path = self.library / "by_id" / "paper_0001" / "full_text.md"
        result = verify_quote(str(file_path), "batch size was 256")
        assert result["verified"] is True

    def test_verify_quote_missing(self):
        """Test quote verification for missing quote."""
        file_path = self.library / "by_id" / "paper_0001" / "full_text.md"
        result = verify_quote(str(file_path), "this text does not exist")
        assert result["verified"] is False


class TestCodeInterpreter:
    """Test Python code interpreter."""

    def setup_method(self):
        """Create temp workspace."""
        self.temp_dir = tempfile.mkdtemp()
        self.interpreter = CodeInterpreter(workspace_dir=self.temp_dir)

    def test_simple_calculation(self):
        """Test simple Python calculation."""
        result = self.interpreter.execute("1 + 1")
        assert result.success
        assert "2" in result.result_repr

    def test_pandas_available(self):
        """Test pandas is available."""
        result = self.interpreter.execute(
            "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nlen(df)"
        )
        assert result.success
        assert "3" in result.result_repr

    def test_numpy_calculation(self):
        """Test numpy calculation."""
        result = self.interpreter.execute(
            "import numpy as np\nnp.mean([1, 2, 3, 4, 5])"
        )
        assert result.success
        assert "3.0" in result.result_repr

    def test_forbidden_import(self):
        """Test forbidden imports are blocked."""
        result = self.interpreter.execute("import os")
        assert not result.success
        assert "not allowed" in result.error.lower()

    def test_syntax_error(self):
        """Test syntax errors are caught."""
        result = self.interpreter.execute("def broken(")
        assert not result.success


class TestScratchpad:
    """Test scratchpad functionality."""

    def setup_method(self):
        """Reset scratchpad before each test."""
        reset_scratchpad()
        self.temp_dir = tempfile.mkdtemp()

    @pytest.mark.asyncio
    async def test_add_and_read_notes(self):
        """Test adding and reading notes."""
        await add_to_notes("Test finding 1", source="paper_001", workspace_dir=self.temp_dir)
        await add_to_notes("Test finding 2", category="quotes", workspace_dir=self.temp_dir)

        notes = await read_notes(workspace_dir=self.temp_dir)
        assert "Test finding 1" in notes
        assert "Test finding 2" in notes
        assert "paper_001" in notes

    @pytest.mark.asyncio
    async def test_read_by_category(self):
        """Test reading notes by category."""
        await add_to_notes("General note", workspace_dir=self.temp_dir)
        await add_to_notes("Quote note", category="quotes", workspace_dir=self.temp_dir)

        notes = await read_notes(category="quotes", workspace_dir=self.temp_dir)
        assert "Quote note" in notes
        assert "General note" not in notes

    @pytest.mark.asyncio
    async def test_clear_notes(self):
        """Test clearing notes."""
        await add_to_notes("Note 1", workspace_dir=self.temp_dir)
        await add_to_notes("Note 2", workspace_dir=self.temp_dir)

        await clear_notes(workspace_dir=self.temp_dir)

        notes = await read_notes(workspace_dir=self.temp_dir)
        assert "No notes" in notes


class TestToolEnv:
    """Test tool environment."""

    def setup_method(self):
        """Create test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.env = ToolEnv(
            library_path=self.temp_dir,
            workspace_path=self.temp_dir,
            max_turns=10,
        )

    def test_tool_registration(self):
        """Test tools are registered."""
        assert "grep_search" in self.env.tools
        assert "execute_python" in self.env.tools
        assert "add_to_notes" in self.env.tools

    def test_get_tool_descriptions(self):
        """Test tool descriptions are generated."""
        desc = self.env.get_tool_descriptions()
        assert "grep_search" in desc
        assert "execute_python" in desc

    def test_reset_creates_episode(self):
        """Test reset creates new episode."""
        episode = self.env.reset("Test question")
        assert episode.question == "Test question"
        assert len(episode.turns) == 0

    @pytest.mark.asyncio
    async def test_step_executes_tool(self):
        """Test step executes tool and records turn."""
        self.env.reset("Test question")

        result = await self.env.step(
            "add_to_notes",
            {"text": "Test note"},
            "Testing note functionality",
        )

        assert result.status == ToolStatus.SUCCESS
        assert self.env.turn_count == 1
        assert len(self.env.current_episode.turns) == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_error(self):
        """Test unknown tool returns error."""
        self.env.reset("Test question")

        result = await self.env.step(
            "nonexistent_tool",
            {},
            "Testing error",
        )

        assert result.status == ToolStatus.ERROR
        assert "Unknown tool" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
