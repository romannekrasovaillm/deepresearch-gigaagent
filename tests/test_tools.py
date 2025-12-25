"""Tests for agent tools."""

import pytest
import tempfile
from pathlib import Path

from src.tools.bash_tools import (
    GrepSearch,
    ReadFileChunk,
    FindFiles,
    VerifyQuote,
)
from src.tools.code_interpreter import CodeInterpreter
from src.tools.scratchpad import Scratchpad


@pytest.fixture
def temp_library():
    """Create temporary library structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        library_path = Path(tmpdir)

        # Create structure
        (library_path / "by_id" / "paper_0001" / "sections").mkdir(parents=True)

        # Create test files
        (library_path / "index.json").write_text('[]')

        full_text = library_path / "by_id" / "paper_0001" / "full_text.md"
        full_text.write_text("""# Test Paper

## Abstract
This is a test paper about reinforcement learning.

## Methods
We use PPO with batch size 256.
The learning rate is 3e-4.

## Results
Accuracy improved by 15%.
""")

        methods = library_path / "by_id" / "paper_0001" / "sections" / "methods.txt"
        methods.write_text("We use PPO with batch size 256.\nThe learning rate is 3e-4.")

        yield library_path


@pytest.fixture
def temp_workspace():
    """Create temporary workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestGrepSearch:
    """Tests for grep_search tool."""

    def test_basic_search(self, temp_library):
        tool = GrepSearch(temp_library)
        result = tool.execute(pattern="PPO")

        assert result.success
        assert "PPO" in result.output

    def test_case_insensitive(self, temp_library):
        tool = GrepSearch(temp_library)
        result = tool.execute(pattern="ppo", case_insensitive=True)

        assert result.success
        assert "PPO" in result.output or "ppo" in result.output

    def test_no_matches(self, temp_library):
        tool = GrepSearch(temp_library)
        result = tool.execute(pattern="nonexistent_pattern_xyz")

        assert result.success
        assert "No matches" in result.output or result.output == ""


class TestReadFileChunk:
    """Tests for read_file_chunk tool."""

    def test_read_file(self, temp_library):
        tool = ReadFileChunk(temp_library)
        result = tool.execute(
            path="by_id/paper_0001/sections/methods.txt",
            start=1,
            num_lines=10
        )

        assert result.success
        assert "PPO" in result.output

    def test_file_not_found(self, temp_library):
        tool = ReadFileChunk(temp_library)
        result = tool.execute(path="nonexistent_file.txt")

        assert not result.success
        assert result.error


class TestVerifyQuote:
    """Tests for verify_quote tool."""

    def test_exact_match(self, temp_library):
        tool = VerifyQuote(temp_library)
        result = tool.execute(
            path="by_id/paper_0001/sections/methods.txt",
            snippet="batch size 256",
            fuzzy=False
        )

        assert result.success
        assert "VERIFIED" in result.output

    def test_fuzzy_match(self, temp_library):
        tool = VerifyQuote(temp_library)
        result = tool.execute(
            path="by_id/paper_0001/sections/methods.txt",
            snippet="PPO batch size",
            fuzzy=True
        )

        assert result.success

    def test_not_found(self, temp_library):
        tool = VerifyQuote(temp_library)
        result = tool.execute(
            path="by_id/paper_0001/sections/methods.txt",
            snippet="completely wrong quote that doesn't exist",
            fuzzy=False
        )

        assert result.success
        assert "NOT FOUND" in result.output


class TestCodeInterpreter:
    """Tests for code interpreter."""

    def test_simple_code(self, temp_workspace):
        tool = CodeInterpreter(temp_workspace)
        result = tool.execute(code="print(2 + 2)")

        assert result.success
        assert "4" in result.output

    def test_pandas(self, temp_workspace):
        tool = CodeInterpreter(temp_workspace)
        result = tool.execute(code="""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df['a'].mean())
""")

        assert result.success
        assert "2" in result.output

    def test_blocked_operation(self, temp_workspace):
        tool = CodeInterpreter(temp_workspace)
        result = tool.execute(code="import os; os.system('ls')")

        assert not result.success
        assert "Blocked" in result.error

    def test_timeout(self, temp_workspace):
        tool = CodeInterpreter(temp_workspace, timeout=1)
        result = tool.execute(code="import time; time.sleep(10)")

        assert not result.success
        assert "timeout" in result.error.lower()


class TestScratchpad:
    """Tests for scratchpad."""

    def test_add_and_read(self, temp_workspace):
        tool = Scratchpad(temp_workspace)

        # Add note
        result = tool.execute(action="add", text="Test note", tag="finding")
        assert result.success
        assert "added" in result.output.lower()

        # Read notes
        result = tool.execute(action="read")
        assert result.success
        assert "Test note" in result.output

    def test_clear(self, temp_workspace):
        tool = Scratchpad(temp_workspace)

        # Add note
        tool.execute(action="add", text="Note to clear")

        # Clear
        result = tool.execute(action="clear")
        assert result.success

        # Verify empty
        result = tool.execute(action="read")
        assert "No notes" in result.output or result.output.strip() == ""
