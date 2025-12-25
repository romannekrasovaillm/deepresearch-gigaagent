"""Tests for agent tools."""

import pytest
import tempfile
from pathlib import Path

from src.tools.bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    verify_quote,
)
from src.tools.code_interpreter import execute_python, validate_code
from src.tools.scratchpad import add_to_notes, read_notes, clear_notes


class TestBashTools:
    """Tests for bash-based tools."""

    def test_grep_search_no_results(self):
        """Test grep_search with no matches."""
        result = grep_search("nonexistent_pattern_xyz123")
        assert "No matches found" in result or "Error" in result

    def test_read_file_chunk_not_found(self):
        """Test read_file_chunk with nonexistent file."""
        result = read_file_chunk("nonexistent/file.txt")
        assert "Error" in result or "not found" in result.lower()

    def test_find_files_pattern(self):
        """Test find_files with pattern."""
        result = find_files("*.md")
        # Should not error
        assert isinstance(result, str)


class TestCodeInterpreter:
    """Tests for Python code interpreter."""

    def test_simple_code(self):
        """Test simple Python execution."""
        result = execute_python("print('Hello, World!')")
        assert "Hello, World!" in result

    def test_pandas_available(self):
        """Test pandas is available."""
        result = execute_python("""
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.sum())
""")
        assert "6" in result or "a" in result

    def test_forbidden_import(self):
        """Test that forbidden imports are blocked."""
        is_valid, error = validate_code("import os")
        assert not is_valid
        assert "Forbidden" in error or "not allowed" in error.lower()

    def test_forbidden_eval(self):
        """Test that eval is blocked."""
        is_valid, error = validate_code("eval('1+1')")
        assert not is_valid

    def test_numpy_calculation(self):
        """Test numpy calculation."""
        result = execute_python("""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {arr.mean()}")
""")
        assert "3.0" in result or "Mean" in result


class TestScratchpad:
    """Tests for scratchpad/notes functionality."""

    def test_add_and_read_notes(self):
        """Test adding and reading notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clear first
            clear_notes(tmpdir)

            # Add note
            result = add_to_notes("Test finding", workspace_dir=tmpdir)
            assert "Added" in result

            # Read notes
            notes = read_notes(tmpdir)
            assert "Test finding" in notes

    def test_empty_notes(self):
        """Test reading empty notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            clear_notes(tmpdir)
            result = read_notes(tmpdir)
            assert "empty" in result.lower() or "no notes" in result.lower()


class TestCodeValidation:
    """Tests for code security validation."""

    @pytest.mark.parametrize("code,should_pass", [
        ("import pandas as pd", True),
        ("import numpy as np", True),
        ("import os", False),
        ("import subprocess", False),
        ("eval('code')", False),
        ("exec('code')", False),
        ("open('file.txt')", False),
        ("__import__('os')", False),
        ("import scipy.stats", True),
        ("from collections import Counter", True),
    ])
    def test_code_validation(self, code, should_pass):
        """Test various code patterns for validation."""
        is_valid, _ = validate_code(code)
        assert is_valid == should_pass, f"Code '{code}' validation mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
