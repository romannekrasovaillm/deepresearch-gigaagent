"""Tests for the research environment."""

import pytest
import tempfile
from pathlib import Path
import json

from src.env.tool_env import ResearchToolEnv, create_env


class TestResearchToolEnv:
    """Tests for ResearchToolEnv."""

    @pytest.fixture
    def temp_library(self):
        """Create a temporary library for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lib_path = Path(tmpdir) / "library"
            lib_path.mkdir()

            # Create minimal library structure
            by_id = lib_path / "by_id"
            by_id.mkdir()

            paper_dir = by_id / "paper_0001"
            paper_dir.mkdir()

            # Metadata
            (paper_dir / "metadata.json").write_text(json.dumps({
                "paper_id": "paper_0001",
                "title": "Test Paper",
                "author": "Test Author",
            }))

            # Full text
            (paper_dir / "full_text.md").write_text("""
# Test Paper

## Abstract
This is a test paper about machine learning.

## Methods
We use a novel approach called TestMethod.

## Results
The results show 95% accuracy.
""")

            # Abstract
            (paper_dir / "abstract.txt").write_text(
                "This is a test paper about machine learning."
            )

            # Index
            (lib_path / "index.json").write_text(json.dumps([{
                "paper_id": "paper_0001",
                "title": "Test Paper",
            }]))

            yield str(lib_path)

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_env_creation(self, temp_library, temp_workspace):
        """Test environment can be created."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
        )
        assert env is not None
        assert env.max_turns == 15

    def test_env_reset(self, temp_library, temp_workspace):
        """Test environment reset."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
        )

        task = {
            "question": "What method does paper_0001 use?",
            "golden_answer": "TestMethod",
            "required_papers": ["paper_0001"],
        }

        obs = env.reset(task)

        assert obs is not None
        assert "question" in obs.content.lower() or "research" in obs.content.lower()
        assert env.state.question == task["question"]

    def test_env_step_unknown_tool(self, temp_library, temp_workspace):
        """Test step with unknown tool."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
        )

        env.reset({"question": "Test", "golden_answer": "Test"})

        obs, reward, done, info = env.step({
            "tool": "unknown_tool",
            "input": {},
        })

        assert "Unknown tool" in obs.content

    def test_env_step_final_answer(self, temp_library, temp_workspace):
        """Test step with final answer."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
        )

        env.reset({"question": "Test", "golden_answer": "Test"})

        obs, reward, done, info = env.step({
            "final_answer": "This is my answer."
        })

        assert done is True
        assert env.state.final_answer == "This is my answer."

    def test_env_max_turns(self, temp_library, temp_workspace):
        """Test environment respects max turns."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
            max_turns=3,
        )

        env.reset({"question": "Test", "golden_answer": "Test"})

        # Make 3 steps
        for i in range(3):
            obs, reward, done, info = env.step({
                "tool": "list_papers",
                "input": {},
            })

        # Should be done after max turns
        assert done is True
        assert "Maximum turns" in obs.content or obs.is_terminal

    def test_get_trajectory(self, temp_library, temp_workspace):
        """Test trajectory retrieval."""
        env = create_env(
            library_path=temp_library,
            workspace_path=temp_workspace,
        )

        env.reset({
            "question": "What is in paper_0001?",
            "golden_answer": "Test content",
            "required_papers": ["paper_0001"],
        })

        # Make some steps
        env.step({"tool": "list_papers", "input": {}})
        env.step({"final_answer": "The paper contains test content."})

        traj = env.get_trajectory()

        assert traj["question"] == "What is in paper_0001?"
        assert len(traj["tool_calls"]) == 1
        assert traj["final_answer"] == "The paper contains test content."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
