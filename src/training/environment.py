"""
Research Environment for RL training.

Implements a multi-turn tool-use environment compatible
with verifiers library from PrimeIntellect.
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..tools.bash_tools import (
    GrepSearch,
    ReadFileChunk,
    FindFiles,
    VerifyQuote,
    ListPapers,
    ToolResult,
)
from ..tools.code_interpreter import CodeInterpreter
from ..tools.scratchpad import Scratchpad, AddToNotes, ReadNotes, SaveFigure


class StepStatus(Enum):
    """Status of environment step."""
    CONTINUE = "continue"
    DONE = "done"
    ERROR = "error"
    MAX_STEPS = "max_steps"


@dataclass
class EnvStep:
    """Single environment step result."""
    observation: str
    reward: float
    done: bool
    status: StepStatus
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete episode trajectory."""
    task: str
    steps: list[dict]
    final_answer: Optional[str]
    total_reward: float
    metadata: dict = field(default_factory=dict)


class ResearchEnv:
    """
    Multi-turn research environment.

    Provides tools for:
    - Searching paper library (grep, find)
    - Reading file contents
    - Executing code for analysis
    - Taking notes
    - Verifying quotes

    Compatible with verifiers.ToolEnv interface.
    """

    def __init__(
        self,
        library_path: str | Path,
        workspace_path: str | Path,
        max_turns: int = 15,
        tools: Optional[list[str]] = None,
    ):
        self.library_path = Path(library_path)
        self.workspace_path = Path(workspace_path)
        self.max_turns = max_turns

        # Initialize tools
        self._init_tools(tools)

        # Episode state
        self._current_task: Optional[str] = None
        self._steps: list[dict] = []
        self._turn: int = 0

    def _init_tools(self, tool_names: Optional[list[str]] = None):
        """Initialize tool instances."""
        all_tools = {
            "grep_search": GrepSearch(self.library_path),
            "read_file_chunk": ReadFileChunk(self.library_path),
            "find_files": FindFiles(self.library_path),
            "verify_quote": VerifyQuote(self.library_path),
            "list_papers": ListPapers(self.library_path),
            "execute_python": CodeInterpreter(self.workspace_path),
            "add_to_notes": AddToNotes(self.workspace_path),
            "read_notes": ReadNotes(self.workspace_path),
            "save_figure": SaveFigure(self.workspace_path),
        }

        if tool_names is None or "all" in tool_names:
            self._tools = all_tools
        else:
            self._tools = {
                name: tool for name, tool in all_tools.items()
                if name in tool_names
            }

    @property
    def tool_schemas(self) -> list[dict]:
        """Get JSON schemas for all available tools."""
        return [tool.to_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        """Get names of available tools."""
        return list(self._tools.keys())

    def reset(self, task: str) -> str:
        """
        Reset environment for new episode.

        Args:
            task: The research task/question

        Returns:
            Initial observation (task description)
        """
        self._current_task = task
        self._steps = []
        self._turn = 0

        # Clear workspace for new episode
        notes_file = self.workspace_path / "notes.md"
        if notes_file.exists():
            notes_file.unlink()

        # Initial observation
        return self._format_initial_observation(task)

    def _format_initial_observation(self, task: str) -> str:
        """Format the initial task observation."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        ])

        return f"""You are a research agent with access to a library of scientific papers.

## Your Task
{task}

## Available Tools
{tools_desc}

## Instructions
1. Search the library to find relevant papers
2. Read sections to gather information
3. Use code execution for calculations if needed
4. Take notes on important findings
5. Provide a final answer with citations

Begin by searching for relevant papers."""

    def step(
        self,
        action: str,
        action_input: dict[str, Any]
    ) -> EnvStep:
        """
        Execute one environment step.

        Args:
            action: Tool name to execute
            action_input: Tool parameters

        Returns:
            EnvStep with observation, reward, done flag
        """
        self._turn += 1

        # Check max turns
        if self._turn > self.max_turns:
            return EnvStep(
                observation="Maximum turns reached. Please provide your final answer.",
                reward=-0.1,
                done=True,
                status=StepStatus.MAX_STEPS,
            )

        # Handle special "final_answer" action
        if action == "final_answer":
            answer = action_input.get("answer", "")
            self._steps.append({
                "turn": self._turn,
                "action": action,
                "action_input": action_input,
                "observation": "Answer submitted.",
            })
            return EnvStep(
                observation="Answer submitted.",
                reward=0.0,  # Actual reward computed by reward function
                done=True,
                status=StepStatus.DONE,
                info={"final_answer": answer}
            )

        # Check if tool exists
        if action not in self._tools:
            return EnvStep(
                observation=f"Unknown tool: {action}. Available: {', '.join(self._tools.keys())}",
                reward=-0.05,
                done=False,
                status=StepStatus.ERROR,
            )

        # Execute tool
        tool = self._tools[action]
        try:
            result: ToolResult = tool.execute(**action_input)

            # Record step
            self._steps.append({
                "turn": self._turn,
                "action": action,
                "action_input": action_input,
                "observation": result.output if result.success else f"Error: {result.error}",
            })

            # Small negative reward for each step (efficiency incentive)
            step_reward = -0.01 if result.success else -0.05

            return EnvStep(
                observation=result.output if result.success else f"Error: {result.error}",
                reward=step_reward,
                done=False,
                status=StepStatus.CONTINUE if result.success else StepStatus.ERROR,
            )

        except Exception as e:
            self._steps.append({
                "turn": self._turn,
                "action": action,
                "action_input": action_input,
                "observation": f"Exception: {str(e)}",
            })
            return EnvStep(
                observation=f"Tool execution error: {str(e)}",
                reward=-0.1,
                done=False,
                status=StepStatus.ERROR,
            )

    def get_trajectory(self) -> Trajectory:
        """Get the current episode trajectory."""
        # Find final answer if any
        final_answer = None
        for step in reversed(self._steps):
            if step.get("action") == "final_answer":
                final_answer = step.get("action_input", {}).get("answer")
                break

        return Trajectory(
            task=self._current_task or "",
            steps=self._steps.copy(),
            final_answer=final_answer,
            total_reward=sum(-0.01 for _ in self._steps),  # Placeholder
            metadata={
                "turns": self._turn,
                "max_turns": self.max_turns,
                "tools_used": list(set(s.get("action") for s in self._steps)),
            }
        )

    def render(self) -> str:
        """Render current state for debugging."""
        lines = [
            f"=== Research Environment ===",
            f"Task: {self._current_task}",
            f"Turn: {self._turn}/{self.max_turns}",
            f"Steps: {len(self._steps)}",
            "",
            "--- Trajectory ---",
        ]

        for step in self._steps[-5:]:  # Last 5 steps
            lines.append(f"[{step['turn']}] {step['action']}")
            lines.append(f"    -> {step['observation'][:100]}...")
            lines.append("")

        return '\n'.join(lines)


class AsyncResearchEnv(ResearchEnv):
    """
    Async version of ResearchEnv for parallel rollouts.

    Compatible with verifiers async environment interface.
    """

    async def reset_async(self, task: str) -> str:
        """Async reset."""
        return self.reset(task)

    async def step_async(
        self,
        action: str,
        action_input: dict[str, Any]
    ) -> EnvStep:
        """Async step - currently just wraps sync version."""
        return self.step(action, action_input)


def create_env(
    library_path: str | Path,
    workspace_path: str | Path,
    max_turns: int = 15,
    curriculum_level: int = 1,
) -> ResearchEnv:
    """
    Create environment with curriculum-appropriate tools.

    Args:
        library_path: Path to paper library
        workspace_path: Path to workspace directory
        max_turns: Maximum turns per episode
        curriculum_level: 1-6, determines available tools

    Returns:
        Configured ResearchEnv
    """
    from ..data_generation.prompts import CURRICULUM_LEVELS

    level_config = CURRICULUM_LEVELS.get(curriculum_level, CURRICULUM_LEVELS[1])
    tools = level_config.get("tools", ["all"])
    max_steps = level_config.get("max_steps", max_turns)

    return ResearchEnv(
        library_path=library_path,
        workspace_path=workspace_path,
        max_turns=max_steps,
        tools=tools if tools != ["all"] else None,
    )
