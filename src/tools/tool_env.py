#!/usr/bin/env python3
"""
Tool Environment for Deep Research Agent
Integrates all tools into a unified environment for RL training.
"""

import json
import asyncio
import inspect
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re

from .bash_tools import BASH_TOOLS
from .code_interpreter import CODE_TOOLS
from .scratchpad import SCRATCHPAD_TOOLS, reset_scratchpad


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_PARAMS = "invalid_params"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    tool_name: str
    status: ToolStatus
    output: str
    raw_output: Any = None
    execution_time: float = 0.0
    error: Optional[str] = None


@dataclass
class Tool:
    """Tool definition."""
    name: str
    function: Callable
    description: str
    parameters: dict[str, dict]

    def get_schema(self) -> dict:
        """Get JSON schema for tool parameters."""
        properties = {}
        required = []

        for name, spec in self.parameters.items():
            prop = {"type": spec.get("type", "string")}
            if "description" in spec:
                prop["description"] = spec["description"]
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            if "default" in spec:
                prop["default"] = spec["default"]

            properties[name] = prop

            if spec.get("required", False):
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


@dataclass
class Episode:
    """Single episode of agent interaction."""
    question: str
    turns: list[dict] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_reward: float = 0.0
    metadata: dict = field(default_factory=dict)


class ToolEnv:
    """
    Environment for Deep Research Agent.
    Manages tools, execution, and episode state.
    """

    def __init__(
        self,
        library_path: str = "/mnt/library",
        workspace_path: str = "/workspace",
        max_turns: int = 15,
        timeout_per_tool: int = 30,
    ):
        self.library_path = library_path
        self.workspace_path = workspace_path
        self.max_turns = max_turns
        self.timeout = timeout_per_tool

        # Register all tools
        self.tools: dict[str, Tool] = {}
        self._register_tools()

        # Episode state
        self.current_episode: Optional[Episode] = None
        self.turn_count = 0

    def _register_tools(self):
        """Register all available tools."""
        all_tools = {}
        all_tools.update(BASH_TOOLS)
        all_tools.update(CODE_TOOLS)
        all_tools.update(SCRATCHPAD_TOOLS)

        for name, spec in all_tools.items():
            self.tools[name] = Tool(
                name=name,
                function=spec["function"],
                description=spec["description"],
                parameters=spec.get("parameters", {}),
            )

    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt."""
        lines = ["Available tools:\n"]

        for name, tool in self.tools.items():
            lines.append(f"## {name}")
            lines.append(f"{tool.description}\n")

            if tool.parameters:
                lines.append("Parameters:")
                for param_name, spec in tool.parameters.items():
                    required = " (required)" if spec.get("required") else ""
                    default = f" [default: {spec['default']}]" if "default" in spec else ""
                    lines.append(f"  - {param_name}: {spec.get('description', '')}{required}{default}")
            lines.append("")

        return "\n".join(lines)

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-style function schemas for all tools."""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.get_schema(),
                },
            })
        return schemas

    def reset(self, question: str) -> Episode:
        """Reset environment for new episode."""
        self.current_episode = Episode(question=question)
        self.turn_count = 0

        # Reset scratchpad
        reset_scratchpad()

        return self.current_episode

    async def step(
        self,
        tool_name: str,
        tool_input: dict,
        thought: str = "",
    ) -> ToolResult:
        """
        Execute a tool and record the turn.

        Args:
            tool_name: Name of tool to execute
            tool_input: Parameters for tool
            thought: Agent's reasoning for this action

        Returns:
            ToolResult with execution output
        """
        import time

        self.turn_count += 1

        # Check if tool exists
        if tool_name not in self.tools:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                output=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}",
                error=f"Unknown tool: {tool_name}",
            )
            self._record_turn(thought, tool_name, tool_input, result)
            return result

        tool = self.tools[tool_name]
        start_time = time.time()

        try:
            # Prepare arguments
            kwargs = self._prepare_arguments(tool, tool_input)

            # Execute tool
            func = tool.function
            if asyncio.iscoroutinefunction(func):
                output = await asyncio.wait_for(
                    func(**kwargs),
                    timeout=self.timeout
                )
            else:
                output = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: func(**kwargs)),
                    timeout=self.timeout
                )

            execution_time = time.time() - start_time

            # Format output
            if isinstance(output, (list, dict)):
                formatted_output = json.dumps(output, ensure_ascii=False, indent=2)
            else:
                formatted_output = str(output)

            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                output=formatted_output,
                raw_output=output,
                execution_time=execution_time,
            )

        except asyncio.TimeoutError:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                output=f"Tool execution timed out after {self.timeout}s",
                error="timeout",
                execution_time=self.timeout,
            )
        except TypeError as e:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.INVALID_PARAMS,
                output=f"Invalid parameters: {e}",
                error=str(e),
            )
        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                output=f"Error: {type(e).__name__}: {e}",
                error=str(e),
                execution_time=time.time() - start_time,
            )

        self._record_turn(thought, tool_name, tool_input, result)
        return result

    def _prepare_arguments(self, tool: Tool, tool_input: dict) -> dict:
        """Prepare and validate tool arguments."""
        kwargs = {}

        for param_name, spec in tool.parameters.items():
            if param_name in tool_input:
                value = tool_input[param_name]

                # Type conversion
                param_type = spec.get("type", "string")
                if param_type == "integer":
                    value = int(value)
                elif param_type == "number":
                    value = float(value)
                elif param_type == "boolean":
                    value = bool(value)

                kwargs[param_name] = value
            elif "default" in spec:
                kwargs[param_name] = spec["default"]
            elif spec.get("required", False):
                raise TypeError(f"Missing required parameter: {param_name}")

        return kwargs

    def _record_turn(
        self,
        thought: str,
        tool_name: str,
        tool_input: dict,
        result: ToolResult,
    ):
        """Record turn in episode history."""
        if self.current_episode:
            self.current_episode.turns.append({
                "turn": self.turn_count,
                "thought": thought,
                "action": tool_name,
                "action_input": tool_input,
                "observation": result.output,
                "status": result.status.value,
                "execution_time": result.execution_time,
            })

    def finish(self, final_answer: str) -> Episode:
        """Finish episode with final answer."""
        if self.current_episode:
            self.current_episode.final_answer = final_answer
        return self.current_episode

    def is_done(self) -> bool:
        """Check if episode should end."""
        return self.turn_count >= self.max_turns

    def get_trajectory(self) -> list[dict]:
        """Get trajectory for training."""
        if not self.current_episode:
            return []
        return self.current_episode.turns

    def format_trajectory_for_prompt(self) -> str:
        """Format trajectory as string for model input."""
        if not self.current_episode:
            return ""

        lines = [f"Question: {self.current_episode.question}\n"]

        for turn in self.current_episode.turns:
            lines.append(f"Thought: {turn['thought']}")
            lines.append(f"Action: {turn['action']}")
            lines.append(f"Action Input: {json.dumps(turn['action_input'])}")
            lines.append(f"Observation: {turn['observation']}")
            lines.append("")

        return "\n".join(lines)


# Async helpers for parallel tool execution
async def execute_tools_parallel(
    env: ToolEnv,
    tool_calls: list[tuple[str, dict, str]],
) -> list[ToolResult]:
    """
    Execute multiple tools in parallel.

    Args:
        env: Tool environment
        tool_calls: List of (tool_name, tool_input, thought) tuples

    Returns:
        List of ToolResults in same order
    """
    tasks = [
        env.step(name, input_dict, thought)
        for name, input_dict, thought in tool_calls
    ]
    return await asyncio.gather(*tasks)


def parse_tool_call(text: str) -> Optional[tuple[str, dict]]:
    """
    Parse tool call from model output.

    Supports formats:
    - Action: tool_name
      Action Input: {"key": "value"}
    - <tool_call>{"name": "tool", "arguments": {...}}</tool_call>

    Returns:
        Tuple of (tool_name, arguments) or None
    """
    # Format 1: Action/Action Input
    action_match = re.search(r'Action:\s*(\w+)', text)
    input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)

    if action_match:
        tool_name = action_match.group(1)
        if input_match:
            try:
                tool_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                tool_input = {}
        else:
            tool_input = {}
        return (tool_name, tool_input)

    # Format 2: <tool_call> JSON
    tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if tool_call_match:
        try:
            data = json.loads(tool_call_match.group(1))
            return (data.get("name", ""), data.get("arguments", {}))
        except json.JSONDecodeError:
            pass

    return None


def format_tool_result(result: ToolResult) -> str:
    """Format tool result for model input."""
    if result.status == ToolStatus.SUCCESS:
        return f"Observation: {result.output}"
    else:
        return f"Observation: [Error] {result.output}"
