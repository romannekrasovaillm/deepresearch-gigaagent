"""RL Environment with tool execution for research agent."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..tools import TOOL_REGISTRY
from ..tools.bash_tools import init_tools as init_bash_tools
from ..tools.code_interpreter import init_interpreter, SandboxConfig
from ..tools.scratchpad import init_scratchpad


@dataclass
class ToolCall:
    """Parsed tool call from model output."""

    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class ToolEnvConfig:
    """Configuration for research tool environment."""

    library_path: str = "./data/library"
    workspace_path: str = "./data/workspace"
    max_turns: int = 15
    timeout_per_action: int = 30

    # Tool configuration
    enabled_tools: list[str] = field(
        default_factory=lambda: [
            "grep_search",
            "read_file_chunk",
            "find_files",
            "list_papers",
            "execute_python",
            "add_to_notes",
            "read_notes",
            "verify_quote",
        ]
    )

    # Sandbox configuration
    use_docker: bool = False
    docker_image: str = "python:3.11-slim"
    memory_limit: str = "512m"


@dataclass
class EnvState:
    """Current state of the environment."""

    turn: int = 0
    done: bool = False
    history: list[dict] = field(default_factory=list)
    opened_files: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    code_executions: list[dict] = field(default_factory=list)


class ResearchToolEnv:
    """
    Environment for training research agents with tool use.

    Compatible with verifiers library interface.
    """

    # Tool call patterns
    TOOL_CALL_PATTERNS = [
        # JSON format: {"name": "tool_name", "arguments": {...}}
        r'\{["\']name["\']\s*:\s*["\'](\w+)["\'].*?["\']arguments["\']\s*:\s*(\{[^}]*\})\}',
        # Function call format: tool_name(arg1=val1, arg2=val2)
        r'(\w+)\((.*?)\)',
        # XML format: <tool_call name="tool_name"><arg>value</arg></tool_call>
        r'<tool_call\s+name=["\'](\w+)["\']>(.*?)</tool_call>',
    ]

    def __init__(self, config: Optional[ToolEnvConfig] = None):
        self.config = config or ToolEnvConfig()
        self.state = EnvState()

        # Initialize tools
        self._setup_tools()

        # Build tool descriptions for prompt
        self.tool_descriptions = self._build_tool_descriptions()

    def _setup_tools(self) -> None:
        """Initialize all tool backends."""
        # Initialize bash tools
        init_bash_tools(self.config.library_path)

        # Initialize code interpreter
        sandbox_config = SandboxConfig(
            timeout=self.config.timeout_per_action,
            memory_limit=self.config.memory_limit,
            use_docker=self.config.use_docker,
            docker_image=self.config.docker_image,
            workspace_path=Path(self.config.workspace_path),
        )
        init_interpreter(sandbox_config)

        # Initialize scratchpad
        init_scratchpad(self.config.workspace_path)

        # Build active tool registry
        self.tools: dict[str, Callable] = {
            name: func
            for name, func in TOOL_REGISTRY.items()
            if name in self.config.enabled_tools
        }

    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for system prompt."""
        descriptions = []

        tool_docs = {
            "grep_search": """grep_search(pattern: str, path: str = None) -> str
    Search for pattern in library files using grep/ripgrep.
    - pattern: Search pattern (regex supported)
    - path: Subpath within library (optional)
    Returns: Matching lines with file paths""",
            "read_file_chunk": """read_file_chunk(path: str, start: int = 1, num_lines: int = 50) -> str
    Read a chunk of a file.
    - path: Relative path within library
    - start: Starting line number (1-indexed)
    - num_lines: Number of lines to read
    Returns: File content""",
            "find_files": """find_files(name_pattern: str, file_type: str = None) -> str
    Find files by name pattern.
    - name_pattern: Part of filename to match
    - file_type: Filter by extension (e.g., "txt", "md")
    Returns: List of matching file paths""",
            "list_papers": """list_papers(topic: str = None, limit: int = 20) -> str
    List papers from library index.
    - topic: Filter by keyword in title/abstract
    - limit: Maximum results
    Returns: List of paper_id: title""",
            "execute_python": """execute_python(code: str) -> str
    Execute Python code for analysis.
    Available: pandas, numpy, scipy, matplotlib, sympy
    - code: Python code to execute
    Returns: stdout, result value, or error""",
            "add_to_notes": """add_to_notes(text: str, section: str = None) -> str
    Save important findings to notes.
    - text: Text to save
    - section: Optional section header
    Returns: Confirmation""",
            "read_notes": """read_notes() -> str
    Read your saved notes.
    Returns: All notes content""",
            "verify_quote": """verify_quote(path: str, snippet: str) -> str
    Verify a quote exists in a file.
    - path: File path
    - snippet: Quote to verify
    Returns: VERIFIED or NOT VERIFIED""",
        }

        for name in self.config.enabled_tools:
            if name in tool_docs:
                descriptions.append(tool_docs[name])

        return "\n\n".join(descriptions)

    def reset(self, question: str) -> dict:
        """
        Reset environment for new episode.

        Args:
            question: Research question for this episode

        Returns:
            Initial observation dict
        """
        self.state = EnvState()

        # Clear workspace
        workspace = Path(self.config.workspace_path)
        notes_file = workspace / "notes.md"
        if notes_file.exists():
            notes_file.unlink()

        system_prompt = f"""You are a research agent with access to a library of scientific papers.
Your task is to answer research questions by searching and analyzing papers.

Available tools:
{self.tool_descriptions}

To use a tool, output a JSON object:
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Guidelines:
1. Start by searching for relevant papers
2. Read specific sections, not entire files
3. Use execute_python for calculations and comparisons
4. Save important findings with add_to_notes
5. Verify quotes before citing them
6. Provide your final answer with [ANSWER] prefix

Current question: {question}"""

        return {
            "observation": system_prompt,
            "done": False,
            "turn": 0,
        }

    def parse_tool_call(self, text: str) -> Optional[ToolCall]:
        """Parse tool call from model output."""
        # Try JSON format first
        try:
            # Find JSON object
            json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "name" in data:
                    return ToolCall(
                        name=data["name"],
                        arguments=data.get("arguments", {}),
                        raw=json_match.group(),
                    )
        except json.JSONDecodeError:
            pass

        # Try function call format
        func_match = re.search(r'(\w+)\(([^)]*)\)', text)
        if func_match:
            name = func_match.group(1)
            if name in self.tools:
                args_str = func_match.group(2)
                arguments = {}
                # Parse key=value pairs
                for arg in args_str.split(","):
                    if "=" in arg:
                        key, val = arg.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        arguments[key] = val
                return ToolCall(
                    name=name,
                    arguments=arguments,
                    raw=func_match.group(),
                )

        return None

    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return result."""
        if tool_call.name not in self.tools:
            return f"Error: Unknown tool '{tool_call.name}'"

        tool_func = self.tools[tool_call.name]

        try:
            result = tool_func(**tool_call.arguments)

            # Track tool usage for rewards
            if tool_call.name == "grep_search":
                self.state.search_queries.append(
                    tool_call.arguments.get("pattern", "")
                )
            elif tool_call.name == "read_file_chunk":
                self.state.opened_files.add(
                    tool_call.arguments.get("path", "")
                )
            elif tool_call.name == "execute_python":
                self.state.code_executions.append({
                    "code": tool_call.arguments.get("code", ""),
                    "result": result[:500],  # Truncate for storage
                })

            return result

        except Exception as e:
            return f"Error executing {tool_call.name}: {e}"

    def step(self, action: str) -> dict:
        """
        Execute one step in the environment.

        Args:
            action: Model output (may contain tool call or answer)

        Returns:
            Observation dict with result, done flag, etc.
        """
        self.state.turn += 1

        # Check for final answer
        if "[ANSWER]" in action:
            self.state.done = True
            answer = action.split("[ANSWER]")[-1].strip()
            return {
                "observation": f"Final answer recorded: {answer[:200]}...",
                "done": True,
                "turn": self.state.turn,
                "answer": answer,
            }

        # Check for max turns
        if self.state.turn >= self.config.max_turns:
            self.state.done = True
            return {
                "observation": "Maximum turns reached. Please provide your final answer with [ANSWER] prefix.",
                "done": True,
                "turn": self.state.turn,
                "timeout": True,
            }

        # Parse and execute tool call
        tool_call = self.parse_tool_call(action)

        if tool_call is None:
            return {
                "observation": "No valid tool call found. Use JSON format: {\"name\": \"tool_name\", \"arguments\": {...}}",
                "done": False,
                "turn": self.state.turn,
            }

        # Execute tool
        result = self.execute_tool(tool_call)

        # Store in history
        self.state.history.append({
            "turn": self.state.turn,
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "result": result[:1000],  # Truncate for storage
        })

        return {
            "observation": f"[{tool_call.name}] {result}",
            "done": False,
            "turn": self.state.turn,
        }

    def get_metrics(self) -> dict:
        """Get episode metrics for reward computation."""
        return {
            "total_turns": self.state.turn,
            "unique_files_opened": len(self.state.opened_files),
            "opened_files": list(self.state.opened_files),
            "search_queries": self.state.search_queries,
            "repeated_searches": len(self.state.search_queries)
            - len(set(self.state.search_queries)),
            "code_executions": len(self.state.code_executions),
            "history": self.state.history,
        }
