"""
Tool Registry - Central registry for all agent tools.

Provides unified interface for tool discovery, schema access,
and execution. Used by verifiers ToolEnv.
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass

from .bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    verify_quote,
    list_papers,
    read_abstract,
    get_paper_metadata,
    BASH_TOOL_SCHEMAS,
)

from .code_interpreter import (
    execute_python,
    CODE_INTERPRETER_SCHEMA,
)

from .scratchpad import (
    add_to_notes,
    read_notes,
    clear_notes,
    save_figure,
    save_data,
    load_data,
    SCRATCHPAD_SCHEMAS,
)


@dataclass
class Tool:
    """Tool wrapper with metadata."""
    name: str
    description: str
    function: Callable
    schema: dict
    category: str


# Build tool registry
TOOL_REGISTRY: dict[str, Tool] = {}


def _register_tool(
    name: str,
    function: Callable,
    schema: dict,
    category: str,
):
    """Register a tool in the global registry."""
    TOOL_REGISTRY[name] = Tool(
        name=name,
        description=schema.get("description", ""),
        function=function,
        schema=schema,
        category=category,
    )


# Register bash tools
_register_tool("grep_search", grep_search, BASH_TOOL_SCHEMAS["grep_search"], "search")
_register_tool("read_file_chunk", read_file_chunk, BASH_TOOL_SCHEMAS["read_file_chunk"], "read")
_register_tool("find_files", find_files, BASH_TOOL_SCHEMAS["find_files"], "search")
_register_tool("verify_quote", verify_quote, BASH_TOOL_SCHEMAS["verify_quote"], "verify")
_register_tool("list_papers", list_papers, BASH_TOOL_SCHEMAS["list_papers"], "search")
_register_tool("read_abstract", read_abstract, BASH_TOOL_SCHEMAS["read_abstract"], "read")
_register_tool("get_paper_metadata", get_paper_metadata, BASH_TOOL_SCHEMAS["get_paper_metadata"], "read")

# Register code interpreter
_register_tool("execute_python", execute_python, CODE_INTERPRETER_SCHEMA, "compute")

# Register scratchpad tools
_register_tool("add_to_notes", add_to_notes, SCRATCHPAD_SCHEMAS["add_to_notes"], "memory")
_register_tool("read_notes", read_notes, SCRATCHPAD_SCHEMAS["read_notes"], "memory")
_register_tool("save_figure", save_figure, SCRATCHPAD_SCHEMAS["save_figure"], "memory")
_register_tool("save_data", save_data, SCRATCHPAD_SCHEMAS["save_data"], "memory")
_register_tool("load_data", load_data, SCRATCHPAD_SCHEMAS["load_data"], "memory")


def get_tool(name: str) -> Optional[Tool]:
    """Get tool by name."""
    return TOOL_REGISTRY.get(name)


def get_all_tools() -> list[Tool]:
    """Get all registered tools."""
    return list(TOOL_REGISTRY.values())


def get_tools_by_category(category: str) -> list[Tool]:
    """Get tools by category."""
    return [t for t in TOOL_REGISTRY.values() if t.category == category]


def get_tool_schemas() -> list[dict]:
    """Get OpenAI-format tool schemas for all tools."""
    return [
        {
            "type": "function",
            "function": tool.schema
        }
        for tool in TOOL_REGISTRY.values()
    ]


def execute_tool(name: str, **kwargs) -> str:
    """
    Execute a tool by name with given arguments.

    Args:
        name: Tool name
        **kwargs: Tool arguments

    Returns:
        Tool result as string
    """
    tool = get_tool(name)
    if not tool:
        return f"Error: Unknown tool '{name}'"

    try:
        result = tool.function(**kwargs)
        return str(result)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# Export for verifiers
def get_verifiers_tools() -> dict[str, Callable]:
    """Get tools dict for verifiers ToolEnv."""
    return {
        name: tool.function
        for name, tool in TOOL_REGISTRY.items()
    }


# Tool descriptions for system prompt
TOOLS_DESCRIPTION = """
You have access to the following tools for researching scientific papers:

## Search Tools
- `grep_search(pattern, path?, max_results?)`: Search for text patterns in papers
- `find_files(name_pattern)`: Find files by name pattern
- `list_papers(topic?, keyword?)`: List papers with optional filtering

## Read Tools
- `read_file_chunk(path, start?, num_lines?)`: Read portion of a file
- `read_abstract(paper_id)`: Quick preview of paper abstract
- `get_paper_metadata(paper_id)`: Get paper metadata (title, author, sections)

## Verification Tools
- `verify_quote(path, snippet)`: Verify a quote exists in file (anti-hallucination)

## Computation Tools
- `execute_python(code)`: Run Python code for analysis
  - Available: pandas, numpy, scipy.stats, matplotlib, sympy
  - Use for: statistics, comparisons, visualizations

## Memory Tools
- `add_to_notes(text, section?)`: Save findings for later
- `read_notes()`: Recall saved findings
- `save_data(key, value)`: Store structured data
- `load_data(key?)`: Retrieve stored data
- `save_figure(source_path, name)`: Save generated figures
"""
