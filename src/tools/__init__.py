"""Agent tools module - Bash navigation + Code Interpreter."""

from .bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    verify_quote,
    list_papers,
    get_paper_metadata,
)

from .code_interpreter import (
    execute_python,
    CodeInterpreter,
)

from .scratchpad import (
    add_to_notes,
    read_notes,
    clear_notes,
    save_figure,
)

from .tool_env import (
    ToolEnv,
    Tool,
    ToolResult,
)

__all__ = [
    # Bash tools
    "grep_search",
    "read_file_chunk",
    "find_files",
    "verify_quote",
    "list_papers",
    "get_paper_metadata",
    # Code interpreter
    "execute_python",
    "CodeInterpreter",
    # Scratchpad
    "add_to_notes",
    "read_notes",
    "clear_notes",
    "save_figure",
    # Environment
    "ToolEnv",
    "Tool",
    "ToolResult",
]
