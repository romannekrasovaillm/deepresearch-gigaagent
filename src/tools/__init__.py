"""Agent Tools Module - Bash navigation + Code Interpreter for reasoning."""

from .bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    verify_quote,
    list_papers,
    read_abstract,
    get_paper_metadata,
)

from .code_interpreter import (
    execute_python,
    CodeExecutionResult,
    ALLOWED_MODULES,
)

from .scratchpad import (
    add_to_notes,
    read_notes,
    clear_notes,
    save_figure,
)

from .registry import TOOL_REGISTRY, get_tool, get_all_tools

__all__ = [
    # Bash tools
    "grep_search",
    "read_file_chunk",
    "find_files",
    "verify_quote",
    "list_papers",
    "read_abstract",
    "get_paper_metadata",
    # Code interpreter
    "execute_python",
    "CodeExecutionResult",
    "ALLOWED_MODULES",
    # Scratchpad
    "add_to_notes",
    "read_notes",
    "clear_notes",
    "save_figure",
    # Registry
    "TOOL_REGISTRY",
    "get_tool",
    "get_all_tools",
]
