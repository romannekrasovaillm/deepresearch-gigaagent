"""Agent tools for research and code reasoning."""

from .bash_tools import (
    grep_search,
    read_file_chunk,
    find_files,
    list_papers,
    verify_quote,
    BashTools,
)
from .code_interpreter import (
    execute_python,
    CodeInterpreter,
    SandboxConfig,
)
from .scratchpad import (
    add_to_notes,
    read_notes,
    save_figure,
    clear_notes,
    Scratchpad,
)

# Tool registry for environment
TOOL_REGISTRY = {
    "grep_search": grep_search,
    "read_file_chunk": read_file_chunk,
    "find_files": find_files,
    "list_papers": list_papers,
    "verify_quote": verify_quote,
    "execute_python": execute_python,
    "add_to_notes": add_to_notes,
    "read_notes": read_notes,
    "save_figure": save_figure,
}

__all__ = [
    "grep_search",
    "read_file_chunk",
    "find_files",
    "list_papers",
    "verify_quote",
    "execute_python",
    "add_to_notes",
    "read_notes",
    "save_figure",
    "clear_notes",
    "BashTools",
    "CodeInterpreter",
    "SandboxConfig",
    "Scratchpad",
    "TOOL_REGISTRY",
]
