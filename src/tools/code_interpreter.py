"""
Code Interpreter - Sandboxed Python execution for analytical reasoning.

Provides the agent with "computational mind" for:
- Statistical analysis (comparing metrics from papers)
- Mathematical calculations (verifying formulas)
- Data structuring (building comparison tables)
- Visualization (plotting trends)
- Logical reasoning (formalizing arguments)
"""

import ast
import io
import sys
import traceback
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
from contextlib import redirect_stdout, redirect_stderr
import tempfile


# Allowed modules for sandboxed execution
ALLOWED_MODULES = {
    # Data analysis
    "pandas": "pd",
    "numpy": "np",
    "scipy": None,
    "scipy.stats": "stats",

    # Visualization
    "matplotlib": None,
    "matplotlib.pyplot": "plt",

    # Math
    "sympy": None,
    "math": None,
    "statistics": None,

    # Utils
    "json": None,
    "re": None,
    "collections": None,
    "itertools": None,
    "functools": None,
    "datetime": None,
}

# Forbidden patterns in code
FORBIDDEN_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import urllib",
    "import requests",
    "import http",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",  # Except for specific allowed paths
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "__builtins__",
    "__code__",
    "__class__",
    "breakpoint(",
]


@dataclass
class CodeExecutionResult:
    """Result from code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Optional[str] = None
    figures: list[str] = field(default_factory=list)  # Base64 encoded PNGs
    error: Optional[str] = None
    execution_time: float = 0.0


def validate_code(code: str) -> tuple[bool, str]:
    """
    Validate code for security issues before execution.

    Args:
        code: Python code to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            return False, f"Forbidden pattern detected: {pattern}"

    # Parse AST to check for dangerous constructs
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES and not any(
                    alias.name.startswith(m + ".") for m in ALLOWED_MODULES
                ):
                    return False, f"Import not allowed: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module not in ALLOWED_MODULES and not any(
                node.module.startswith(m + ".") for m in ALLOWED_MODULES if m
            ):
                return False, f"Import not allowed: {node.module}"

    return True, ""


def create_sandbox_globals(workspace_dir: Path) -> dict:
    """
    Create a restricted globals dict for code execution.

    Args:
        workspace_dir: Directory for saving outputs

    Returns:
        Restricted globals dictionary
    """
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import json
    import re
    import math
    import statistics
    from collections import Counter, defaultdict
    from datetime import datetime, timedelta

    sandbox_globals = {
        # Allowed modules
        "pd": pd,
        "np": np,
        "plt": plt,
        "stats": stats,
        "json": json,
        "re": re,
        "math": math,
        "statistics": statistics,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "datetime": datetime,
        "timedelta": timedelta,

        # Safe builtins
        "print": print,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "any": any,
        "all": all,
        "isinstance": isinstance,
        "type": type,

        # Workspace path for saving figures
        "__workspace__": workspace_dir,
    }

    return sandbox_globals


def execute_python(
    code: str,
    timeout: int = 30,
    workspace_dir: str = "/workspace",
) -> str:
    """
    Execute Python code in a sandboxed environment.

    Provides the agent with computational capabilities for data analysis,
    statistical comparisons, and visualization.

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        workspace_dir: Directory for saving outputs (figures, data)

    Returns:
        Formatted string with stdout, result, and figure paths

    Example:
        >>> execute_python('''
        ... import pandas as pd
        ... results = pd.DataFrame({
        ...     'paper': ['MOA', 'PPO', 'DPO'],
        ...     'GSM8K': [0.89, 0.85, 0.87]
        ... })
        ... print(results.to_string())
        ... ''')
        "   paper  GSM8K
         0   MOA   0.89
         1   PPO   0.85
         2   DPO   0.87"
    """
    import time
    import signal

    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    # Validate code
    is_valid, error_msg = validate_code(code)
    if not is_valid:
        return f"Error: Code validation failed - {error_msg}"

    # Prepare execution environment
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = CodeExecutionResult(success=False)
    start_time = time.time()

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {timeout}s")

    try:
        # Set timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        sandbox_globals = create_sandbox_globals(workspace)

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute code
            exec_result = exec(code, sandbox_globals)

            # Try to get last expression result
            try:
                tree = ast.parse(code)
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.unparse(tree.body[-1].value)
                    expr_result = eval(last_expr, sandbox_globals)
                    if expr_result is not None:
                        result.result = str(expr_result)
            except Exception:
                pass

        result.success = True
        result.stdout = stdout_capture.getvalue()
        result.stderr = stderr_capture.getvalue()

        # Check for saved figures
        import matplotlib.pyplot as plt
        if plt.get_fignums():
            for i, fig_num in enumerate(plt.get_fignums()):
                fig = plt.figure(fig_num)
                fig_path = workspace / f"figure_{int(time.time())}_{i}.png"
                fig.savefig(fig_path, dpi=100, bbox_inches='tight')
                result.figures.append(str(fig_path))
            plt.close('all')

    except TimeoutError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        result.stderr = traceback.format_exc()
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel timeout
        result.execution_time = time.time() - start_time

    # Format output
    output_parts = []

    if result.stdout:
        output_parts.append(f"Output:\n{result.stdout}")

    if result.result:
        output_parts.append(f"Result: {result.result}")

    if result.figures:
        output_parts.append(f"Figures saved: {', '.join(result.figures)}")

    if result.error:
        output_parts.append(f"Error: {result.error}")

    if result.stderr and not result.error:
        output_parts.append(f"Warnings:\n{result.stderr}")

    return '\n'.join(output_parts) if output_parts else "Code executed successfully (no output)"


# Tool schema for verifiers integration
CODE_INTERPRETER_SCHEMA = {
    "name": "execute_python",
    "description": """Execute Python code for data analysis and computation.

Available libraries:
- pandas (pd): DataFrames, data manipulation
- numpy (np): Numerical operations
- scipy.stats (stats): Statistical tests
- matplotlib.pyplot (plt): Visualization
- sympy: Symbolic math
- math, statistics: Basic math functions

Use for:
- Comparing metrics from papers (t-tests, means)
- Building comparison tables
- Plotting trends and results
- Verifying calculations
- Formalizing logical arguments

Figures are automatically saved to /workspace/""",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            }
        },
        "required": ["code"]
    }
}


# Async version for verifiers integration
async def execute_python_async(
    code: str,
    timeout: int = 30,
    workspace_dir: str = "/workspace",
) -> str:
    """Async wrapper for execute_python."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: execute_python(code, timeout, workspace_dir)
    )
