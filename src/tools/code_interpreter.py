#!/usr/bin/env python3
"""
Code Interpreter for Scientific Analysis
Sandboxed Python execution for analytical reasoning.
"""

import io
import sys
import ast
import traceback
import base64
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import subprocess
import json

# Whitelist of allowed modules
ALLOWED_MODULES = {
    # Data analysis
    'pandas', 'pd',
    'numpy', 'np',
    'scipy', 'scipy.stats',
    # Math
    'math', 'statistics', 'sympy',
    # Visualization
    'matplotlib', 'matplotlib.pyplot', 'plt',
    'seaborn', 'sns',
    # Standard library (safe subset)
    'json', 're', 'collections', 'itertools', 'functools',
    'datetime', 'time', 'random',
    'typing', 'dataclasses',
    'io', 'csv', 'string',
}

# Forbidden operations
FORBIDDEN_NAMES = {
    'exec', 'eval', 'compile', '__import__',
    'open', 'file', 'input',
    'subprocess', 'os', 'sys',
    'globals', 'locals', 'vars',
    'getattr', 'setattr', 'delattr',
    '__builtins__',
}


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: Any = None
    result_repr: str = ""
    figures: list[str] = field(default_factory=list)  # Base64 encoded PNGs
    error: Optional[str] = None
    execution_time: float = 0.0


class CodeSanitizer:
    """Validates Python code for safety."""

    def __init__(self, allowed_modules: set[str] = None):
        self.allowed_modules = allowed_modules or ALLOWED_MODULES

    def validate(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate code for safety.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check for forbidden names
            if isinstance(node, ast.Name):
                if node.id in FORBIDDEN_NAMES:
                    return False, f"Forbidden name: {node.id}"

            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_allowed_module(alias.name):
                        return False, f"Module not allowed: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module and not self._is_allowed_module(node.module):
                    return False, f"Module not allowed: {node.module}"

            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    return False, f"Private attribute access not allowed: {node.attr}"

        return True, None

    def _is_allowed_module(self, module: str) -> bool:
        """Check if module is in whitelist."""
        # Check exact match
        if module in self.allowed_modules:
            return True

        # Check parent module
        parts = module.split('.')
        for i in range(len(parts)):
            parent = '.'.join(parts[:i + 1])
            if parent in self.allowed_modules:
                return True

        return False


class CodeInterpreter:
    """
    Sandboxed Python code interpreter for scientific analysis.

    Supports: pandas, numpy, scipy, matplotlib, sympy
    """

    def __init__(
        self,
        workspace_dir: str = "/workspace",
        timeout: int = 30,
        use_docker: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.use_docker = use_docker
        self.sanitizer = CodeSanitizer()

        # Persistent state between executions
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self):
        """Initialize safe execution namespace."""
        # Safe builtins
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'frozenset': frozenset, 'int': int,
            'len': len, 'list': list, 'map': map, 'max': max,
            'min': min, 'print': print, 'range': range, 'reversed': reversed,
            'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
            'zip': zip, 'True': True, 'False': False, 'None': None,
        }

        self._namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
        }

        # Pre-import common modules
        try:
            import pandas as pd
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            self._namespace['pd'] = pd
            self._namespace['pandas'] = pd
            self._namespace['np'] = np
            self._namespace['numpy'] = np
            self._namespace['plt'] = plt
            self._namespace['matplotlib'] = matplotlib
        except ImportError:
            pass

        try:
            import scipy
            import scipy.stats
            self._namespace['scipy'] = scipy
        except ImportError:
            pass

        try:
            import seaborn as sns
            self._namespace['sns'] = sns
            self._namespace['seaborn'] = sns
        except ImportError:
            pass

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute

        Returns:
            ExecutionResult with stdout, stderr, result, figures
        """
        import time
        start_time = time.time()

        # Validate code
        is_valid, error = self.sanitizer.validate(code)
        if not is_valid:
            return ExecutionResult(
                success=False,
                error=f"Code validation failed: {error}",
                execution_time=time.time() - start_time,
            )

        if self.use_docker:
            return self._execute_docker(code)
        else:
            return self._execute_restricted(code, start_time)

    def _execute_restricted(self, code: str, start_time: float) -> ExecutionResult:
        """Execute code with RestrictedPython (in-process)."""
        import time

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        figures = []
        result = None

        try:
            # Clear any existing figures
            if 'plt' in self._namespace:
                self._namespace['plt'].close('all')

            # Execute with captured output
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute code
                exec(compile(code, '<agent_code>', 'exec'), self._namespace)

                # Get last expression result if any
                try:
                    tree = ast.parse(code)
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = ast.Expression(body=tree.body[-1].value)
                        result = eval(compile(last_expr, '<agent_code>', 'eval'), self._namespace)
                except Exception:
                    pass

            # Capture any matplotlib figures
            if 'plt' in self._namespace:
                plt = self._namespace['plt']
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    figures.append(base64.b64encode(buf.read()).decode('utf-8'))
                    buf.close()
                plt.close('all')

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                result=result,
                result_repr=repr(result) if result is not None else "",
                figures=figures,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                error=error_msg,
                execution_time=execution_time,
            )

    def _execute_docker(self, code: str) -> ExecutionResult:
        """Execute code in Docker container (more secure)."""
        import time
        start_time = time.time()

        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            code_file = f.name

        try:
            result = subprocess.run(
                [
                    'docker', 'run', '--rm',
                    '--network=none',  # No network access
                    '--memory=512m',   # Memory limit
                    '--cpus=1',        # CPU limit
                    '-v', f'{code_file}:/code.py:ro',
                    '-v', f'{self.workspace_dir}:/workspace',
                    'python:3.11-slim',
                    'python', '/code.py',
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timeout ({self.timeout}s)",
                execution_time=self.timeout,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
        finally:
            Path(code_file).unlink(missing_ok=True)

    def reset(self):
        """Reset interpreter state."""
        self._namespace.clear()
        self._setup_namespace()

    def save_figure(self, figure_base64: str, filename: str) -> str:
        """Save a figure to workspace."""
        figure_path = self.workspace_dir / filename
        figure_data = base64.b64decode(figure_base64)
        figure_path.write_bytes(figure_data)
        return str(figure_path)


# Singleton interpreter instance
_interpreter: Optional[CodeInterpreter] = None


def get_interpreter(workspace_dir: str = "/workspace") -> CodeInterpreter:
    """Get or create interpreter instance."""
    global _interpreter
    if _interpreter is None:
        _interpreter = CodeInterpreter(workspace_dir=workspace_dir)
    return _interpreter


async def execute_python(
    code: str,
    workspace_dir: str = "/workspace",
    timeout: int = 30,
) -> str:
    """
    Execute Python code and return results.

    This is the main tool function for the agent.

    Args:
        code: Python code to execute
        workspace_dir: Directory for saving outputs
        timeout: Execution timeout in seconds

    Returns:
        Formatted string with execution results
    """
    interpreter = get_interpreter(workspace_dir)
    interpreter.timeout = timeout

    result = interpreter.execute(code)

    # Format output for agent
    output_parts = []

    if result.stdout:
        output_parts.append(f"[STDOUT]\n{result.stdout.strip()}")

    if result.stderr:
        output_parts.append(f"[STDERR]\n{result.stderr.strip()}")

    if result.result_repr:
        output_parts.append(f"[RESULT]\n{result.result_repr}")

    if result.figures:
        output_parts.append(f"[FIGURES] Generated {len(result.figures)} figure(s)")
        for i, fig_b64 in enumerate(result.figures):
            path = interpreter.save_figure(fig_b64, f"figure_{i}.png")
            output_parts.append(f"  Saved: {path}")

    if result.error:
        output_parts.append(f"[ERROR]\n{result.error}")

    if not output_parts:
        output_parts.append("[OK] Code executed successfully (no output)")

    output_parts.append(f"[TIME] {result.execution_time:.2f}s")

    return "\n\n".join(output_parts)


# Tool definition for ToolEnv
CODE_TOOLS = {
    "execute_python": {
        "function": execute_python,
        "description": """Execute Python code for data analysis and visualization.
Available libraries: pandas (pd), numpy (np), scipy, matplotlib (plt), seaborn (sns), sympy.
Use for: statistical analysis, comparing metrics, building tables, creating visualizations.
Figures are automatically saved to /workspace/""",
        "parameters": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
                "required": True,
            },
        },
    },
}
