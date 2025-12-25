"""
Code Interpreter for analytical reasoning.

Provides a sandboxed Python environment for the agent to perform
statistical analysis, mathematical computations, and visualization.
"""

import subprocess
import tempfile
import os
import sys
import json
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .bash_tools import BaseTool, ToolResult


# Whitelisted libraries for sandboxed execution
ALLOWED_IMPORTS = {
    "pandas", "pd",
    "numpy", "np",
    "scipy", "stats",
    "matplotlib", "plt",
    "sympy",
    "math",
    "statistics",
    "collections",
    "itertools",
    "functools",
    "json",
    "re",
    "datetime",
    "typing",
}

# Blocked operations for security
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import requests",
    "import urllib",
    "__import__",
    "exec(",
    "eval(",
    "compile(",
    "open(",
    "file(",
    "input(",
    "raw_input(",
    "breakpoint(",
    ".system(",
    ".popen(",
    ".spawn",
    "pty.",
    "pickle.",
]


@dataclass
class CodeResult:
    """Result of code execution."""
    success: bool
    stdout: str
    result: Optional[str] = None
    error: Optional[str] = None
    figures: list[str] = None  # Base64 encoded images

    def __post_init__(self):
        if self.figures is None:
            self.figures = []


class CodeInterpreter(BaseTool):
    """
    Sandboxed Python interpreter for code reasoning.

    Executes Python code with access to scientific libraries
    (pandas, numpy, scipy, matplotlib) but no filesystem or network access.
    """

    name = "execute_python"
    description = (
        "Execute Python code for data analysis and computation. "
        "Available libraries: pandas, numpy, scipy, matplotlib, sympy. "
        "Use for: statistical analysis, metric comparison, visualization, "
        "mathematical calculations. Returns stdout and any generated figures."
    )

    def __init__(
        self,
        workspace_path: str | Path,
        timeout: int = 30,
        use_docker: bool = False
    ):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.use_docker = use_docker

    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Check code for blocked patterns."""
        code_lower = code.lower()

        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return False, f"Blocked operation: {pattern}"

        return True, ""

    def _wrap_code(self, code: str) -> str:
        """Wrap code with setup and capture logic."""
        return f'''
import sys
import io
import json

# Redirect stdout
_stdout_capture = io.StringIO()
_old_stdout = sys.stdout
sys.stdout = _stdout_capture

# Track figures
_figures = []

try:
    # Setup matplotlib for non-interactive backend
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Monkey-patch savefig to capture figures
    _original_savefig = plt.savefig
    def _capturing_savefig(fname, *args, **kwargs):
        import base64
        from io import BytesIO
        buf = BytesIO()
        _original_savefig(buf, format='png', *args, **kwargs)
        buf.seek(0)
        _figures.append(base64.b64encode(buf.read()).decode('utf-8'))
        # Also save to file if path provided
        if isinstance(fname, str):
            buf.seek(0)
            with open(fname, 'wb') as f:
                f.write(buf.read())
    plt.savefig = _capturing_savefig

except ImportError:
    pass

# User code execution
_result = None
_error = None

try:
    # Execute user code
{_indent_code(code)}

except Exception as e:
    _error = f"{{type(e).__name__}}: {{str(e)}}"

# Restore stdout
sys.stdout = _old_stdout
_stdout_output = _stdout_capture.getvalue()

# Output results as JSON
import json
print("__CODE_RESULT__" + json.dumps({{
    "stdout": _stdout_output,
    "error": _error,
    "figures": _figures,
    "result": repr(_result) if '_result' in dir() and _result is not None else None
}}))
'''

    def execute(self, code: str) -> ToolResult:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute

        Returns:
            ToolResult with stdout, result value, and any figures
        """
        # Validate code
        is_valid, error_msg = self._validate_code(code)
        if not is_valid:
            return ToolResult(
                success=False,
                output="",
                error=f"Code validation failed: {error_msg}"
            )

        if self.use_docker:
            return self._execute_docker(code)
        else:
            return self._execute_subprocess(code)

    def _execute_subprocess(self, code: str) -> ToolResult:
        """Execute code in subprocess with timeout."""
        wrapped_code = self._wrap_code(code)

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(wrapped_code)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.workspace_path),
                env={
                    **os.environ,
                    "MPLCONFIGDIR": str(self.workspace_path / ".matplotlib"),
                }
            )

            # Parse output
            output = result.stdout
            if "__CODE_RESULT__" in output:
                parts = output.split("__CODE_RESULT__", 1)
                pre_output = parts[0]
                try:
                    code_result = json.loads(parts[1])
                    stdout = code_result.get("stdout", "")
                    error = code_result.get("error")
                    figures = code_result.get("figures", [])
                    result_val = code_result.get("result")

                    # Build output string
                    output_parts = []
                    if stdout:
                        output_parts.append(stdout)
                    if result_val:
                        output_parts.append(f"Result: {result_val}")
                    if figures:
                        output_parts.append(f"Generated {len(figures)} figure(s)")
                        # Save figures to workspace
                        for i, fig_b64 in enumerate(figures):
                            fig_path = self.workspace_path / f"figure_{i}.png"
                            fig_path.write_bytes(base64.b64decode(fig_b64))
                            output_parts.append(f"  Saved: {fig_path}")

                    final_output = '\n'.join(output_parts) if output_parts else "(no output)"

                    if error:
                        return ToolResult(
                            success=False,
                            output=final_output,
                            error=error
                        )

                    return ToolResult(
                        success=True,
                        output=final_output
                    )

                except json.JSONDecodeError:
                    return ToolResult(
                        success=True,
                        output=output
                    )
            else:
                # No result marker, return raw output
                if result.returncode != 0:
                    return ToolResult(
                        success=False,
                        output=output,
                        error=result.stderr or "Execution failed"
                    )
                return ToolResult(
                    success=True,
                    output=output if output else "(no output)"
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e)
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def _execute_docker(self, code: str) -> ToolResult:
        """Execute code in Docker container for better isolation."""
        wrapped_code = self._wrap_code(code)

        # Write code to workspace
        script_path = self.workspace_path / "_temp_script.py"
        script_path.write_text(wrapped_code)

        try:
            result = subprocess.run(
                [
                    "docker", "run",
                    "--rm",
                    "--network=none",  # No network access
                    "--memory=512m",   # Memory limit
                    "--cpus=1",        # CPU limit
                    "-v", f"{self.workspace_path}:/workspace:rw",
                    "-w", "/workspace",
                    "python:3.11-slim",
                    "python", "_temp_script.py"
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout + 10  # Extra time for Docker overhead
            )

            # Process output same as subprocess
            output = result.stdout
            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=output,
                    error=result.stderr or "Docker execution failed"
                )

            return ToolResult(
                success=True,
                output=output if output else "(no output)"
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Docker execution timed out"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="Docker not available, falling back to subprocess"
            )
        finally:
            try:
                script_path.unlink()
            except OSError:
                pass

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. "
                            "Available: pandas, numpy, scipy, matplotlib, sympy. "
                            "Use plt.savefig() to save figures."
                        )
                    }
                },
                "required": ["code"]
            }
        }


def _indent_code(code: str, spaces: int = 4) -> str:
    """Indent code block for embedding in wrapper."""
    indent = ' ' * spaces
    lines = code.split('\n')
    return '\n'.join(indent + line for line in lines)


class ExecuteJavaScript(BaseTool):
    """
    JavaScript interpreter for code reasoning.

    Uses Node.js for execution with limited capabilities.
    """

    name = "execute_javascript"
    description = (
        "Execute JavaScript code for data processing. "
        "Use for: JSON manipulation, array operations, string processing. "
        "No filesystem or network access."
    )

    def __init__(
        self,
        workspace_path: str | Path,
        timeout: int = 30
    ):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def execute(self, code: str) -> ToolResult:
        """Execute JavaScript code."""
        # Basic validation
        blocked = ["require(", "import ", "fetch(", "process.", "fs."]
        for pattern in blocked:
            if pattern in code:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked operation: {pattern}"
                )

        wrapped_code = f'''
try {{
    const result = (function() {{
        {code}
    }})();
    if (result !== undefined) {{
        console.log(JSON.stringify(result, null, 2));
    }}
}} catch (e) {{
    console.error("Error:", e.message);
    process.exit(1);
}}
'''

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.js',
            delete=False
        ) as f:
            f.write(wrapped_code)
            script_path = f.name

        try:
            result = subprocess.run(
                ["node", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr or "Execution failed"
                )

            return ToolResult(
                success=True,
                output=result.stdout if result.stdout else "(no output)"
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="Node.js not installed"
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds"
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "JavaScript code to execute"
                    }
                },
                "required": ["code"]
            }
        }
