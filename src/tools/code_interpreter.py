"""Python code interpreter with sandboxing."""

import io
import os
import subprocess
import sys
import tempfile
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SandboxConfig:
    """Configuration for code execution sandbox."""

    timeout: int = 30  # seconds
    memory_limit: str = "512m"
    use_docker: bool = False
    docker_image: str = "python:3.11-slim"
    allowed_imports: set[str] = field(
        default_factory=lambda: {
            "pandas",
            "numpy",
            "scipy",
            "matplotlib",
            "sympy",
            "math",
            "statistics",
            "collections",
            "itertools",
            "functools",
            "json",
            "re",
            "datetime",
        }
    )
    workspace_path: Optional[Path] = None


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str
    stderr: str
    return_value: Optional[str] = None
    error: Optional[str] = None
    figures: list[str] = field(default_factory=list)


class CodeInterpreter:
    """Safe Python code execution environment."""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._workspace = self.config.workspace_path or Path(tempfile.mkdtemp())
        self._workspace.mkdir(parents=True, exist_ok=True)

        # Shared namespace for persistent variables
        self._namespace: dict[str, Any] = {}
        self._figure_counter = 0

    def _validate_imports(self, code: str) -> Optional[str]:
        """Check for disallowed imports."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in self.config.allowed_imports:
                        return f"Import not allowed: {module}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in self.config.allowed_imports:
                        return f"Import not allowed: {module}"

        return None

    def _setup_matplotlib(self) -> None:
        """Configure matplotlib for non-interactive backend."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.ioff()
        except ImportError:
            pass

    def execute_local(self, code: str) -> ExecutionResult:
        """Execute code in local Python environment with restrictions."""
        # Validate imports
        import_error = self._validate_imports(code)
        if import_error:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=import_error,
            )

        # Setup matplotlib
        self._setup_matplotlib()

        # Capture output
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Prepare namespace with safe builtins
        safe_builtins = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "any": any,
                "all": all,
                "open": None,  # Disabled
                "exec": None,  # Disabled
                "eval": None,  # Disabled
                "__import__": __import__,
            }
        }

        exec_namespace = {**safe_builtins, **self._namespace}
        exec_namespace["__workspace__"] = str(self._workspace)

        figures: list[str] = []

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, exec_namespace)

            # Check for matplotlib figures
            try:
                import matplotlib.pyplot as plt

                if plt.get_fignums():
                    self._figure_counter += 1
                    fig_path = self._workspace / f"figure_{self._figure_counter}.png"
                    plt.savefig(fig_path, dpi=100, bbox_inches="tight")
                    figures.append(str(fig_path))
                    plt.close("all")
            except ImportError:
                pass

            # Update persistent namespace (exclude builtins)
            for key, value in exec_namespace.items():
                if not key.startswith("_") and key != "__builtins__":
                    self._namespace[key] = value

            # Get last expression value if available
            return_value = None
            lines = code.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                if (
                    last_line
                    and not last_line.startswith(("#", "import", "from", "def", "class", "if", "for", "while", "with", "try"))
                    and "=" not in last_line.split("#")[0]
                ):
                    try:
                        return_value = str(eval(last_line, exec_namespace))
                    except Exception:
                        pass

            return ExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                return_value=return_value,
                figures=figures,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

    def execute_docker(self, code: str) -> ExecutionResult:
        """Execute code in Docker container."""
        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            code_path = f.name

        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # No network access
                f"--memory={self.config.memory_limit}",
                "--cpus=1",
                "-v",
                f"{code_path}:/code.py:ro",
                "-v",
                f"{self._workspace}:/workspace",
                "-w",
                "/workspace",
                self.config.docker_image,
                "python",
                "/code.py",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            # Check for generated figures
            figures = [
                str(f)
                for f in self._workspace.glob("*.png")
            ]

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.stderr if result.returncode != 0 else None,
                figures=figures,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution timed out after {self.config.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=str(e),
            )
        finally:
            os.unlink(code_path)

    def execute(self, code: str) -> ExecutionResult:
        """Execute code using configured method."""
        if self.config.use_docker:
            return self.execute_docker(code)
        return self.execute_local(code)

    def reset(self) -> None:
        """Reset interpreter state."""
        self._namespace.clear()
        self._figure_counter = 0

    @property
    def workspace(self) -> Path:
        """Get workspace path."""
        return self._workspace


# Global instance
_interpreter: Optional[CodeInterpreter] = None


def _get_interpreter() -> CodeInterpreter:
    """Get or create global interpreter."""
    global _interpreter
    if _interpreter is None:
        _interpreter = CodeInterpreter()
    return _interpreter


def init_interpreter(config: Optional[SandboxConfig] = None) -> None:
    """Initialize global interpreter with config."""
    global _interpreter
    _interpreter = CodeInterpreter(config)


def execute_python(code: str) -> str:
    """
    Execute Python code and return result.

    Args:
        code: Python code to execute

    Returns:
        String with stdout, return value, and any errors
    """
    result = _get_interpreter().execute(code)

    output_parts = []

    if result.stdout:
        output_parts.append(f"Output:\n{result.stdout}")

    if result.return_value:
        output_parts.append(f"Result: {result.return_value}")

    if result.figures:
        output_parts.append(f"Figures saved: {', '.join(result.figures)}")

    if result.error:
        output_parts.append(f"Error: {result.error}")

    if not output_parts:
        if result.success:
            output_parts.append("Code executed successfully (no output).")
        else:
            output_parts.append("Execution failed with no output.")

    return "\n".join(output_parts)


def get_workspace_path() -> str:
    """Get path to code interpreter workspace."""
    return str(_get_interpreter().workspace)
