"""Sandbox implementations for code execution."""

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None


class BaseSandbox(ABC):
    """Abstract base class for sandboxed execution."""

    @abstractmethod
    def execute(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute code in sandbox."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass


class DockerSandbox(BaseSandbox):
    """Docker-based sandbox for code execution."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "512m",
        network: bool = False,
        workspace: Optional[Path] = None,
    ):
        self.image = image
        self.memory_limit = memory_limit
        self.network = network
        self.workspace = workspace or Path(tempfile.mkdtemp())
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Pre-pull image
        self._ensure_image()

    def _ensure_image(self) -> None:
        """Ensure Docker image exists."""
        try:
            subprocess.run(
                ["docker", "image", "inspect", self.image],
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["docker", "pull", self.image],
                check=True,
            )

    def execute(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute Python code in Docker container."""
        # Write code to temp file
        code_file = self.workspace / "code.py"
        code_file.write_text(code, encoding="utf-8")

        cmd = [
            "docker",
            "run",
            "--rm",
            f"--memory={self.memory_limit}",
            "--cpus=1",
            "--pids-limit=50",
            "-v",
            f"{code_file}:/code.py:ro",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
        ]

        if not self.network:
            cmd.append("--network=none")

        cmd.extend([self.image, "python", "/code.py"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return SandboxResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error=f"Execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                error=str(e),
            )

    def cleanup(self) -> None:
        """Clean up workspace."""
        import shutil

        if self.workspace.exists():
            shutil.rmtree(self.workspace)


class RestrictedSandbox(BaseSandbox):
    """
    RestrictedPython-based sandbox for local execution.

    More lightweight than Docker but less isolated.
    """

    ALLOWED_BUILTINS = {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "int",
        "float",
        "str",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "isinstance",
        "any",
        "all",
    }

    ALLOWED_IMPORTS = {
        "math",
        "statistics",
        "collections",
        "itertools",
        "functools",
        "json",
        "re",
        "datetime",
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
    }

    def __init__(self, workspace: Optional[Path] = None):
        self.workspace = workspace or Path(tempfile.mkdtemp())
        self.workspace.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, timeout: int = 30) -> SandboxResult:
        """Execute code with restrictions."""
        import io
        import signal
        from contextlib import redirect_stdout, redirect_stderr

        # Validate code
        validation_error = self._validate_code(code)
        if validation_error:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=validation_error,
                exit_code=1,
                error=validation_error,
            )

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Build restricted namespace
        namespace = self._build_namespace()

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timed out after {timeout} seconds")

        try:
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, namespace)

            signal.alarm(0)  # Cancel timeout

            return SandboxResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                exit_code=0,
            )

        except TimeoutError as e:
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=str(e),
                exit_code=-1,
                error=str(e),
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=str(e),
                exit_code=1,
                error=str(e),
            )
        finally:
            signal.alarm(0)

    def _validate_code(self, code: str) -> Optional[str]:
        """Validate code for dangerous patterns."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in self.ALLOWED_IMPORTS:
                        return f"Import not allowed: {module}"

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in self.ALLOWED_IMPORTS:
                        return f"Import not allowed: {module}"

            # Check for dangerous builtins
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {"eval", "exec", "compile", "open", "__import__"}:
                        return f"Function not allowed: {node.func.id}"

        return None

    def _build_namespace(self) -> dict:
        """Build restricted namespace."""
        import builtins

        safe_builtins = {
            name: getattr(builtins, name)
            for name in self.ALLOWED_BUILTINS
            if hasattr(builtins, name)
        }

        safe_builtins["__import__"] = self._restricted_import

        return {
            "__builtins__": safe_builtins,
            "__workspace__": str(self.workspace),
        }

    def _restricted_import(self, name: str, *args, **kwargs) -> None:
        """Restricted import function."""
        module = name.split(".")[0]
        if module not in self.ALLOWED_IMPORTS:
            raise ImportError(f"Import not allowed: {name}")
        return __import__(name, *args, **kwargs)

    def cleanup(self) -> None:
        """Clean up workspace."""
        import shutil

        if self.workspace.exists():
            shutil.rmtree(self.workspace)
