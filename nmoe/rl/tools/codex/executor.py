"""Codex-RS PyO3 bindings wrapper for sandboxed execution.

Wraps the native codex_python module built from Rust with PyO3/maturin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Import native PyO3 bindings
import codex_python as _native

# Re-export native types
ExecResult = _native.ExecResult
SandboxConfig = _native.SandboxConfig
SandboxExecutor = _native.SandboxExecutor

# Re-export standalone functions
execute_python_tests = _native.execute_python_tests
eval_python = _native.eval_python


@dataclass
class CodexConfig:
    """High-level configuration for CodexExecutor."""

    timeout_ms: int = 30000
    working_dir: Optional[str] = None
    allow_network: bool = False
    read_write_paths: list[str] = field(default_factory=list)

    def to_native(self) -> SandboxConfig:
        """Convert to native SandboxConfig."""
        config = SandboxConfig(
            timeout_ms=self.timeout_ms,
            working_dir=self.working_dir,
            allow_network=self.allow_network,
        )
        for path in self.read_write_paths:
            config.add_read_write_path(path)
        return config


class CodexExecutor:
    """High-level executor wrapping native SandboxExecutor.

    Example:
        executor = CodexExecutor()
        result = executor.exec_python("print('hello')")
        assert result.success
        assert result.stdout.strip() == "hello"
    """

    def __init__(self, config: Optional[CodexConfig] = None):
        self.config = config or CodexConfig()
        native_config = self.config.to_native()
        self._executor = SandboxExecutor(native_config)

    def exec_command(self, command: str, env: Optional[dict[str, str]] = None) -> ExecResult:
        """Execute a shell command in the sandbox."""
        return self._executor.exec_command(command, env)

    def exec_python(self, code: str, env: Optional[dict[str, str]] = None) -> ExecResult:
        """Execute Python code in the sandbox."""
        return self._executor.exec_python(code, env)

    def exec_python_file(
        self,
        path: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ExecResult:
        """Execute a Python file in the sandbox."""
        return self._executor.exec_python_file(path, args, env)

    def exec_bash(self, script: str, env: Optional[dict[str, str]] = None) -> ExecResult:
        """Execute a bash script in the sandbox."""
        return self._executor.exec_bash(script, env)

    def exec_tests(self, code: str, tests: str) -> ExecResult:
        """Execute Python code with test assertions."""
        return execute_python_tests(code, tests, self.config.timeout_ms)

    def eval_expr(self, expr: str) -> str:
        """Evaluate a Python expression, return result as string."""
        return eval_python(expr, self.config.timeout_ms)
