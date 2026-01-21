"""Tool execution for agentic RL training.

Provides async tool execution with scatter/gather pattern for
parallel tool calls during generation.

Supports:
- Python code execution (sandboxed)
- Bash command execution (sandboxed)
- File operations (read, search)
- Integration with codex-rs for production sandboxing
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolType(Enum):
    """Supported tool types."""
    PYTHON = "python"
    BASH = "bash"
    READ = "read"
    SEARCH = "search"
    EDIT = "edit"


@dataclass
class ToolCall:
    """A tool invocation request."""

    type: ToolType | str
    call_id: str = ""

    # Tool-specific arguments
    command: str = ""  # For bash
    code: str = ""  # For python
    path: str = ""  # For read/edit
    query: str = ""  # For search
    content: str = ""  # For edit

    # Execution parameters
    timeout_ms: int = 30000
    # Working directory for execution. If empty, uses ToolConfig.cwd.
    cwd: str = ""

    # Raw arguments (for custom tools)
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = ToolType(self.type)
            except ValueError:
                pass  # Keep as string for custom tools


@dataclass
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    success: bool
    output: str = ""
    error: str | None = None
    exit_code: int = 0
    execution_time_ms: float = 0.0
    timed_out: bool = False
    failure_category: str = "ok"

    # For code tools
    compiled: bool = False
    executed: bool = False

    # Raw result data
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, call_id: str, error: str) -> "ToolResult":
        """Create error result."""
        return cls(
            call_id=call_id,
            success=False,
            error=error,
            exit_code=1,
            failure_category="internal_error",
        )

    @classmethod
    def from_timeout(cls, call_id: str) -> "ToolResult":
        """Create timeout result."""
        return cls(
            call_id=call_id,
            success=False,
            error="Execution timed out",
            exit_code=124,  # Standard timeout exit code
            timed_out=True,
            failure_category="timeout",
        )


@dataclass
class ToolConfig:
    """Configuration for tool execution."""

    # Executor type:
    # - "codex_python": production sandboxed execution via codex-rs PyO3 bindings
    # - "subprocess": legacy codex CLI path (optional)
    # - "native": local subprocess fallback (NOT sandboxed; dev-only)
    executor_type: str = "codex_python"

    # Codex binary path (for subprocess executor)
    codex_binary: str = "codex"

    # Sandbox mode for codex
    sandbox_mode: str = "workspace-write"

    # Worker counts for parallel execution
    python_workers: int = 4
    bash_workers: int = 4
    read_workers: int = 8

    # Timeouts
    timeout_default_ms: int = 30000
    timeout_python_ms: int = 60000
    timeout_bash_ms: int = 30000

    # Security
    allow_network: bool = False
    allowed_paths: list[str] = field(default_factory=list)

    # Working directory
    cwd: str = "."


# Re-export executor
from nmoe.rl.tools.executor import AsyncToolExecutor

__all__ = [
    "ToolType",
    "ToolCall",
    "ToolResult",
    "ToolConfig",
    "AsyncToolExecutor",
]
