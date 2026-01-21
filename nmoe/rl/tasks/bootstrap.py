"""Bootstrap configuration for code workspaces.

This module is intentionally pure-Python (no sandbox/tool imports) so task
objects can be imported in environments where codex_python is not built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BootstrapStrategy(str, Enum):
    """Bootstrap strategy for dependency installation/setup."""

    NONE = "none"  # No bootstrap (tests run without deps)
    AUTO = "auto"  # Auto-detect based on workspace files
    PIP_EDITABLE = "pip_editable"  # python -m pip install -e .
    UV_SYNC = "uv_sync"  # uv sync --frozen (requires uv.lock)
    REQUIREMENTS_TXT = "requirements_txt"  # pip install -r requirements.txt
    PYPROJECT_EDITABLE = "pyproject_editable"  # pip install -e . with pyproject.toml


@dataclass
class BootstrapConfig:
    """Configuration for workspace bootstrap.

    Execution semantics:
    - If `commands` is non-empty, they run in order.
    - If `commands` is empty and `strategy` != NONE, commands are expanded from
      the strategy (AUTO resolves via file detection).
    """

    strategy: BootstrapStrategy = BootstrapStrategy.NONE
    commands: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)  # e.g., ["dev", "test"]
    requirements_file: str = "requirements.txt"  # For REQUIREMENTS_TXT
    timeout_ms: int = 300_000  # 5 min default for installs
    allow_network: bool = False  # Explicit opt-in for pip/uv network access


@dataclass
class BootstrapResult:
    """Result of running a bootstrap configuration."""

    success: bool
    strategy: str
    commands_run: list[str]
    exit_codes: list[int]
    duration_ms: int
    error: str = ""

