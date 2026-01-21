"""Failure taxonomy for agentic RL.

This is a small, closed taxonomy used to prevent silent reward noise:
- Every non-successful tool execution should map to one of these categories.
- Categories are stable string enums so they can be logged/aggregated easily.
"""

from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    SANDBOX_DENIED = "sandbox_denied"
    SANDBOX_SETUP_FAILED = "sandbox_setup_failed"
    COMMAND_NOT_FOUND = "command_not_found"
    OOM = "oom"
    NONZERO_EXIT = "nonzero_exit"
    PARSE_ERROR = "parse_error"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"


def categorize_failure(
    *,
    success: bool,
    timed_out: bool = False,
    exit_code: int | None = None,
    error: str | None = None,
    stderr: str | None = None,
    stdout: str | None = None,
) -> str:
    """Categorize an execution outcome into a stable FailureCategory string."""
    _ = stdout
    if success:
        return FailureCategory.OK.value

    err = (error or "").lower()
    se = (stderr or "").lower()
    ec = int(exit_code) if exit_code is not None else None

    if timed_out or ec == 124 or "timed out" in err or "timed out" in se:
        return FailureCategory.TIMEOUT.value

    if ec == 127 or "command not found" in err or "command not found" in se:
        return FailureCategory.COMMAND_NOT_FOUND.value

    if "out of memory" in err or "out of memory" in se or "cuda oom" in se:
        return FailureCategory.OOM.value

    # Sandboxed write denials show up as EPERM/EACCES/permission denied.
    if "permission denied" in err or "permission denied" in se:
        return FailureCategory.SANDBOX_DENIED.value
    if "sandboxdenied" in err:
        return FailureCategory.SANDBOX_DENIED.value
    if "landlock not applied" in err or "seccomp not applied" in err:
        return FailureCategory.SANDBOX_SETUP_FAILED.value
    if "landlock not applied" in se or "seccomp not applied" in se:
        return FailureCategory.SANDBOX_SETUP_FAILED.value

    if ec is not None and ec != 0:
        return FailureCategory.NONZERO_EXIT.value

    if error:
        return FailureCategory.INTERNAL_ERROR.value

    return FailureCategory.UNKNOWN.value

