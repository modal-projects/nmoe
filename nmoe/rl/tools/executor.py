"""Async tool executor with scatter/gather pattern.

Provides parallel tool execution for agentic RL training.
All execution goes through codex_python with Landlock/Seccomp sandboxing.
"""
from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from nmoe.rl.tools import ToolCall, ToolConfig, ToolResult
from nmoe.rl.tools.codex import CodexConfig, CodexExecutor
from nmoe.rl.failures import categorize_failure


class AsyncToolExecutor:
    """Async tool execution with scatter/gather pattern.

    All execution goes through CodexExecutor with kernel-level sandboxing
    (Landlock for filesystem, Seccomp for syscalls/network).

    Usage:
        executor = AsyncToolExecutor(config)

        # Scatter: dispatch tools async
        call_ids = executor.scatter([call1, call2, call3])

        # Gather: wait for results
        results = await executor.gather(call_ids)

        # Or poll for non-blocking check
        completed, pending = await executor.poll(call_ids)
    """

    def __init__(self, config: ToolConfig | None = None):
        """Initialize executor.

        Args:
            config: Tool configuration (default: ToolConfig())
        """
        self.config = config or ToolConfig()
        self._pending: dict[str, asyncio.Task] = {}

    def _generate_call_id(self, tool_type: str) -> str:
        """Generate unique call ID."""
        return f"{tool_type}_{uuid4().hex[:8]}"

    async def _execute_call(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call via CodexExecutor."""
        start = time.perf_counter()

        try:
            result_dict = await self._execute_via_codex_python(call)

            return ToolResult(
                call_id=call.call_id,
                success=result_dict.get("success", False),
                output=result_dict.get("output", ""),
                error=result_dict.get("error"),
                exit_code=result_dict.get("exit_code", 0),
                execution_time_ms=result_dict.get("execution_time_ms", 0),
                timed_out=result_dict.get("timed_out", False),
                failure_category=categorize_failure(
                    success=bool(result_dict.get("success", False)),
                    timed_out=bool(result_dict.get("timed_out", False)),
                    exit_code=int(result_dict.get("exit_code", 0) or 0),
                    error=result_dict.get("error"),
                    stderr=result_dict.get("stderr"),
                    stdout=result_dict.get("output", ""),
                ),
                compiled=result_dict.get("compiled", False),
                executed=result_dict.get("executed", False),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=str(e),
                exit_code=1,
                execution_time_ms=elapsed_ms,
                failure_category=categorize_failure(
                    success=False,
                    timed_out=False,
                    exit_code=1,
                    error=str(e),
                    stderr=str(e),
                ),
            )

    async def _execute_via_codex_python(self, call: ToolCall) -> dict:
        """Execute via codex-python (PyO3 bindings) with kernel-enforced sandboxing."""
        tool_type = call.type if isinstance(call.type, str) else call.type.value

        timeout_ms = int(call.timeout_ms or self.config.timeout_default_ms)
        working_dir = call.cwd or self.config.cwd
        cfg = CodexConfig(
            timeout_ms=timeout_ms,
            working_dir=working_dir,
            allow_network=bool(self.config.allow_network),
            read_write_paths=list(self.config.allowed_paths),
        )

        def _run() -> dict:
            executor = CodexExecutor(cfg)

            if tool_type == "python":
                res = executor.exec_python(call.code)
            elif tool_type == "bash":
                res = executor.exec_bash(call.command)
            elif tool_type == "read":
                # Route through sandboxed bash to enforce Landlock restrictions
                res = executor.exec_bash(f"cat {repr(call.path)}")
            else:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Unsupported tool type: {tool_type}",
                    "exit_code": 1,
                    "execution_time_ms": 0.0,
                    "timed_out": False,
                }

            err = res.stderr if (not res.success and res.stderr) else None
            compiled = False
            executed = False
            if tool_type == "python":
                stderr_text = res.stderr or ""
                is_syntax_error = "SyntaxError" in stderr_text
                compiled = not is_syntax_error
                executed = bool(res.success) and compiled

            return {
                "success": bool(res.success),
                "output": res.stdout,
                "error": err,
                "stderr": res.stderr,
                "exit_code": int(res.exit_code),
                "execution_time_ms": float(res.duration_ms),
                "timed_out": False,
                "compiled": compiled,
                "executed": executed,
            }

        # codex-python spawns child processes; keep it off the event loop.
        return await asyncio.to_thread(_run)

    def scatter(self, calls: list[ToolCall]) -> list[str]:
        """Dispatch tool calls asynchronously (non-blocking).

        Args:
            calls: List of tool calls to execute

        Returns:
            List of call IDs for later gathering
        """
        call_ids = []

        for call in calls:
            if not call.call_id:
                tool_type = call.type if isinstance(call.type, str) else call.type.value
                call.call_id = self._generate_call_id(tool_type)

            task = asyncio.create_task(self._execute_call(call))
            self._pending[call.call_id] = task
            call_ids.append(call.call_id)

        return call_ids

    async def gather(
        self,
        call_ids: list[str],
        timeout: float | None = None,
    ) -> list[ToolResult]:
        """Wait for tool calls to complete (blocking).

        Args:
            call_ids: List of call IDs to wait for
            timeout: Optional timeout in seconds

        Returns:
            List of ToolResults in same order as call_ids
        """
        results = []

        for call_id in call_ids:
            task = self._pending.pop(call_id, None)
            if task is None:
                results.append(ToolResult.from_error(
                    call_id, f"Unknown call_id: {call_id}"
                ))
                continue

            try:
                if timeout:
                    result = await asyncio.wait_for(task, timeout=timeout)
                else:
                    result = await task
                results.append(result)
            except asyncio.TimeoutError:
                task.cancel()
                results.append(ToolResult.from_timeout(call_id))
            except Exception as e:
                results.append(ToolResult.from_error(call_id, str(e)))

        return results

    async def poll(
        self,
        call_ids: list[str],
    ) -> tuple[list[ToolResult], list[str]]:
        """Non-blocking check for completed calls.

        Args:
            call_ids: List of call IDs to check

        Returns:
            Tuple of (completed_results, still_pending_ids)
        """
        completed = []
        pending = []

        for call_id in call_ids:
            task = self._pending.get(call_id)
            if task is None:
                completed.append(ToolResult.from_error(
                    call_id, f"Unknown call_id: {call_id}"
                ))
                continue

            if task.done():
                self._pending.pop(call_id)
                try:
                    result = task.result()
                    completed.append(result)
                except Exception as e:
                    completed.append(ToolResult.from_error(call_id, str(e)))
            else:
                pending.append(call_id)

        return completed, pending

    async def execute_one(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call (convenience method).

        Args:
            call: Tool call to execute

        Returns:
            ToolResult
        """
        call_ids = self.scatter([call])
        results = await self.gather(call_ids)
        return results[0]

    def close(self):
        """Clean up executor resources."""
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()

    def __del__(self):
        self.close()
