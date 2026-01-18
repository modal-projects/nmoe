"""Functional kill test for the agent self-play environment.

Validates:
1) AgentSelfPlayTaskPool produces deterministic, tools-first tasks
2) The task is solvable via CodexExecutor tools in a sandbox
3) compute_all_rewards() receives structure + tool + task signals end-to-end
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_agent_selfplay_gcd_task_tools_and_rewards(tmp_path: Path):
    pytest.importorskip("nmoe.rl.tools.codex", reason="codex_python module not built/available")

    from nmoe.rl.rewards_gdpo import TrajectoryContext, compute_all_rewards
    from nmoe.rl.rewards_tools import ToolCallSite
    from nmoe.rl.tasks.agents import AgentSelfPlayTaskPool, MultiToolGCDTask
    from nmoe.rl.tools import AsyncToolExecutor, ToolCall, ToolConfig, ToolType

    pool = AgentSelfPlayTaskPool(root_dir=tmp_path / "agent_env", seed=0, digits=96)
    task = pool.sample(1)[0]
    assert isinstance(task, MultiToolGCDTask)

    # Execute tools in a sandbox restricted to tmp_path.
    executor = AsyncToolExecutor(
        ToolConfig(
            executor_type="codex_python",
            allow_network=False,
            allowed_paths=[str(tmp_path)],
            cwd=str(tmp_path),
        )
    )
    try:
        bash_call = ToolCall(
            type=ToolType.BASH,
            command=f"head -n 1 {task.inputs_path}",
            timeout_ms=10_000,
        )
        bash_res = asyncio.run(executor.execute_one(bash_call))

        py_code = (
            "import math\n"
            f"p={task.inputs_path!r}\n"
            "with open(p,'r',encoding='utf-8') as f:\n"
            "  a=int(f.readline().strip())\n"
            "  b=int(f.readline().strip())\n"
            "print(math.gcd(a,b), end='')\n"
        )
        py_call = ToolCall(type=ToolType.PYTHON, code=py_code, timeout_ms=30_000)
        py_res = asyncio.run(executor.execute_one(py_call))
    finally:
        executor.close()

    assert bash_res.success
    assert py_res.success
    assert py_res.output.strip() == task.gold_gcd

    # Subsequent text must include enough of each tool output to count as "used".
    bash_snip = bash_res.output.strip()[:40]
    response_text = (
        "<|start|>assistant<|channel|>analysis<|message|>"
        f"I read a prefix={bash_snip} and computed gcd={task.gold_gcd}."
        "<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        f"{task.gold_gcd}"
        "<|end|>"
    )

    tool_sites = [
        ToolCallSite(call=bash_call, result=bash_res, position=0, subsequent_text=response_text),
        ToolCallSite(call=py_call, result=py_res, position=1, subsequent_text=response_text),
    ]
    ctx = TrajectoryContext(
        response_text=response_text,
        tool_sites=tool_sites,
        task=task,
        reasoning_tokens=0,
        format_type="harmony",
    )
    signals = compute_all_rewards(ctx)

    assert signals.has_reasoning == 1.0
    assert signals.has_final_response == 1.0
    assert signals.answer_correct == 1.0
    assert signals.has_tool_use == 1.0
    assert signals.tool_executed == 1.0
    assert signals.tool_output_used == 1.0
    assert signals.python_runs == 1.0
    assert signals.bash_succeeds == 1.0
