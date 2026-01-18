"""Git-based episode generation for code self-play.

Generates CodeEditTask instances from git commit history.
Option A vs B is an emergent property of the repo history:
- Option B: Commits with test changes that fail on parent
- Option A: Commits without testable oracle (exact patch match)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from dataclasses import replace as dc_replace
from typing import Iterator

from nmoe.rl.tasks import TaskPool
from nmoe.rl.tasks.agentic import CodeEditTask
from nmoe.rl.tools.codex import CodexConfig, CodexExecutor


# Patterns that identify spec/test files
SPEC_PATTERNS = [
    r"^tests?/",
    r"_test\.py$",
    r"test_[^/]+\.py$",
    r"conftest\.py$",
    r"pytest\.ini$",
]
SPEC_RE = [re.compile(p) for p in SPEC_PATTERNS]


def is_spec_file(path: str) -> bool:
    """Check if a file path is a spec/test file."""
    return any(r.search(path) for r in SPEC_RE)


@dataclass(frozen=True)
class EpisodeFilterConfig:
    """Configuration for filtering git episodes to ensure training-grade quality.

    Default values are tuned for meaningful code changes (not formatting/trivial).
    """
    # Minimum requirements
    min_impl_files: int = 1          # At least N implementation files changed
    min_changed_lines: int = 5       # At least N lines changed (insertions + deletions)
    require_non_test_impl: bool = True  # At least one impl file must not be a test

    # Maximum limits (reject huge commits)
    max_files: int = 30              # Skip commits touching too many files
    max_changed_lines: int = 2000    # Skip massive diffs

    # Content filters
    reject_whitespace_only: bool = True   # Reject if all changes are whitespace
    reject_formatting_only: bool = True   # Reject commits with only formatting keywords

    # Option-B strictness
    strict_option_b: bool = True     # If True, failed Option-B → skip (not downgrade to A)

    # Formatting keywords that suggest trivial commits
    FORMATTING_KEYWORDS: tuple[str, ...] = (
        "format", "formatting", "lint", "style", "whitespace",
        "indent", "spacing", "trailing", "prettier", "black",
    )


@dataclass
class CommitMeta:
    """Metadata for a single commit."""
    sha: str
    parent_sha: str
    message: str
    files: list[str]
    spec_files: list[str]
    impl_files: list[str]

    # Diff statistics
    insertions: int = 0
    deletions: int = 0

    @property
    def has_spec(self) -> bool:
        return len(self.spec_files) > 0

    @property
    def has_impl(self) -> bool:
        return len(self.impl_files) > 0

    @property
    def total_changed_lines(self) -> int:
        return self.insertions + self.deletions

    @property
    def has_non_test_impl(self) -> bool:
        """True if at least one impl file is not a test file."""
        return any(not is_spec_file(f) for f in self.impl_files)


@dataclass
class GitEpisode:
    """A training episode derived from git history."""
    repo_id: str
    base_sha: str           # Parent commit (starting state)
    target_sha: str         # Child commit (goal state)
    commit_message: str
    spec_files: list[str]   # Test files changed
    impl_files: list[str]   # Implementation files changed

    # Verification
    test_command: str = "python -m pytest"
    hidden_test_command: str = ""  # Eval-only tests

    # Option classification (emergent from verify_option_b)
    option: str = "unknown"  # "A", "B", or "skip"
    verified: bool = False

    # Workspace isolation
    workspace_path: Path | None = None  # Per-episode workspace (set during setup)

    def to_task_id(self) -> str:
        """Generate unique task ID."""
        h = hashlib.sha256(f"{self.repo_id}:{self.base_sha}:{self.target_sha}".encode())
        return f"git_{h.hexdigest()[:12]}"

    @property
    def commit_range(self) -> str:
        """Git commit range for this episode (NOT a patch artifact)."""
        return f"{self.base_sha}..{self.target_sha}"


class GitRepoScanner:
    """Scans a git repo and extracts commit metadata."""

    def __init__(self, repo_path: Path, executor: CodexExecutor):
        self.repo_path = repo_path
        self.executor = executor

    def _run_git(self, *args: str) -> tuple[bool, str]:
        """Run a git command via CodexExecutor."""
        # Quote args that contain special chars
        quoted_args = []
        for arg in args:
            if any(c in arg for c in "|<>$`\"'\\"):
                quoted_args.append(f'"{arg}"')
            else:
                quoted_args.append(arg)
        # GIT_TERMINAL_PROMPT=0 prevents git from waiting for user input
        cmd = f"GIT_TERMINAL_PROMPT=0 git -C {self.repo_path} {' '.join(quoted_args)}"
        result = self.executor.exec_bash(cmd)
        return result.success, result.stdout or ""

    def iter_commits(self, max_commits: int = 100, branch: str = "HEAD") -> Iterator[CommitMeta]:
        """Iterate over commits with metadata."""
        # Get commit list with files
        # Use %P for parent(s) - for merge commits, take first parent only
        ok, output = self._run_git(
            "log", f"--max-count={max_commits}",
            "--first-parent",  # Follow first parent only (skip merge commit side branches)
            "--format=COMMIT:%H|%P%nMSG:%s",
            "--name-only", branch
        )
        if not ok:
            return

        lines = output.strip().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line.startswith("COMMIT:"):
                i += 1
                continue

            # Parse commit line: "COMMIT:sha|parent1 parent2 ..."
            commit_part = line[7:]  # Strip "COMMIT:"
            if "|" not in commit_part:
                i += 1
                continue

            sha_part, parents_part = commit_part.split("|", 1)
            sha = sha_part.strip()
            # Take first parent only
            parents = parents_part.strip().split()
            parent_sha = parents[0] if parents else ""

            # Parse message
            i += 1
            message = ""
            if i < len(lines) and lines[i].startswith("MSG:"):
                message = lines[i][4:].strip()
                i += 1

            # Collect files until next commit
            files = []
            while i < len(lines) and not lines[i].startswith("COMMIT:"):
                f = lines[i].strip()
                if f and not f.startswith("MSG:"):
                    files.append(f)
                i += 1

            if not parent_sha:
                continue  # Skip initial commit

            # Skip merge commits (no files changed in the merge itself)
            if not files:
                continue

            spec_files = [f for f in files if is_spec_file(f)]
            impl_files = [f for f in files if not is_spec_file(f)]

            # Get diff stats (insertions/deletions)
            insertions, deletions = self._get_diff_stats(parent_sha, sha)

            yield CommitMeta(
                sha=sha,
                parent_sha=parent_sha,
                message=message,
                files=files,
                spec_files=spec_files,
                impl_files=impl_files,
                insertions=insertions,
                deletions=deletions,
            )

    def _get_diff_stats(self, base_sha: str, target_sha: str) -> tuple[int, int]:
        """Get insertions and deletions between two commits."""
        ok, output = self._run_git("diff", "--shortstat", base_sha, target_sha)
        if not ok:
            return 0, 0

        # Parse output like: " 3 files changed, 10 insertions(+), 5 deletions(-)"
        insertions = 0
        deletions = 0
        import re
        ins_match = re.search(r"(\d+) insertion", output)
        del_match = re.search(r"(\d+) deletion", output)
        if ins_match:
            insertions = int(ins_match.group(1))
        if del_match:
            deletions = int(del_match.group(1))
        return insertions, deletions

    def get_diff_summary(self, base_sha: str, target_sha: str, max_lines: int = 50) -> str:
        """Get a summary of the diff between two commits."""
        ok, output = self._run_git("diff", "--stat", base_sha, target_sha)
        if not ok:
            return ""
        lines = output.strip().split("\n")
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... and {len(lines) - max_lines} more files"]
        return "\n".join(lines)

    def get_file_content(self, sha: str, path: str) -> str | None:
        """Get file content at a specific commit."""
        ok, output = self._run_git("show", f"{sha}:{path}")
        return output if ok else None


class GitCommitTaskPool(TaskPool):
    """Task pool that generates CodeEditTask from git commit history.

    Usage:
        executor = CodexExecutor(CodexConfig())
        pool = GitCommitTaskPool(
            repo_path="/path/to/repo",
            executor=executor,
            test_command="python -m pytest tests/",
        )
        pool.scan()  # Populate with episodes

        tasks = pool.sample(batch_size=8)
    """

    def __init__(
        self,
        repo_path: str | Path,
        executor: CodexExecutor,
        repo_id: str | None = None,
        test_command: str = "python -m pytest",
        hidden_test_command: str = "",
        workspaces_dir: str | Path | None = None,
        max_commits: int = 500,
        branch: str = "HEAD",
        verify_option_b: bool = False,
        option_b_timeout_s: float = 60.0,
        seed: int | None = None,
        filter_config: EpisodeFilterConfig | None = None,
    ):
        """Initialize GitCommitTaskPool.

        Args:
            repo_path: Path to git repository
            executor: CodexExecutor for sandboxed operations
            repo_id: Identifier for the repo (defaults to path basename)
            test_command: Command to run tests
            hidden_test_command: Eval-only test command (withheld from agent)
            max_commits: Maximum commits to scan
            branch: Git branch to scan
            verify_option_b: Actually run tests to verify Option B episodes
            option_b_timeout_s: Timeout for Option B verification
            seed: Random seed for sampling
            filter_config: Episode filtering configuration (default: EpisodeFilterConfig())
        """
        super().__init__(tasks=[], seed=seed)
        self.repo_path = Path(repo_path)
        self.executor = executor
        self.repo_id = repo_id or self.repo_path.name
        self.test_command = test_command
        self.hidden_test_command = hidden_test_command
        self.workspaces_dir = Path(workspaces_dir) if workspaces_dir is not None else None
        self.max_commits = max_commits
        self.branch = branch
        self.verify_option_b = verify_option_b
        self.option_b_timeout_s = option_b_timeout_s
        self.filter_config = filter_config or EpisodeFilterConfig()

        self.scanner = GitRepoScanner(self.repo_path, executor)
        self.episodes: list[GitEpisode] = []
        self._tasks: list[CodeEditTask] = []

    def scan(self) -> dict[str, int]:
        """Scan repository and populate episodes.

        Applies EpisodeFilterConfig to ensure training-grade quality:
        - Minimum/maximum size constraints
        - Content quality filters (no whitespace-only, no formatting-only)
        - Strict Option-B gating (fail-before/pass-after as hard contract)

        Returns:
            Dict with counts: total, option_a, option_b, skip, filtered
        """
        stats = {"total": 0, "option_a": 0, "option_b": 0, "skip": 0, "filtered": 0}
        cfg = self.filter_config

        for commit in self.scanner.iter_commits(self.max_commits, self.branch):
            stats["total"] += 1

            # === Apply filters ===
            skip_reason = self._should_skip_commit(commit)
            if skip_reason:
                stats["filtered"] += 1
                continue

            episode = GitEpisode(
                repo_id=self.repo_id,
                base_sha=commit.parent_sha,
                target_sha=commit.sha,
                commit_message=commit.message,
                spec_files=commit.spec_files,
                impl_files=commit.impl_files,
                test_command=self.test_command,
                hidden_test_command=self.hidden_test_command,
            )

            # Classify as Option A or B
            if commit.has_spec:
                episode.option = "B"

                # Option-B verification (strict or fallback)
                if self.verify_option_b:
                    verified, fail_reason = self._verify_option_b_strict(episode)
                    if verified:
                        episode.verified = True
                        stats["option_b"] += 1
                    elif cfg.strict_option_b:
                        # Strict mode: skip entirely (don't downgrade to A)
                        stats["skip"] += 1
                        continue
                    else:
                        # Fallback mode: downgrade to Option A
                        episode.option = "A"
                        stats["option_a"] += 1
                else:
                    stats["option_b"] += 1
            else:
                episode.option = "A"
                stats["option_a"] += 1

            self.episodes.append(episode)
            self._tasks.append(self._episode_to_task(episode))

        # Update parent class task list
        self.tasks = self._tasks
        return stats

    def _should_skip_commit(self, commit: CommitMeta) -> str | None:
        """Check if commit should be filtered out. Returns skip reason or None."""
        cfg = self.filter_config

        # No implementation changes
        if not commit.has_impl:
            return "no_impl"

        # Too few impl files
        if len(commit.impl_files) < cfg.min_impl_files:
            return "too_few_impl_files"

        # Require at least one non-test impl file
        if cfg.require_non_test_impl and not commit.has_non_test_impl:
            return "all_impl_are_tests"

        # Too few changed lines
        if commit.total_changed_lines < cfg.min_changed_lines:
            return "too_few_lines"

        # Too many files
        if len(commit.files) > cfg.max_files:
            return "too_many_files"

        # Too many changed lines
        if commit.total_changed_lines > cfg.max_changed_lines:
            return "too_many_lines"

        # Formatting-only commit (by message keywords)
        if cfg.reject_formatting_only:
            msg_lower = commit.message.lower()
            if any(kw in msg_lower for kw in cfg.FORMATTING_KEYWORDS):
                return "formatting_only"

        return None

    def sample(self, n: int, replace: bool = True) -> list[CodeEditTask]:
        """Sample tasks and materialize an isolated workspace per episode.

        For code self-play we must not run inside the scanner repo (episodes
        would contaminate each other). On sampling, we:
        1) select episodes via TaskPool sampling
        2) create a hermetic workspace at base_sha (+ spec files for Option B)
        3) return a task whose repo_path points at that workspace
        """
        sampled = super().sample(n, replace=replace)
        out: list[CodeEditTask] = []
        for task in sampled:
            ep = self.get_episode(task.task_id)
            if ep is None:
                out.append(task)
                continue
            if not self.setup_workspace(ep, workspaces_dir=self.workspaces_dir):
                out.append(task)
                continue
            if ep.workspace_path is None:
                out.append(task)
                continue
            out.append(dc_replace(task, repo_path=str(ep.workspace_path)))
        return out

    def _verify_option_b_strict(self, episode: GitEpisode) -> tuple[bool, str]:
        """Verify Option B with strict requirements.

        Requirements for a valid Option-B episode:
        1. Tests on (parent + spec) must FAIL with nonzero exit (not timeout/crash)
        2. Tests on child must PASS within timeout

        Critical contract:
        - This must NOT mutate the source repo checkout. Verification runs in
          ephemeral workspaces materialized via `git archive`.

        Returns:
            (success, fail_reason) tuple
        """
        from nmoe.rl.tasks.code_workspace import materialize_from_git_archive, run_in_workspace
        import shlex

        ws_root = self.workspaces_dir or Path("/tmp/nmoe_workspaces")
        verify_root = ws_root / "_verify_option_b"
        base_ws = verify_root / f"{episode.to_task_id()}_base"
        child_ws = verify_root / f"{episode.to_task_id()}_child"

        def _rm_rf(p: Path) -> None:
            self.executor.exec_bash(f"rm -rf {shlex.quote(str(p))}")

        def _mkdir(p: Path) -> bool:
            return bool(self.executor.exec_bash(f"mkdir -p {shlex.quote(str(p))}").success)

        # Keep verification workspaces bounded and ephemeral.
        _rm_rf(base_ws)
        _rm_rf(child_ws)
        if not _mkdir(base_ws) or not _mkdir(child_ws):
            return False, "mkdir_workspace_failed"

        try:
            # Materialize base workspace (parent commit).
            if not materialize_from_git_archive(
                executor=self.executor,
                repo_path=self.repo_path,
                sha=episode.base_sha,
                workspace_path=base_ws,
            ):
                return False, "materialize_parent_failed"

            # Apply spec files from child into base workspace.
            for spec_file in episode.spec_files:
                content = self.scanner.get_file_content(episode.target_sha, spec_file)
                if content is None:
                    continue
                target_path = base_ws / spec_file
                if not _mkdir(target_path.parent):
                    return False, "apply_spec_failed"
                cmd = f"cat > {shlex.quote(str(target_path))} << 'NMOE_EPISODE_EOF'\n{content}\nNMOE_EPISODE_EOF"
                if not self.executor.exec_bash(cmd).success:
                    return False, "apply_spec_failed"

            # Run tests on parent+spec - should FAIL for the right reason.
            # Use shell `timeout` so we get a stable exit code (124) instead of a raised exception.
            cmd = f"timeout {int(self.option_b_timeout_s)} {self.test_command}"
            out = run_in_workspace(
                workspace_path=base_ws,
                command=cmd,
                timeout_ms=int((self.option_b_timeout_s + 5.0) * 1000),
            )
            if out.success:
                return False, "tests_pass_before_fix"
            if out.exit_code == 124:
                return False, "tests_timeout_before"
            if out.exit_code == 127 or "command not found" in (out.stderr or "").lower():
                return False, "test_command_not_found"

            # Materialize child workspace (target commit) and verify tests pass.
            if not materialize_from_git_archive(
                executor=self.executor,
                repo_path=self.repo_path,
                sha=episode.target_sha,
                workspace_path=child_ws,
            ):
                return False, "materialize_child_failed"

            out = run_in_workspace(
                workspace_path=child_ws,
                command=cmd,
                timeout_ms=int((self.option_b_timeout_s + 5.0) * 1000),
            )
            if not out.success:
                return False, "tests_fail_after_fix"

            return True, ""
        finally:
            _rm_rf(base_ws)
            _rm_rf(child_ws)

    def _verify_option_b(self, episode: GitEpisode) -> bool:
        """Legacy wrapper for backward compatibility."""
        success, _ = self._verify_option_b_strict(episode)
        return success

    def _episode_to_task(self, episode: GitEpisode) -> CodeEditTask:
        """Convert GitEpisode to CodeEditTask.

        Note: gold_patch is intentionally left empty. The commit_range is stored
        in the GitEpisode, not passed to CodeEditTask, to avoid confusion
        (it's a rev-range label, not an actual patch artifact).
        """
        # Build issue description
        diff_summary = self.scanner.get_diff_summary(episode.base_sha, episode.target_sha)

        if episode.option == "B":
            # Option B: describe the failing tests
            issue_desc = (
                f"The following tests are failing. Fix the code to make them pass.\n\n"
                f"Commit context: {episode.commit_message}\n\n"
                f"Files changed:\n{diff_summary}\n\n"
                f"Test files added/modified: {', '.join(episode.spec_files)}"
            )
        else:
            # Option A: describe the change needed
            issue_desc = (
                f"Implement the following change:\n\n"
                f"{episode.commit_message}\n\n"
                f"Files to modify:\n{diff_summary}"
            )

        return CodeEditTask(
            task_id=episode.to_task_id(),
            issue_description=issue_desc,
            repo_path=str(self.repo_path),
            test_command=episode.test_command,
            hidden_test_command=episode.hidden_test_command,
            files_to_edit=episode.impl_files,
            gold_patch="",  # Intentionally empty - use episode.commit_range for analysis
            max_turns=10,
        )

    def get_episode(self, task_id: str) -> GitEpisode | None:
        """Get the GitEpisode for a task ID."""
        for ep in self.episodes:
            if ep.to_task_id() == task_id:
                return ep
        return None

    def setup_workspace(
        self,
        episode: GitEpisode,
        workspaces_dir: Path | None = None,
    ) -> bool:
        """Setup isolated workspace for an episode.

        Creates a copy of the repo at the base commit state for hermetic execution.
        Each episode gets its own directory to prevent cross-episode contamination.

        Note: We use cp + git checkout rather than git worktree because the
        Landlock sandbox restricts writes to .git/worktrees.

        Args:
            episode: The episode to setup
            workspaces_dir: Base directory for workspaces (default: repo_path/../workspaces)

        Returns:
            True if setup succeeded
        """
        if workspaces_dir is None:
            # Use /tmp for sandbox compatibility (Landlock allows /tmp writes)
            workspaces_dir = Path("/tmp/nmoe_workspaces")

        workspace_path = workspaces_dir / episode.to_task_id()

        # Clean up existing workspace if present
        from nmoe.rl.tasks.code_workspace import materialize_from_git_archive
        import shlex

        cmd = f"rm -rf {shlex.quote(str(workspace_path))}"
        self.executor.exec_bash(cmd)

        # Create workspace directory
        cmd = f"mkdir -p {shlex.quote(str(workspace_path))}"
        result = self.executor.exec_bash(cmd)
        if not result.success:
            return False

        # Copy repo (excluding .git internals we don't need).
        # Use git archive to get a clean snapshot at base commit.
        if not materialize_from_git_archive(
            executor=self.executor,
            repo_path=self.repo_path,
            sha=episode.base_sha,
            workspace_path=workspace_path,
        ):
            return False

        episode.workspace_path = workspace_path

        # For Option B, apply spec files from target commit
        if episode.option == "B" and episode.spec_files:
            for spec_file in episode.spec_files:
                # Get file content from target commit
                content = self.scanner.get_file_content(episode.target_sha, spec_file)
                if content is None:
                    continue
                # Write to workspace
                target_path = workspace_path / spec_file
                cmd = f"mkdir -p $(dirname '{target_path}')"
                self.executor.exec_bash(cmd)
                # Use heredoc for safe content writing
                cmd = f"cat > '{target_path}' << 'NMOE_EPISODE_EOF'\n{content}\nNMOE_EPISODE_EOF"
                result = self.executor.exec_bash(cmd)
                if not result.success:
                    return False

        # Record provenance (mandatory): base/target SHA, sandbox config, and test commands.
        from nmoe.rl.tasks.bootstrap import BootstrapConfig, BootstrapStrategy
        from nmoe.rl.tasks.code_workspace import setup_workspace_with_provenance

        ok, _prov = setup_workspace_with_provenance(
            workspace_path=workspace_path,
            base_sha=episode.base_sha,
            target_sha=episode.target_sha,
            repo_id=self.repo_id,
            test_command=episode.test_command,
            hidden_test_command=episode.hidden_test_command or "",
            bootstrap_config=BootstrapConfig(strategy=BootstrapStrategy.NONE),
            auto_detect_bootstrap=False,
        )
        return bool(ok)

    def cleanup_workspace(self, episode: GitEpisode) -> None:
        """Clean up an episode's workspace."""
        if episode.workspace_path:
            cmd = f"rm -rf {episode.workspace_path}"
            self.executor.exec_bash(cmd)
            episode.workspace_path = None

    def verify_solution(self, episode: GitEpisode) -> tuple[bool, dict]:
        """Verify a solution for an episode.

        Runs tests in the episode's isolated workspace.

        Returns:
            Tuple of (success, details_dict)
        """
        details = {
            "test_passed": False,
            "hidden_test_passed": False,
            "test_output": "",
            "hidden_test_output": "",
        }

        # Use workspace if available, else fall back to main repo
        work_dir = episode.workspace_path or self.repo_path

        # Run main tests
        cmd = f"cd {work_dir} && {episode.test_command}"
        result = self.executor.exec_bash(cmd)
        details["test_passed"] = result.success
        details["test_output"] = result.stdout or result.stderr or ""

        # Run hidden tests if configured
        if episode.hidden_test_command:
            cmd = f"cd {work_dir} && {episode.hidden_test_command}"
            result = self.executor.exec_bash(cmd)
            details["hidden_test_passed"] = result.success
            details["hidden_test_output"] = result.stdout or result.stderr or ""

        success = details["test_passed"]
        if episode.hidden_test_command:
            success = success and details["hidden_test_passed"]

        return success, details


def clone_repo(
    repo_url: str,
    target_dir: Path,
    executor: CodexExecutor,
    shallow: bool = False,
    depth: int = 500,
) -> bool:
    """Clone a git repository.

    Args:
        repo_url: Git URL to clone
        target_dir: Directory to clone into
        executor: CodexExecutor for sandboxed git operations
        shallow: Use shallow clone (faster but limited history)
        depth: Depth for shallow clone

    Returns:
        True if successful
    """
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if shallow:
        cmd = f"git clone --depth {depth} {repo_url} {target_dir}"
    else:
        cmd = f"git clone {repo_url} {target_dir}"

    result = executor.exec_bash(cmd)
    return result.success


def create_git_task_pool(
    repo_path: str | Path,
    test_command: str = "python -m pytest",
    max_commits: int = 500,
    verify: bool = False,
) -> GitCommitTaskPool:
    """Convenience factory for GitCommitTaskPool.

    Args:
        repo_path: Path to git repository
        test_command: Test command for the repo
        max_commits: Maximum commits to scan
        verify: Whether to verify Option B episodes

    Returns:
        Populated GitCommitTaskPool
    """
    config = CodexConfig(timeout_ms=60000)
    executor = CodexExecutor(config)

    pool = GitCommitTaskPool(
        repo_path=repo_path,
        executor=executor,
        test_command=test_command,
        max_commits=max_commits,
        verify_option_b=verify,
    )
    pool.scan()
    return pool


@dataclass
class RepoConfig:
    """Configuration for a repository to mine episodes from."""
    url: str
    name: str
    test_command: str = "python -m pytest"
    hidden_test_command: str = ""
    branch: str = "main"
    max_commits: int = 500


# Common Python repos with good test coverage
COMMON_REPOS = [
    RepoConfig(
        url="https://github.com/psf/requests",
        name="requests",
        test_command="python -m pytest tests/ -x -q",
        branch="main",
    ),
    RepoConfig(
        url="https://github.com/pallets/flask",
        name="flask",
        test_command="python -m pytest tests/ -x -q",
        branch="main",
    ),
    RepoConfig(
        url="https://github.com/tiangolo/fastapi",
        name="fastapi",
        test_command="python -m pytest tests/ -x -q",
        branch="master",
    ),
    RepoConfig(
        url="https://github.com/pallets/click",
        name="click",
        test_command="python -m pytest tests/ -x -q",
        branch="main",
    ),
]


class MultiRepoTaskPool(TaskPool):
    """Task pool that aggregates episodes from multiple repositories."""

    def __init__(
        self,
        repos_dir: Path,
        executor: CodexExecutor,
        repo_configs: list[RepoConfig] | None = None,
        workspaces_dir: str | Path | None = None,
        seed: int | None = None,
    ):
        """Initialize MultiRepoTaskPool.

        Args:
            repos_dir: Base directory for cloned repos
            executor: CodexExecutor for operations
            repo_configs: List of repo configurations (default: COMMON_REPOS)
            seed: Random seed
        """
        super().__init__(tasks=[], seed=seed)
        self.repos_dir = Path(repos_dir)
        self.executor = executor
        self.repo_configs = repo_configs or COMMON_REPOS
        self.workspaces_dir = Path(workspaces_dir) if workspaces_dir is not None else None
        self.pools: dict[str, GitCommitTaskPool] = {}
        self.episodes: list[GitEpisode] = []
        self._pool_by_task_id: dict[str, GitCommitTaskPool] = {}

    def setup(self, clone_missing: bool = True) -> dict[str, dict]:
        """Setup all repositories.

        Args:
            clone_missing: Clone repos that aren't already present

        Returns:
            Dict of repo_name -> scan stats
        """
        all_stats = {}

        for config in self.repo_configs:
            repo_path = self.repos_dir / config.name

            # Clone if needed
            if not repo_path.exists() and clone_missing:
                print(f"Cloning {config.name}...")
                ok = clone_repo(config.url, repo_path, self.executor, shallow=True)
                if not ok:
                    print(f"  Failed to clone {config.name}")
                    continue

            if not repo_path.exists():
                continue

            # Create pool and scan
            pool = GitCommitTaskPool(
                repo_path=repo_path,
                executor=self.executor,
                repo_id=config.name,
                test_command=config.test_command,
                hidden_test_command=config.hidden_test_command,
                workspaces_dir=(self.workspaces_dir / config.name) if self.workspaces_dir is not None else None,
                max_commits=config.max_commits,
                branch=config.branch,
            )

            stats = pool.scan()
            all_stats[config.name] = stats
            self.pools[config.name] = pool
            self.episodes.extend(pool.episodes)

        # Aggregate all tasks
        self.tasks = []
        for pool in self.pools.values():
            self.tasks.extend(pool.tasks)
            for task in pool.tasks:
                self._pool_by_task_id[task.task_id] = pool

        # Rebuild type index
        self._by_type = {}
        for task in self.tasks:
            if task.task_type not in self._by_type:
                self._by_type[task.task_type] = []
            self._by_type[task.task_type].append(task)

        return all_stats

    def sample(self, n: int, replace: bool = True) -> list[CodeEditTask]:
        """Sample tasks across repos and materialize isolated workspaces."""
        sampled = super().sample(n, replace=replace)
        out: list[CodeEditTask] = []
        for task in sampled:
            pool = self._pool_by_task_id.get(task.task_id)
            if pool is None:
                out.append(task)
                continue
            ep = pool.get_episode(task.task_id)
            if ep is None:
                out.append(task)
                continue
            if not pool.setup_workspace(ep):
                out.append(task)
                continue
            if ep.workspace_path is None:
                out.append(task)
                continue
            out.append(dc_replace(task, repo_path=str(ep.workspace_path)))
        return out

    def get_pool(self, repo_name: str) -> GitCommitTaskPool | None:
        """Get the pool for a specific repo."""
        return self.pools.get(repo_name)
