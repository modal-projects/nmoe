//! Python bindings for codex-rs sandboxed execution.
//!
//! Provides a Python interface to execute code in a sandboxed environment
//! using Linux Landlock (filesystem) and Seccomp (syscall) restrictions.

use pyo3::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use std::collections::HashMap;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::io::Read;

/// Result of executing a command in the sandbox.
#[pyclass]
#[derive(Clone)]
pub struct ExecResult {
    #[pyo3(get)]
    pub success: bool,
    #[pyo3(get)]
    pub exit_code: i32,
    #[pyo3(get)]
    pub stdout: String,
    #[pyo3(get)]
    pub stderr: String,
    #[pyo3(get)]
    pub duration_ms: u64,
    #[pyo3(get)]
    pub sandboxed: bool,
}

#[pymethods]
impl ExecResult {
    fn __repr__(&self) -> String {
        format!(
            "ExecResult(success={}, exit_code={}, stdout_len={}, stderr_len={}, sandboxed={})",
            self.success, self.exit_code, self.stdout.len(), self.stderr.len(), self.sandboxed
        )
    }
}

/// Configuration for the sandbox executor.
#[pyclass]
#[derive(Clone)]
pub struct SandboxConfig {
    #[pyo3(get, set)]
    pub timeout_ms: u64,
    #[pyo3(get, set)]
    pub working_dir: Option<String>,
    #[pyo3(get, set)]
    pub allow_network: bool,
    #[pyo3(get, set)]
    pub read_only_paths: Vec<String>,
    #[pyo3(get, set)]
    pub read_write_paths: Vec<String>,
}

#[pymethods]
impl SandboxConfig {
    #[new]
    #[pyo3(signature = (timeout_ms=30000, working_dir=None, allow_network=false))]
    fn new(timeout_ms: u64, working_dir: Option<String>, allow_network: bool) -> Self {
        Self {
            timeout_ms,
            working_dir,
            allow_network,
            read_only_paths: Vec::new(),
            read_write_paths: Vec::new(),
        }
    }

    fn add_read_only_path(&mut self, path: String) {
        self.read_only_paths.push(path);
    }

    fn add_read_write_path(&mut self, path: String) {
        self.read_write_paths.push(path);
    }
}

/// Sandbox executor for running commands securely.
///
/// Uses Linux Landlock for filesystem isolation and Seccomp for network blocking.
/// These are applied in the child process via pre_exec hooks.
#[pyclass]
pub struct SandboxExecutor {
    config: SandboxConfig,
}

#[pymethods]
impl SandboxExecutor {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<SandboxConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_else(|| SandboxConfig::new(30000, None, false));
        Ok(Self { config })
    }

    /// Execute a shell command in the sandbox.
    #[pyo3(signature = (command, env=None))]
    fn exec_command(&self, command: &str, env: Option<HashMap<String, String>>) -> PyResult<ExecResult> {
        exec_sandboxed(command, &self.config, env)
    }

    /// Execute Python code in the sandbox.
    #[pyo3(signature = (code, env=None))]
    fn exec_python(&self, code: &str, env: Option<HashMap<String, String>>) -> PyResult<ExecResult> {
        let command = format!("python3 -c {}", shell_escape(code));
        self.exec_command(&command, env)
    }

    /// Execute a Python file in the sandbox.
    #[pyo3(signature = (path, args=None, env=None))]
    fn exec_python_file(&self, path: &str, args: Option<Vec<String>>, env: Option<HashMap<String, String>>) -> PyResult<ExecResult> {
        let args_str = args.map(|a| a.join(" ")).unwrap_or_default();
        let command = format!("python3 {} {}", path, args_str);
        self.exec_command(&command, env)
    }

    /// Execute bash script in the sandbox.
    #[pyo3(signature = (script, env=None))]
    fn exec_bash(&self, script: &str, env: Option<HashMap<String, String>>) -> PyResult<ExecResult> {
        let command = format!("bash -c {}", shell_escape(script));
        self.exec_command(&command, env)
    }
}

/// Shell-escape a string for safe command execution.
fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Execute command with sandbox restrictions applied in child process.
fn exec_sandboxed(
    command: &str,
    config: &SandboxConfig,
    env: Option<HashMap<String, String>>,
) -> PyResult<ExecResult> {
    let start = Instant::now();
    let timeout = Duration::from_millis(config.timeout_ms);

    // Build the command
    let mut cmd = Command::new("bash");
    cmd.arg("-c").arg(command);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if let Some(cwd) = &config.working_dir {
        cmd.current_dir(cwd);
    }

    if let Some(env_vars) = env {
        for (key, value) in env_vars {
            cmd.env(key, value);
        }
    }

    // Clone config values for pre_exec closure
    let allow_network = config.allow_network;
    let writable_paths: Vec<String> = config.read_write_paths.clone();
    let cwd = config.working_dir.clone();

    // Apply sandbox in pre_exec (runs in child after fork, before exec)
    // SAFETY: pre_exec runs in child process after fork
    unsafe {
        cmd.pre_exec(move || {
            // Apply Landlock filesystem restrictions (fail-closed).
            //
            // Silent downshifts are not acceptable for training correctness.
            if let Err(e) = apply_landlock_rules(&writable_paths, cwd.as_deref()) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Landlock not applied: {e}"),
                ));
            }

            // Apply Seccomp network restrictions (fail-closed).
            if !allow_network {
                if let Err(e) = apply_network_seccomp() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Seccomp not applied: {e}"),
                    ));
                }
            }

            Ok(())
        });
    }

    // Spawn the process
    let mut child = cmd.spawn()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to spawn: {e}")))?;

    // Wait with timeout
    let result = wait_with_timeout(&mut child, timeout);
    let duration_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok((exit_code, stdout, stderr)) => {
            Ok(ExecResult {
                success: exit_code == 0,
                exit_code,
                stdout,
                stderr,
                duration_ms,
                sandboxed: true,
            })
        }
        Err(WaitError::Timeout) => {
            Err(PyTimeoutError::new_err(format!(
                "Command timed out after {}ms",
                config.timeout_ms
            )))
        }
        Err(WaitError::Io(e)) => {
            Err(PyRuntimeError::new_err(format!("Execution failed: {e}")))
        }
    }
}

enum WaitError {
    Timeout,
    Io(std::io::Error),
}

fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<(i32, String, String), WaitError> {
    use std::thread;

    // Take ownership of stdout/stderr handles
    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    // Spawn threads to read stdout/stderr concurrently
    // This prevents deadlock when output exceeds pipe buffer (~64KB)
    let stdout_thread = thread::spawn(move || {
        let mut output = String::new();
        if let Some(mut handle) = stdout_handle {
            let _ = handle.read_to_string(&mut output);
        }
        output
    });

    let stderr_thread = thread::spawn(move || {
        let mut output = String::new();
        if let Some(mut handle) = stderr_handle {
            let _ = handle.read_to_string(&mut output);
        }
        output
    });

    // Poll for process completion with timeout
    let start = Instant::now();
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => {
                if start.elapsed() > timeout {
                    // Ensure the child is terminated before joining reader threads,
                    // otherwise they can block forever on open stdout/stderr pipes.
                    let _ = child.kill();
                    let _ = child.wait();
                    break Err(WaitError::Timeout);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => break Err(WaitError::Io(e)),
        }
    };

    // Join reader threads (they'll complete once the process exits or is killed)
    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();

    match exit_status {
        Ok(status) => Ok((status.code().unwrap_or(-1), stdout, stderr)),
        Err(e) => Err(e),
    }
}

/// Apply Landlock filesystem restrictions.
/// Allows read everywhere, write only to specified paths + /tmp + cwd.
fn apply_landlock_rules(writable_paths: &[String], cwd: Option<&str>) -> Result<(), String> {
    use landlock::{
        ABI, Access, AccessFs, CompatLevel, Compatible, PathBeneath, PathFd,
        Ruleset, RulesetAttr, RulesetCreatedAttr, RulesetStatus,
    };

    let abi = ABI::V5;
    let access_rw = AccessFs::from_all(abi);
    let access_ro = AccessFs::from_read(abi);

    let mut ruleset = Ruleset::default()
        .set_compatibility(CompatLevel::BestEffort)
        .handle_access(access_rw)
        .map_err(|e| format!("handle_access: {e}"))?
        .create()
        .map_err(|e| format!("create: {e}"))?;

    // Read access to everything
    ruleset = ruleset
        .add_rule(PathBeneath::new(PathFd::new("/").map_err(|e| format!("open /: {e}"))?, access_ro))
        .map_err(|e| format!("add_rule /: {e}"))?;

    // Write access to /dev/null
    ruleset = ruleset
        .add_rule(PathBeneath::new(PathFd::new("/dev/null").map_err(|e| format!("open /dev/null: {e}"))?, access_rw))
        .map_err(|e| format!("add_rule /dev/null: {e}"))?;

    // Write access to /tmp
    ruleset = ruleset
        .add_rule(PathBeneath::new(PathFd::new("/tmp").map_err(|e| format!("open /tmp: {e}"))?, access_rw))
        .map_err(|e| format!("add_rule /tmp: {e}"))?;

    // Write access to cwd if specified
    if let Some(cwd_path) = cwd {
        if let Ok(fd) = PathFd::new(cwd_path) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_rw))
                .map_err(|e| format!("add_rule cwd: {e}"))?;
        }
    }

    // Write access to user-specified paths
    for path in writable_paths {
        if let Ok(fd) = PathFd::new(path) {
            ruleset = ruleset
                .add_rule(PathBeneath::new(fd, access_rw))
                .map_err(|e| format!("add_rule {path}: {e}"))?;
        }
    }

    ruleset = ruleset.set_no_new_privs(true);

    let status = ruleset.restrict_self().map_err(|e| format!("restrict_self: {e}"))?;

    if status.ruleset == RulesetStatus::NotEnforced {
        return Err("Landlock not enforced (kernel too old?)".into());
    }

    Ok(())
}

/// Apply Seccomp filter to block network syscalls.
fn apply_network_seccomp() -> Result<(), String> {
    use seccompiler::{
        BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp,
        SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
    };
    use std::collections::BTreeMap;

    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();

    // Block network syscalls
    let deny_syscalls = [
        libc::SYS_connect,
        libc::SYS_accept,
        libc::SYS_accept4,
        libc::SYS_bind,
        libc::SYS_listen,
        libc::SYS_sendto,
        libc::SYS_sendmmsg,
        libc::SYS_recvmmsg,
    ];

    for &syscall in &deny_syscalls {
        rules.insert(syscall, vec![]); // empty = unconditional deny
    }

    // Allow AF_UNIX sockets only
    let unix_only = SeccompRule::new(vec![
        SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Ne, libc::AF_UNIX as u64)
            .map_err(|e| format!("SeccompCondition: {e}"))?
    ]).map_err(|e| format!("SeccompRule: {e}"))?;

    rules.insert(libc::SYS_socket, vec![unix_only]);

    let arch = if cfg!(target_arch = "x86_64") {
        TargetArch::x86_64
    } else if cfg!(target_arch = "aarch64") {
        TargetArch::aarch64
    } else {
        return Err("Unsupported architecture".into());
    };

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        arch,
    ).map_err(|e| format!("SeccompFilter: {e}"))?;

    let prog: BpfProgram = filter.try_into().map_err(|e| format!("BpfProgram: {e}"))?;
    seccompiler::apply_filter(&prog).map_err(|e| format!("apply_filter: {e}"))?;

    Ok(())
}

/// Execute Python code with test assertions.
#[pyfunction]
#[pyo3(signature = (code, tests, timeout_ms=30000))]
fn execute_python_tests(code: &str, tests: &str, timeout_ms: u64) -> PyResult<ExecResult> {
    let config = SandboxConfig::new(timeout_ms, None, false);
    let full_code = format!("{}\n\n{}", code, tests);
    let command = format!("python3 -c {}", shell_escape(&full_code));
    exec_sandboxed(&command, &config, None)
}

/// Quick utility to run a single Python expression and get the result.
#[pyfunction]
#[pyo3(signature = (expr, timeout_ms=5000))]
fn eval_python(expr: &str, timeout_ms: u64) -> PyResult<String> {
    let config = SandboxConfig::new(timeout_ms, None, false);
    let code = format!("print({})", expr);
    let command = format!("python3 -c {}", shell_escape(&code));

    let result = exec_sandboxed(&command, &config, None)?;

    if result.success {
        Ok(result.stdout.trim().to_string())
    } else {
        Err(PyRuntimeError::new_err(result.stderr))
    }
}

/// Python module definition.
#[pymodule]
fn codex_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExecResult>()?;
    m.add_class::<SandboxConfig>()?;
    m.add_class::<SandboxExecutor>()?;
    m.add_function(wrap_pyfunction!(execute_python_tests, m)?)?;
    m.add_function(wrap_pyfunction!(eval_python, m)?)?;
    Ok(())
}
