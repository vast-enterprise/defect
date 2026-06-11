//! A [`ShellBackend`] implementation for local processes.
//!
//! Originates from the same inline `tokio::process::Command` flow historically used in
//! the `bash` tool, but moves process management, buffered reads, and exit
//! synchronization into the backend layer so that `BashTool` interacts only through the
//! [`ShellBackend`] trait — a local shell execution backend.
//!
//! Internal data structures:
//!
//! - `LocalShellBackend.terminals: Mutex<HashMap<TerminalId, Arc<TerminalState>>>`
//!   Global terminal table.
//! - `TerminalState` holds the output buffer, `exit` status, `exit_notify`, and
//!   `kill_notify`.
//! - Each terminal spawns a **reader task**: blocks reading stdout/stderr → writes into
//!   buffer → waits on `kill_notify` or both EOFs → calls `child.wait()` → writes `exit`
//!   → calls `notify_waiters()`. The child is exclusively owned by the reader task to
//!   avoid lock contention.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use defect_agent::error::BoxError;
use defect_agent::shell::{ShellBackend, ShellError, ShellOutput, TerminalExitStatus, TerminalId};
use futures::future::BoxFuture;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Notify;

/// Default per-terminal captured-output cap (1 MiB). Output beyond this is dropped and
/// counted as `truncated`.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 1024 * 1024;

/// Local shell backend: each command spawns a `sh -c` child process, with state managed
/// in the `terminals` table until `release`.
pub struct LocalShellBackend {
    terminals: Mutex<HashMap<TerminalId, Arc<TerminalState>>>,
    /// Maximum bytes of merged stdout/stderr captured per terminal; excess is truncated.
    max_output_bytes: usize,
}

impl LocalShellBackend {
    pub fn new() -> Self {
        Self::with_max_output_bytes(DEFAULT_MAX_OUTPUT_BYTES)
    }

    /// Constructs a backend with an explicit captured-output cap. `0` is clamped to `1` so
    /// at least one byte can always be captured.
    pub fn with_max_output_bytes(max_output_bytes: usize) -> Self {
        Self {
            terminals: Mutex::new(HashMap::new()),
            max_output_bytes: max_output_bytes.max(1),
        }
    }

    fn lookup(&self, id: &TerminalId) -> Result<Arc<TerminalState>, ShellError> {
        let guard = self
            .terminals
            .lock()
            .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
        guard
            .get(id)
            .cloned()
            .ok_or_else(|| ShellError::NotFound(id.clone()))
    }
}

impl Default for LocalShellBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for a single terminal. The reader task and `output` / `wait_for_exit` /
/// `kill` all share access via `Arc<TerminalState>`.
struct TerminalState {
    output: Mutex<OutputBuffer>,
    exit: Mutex<Option<TerminalExitStatus>>,
    exit_notify: Notify,
    /// Set by `kill`; the reader task observes it in a `select` and calls
    /// `Child::start_kill()`. Uses `notify_one()` to buffer a permit, preventing signal
    /// loss when the reader task has not yet registered a waiter (`notify_waiters` only
    /// wakes already-registered waiters). The reader task deduplicates via a `killed`
    /// flag, so multiple kills are equivalent to one.
    kill_notify: Notify,
}

#[derive(Debug, thiserror::Error)]
#[error("local shell backend mutex poisoned")]
struct PoisonedTable;

impl ShellBackend for LocalShellBackend {
    fn create(
        &self,
        command: String,
        cwd: PathBuf,
    ) -> BoxFuture<'_, Result<TerminalId, ShellError>> {
        Box::pin(async move {
            let mut cmd = build_command(&command);
            cmd.current_dir(&cwd)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .kill_on_drop(true);

            let mut child = cmd
                .spawn()
                .map_err(|err| ShellError::Backend(BoxError::new(err)))?;

            let stdout = child.stdout.take().expect("piped stdout");
            let stderr = child.stderr.take().expect("piped stderr");

            let id = next_terminal_id();
            let state = Arc::new(TerminalState {
                output: Mutex::new(OutputBuffer::new(self.max_output_bytes)),
                exit: Mutex::new(None),
                exit_notify: Notify::new(),
                kill_notify: Notify::new(),
            });

            {
                let mut guard = self
                    .terminals
                    .lock()
                    .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                guard.insert(id.clone(), state.clone());
            }

            tokio::spawn(reader_task(state, child, stdout, stderr));

            Ok(id)
        })
    }

    fn output(&self, id: &TerminalId) -> BoxFuture<'_, Result<ShellOutput, ShellError>> {
        let id = id.clone();
        Box::pin(async move {
            let state = self.lookup(&id)?;
            let (text, truncated) = {
                let buf = state
                    .output
                    .lock()
                    .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                (
                    String::from_utf8_lossy(buf.as_bytes()).into_owned(),
                    buf.truncated() > 0,
                )
            };
            let exit_status = {
                let exit = state
                    .exit
                    .lock()
                    .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                exit.clone()
            };
            Ok(ShellOutput {
                text,
                truncated,
                exit_status,
            })
        })
    }

    fn wait_for_exit(
        &self,
        id: &TerminalId,
    ) -> BoxFuture<'_, Result<TerminalExitStatus, ShellError>> {
        let id = id.clone();
        Box::pin(async move {
            let state = self.lookup(&id)?;
            loop {
                {
                    let exit = state
                        .exit
                        .lock()
                        .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                    if let Some(status) = exit.as_ref() {
                        return Ok(status.clone());
                    }
                }
                // `notified()` only observes `notify_waiters` calls made **after** it is
                // registered – so register first, then double-check for an already-set
                // value to avoid a race.
                let notified = state.exit_notify.notified();
                tokio::pin!(notified);
                {
                    let exit = state
                        .exit
                        .lock()
                        .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                    if let Some(status) = exit.as_ref() {
                        return Ok(status.clone());
                    }
                }
                notified.await;
            }
        })
    }

    fn release(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let id = id.clone();
        Box::pin(async move {
            let removed = {
                let mut guard = self
                    .terminals
                    .lock()
                    .map_err(|_| ShellError::Backend(BoxError::new(PoisonedTable)))?;
                guard.remove(&id)
            };
            // Notify the reader task to wind down if it is still running. The `Child`
            // held by the reader task will be dropped when the task exits, triggering the
            // `kill_on_drop` fallback.
            if let Some(state) = removed {
                state.kill_notify.notify_one();
            }
            Ok(())
        })
    }

    fn kill(&self, id: &TerminalId) -> BoxFuture<'_, Result<(), ShellError>> {
        let id = id.clone();
        Box::pin(async move {
            let state = self.lookup(&id)?;
            state.kill_notify.notify_one();
            Ok(())
        })
    }
}

async fn reader_task(
    state: Arc<TerminalState>,
    mut child: Child,
    stdout: tokio::process::ChildStdout,
    stderr: tokio::process::ChildStderr,
) {
    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut stderr_lines = BufReader::new(stderr).lines();
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut killed = false;

    while stdout_open || stderr_open {
        tokio::select! {
            _ = state.kill_notify.notified(), if !killed => {
                killed = true;
                let _ = child.start_kill();
                // Continue draining: after `start_kill`, the child process receives
                // SIGKILL, the pipe fds close, and both `next_line` calls will naturally
                // return EOF. Note that commands like `sh -c "sleep N"` leave `sleep`
                // alive because `sh` does not `exec` it; the caller is responsible for
                // `exec`-ing the long-running part in the shell command (or accepting
                // that `kill_on_drop` will handle it on release).
            }
            line = stdout_lines.next_line(), if stdout_open => {
                match line {
                    Ok(Some(mut l)) => {
                        l.push('\n');
                        if let Ok(mut buf) = state.output.lock() {
                            buf.push(l.as_bytes());
                        }
                    }
                    _ => stdout_open = false,
                }
            }
            line = stderr_lines.next_line(), if stderr_open => {
                match line {
                    Ok(Some(mut l)) => {
                        l.push('\n');
                        if let Ok(mut buf) = state.output.lock() {
                            buf.push(l.as_bytes());
                        }
                    }
                    _ => stderr_open = false,
                }
            }
        }
    }
    // When already killed, `killed` also means "terminated by external request" — the
    // exit status from the `wait` below reflects the actual signal (SIGKILL/SIGTERM,
    // etc.).
    let _ = killed;

    let wait_result = child.wait().await;
    let status = decode_status(wait_result.ok().as_ref());
    if let Ok(mut exit) = state.exit.lock() {
        *exit = Some(status);
    }
    state.exit_notify.notify_waiters();
}

#[cfg(unix)]
fn decode_status(status: Option<&std::process::ExitStatus>) -> TerminalExitStatus {
    use std::os::unix::process::ExitStatusExt;
    match status {
        None => TerminalExitStatus {
            exit_code: None,
            signal: None,
        },
        Some(s) => {
            if let Some(code) = s.code() {
                TerminalExitStatus {
                    exit_code: Some(code),
                    signal: None,
                }
            } else if let Some(sig) = s.signal() {
                TerminalExitStatus {
                    exit_code: None,
                    signal: Some(signal_name(sig)),
                }
            } else {
                TerminalExitStatus {
                    exit_code: None,
                    signal: None,
                }
            }
        }
    }
}

#[cfg(windows)]
fn decode_status(status: Option<&std::process::ExitStatus>) -> TerminalExitStatus {
    match status {
        None => TerminalExitStatus {
            exit_code: None,
            signal: None,
        },
        Some(s) => TerminalExitStatus {
            exit_code: s.code(),
            signal: None,
        },
    }
}

#[cfg(unix)]
fn signal_name(sig: i32) -> String {
    match sig {
        1 => "SIGHUP".into(),
        2 => "SIGINT".into(),
        3 => "SIGQUIT".into(),
        6 => "SIGABRT".into(),
        9 => "SIGKILL".into(),
        13 => "SIGPIPE".into(),
        14 => "SIGALRM".into(),
        15 => "SIGTERM".into(),
        other => format!("SIG#{other}"),
    }
}

#[cfg(unix)]
fn build_command(command: &str) -> Command {
    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg(command);
    cmd
}

#[cfg(windows)]
fn build_command(command: &str) -> Command {
    let mut cmd = Command::new("cmd");
    cmd.arg("/C").arg(command);
    cmd
}

/// An append-only buffer with a 1 MiB cap. Excess bytes are dropped but counted in
/// `truncated`.
struct OutputBuffer {
    bytes: Vec<u8>,
    truncated: u64,
    max_bytes: usize,
}

impl OutputBuffer {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            truncated: 0,
            max_bytes,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        let remaining = self.max_bytes.saturating_sub(self.bytes.len());
        if remaining == 0 {
            self.truncated += chunk.len() as u64;
            return;
        }
        if chunk.len() <= remaining {
            self.bytes.extend_from_slice(chunk);
        } else {
            self.bytes
                .extend_from_slice(chunk.get(..remaining).unwrap_or(chunk));
            self.truncated += (chunk.len() - remaining) as u64;
        }
    }

    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn truncated(&self) -> u64 {
        self.truncated
    }
}

/// A monotonically increasing terminal ID generator. The prefix includes the nanos at
/// process start to avoid conflicts with old IDs from future persistence scenarios.
fn next_terminal_id() -> TerminalId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    static PREFIX: OnceLock<String> = OnceLock::new();
    let prefix = PREFIX.get_or_init(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("local-{ts:x}")
    });
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    TerminalId::new(format!("{prefix}-{n:x}"))
}

#[cfg(test)]
mod tests;
