//! 本地进程的 [`ShellBackend`] 实现。
//!
//! 与历史上 `bash` 工具内联的 `tokio::process::Command` 流程同源，但把
//! 进程管理 / 缓冲读 / 退出同步搬到 backend 层，让 `BashTool` 只通过
//! [`ShellBackend`] trait — local shell execution backend.
//!
//! 内部数据结构：
//!
//! - `LocalShellBackend.terminals: Mutex<HashMap<TerminalId, Arc<TerminalState>>>`
//!   全局 terminal 表
//! - `TerminalState` 持有 output 缓冲、`exit` 状态、`exit_notify`、`kill_notify`
//! - 每个 terminal 启动一个 **reader task**：阻塞读 stdout/stderr → 进 buffer
//!   → 同步等 `kill_notify` 或两端 EOF → `child.wait()` → 写 `exit` →
//!   `notify_waiters()`。Child 由 reader task 独占持有，避免锁竞争。

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

const MAX_OUTPUT_BYTES: usize = 1024 * 1024;

/// 本地 shell 后端：每条命令 spawn 一个 `sh -c` 子进程，状态托管在 `terminals`
/// 表里直至 `release`。
pub struct LocalShellBackend {
    terminals: Mutex<HashMap<TerminalId, Arc<TerminalState>>>,
}

impl LocalShellBackend {
    pub fn new() -> Self {
        Self {
            terminals: Mutex::new(HashMap::new()),
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

/// 单个 terminal 的运行态。reader task 与 `output` / `wait_for_exit` /
/// `kill` 都通过 `Arc<TerminalState>` 共享访问。
struct TerminalState {
    output: Mutex<OutputBuffer>,
    exit: Mutex<Option<TerminalExitStatus>>,
    exit_notify: Notify,
    /// `kill` 调用置位；reader task 在 select 里观察到后调 `Child::start_kill()`。
    /// 用 `notify_one()` 缓冲一个 permit，避免 reader task 还没注册 waiter 时
    /// 信号丢失（`notify_waiters` 只唤醒已注册等待者）。reader task 用 `killed`
    /// 标志去重，多次 kill 等价于一次。
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
                output: Mutex::new(OutputBuffer::new()),
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
                // notified() 只对**之后**的 notify_waiters 生效——所以先注册
                // 再做 already-set 二次探测，避免错过 race。
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
            // 通知 reader task：如果还在跑，让它收尾。reader task 持有的 Child
            // 在 task 退出后 drop，触发 kill_on_drop 兜底。
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
                // 继续 drain：start_kill 之后子进程会 SIGKILL，pipe fd 关闭，
                // 两个 next_line 自然 EOF。注意 `sh -c "sleep N"` 这类命令会
                // 因 sh 没 exec 而把 sleep 留下；调用方有责任在 shell 命令里
                // `exec` 真正长跑的部分（或接受 release 时 kill_on_drop 兜底）。
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
    // 已 kill 的情况下 killed 也代表"被外部要求终止"——下面 wait 的退出态
    // 反映实际信号（SIGKILL/SIGTERM 等）。
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

/// 1 MiB 上限的 append-only buffer。超额字节 drop 但记入 `truncated`。
struct OutputBuffer {
    bytes: Vec<u8>,
    truncated: u64,
}

impl OutputBuffer {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            truncated: 0,
        }
    }

    fn push(&mut self, chunk: &[u8]) {
        let remaining = MAX_OUTPUT_BYTES.saturating_sub(self.bytes.len());
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

/// 单调递增的 terminal id 生成器。前缀加进程启动时的 nanos，避免与未来
/// 持久化场景的旧 id 冲突。
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
