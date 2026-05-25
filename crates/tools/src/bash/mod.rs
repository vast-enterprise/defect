//! `bash` 内置工具：跑一条非交互 shell 命令、合并 stdout/stderr、单帧返回。
//!
//! 设计与取舍详见 `docs/internal/tools-bash.md`。

use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Stdio;
use std::time::Duration;

use agent_client_protocol::schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallLocation, ToolCallUpdateFields,
    ToolKind,
};
use defect_agent::error::BoxError;
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use defect_config::BashToolConfig;
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MAX_OUTPUT_BYTES: usize = 1024 * 1024;
const TITLE_TRUNC: usize = 80;

/// v0 内置 bash 工具。无内部状态——单例 `Arc::new(BashTool::new())` 即可。
pub struct BashTool {
    schema: ToolSchema,
    default_timeout_ms: u64,
    max_timeout_ms: u64,
}

impl BashTool {
    pub fn new() -> Self {
        Self::from_config(&BashToolConfig {
            default_timeout_ms: DEFAULT_TIMEOUT_MS,
            max_timeout_ms: MAX_TIMEOUT_MS,
        })
    }

    pub fn from_config(config: &BashToolConfig) -> Self {
        let default_timeout_ms = config.default_timeout_ms.max(1);
        let max_timeout_ms = config.max_timeout_ms.max(default_timeout_ms);
        Self {
            schema: ToolSchema {
                name: "bash".to_string(),
                description: format!(
                    "Run a non-interactive shell command. \
                     Captures stdout and stderr (merged); returns combined output and \
                     exit code. Times out after `timeout_ms` (default {default_timeout_ms}; max {max_timeout_ms})."
                ),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute (passed to `sh -c` on unix, `cmd /C` on windows)."
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Optional working directory. Must resolve inside the session cwd; relative paths resolve against the session cwd. Defaults to the session cwd."
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": max_timeout_ms,
                            "description": format!(
                                "Per-call timeout in milliseconds. Defaults to {default_timeout_ms}."
                            )
                        }
                    },
                    "required": ["command"]
                }),
            },
            default_timeout_ms,
            max_timeout_ms,
        }
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
struct BashArgs {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BashOutput {
    /// `None` 时表示子进程被信号杀掉或超时；查 `signal` / `timed_out`。
    exit_code: Option<i32>,
    /// 子进程被信号终止时给出信号名，如 `SIGKILL`；否则 `None`。
    #[serde(skip_serializing_if = "Option::is_none")]
    signal: Option<String>,
    timed_out: bool,
    /// 因 1 MiB cap 被 drop 的字节数（≥0）。
    truncated_bytes: u64,
    /// 实测耗时（毫秒）。spawn 失败时不写。
    duration_ms: u64,
}

impl Tool for BashTool {
    fn schema(&self) -> &ToolSchema {
        &self.schema
    }

    fn safety_hint(&self, _args: &serde_json::Value) -> SafetyClass {
        // 一律 Destructive——v0 不解析命令文本。详见 docs/internal/tools-bash.md §2。
        SafetyClass::Destructive
    }

    fn describe<'a>(
        &'a self,
        args: &'a serde_json::Value,
        _ctx: ToolContext<'a>,
    ) -> BoxFuture<'a, ToolCallDescription> {
        Box::pin(async move {
            let command = args
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let workdir = args
                .get("workdir")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let title = format!("$ {}", truncate_title(&command));
            let mut fields = ToolCallUpdateFields::default();
            fields.title = Some(title);
            fields.kind = Some(ToolKind::Execute);
            if let Some(dir) = workdir {
                fields.locations = Some(vec![ToolCallLocation::new(PathBuf::from(dir))]);
            }
            ToolCallDescription { fields }
        })
    }

    fn execute(&self, args: serde_json::Value, ctx: ToolContext<'_>) -> ToolStream {
        let cwd = ctx.cwd.to_path_buf();
        let cancel = ctx.cancel.clone();
        let default_timeout_ms = self.default_timeout_ms;
        let max_timeout_ms = self.max_timeout_ms;
        let fut =
            async move { run_bash(args, cwd, cancel, default_timeout_ms, max_timeout_ms).await };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

/// 一次完整的 bash 调用：解析 args、resolve workdir、spawn、捕获、终态。
/// 返回单一 [`ToolEvent`]——`Completed` 或 `Failed`。
async fn run_bash(
    args: serde_json::Value,
    session_cwd: PathBuf,
    cancel: tokio_util::sync::CancellationToken,
    default_timeout_ms: u64,
    max_timeout_ms: u64,
) -> ToolEvent {
    let parsed: BashArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(err) => return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(err))),
    };

    let timeout = parsed
        .timeout_ms
        .unwrap_or(default_timeout_ms)
        .min(max_timeout_ms);
    if timeout == 0 {
        return ToolEvent::Failed(ToolError::InvalidArgs(BoxError::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "timeout_ms must be > 0",
        ))));
    }

    let workdir = match resolve_workdir(&session_cwd, parsed.workdir.as_deref()) {
        Ok(p) => p,
        Err(e) => return ToolEvent::Failed(e),
    };

    let started = std::time::Instant::now();

    let mut cmd = build_command(&parsed.command);
    cmd.current_dir(&workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => {
            return ToolEvent::Failed(ToolError::Execution(BoxError::new(err)));
        }
    };

    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let mut stdout_lines = BufReader::new(stdout).lines();
    let mut stderr_lines = BufReader::new(stderr).lines();

    let mut buf = OutputBuffer::new();
    let mut stdout_open = true;
    let mut stderr_open = true;
    let mut timed_out = false;
    let mut canceled = false;
    let mut wait_status: Option<std::process::ExitStatus> = None;

    let timeout_at = tokio::time::sleep(Duration::from_millis(timeout));
    tokio::pin!(timeout_at);

    loop {
        tokio::select! {
            biased;

            _ = cancel.cancelled() => {
                canceled = true;
                break;
            }

            _ = &mut timeout_at, if !timed_out => {
                timed_out = true;
                // child drop = SIGKILL via kill_on_drop. We fall through to break
                // after explicitly killing for deterministic behavior.
                let _ = child.start_kill();
                break;
            }

            line = stdout_lines.next_line(), if stdout_open => {
                match line {
                    Ok(Some(mut l)) => {
                        l.push('\n');
                        buf.push(l.as_bytes());
                    }
                    Ok(None) => stdout_open = false,
                    Err(_) => stdout_open = false,
                }
            }

            line = stderr_lines.next_line(), if stderr_open => {
                match line {
                    Ok(Some(mut l)) => {
                        l.push('\n');
                        buf.push(l.as_bytes());
                    }
                    Ok(None) => stderr_open = false,
                    Err(_) => stderr_open = false,
                }
            }

            status = child.wait(), if !stdout_open && !stderr_open => {
                wait_status = status.ok();
                break;
            }
        }
    }

    if canceled {
        return ToolEvent::Failed(ToolError::Canceled);
    }

    // 取消 / 超时 / 流提前关闭都可能在这里 child 还没 wait——补一下。
    if wait_status.is_none() {
        wait_status = child.wait().await.ok();
    }

    let duration_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

    let (exit_code, signal_name) = decode_status(wait_status.as_ref());

    let mut text = String::from_utf8_lossy(buf.as_bytes()).into_owned();
    if buf.truncated() > 0 {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!(
            "[output truncated; {} bytes dropped]",
            buf.truncated()
        ));
    }
    if timed_out {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("[timed out after {timeout}ms]"));
    } else if let Some(sig) = signal_name.as_deref() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("[killed by signal: {sig}]"));
    } else if let Some(code) = exit_code
        && code != 0
    {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&format!("[exit code: {code}]"));
    }

    let raw_output = serde_json::to_value(BashOutput {
        exit_code,
        signal: signal_name,
        timed_out,
        truncated_bytes: buf.truncated(),
        duration_ms,
    })
    .unwrap_or(serde_json::Value::Null);

    let mut fields = ToolCallUpdateFields::default();
    fields.content = Some(vec![ToolCallContent::Content(Content::new(
        ContentBlock::Text(TextContent::new(text)),
    ))]);
    fields.raw_output = Some(raw_output);
    ToolEvent::Completed(fields)
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

/// Canonicalize 工作目录并校验它在 session cwd 子树内。
fn resolve_workdir(session_cwd: &Path, requested: Option<&str>) -> Result<PathBuf, ToolError> {
    let target = match requested {
        None => session_cwd.to_path_buf(),
        Some(s) => {
            let p = Path::new(s);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                session_cwd.join(p)
            }
        }
    };

    let canon_target =
        std::fs::canonicalize(&target).map_err(|e| ToolError::InvalidArgs(BoxError::new(e)))?;
    let canon_cwd =
        std::fs::canonicalize(session_cwd).unwrap_or_else(|_| session_cwd.to_path_buf());

    if !canon_target.starts_with(&canon_cwd) {
        return Err(ToolError::InvalidArgs(BoxError::new(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "workdir {} escapes session cwd {}",
                canon_target.display(),
                canon_cwd.display()
            ),
        ))));
    }

    Ok(canon_target)
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
            self.bytes.extend_from_slice(&chunk[..remaining]);
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

fn truncate_title(s: &str) -> String {
    if s.chars().count() <= TITLE_TRUNC {
        return s.to_string();
    }
    let truncated: String = s.chars().take(TITLE_TRUNC).collect();
    format!("{truncated}…")
}

#[cfg(unix)]
fn decode_status(status: Option<&std::process::ExitStatus>) -> (Option<i32>, Option<String>) {
    use std::os::unix::process::ExitStatusExt;
    match status {
        None => (None, None),
        Some(s) => {
            if let Some(code) = s.code() {
                (Some(code), None)
            } else if let Some(sig) = s.signal() {
                (None, Some(signal_name(sig)))
            } else {
                (None, None)
            }
        }
    }
}

#[cfg(windows)]
fn decode_status(status: Option<&std::process::ExitStatus>) -> (Option<i32>, Option<String>) {
    match status {
        None => (None, None),
        Some(s) => (s.code(), None),
    }
}

#[cfg(unix)]
fn signal_name(sig: i32) -> String {
    // 保守的内置映射——常见的几个；其余给 SIG#N。
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

#[cfg(test)]
mod tests;
