//! `bash` 内置工具：跑一条非交互 shell 命令、合并 stdout/stderr、单帧返回。
//!
//! Bash tool — runs a shell command, streams stdout/stderr, supports timeout and cancellation.

use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol_schema::{
    Content, ContentBlock, TextContent, ToolCallContent, ToolCallLocation, ToolCallUpdateFields,
    ToolKind,
};
use defect_agent::error::BoxError;
use defect_agent::shell::{ShellBackend, ShellError, TerminalExitStatus, TerminalId};
use defect_agent::tool::{
    SafetyClass, Tool, ToolCallDescription, ToolContext, ToolError, ToolEvent, ToolSchema,
    ToolStream,
};
use defect_config::BashToolConfig;
use futures::future::BoxFuture;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_TIMEOUT_MS: u64 = 30_000;
const MAX_TIMEOUT_MS: u64 = 600_000;
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
        // Always Destructive — v0 does not parse command text.
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
        let shell = ctx.shell.clone();
        let default_timeout_ms = self.default_timeout_ms;
        let max_timeout_ms = self.max_timeout_ms;
        let fut = async move {
            run_bash(args, cwd, cancel, shell, default_timeout_ms, max_timeout_ms).await
        };
        let s: Pin<Box<dyn futures::Stream<Item = ToolEvent> + Send>> = Box::pin(stream::once(fut));
        s
    }
}

/// 一次完整的 bash 调用：解析 args、resolve workdir、走 [`ShellBackend`]、
/// 装配最终输出。返回单一 [`ToolEvent`]——`Completed` 或 `Failed`。
async fn run_bash(
    args: serde_json::Value,
    session_cwd: PathBuf,
    cancel: tokio_util::sync::CancellationToken,
    shell: Arc<dyn ShellBackend>,
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

    let terminal_id = match shell.create(parsed.command.clone(), workdir).await {
        Ok(id) => id,
        Err(err) => return ToolEvent::Failed(ToolError::Execution(BoxError::new(err))),
    };

    let result = run_command(shell.clone(), &terminal_id, &cancel, timeout, started).await;
    // release 在所有出口幂等触发——backend 保证重复 release 同一个 id 不报错。
    let _ = shell.release(&terminal_id).await;
    result
}

async fn run_command(
    shell: Arc<dyn ShellBackend>,
    terminal_id: &TerminalId,
    cancel: &tokio_util::sync::CancellationToken,
    timeout: u64,
    started: std::time::Instant,
) -> ToolEvent {
    let mut timed_out = false;
    let mut canceled = false;

    let timeout_at = tokio::time::sleep(Duration::from_millis(timeout));
    tokio::pin!(timeout_at);

    // wait_fut 必须能"在 cancel 出口活下去"。ACP 反向请求一旦发出，response
    // 必须能交付给在世的 oneshot::Receiver；若我们 drop wait_fut，server 把
    // "无人接收"映射成 internal_error 并撕掉整条连接（详见
    // `agent_client_protocol::jsonrpc::incoming_actor::dispatch_dispatch` 里
    // `router.respond_with_result(result)?`）。
    //
    // 做法：把 wait_fut 装成 `'static` 的 self-owning future（闭包持 Arc<shell>
    // 与 id），cancel 分支用 [`tokio::spawn`] 把它 detach 走继续 drain 响应；
    // timeout 分支保留"先 kill 再 drain"语义，此时同一个 fut 直接 await 即可。
    let mut wait_fut: Pin<
        Box<dyn futures::Future<Output = Result<TerminalExitStatus, ShellError>> + Send>,
    > = {
        let shell = shell.clone();
        let id = terminal_id.clone();
        Box::pin(async move { shell.wait_for_exit(&id).await })
    };

    let exit_status = tokio::select! {
        biased;

        _ = cancel.cancelled() => {
            canceled = true;
            None
        }

        _ = &mut timeout_at => {
            timed_out = true;
            None
        }

        result = &mut wait_fut => {
            match result {
                Ok(status) => Some(status),
                Err(err) => {
                    return ToolEvent::Failed(ToolError::Execution(BoxError::new(err)));
                }
            }
        }
    };

    if canceled {
        // 先发 kill 让进程尽快收尾；wait_fut 不能 drop（reverse-request 路径
        // 的 oneshot 必须有人接），detach 出去 await，让运行时在响应到达时
        // 仍有活的 receiver。LocalShellBackend 的 future 是 in-process 通知，
        // detach 也无副作用。
        let _ = shell.kill(terminal_id).await;
        tokio::spawn(async move {
            let _ = wait_fut.await;
        });
        return ToolEvent::Failed(ToolError::Canceled);
    }

    // 超时路径：先 kill，再 wait_for_exit + output 拿最终输出。
    let exit_status = match exit_status {
        Some(status) => Some(status),
        None => {
            let _ = shell.kill(terminal_id).await;
            wait_fut.await.ok()
        }
    };

    let output = match shell.output(terminal_id).await {
        Ok(o) => o,
        Err(err) => {
            return ToolEvent::Failed(ToolError::Execution(BoxError::new(err)));
        }
    };

    let duration_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

    let (exit_code, signal_name) = match exit_status.as_ref() {
        Some(s) => (s.exit_code, s.signal.clone()),
        None => (None, None),
    };

    let mut text = output.text;
    let truncated_bytes: u64 = if output.truncated { 1 } else { 0 };
    if output.truncated {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str("[output truncated]");
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
        truncated_bytes,
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

fn truncate_title(s: &str) -> String {
    if s.chars().count() <= TITLE_TRUNC {
        return s.to_string();
    }
    let truncated: String = s.chars().take(TITLE_TRUNC).collect();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests;
