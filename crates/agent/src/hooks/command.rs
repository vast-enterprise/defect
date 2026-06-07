//! Command hook handler — 把 step 信封 JSON 喂给一个外部子进程，按
//! The IO protocol passes stdout through as verdict JSON.
//!
//! ## 形态
//!
//! - [`CommandSpec`]：handler 配置——argv 直 spawn / 显式 shell 二选一
//! - [`CommandHandler`]：实现 [`StepHandler`]；spawn /
//!   kill_on_drop / 超时走 §4.2.3 的语义
//!
//! 不依赖任何 shell：argv 直 spawn 是默认；显式 `shell` 字段才走 shell。
//!
//! 平台兜底：在 `cfg(unix)` 与 `cfg(windows)` 下用 `tokio::process::Command`
//! spawn 子进程。

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::error::BoxError;

use super::{HookCtx, HookError, StepHandler};

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// Command handler 的配置。
///
/// See module-level docs.
///
/// 设计上等价于 `defect_config::HookCommandSpec`，但放在 agent crate
/// 这一层，CLI 装配期把 config 形态翻译过来——agent crate 不依赖 config。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSpec {
    /// 直接 spawn argv，不经任何 shell。
    Argv {
        argv: Vec<String>,
        /// Windows 平台覆盖；`None` 时 fall back 到 `argv`。
        argv_windows: Option<Vec<String>>,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
    /// 显式 shell。引擎不再"自动选 sh"；shell 形态写错走配置层报错。
    Shell {
        shell: ShellKind,
        command: String,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
}

/// 显式 shell 形态。引擎按这条标记选可执行 + flag。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellKind {
    /// `sh -c <command>`。
    Sh,
    /// `bash -c <command>`。
    Bash,
    /// `pwsh -NoProfile -NonInteractive -Command <command>`。
    Pwsh,
    /// `cmd /C <command>`。
    Cmd,
    /// 用户提供的 program + 透传 args（不含 command 本身）。
    Custom { program: String, args: Vec<String> },
}

impl CommandSpec {
    fn timeout(&self) -> Option<Duration> {
        let secs = match self {
            Self::Argv { timeout_sec, .. } | Self::Shell { timeout_sec, .. } => *timeout_sec,
        };
        secs.map(Duration::from_secs)
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

/// `Command` handler 实现。
///
/// IO protocol:
/// - stdin = step 信封的 JSON 序列化，单行
/// - stdout = verdict JSON 对象（空 = 不干预），原样透传给引擎
/// - stderr 透传 tracing
/// - exit 0 = 按 stdout 决定；非 0 = `HookError::HandlerFailed`
pub struct CommandHandler {
    spec: CommandSpec,
}

impl CommandHandler {
    #[must_use]
    pub fn new(spec: CommandSpec) -> Self {
        Self { spec }
    }

    /// 该 handler 配置上自带的超时。CLI 装配把它翻进
    /// [`StepHandlerEntry::with_timeout`](super::StepHandlerEntry::with_timeout)，引擎默认值兜底见 §8。
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.spec.timeout()
    }
}

impl StepHandler for CommandHandler {
    /// Step 模型：把 step 信封作为 JSON 喂子进程 stdin，stdout 即 verdict JSON（空 stdout = 不干预）。
    ///
    /// 比旧 `handle` 简单——信封已经是 `Value`，不再需要 `CommandEventEnvelope` 转换；stdout 直接
    /// 当 verdict 透传给引擎的 `apply_verdict`，IO 协议从"解析成 HookOutcome"简化成"原样回传 JSON"。
    fn handle_step<'a>(
        &'a self,
        envelope: &'a Value,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<Option<Value>, HookError>> {
        Box::pin(async move {
            let stdin_payload = serde_json::to_vec(envelope).map_err(|err| {
                HookError::HandlerFailed(BoxError::new(io_invalid("serialize step envelope", err)))
            })?;

            let env_vars = step_env_vars(envelope, &ctx);
            let mut cmd = build_command(&self.spec, &env_vars)?;
            cmd.stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let mut child = cmd
                .spawn()
                .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?;

            if let Some(mut stdin) = child.stdin.take() {
                // 写 stdin 可能撞上子进程不读 stdin 就先退出（如 `exit 2` 类脚本）——
                // 此时管道被对端关闭，write 报 `BrokenPipe`。这是合法情形：脚本有权
                // 忽略 stdin，退出码才是它的输出。把 BrokenPipe 当成"喂完了"静默收尾，
                // 让后续按退出码裁决；其它写错误才视为 handler 失败。
                let write_res = async {
                    stdin.write_all(&stdin_payload).await?;
                    stdin.write_all(b"\n").await
                }
                .await;
                match write_res {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => {}
                    Err(err) => return Err(HookError::HandlerFailed(BoxError::new(err))),
                }
                drop(stdin);
            }

            let cancel = ctx.cancel.clone();
            let output = tokio::select! {
                () = cancel.cancelled() => return Err(HookError::Timeout),
                result = child.wait_with_output() => {
                    result.map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?
                }
            };

            let stderr_text = String::from_utf8_lossy(&output.stderr).into_owned();
            if !stderr_text.is_empty() {
                tracing::debug!(target: "defect_agent::hooks::command", stderr = %stderr_text, "command stderr");
            }

            // 退出码约定（对齐 Claude exit code 2）：
            // - 0   → 按 stdout 决定（stdout 空 / 非 JSON = 不干预）
            // - 2   → veto 本步（具体语义由 step 的 apply_verdict 解读：turn-end→continue、
            //         tool/turn/session→break、compact→skip）；stderr 作为反馈注入
            // - 其它非零 / 信号 → handler 错误（引擎降级跳过）
            match output.status.code() {
                Some(0) => {
                    let trimmed = output.stdout.trim_ascii();
                    if trimmed.is_empty() {
                        return Ok(None);
                    }
                    match serde_json::from_slice::<Value>(trimmed) {
                        Ok(v) => Ok(Some(v)),
                        Err(_) => Ok(None),
                    }
                }
                Some(2) => {
                    let mut obj = serde_json::Map::new();
                    obj.insert("control".to_string(), Value::String("veto".to_string()));
                    if !stderr_text.is_empty() {
                        obj.insert(
                            "additional_context".to_string(),
                            Value::Array(vec![Value::String(stderr_text)]),
                        );
                    }
                    Ok(Some(Value::Object(obj)))
                }
                Some(c) => Err(HookError::HandlerFailed(BoxError::new(io_invalid(
                    format!("hook command exited with status {c}"),
                    "",
                )))),
                None => Err(HookError::HandlerFailed(BoxError::new(io_invalid(
                    "hook command terminated by signal",
                    "",
                )))),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// command construction
// ---------------------------------------------------------------------------

fn build_command(
    spec: &CommandSpec,
    env_vars: &BTreeMap<String, String>,
) -> Result<Command, HookError> {
    match spec {
        CommandSpec::Argv {
            argv,
            argv_windows,
            cwd,
            env,
            ..
        } => {
            let chosen = if cfg!(target_os = "windows") {
                argv_windows.as_ref().unwrap_or(argv)
            } else {
                argv
            };
            let (program, args) = chosen.split_first().ok_or_else(|| {
                HookError::Configuration("command handler `argv` must not be empty".into())
            })?;
            let mut cmd = Command::new(program);
            cmd.args(args);
            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }
            for (k, v) in env_vars {
                cmd.env(k, v);
            }
            for (k, v) in env {
                cmd.env(k, v);
            }
            Ok(cmd)
        }
        CommandSpec::Shell {
            shell,
            command,
            cwd,
            env,
            ..
        } => {
            let mut cmd = build_shell_command(shell, command);
            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }
            for (k, v) in env_vars {
                cmd.env(k, v);
            }
            for (k, v) in env {
                cmd.env(k, v);
            }
            Ok(cmd)
        }
    }
}

fn build_shell_command(shell: &ShellKind, command: &str) -> Command {
    match shell {
        ShellKind::Sh => {
            let mut c = Command::new("sh");
            c.arg("-c").arg(command);
            c
        }
        ShellKind::Bash => {
            let mut c = Command::new("bash");
            c.arg("-c").arg(command);
            c
        }
        ShellKind::Pwsh => {
            let mut c = Command::new("pwsh");
            c.arg("-NoProfile")
                .arg("-NonInteractive")
                .arg("-Command")
                .arg(command);
            c
        }
        ShellKind::Cmd => {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(command);
            c
        }
        ShellKind::Custom { program, args } => {
            let mut c = Command::new(program);
            c.args(args).arg(command);
            c
        }
    }
}

/// Step 模型的环境变量：通用头 + 从信封提取的工具名（若有）。脚本作者可读 env 也可读 stdin JSON。
fn step_env_vars(envelope: &Value, ctx: &HookCtx<'_>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert(
        "DEFECT_SESSION_ID".to_string(),
        ctx.session_id.0.to_string(),
    );
    out.insert(
        "DEFECT_CWD".to_string(),
        ctx.cwd.to_string_lossy().into_owned(),
    );
    if let Some(tool) = envelope.get("tool").and_then(Value::as_str) {
        out.insert("DEFECT_TOOL_NAME".to_string(), tool.to_string());
    }
    out
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn io_invalid(msg: impl Into<String>, detail: impl std::fmt::Display) -> std::io::Error {
    let s = msg.into();
    let body = if s.is_empty() {
        detail.to_string()
    } else if format!("{detail}").is_empty() {
        s
    } else {
        format!("{s}: {detail}")
    };
    std::io::Error::new(std::io::ErrorKind::InvalidData, body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_client_protocol_schema::SessionId;
    use std::path::Path;
    use tokio_util::sync::CancellationToken;

    fn ctx<'a>(session_id: &'a SessionId, cwd: &'a Path) -> HookCtx<'a> {
        HookCtx::new(session_id, cwd, CancellationToken::new())
    }

    fn argv_spec(argv: Vec<&str>) -> CommandSpec {
        CommandSpec::Argv {
            argv: argv.into_iter().map(str::to_string).collect(),
            argv_windows: None,
            cwd: None,
            env: BTreeMap::new(),
            timeout_sec: None,
        }
    }

    /// 空 stdout（exit 0）→ 不干预（`Ok(None)`）。
    #[tokio::test]
    async fn step_empty_stdout_is_no_verdict() {
        if !Path::new("/bin/true").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/true"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let env = serde_json::json!({"tool": "bash", "args": {"x": 1}});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok");
        assert!(v.is_none());
    }

    /// 非 JSON stdout → 不干预（审计脚本可只 echo 日志）。
    #[tokio::test]
    async fn step_non_json_stdout_is_no_verdict() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/sh", "-c", "echo audit-line"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let env = serde_json::json!({"tool": "bash"});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok");
        assert!(v.is_none());
    }

    /// JSON stdout → 原样作为 verdict 透传。
    #[tokio::test]
    async fn step_json_stdout_becomes_verdict() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            r#"echo '{"control":"break"}'"#,
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let env = serde_json::json!({"tool": "bash"});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(v["control"], "break");
    }

    /// 退出码 2 → veto verdict（stderr 作为反馈注入）。
    #[tokio::test]
    async fn step_exit_2_yields_veto() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            "echo 'tests failed' >&2; exit 2",
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let env = serde_json::json!({"tool": "bash"});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(v["control"], "veto");
        assert_eq!(v["additional_context"][0], "tests failed\n");
    }

    /// 脚本不读 stdin 就退出（exit 2）且 envelope 大于管道缓冲 → 写 stdin 撞
    /// `BrokenPipe`，但必须按退出码裁决（veto），不能把 BrokenPipe 当 handler 失败。
    /// 回归测试：曾因把 BrokenPipe 直接上抛 HandlerFailed 而在 CI 偶发挂。
    /// 用一个远超 64KiB 管道缓冲的 envelope，让 write_all 必然在子进程退出前阻塞，
    /// 稳定复现竞态（小 payload 会侥幸塞进缓冲而漏掉这条路径）。
    #[tokio::test]
    async fn step_exit_2_vetoes_even_when_script_ignores_large_stdin() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            "echo 'tests failed' >&2; exit 2",
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        // 1 MiB padding，远超典型 64KiB 管道缓冲。
        let env = serde_json::json!({"tool": "bash", "pad": "x".repeat(1024 * 1024)});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(v["control"], "veto");
        assert_eq!(v["additional_context"][0], "tests failed\n");
    }

    /// 其它非零退出（非 2）→ HandlerFailed。
    #[tokio::test]
    async fn step_nonzero_exit_is_handler_failed() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/sh", "-c", "exit 7"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let env = serde_json::json!({"tool": "bash"});
        let err = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect_err("expected error");
        assert!(matches!(err, HookError::HandlerFailed(_)));
    }

    /// 取消 → Timeout。
    #[tokio::test]
    async fn step_cancellation_returns_timeout() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/sh", "-c", "sleep 5"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let cancel = CancellationToken::new();
        let cancel_for_drop = cancel.clone();
        let hctx = HookCtx::new(&session_id, cwd, cancel);
        let env = serde_json::json!({"tool": "bash"});
        let fut = h.handle_step(&env, hctx);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            cancel_for_drop.cancel();
        });
        let err = fut.await.expect_err("expected cancellation -> Timeout");
        assert!(matches!(err, HookError::Timeout));
    }

    #[test]
    fn shell_kind_dispatch_compiles() {
        let kinds = [
            ShellKind::Sh,
            ShellKind::Bash,
            ShellKind::Pwsh,
            ShellKind::Cmd,
            ShellKind::Custom {
                program: "fish".into(),
                args: vec!["-c".into()],
            },
        ];
        for k in &kinds {
            let _ = build_shell_command(k, "echo hi");
        }
    }
}
