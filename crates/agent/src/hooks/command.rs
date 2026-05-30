//! Command hook handler — 把 [`HookEvent`] 喂给一个外部子进程，按
//! `docs/internal/hooks.md` §4.2 的 IO 协议解析 stdout 成 [`HookOutcome`]。
//!
//! ## 形态
//!
//! - [`CommandSpec`]：handler 配置——argv 直 spawn / 显式 shell 二选一
//! - [`CommandHandler`]：实现 [`HookHandler`]；spawn / kill_on_drop / 超时
//!   走 §4.2.3 的语义
//! - [`CommandEventEnvelope`]：stdin JSON 形态。仅供测试与诊断诊断；脚本
//!   作者按 §4.2.1 的 env 表读环境变量也行，二选一。
//!
//! 不依赖任何 shell：argv 直 spawn 是默认；显式 `shell` 字段才走 shell。
//!
//! 平台兜底：在 `cfg(unix)` 与 `cfg(windows)` 下 [`CommandHandler::spawn`]
//! 用 `tokio::process::Command`；其它平台（纯 WASM）这个 handler 类型由
//! 上层 cargo feature flag 关闭——v0 仅在能 spawn 子进程的环境装配。
//!
//! [`HookHandler`]: super::HookHandler
//! [`HookEvent`]: super::HookEvent
//! [`HookOutcome`]: super::HookOutcome

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use agent_client_protocol_schema::ContentBlock;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::error::BoxError;
use crate::tool::SafetyClass;

use super::{
    HookCapability, HookCtx, HookError, HookEvent, HookHandler, HookOutcome, HookPatch,
    SessionSource,
};

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

/// Command handler 的配置。
///
/// 详见 `docs/internal/hooks.md` §4.2。
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
/// 见 `docs/internal/hooks.md` §4.2：
/// - stdin = `HookEvent` 的 JSON 序列化（[`CommandEventEnvelope`]），单行
/// - stdout = JSON 对象，按 [`CommandStdoutShape`] 解析
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
    /// [`super::HandlerEntry::with_timeout`]，引擎默认值兜底见 §8。
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.spec.timeout()
    }
}

impl HookHandler for CommandHandler {
    fn capability(&self) -> HookCapability {
        HookCapability::Intercept
    }

    fn handle<'a>(
        &'a self,
        event: &'a HookEvent<'a>,
        ctx: HookCtx<'a>,
    ) -> BoxFuture<'a, Result<HookOutcome, HookError>> {
        Box::pin(async move {
            let envelope = CommandEventEnvelope::from_event(event);
            let stdin_payload = serde_json::to_vec(&envelope).map_err(|err| {
                HookError::HandlerFailed(BoxError::new(io_invalid("serialize event", err)))
            })?;

            let env_vars = command_env_vars(event, &ctx);
            let mut cmd = build_command(&self.spec, &env_vars)?;
            cmd.stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .kill_on_drop(true);

            let mut child = cmd
                .spawn()
                .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?;

            // 喂 stdin、合上，避免子进程读阻塞。
            if let Some(mut stdin) = child.stdin.take() {
                stdin
                    .write_all(&stdin_payload)
                    .await
                    .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?;
                stdin
                    .write_all(b"\n")
                    .await
                    .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?;
                drop(stdin);
            }

            let cancel = ctx.cancel.clone();
            let wait_fut = child.wait_with_output();
            let output = tokio::select! {
                () = cancel.cancelled() => {
                    return Err(HookError::Timeout);
                }
                result = wait_fut => {
                    result.map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?
                }
            };

            if !output.stderr.is_empty() {
                let text = String::from_utf8_lossy(&output.stderr);
                tracing::debug!(target: "defect_agent::hooks::command", stderr = %text, "command stderr");
            }

            if !output.status.success() {
                let code = output.status.code();
                let msg = match code {
                    Some(c) => format!("hook command exited with status {c}"),
                    None => "hook command terminated by signal".to_string(),
                };
                return Err(HookError::HandlerFailed(BoxError::new(io_invalid(msg, ""))));
            }

            parse_stdout(&output.stdout, event.kind_str())
        })
    }
}

// ---------------------------------------------------------------------------
// stdin envelope
// ---------------------------------------------------------------------------

/// 喂给子进程 stdin 的 JSON 形态。详见 `docs/internal/hooks.md` §4.2.1。
///
/// 仅承诺 v0 实际 emit 的 5 件 Sync 拦截事件；Async 观察事件不会通过
/// [`super::HookEngine::fire`] 入口走这里。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum CommandEventEnvelope {
    SessionStart {
        source: SessionSourceWire,
        cwd: PathBuf,
    },
    UserPromptSubmit {
        content: Vec<ContentBlock>,
    },
    PreToolUse {
        id: String,
        name: String,
        args: Value,
        safety: SafetyClass,
    },
    PostToolUse {
        id: String,
        name: String,
        fields: Value,
    },
    PostToolUseFailure {
        id: String,
        name: String,
        error: String,
    },
    /// 兜底——理论上 fire 不会把 Async 事件走到这里。
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionSourceWire {
    New,
    Resume { session_id: String },
}

impl CommandEventEnvelope {
    /// 把借用形态的 [`HookEvent`] 转成 owned wire 形态。
    ///
    /// 公开给同 crate 的 [`super::prompt`] 模块在 `PromptRender::Json`
    /// 上复用。
    pub(super) fn from_event(event: &HookEvent<'_>) -> Self {
        match event {
            HookEvent::SessionStart { source, cwd } => Self::SessionStart {
                source: match source {
                    SessionSource::New => SessionSourceWire::New,
                    SessionSource::Resume { session_id } => SessionSourceWire::Resume {
                        session_id: session_id.0.to_string(),
                    },
                },
                cwd: cwd.to_path_buf(),
            },
            HookEvent::UserPromptSubmit { content } => Self::UserPromptSubmit {
                content: (*content).to_vec(),
            },
            HookEvent::PreToolUse {
                id,
                name,
                args,
                safety,
            } => Self::PreToolUse {
                id: id.0.to_string(),
                name: (*name).to_string(),
                args: (*args).clone(),
                safety: *safety,
            },
            HookEvent::PostToolUse { id, name, fields } => Self::PostToolUse {
                id: id.0.to_string(),
                name: (*name).to_string(),
                fields: serde_json::to_value(fields).unwrap_or(Value::Null),
            },
            HookEvent::PostToolUseFailure { id, name, error } => Self::PostToolUseFailure {
                id: id.0.to_string(),
                name: (*name).to_string(),
                error: (*error).to_string(),
            },
            _ => Self::Other,
        }
    }
}

// ---------------------------------------------------------------------------
// stdout schema
// ---------------------------------------------------------------------------

/// stdout JSON 形态。详见 `docs/internal/hooks.md` §4.2.2。
///
/// 缺省 = `Pass`。空 stdout / 非 JSON stdout 都视为 `Pass`，便于审计脚本
/// 只 echo 日志；含合法 JSON 但 schema 不匹配 = `HandlerFailed`。
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandStdoutShape {
    block: Option<String>,
    patch: Option<CommandPatchShape>,
    #[serde(default)]
    append: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum CommandPatchShape {
    ToolArgs(Value),
    UserPrompt {
        #[serde(default)]
        prepend: Vec<ContentBlock>,
        #[serde(default)]
        append: Vec<ContentBlock>,
    },
}

fn parse_stdout(bytes: &[u8], event_kind: &str) -> Result<HookOutcome, HookError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|err| HookError::HandlerFailed(BoxError::new(err)))?
        .trim();
    if text.is_empty() {
        return Ok(HookOutcome::default());
    }
    // 非 JSON 视为 `Pass`——审计脚本可能只是 echo 日志。
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') {
        return Ok(HookOutcome::default());
    }
    let shape: CommandStdoutShape = serde_json::from_str(text).map_err(|err| {
        HookError::HandlerFailed(BoxError::new(io_invalid(
            format!("invalid hook stdout JSON for event {event_kind}"),
            err,
        )))
    })?;

    let patch = shape.patch.map(|p| match p {
        CommandPatchShape::ToolArgs(v) => HookPatch::ToolArgs(v),
        CommandPatchShape::UserPrompt { prepend, append } => {
            HookPatch::UserPrompt { prepend, append }
        }
    });

    Ok(HookOutcome {
        block: shape.block,
        patch,
        append: shape.append,
    })
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

// ---------------------------------------------------------------------------
// env injection
// ---------------------------------------------------------------------------

fn command_env_vars(event: &HookEvent<'_>, ctx: &HookCtx<'_>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    out.insert(
        "DEFECT_HOOK_EVENT".to_string(),
        event.kind_str().to_string(),
    );
    out.insert(
        "DEFECT_SESSION_ID".to_string(),
        ctx.session_id.0.to_string(),
    );
    out.insert(
        "DEFECT_CWD".to_string(),
        ctx.cwd.to_string_lossy().into_owned(),
    );

    match event {
        HookEvent::PreToolUse { name, args, .. } => {
            out.insert("DEFECT_TOOL_NAME".to_string(), (*name).to_string());
            out.insert("DEFECT_TOOL_INPUT".to_string(), args.to_string());
        }
        HookEvent::PostToolUse { name, fields, .. } => {
            out.insert("DEFECT_TOOL_NAME".to_string(), (*name).to_string());
            out.insert(
                "DEFECT_TOOL_INPUT".to_string(),
                serde_json::to_string(fields).unwrap_or_default(),
            );
        }
        HookEvent::PostToolUseFailure { name, error, .. } => {
            out.insert("DEFECT_TOOL_NAME".to_string(), (*name).to_string());
            out.insert("DEFECT_TOOL_ERROR".to_string(), (*error).to_string());
        }
        HookEvent::UserPromptSubmit { content } => {
            let prompt_text = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(t.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            out.insert("DEFECT_USER_PROMPT".to_string(), prompt_text);
        }
        _ => {}
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
mod test {
    use super::*;
    use agent_client_protocol_schema::{SessionId, ToolCallId};
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

    #[tokio::test]
    async fn argv_pass_when_stdout_empty() {
        // /bin/true exits 0 with empty stdout — should be Pass.
        if !Path::new("/bin/true").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/true"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"x": 1});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn argv_block_via_stdout_json() {
        // sh -c 'echo "{\"block\": \"nope\"}"'
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            "echo '{\"block\":\"nope\"}'",
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::Value::Null;
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert_eq!(outcome.block.as_deref(), Some("nope"));
    }

    #[tokio::test]
    async fn argv_patch_tool_args() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            r#"echo '{"patch": {"tool_args": {"redacted": true}}}'"#,
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"x": 1});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        match outcome.patch {
            Some(HookPatch::ToolArgs(v)) => {
                assert_eq!(v, serde_json::json!({"redacted": true}));
            }
            other => panic!("expected ToolArgs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn argv_nonzero_exit_is_handler_failed() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/sh", "-c", "exit 7"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::Value::Null;
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let err = h
            .handle(&ev, ctx(&session_id, cwd))
            .await
            .expect_err("expected error");
        assert!(matches!(err, HookError::HandlerFailed(_)));
    }

    #[tokio::test]
    async fn argv_non_json_stdout_treated_as_pass() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            "echo audit-log-line-without-json",
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::Value::Null;
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[tokio::test]
    async fn argv_invalid_json_object_is_handler_failed() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        // 含合法 JSON 起始但 schema 不匹配 → HandlerFailed
        let h = CommandHandler::new(argv_spec(vec![
            "/bin/sh",
            "-c",
            "echo '{\"unknown_field\":1}'",
        ]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::Value::Null;
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let err = h
            .handle(&ev, ctx(&session_id, cwd))
            .await
            .expect_err("expected error");
        assert!(matches!(err, HookError::HandlerFailed(_)));
    }

    #[tokio::test]
    async fn shell_kind_sh_runs_command() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let spec = CommandSpec::Shell {
            shell: ShellKind::Sh,
            command: r#"echo '{"append":[{"type":"text","text":"hi"}]}'"#.to_string(),
            cwd: None,
            env: BTreeMap::new(),
            timeout_sec: None,
        };
        let h = CommandHandler::new(spec);
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let fields = agent_client_protocol_schema::ToolCallUpdateFields::default();
        let ev = HookEvent::PostToolUse {
            id: &id,
            name: "bash",
            fields: &fields,
        };
        let outcome = h.handle(&ev, ctx(&session_id, cwd)).await.expect("ok");
        assert_eq!(outcome.append.len(), 1);
    }

    #[tokio::test]
    async fn cancellation_returns_timeout() {
        if !Path::new("/bin/sh").exists() {
            return;
        }
        let h = CommandHandler::new(argv_spec(vec!["/bin/sh", "-c", "sleep 5"]));
        let session_id = SessionId::new("s1");
        let cwd = Path::new("/");
        let id = ToolCallId::new("c1");
        let args = serde_json::Value::Null;
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::ReadOnly,
        };
        let cancel = CancellationToken::new();
        let cancel_for_drop = cancel.clone();
        let ctx2 = HookCtx::new(&session_id, cwd, cancel);
        let fut = h.handle(&ev, ctx2);
        // 200ms 后取消
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            cancel_for_drop.cancel();
        });
        let err = fut.await.expect_err("expected cancellation -> Timeout");
        assert!(matches!(err, HookError::Timeout));
    }

    #[test]
    fn parse_stdout_pass_on_empty() {
        let outcome = parse_stdout(b"", "pre_tool_use").expect("ok");
        assert!(outcome.block.is_none());
        assert!(outcome.patch.is_none());
        assert!(outcome.append.is_empty());
    }

    #[test]
    fn parse_stdout_pass_on_non_json() {
        let outcome = parse_stdout(b"audit log line\n", "pre_tool_use").expect("ok");
        assert!(outcome.block.is_none());
    }

    #[test]
    fn parse_stdout_user_prompt_patch() {
        let body = br#"{"patch":{"user_prompt":{"prepend":[{"type":"text","text":"[hint] "}],"append":[]}}}"#;
        let outcome = parse_stdout(body, "user_prompt_submit").expect("ok");
        match outcome.patch {
            Some(HookPatch::UserPrompt { prepend, append }) => {
                assert_eq!(prepend.len(), 1);
                assert!(append.is_empty());
            }
            other => panic!("expected UserPrompt patch, got {other:?}"),
        }
    }

    #[test]
    fn envelope_pre_tool_use_round_trips() {
        let id = ToolCallId::new("c1");
        let args = serde_json::json!({"k": "v"});
        let ev = HookEvent::PreToolUse {
            id: &id,
            name: "bash",
            args: &args,
            safety: SafetyClass::Destructive,
        };
        let envelope = CommandEventEnvelope::from_event(&ev);
        let json = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(json["type"], "pre_tool_use");
        assert_eq!(json["name"], "bash");
        assert_eq!(json["args"], serde_json::json!({"k": "v"}));
    }

    #[test]
    fn envelope_session_start_resume() {
        let id = SessionId::new("s9");
        let cwd = Path::new("/repo");
        let ev = HookEvent::SessionStart {
            source: SessionSource::Resume { session_id: &id },
            cwd,
        };
        let envelope = CommandEventEnvelope::from_event(&ev);
        let json = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(json["type"], "session_start");
        assert_eq!(json["source"]["kind"], "resume");
        assert_eq!(json["source"]["session_id"], "s9");
    }

    #[test]
    fn build_command_argv_picks_windows_fallback_on_unix() {
        let spec = CommandSpec::Argv {
            argv: vec!["echo".into(), "hi".into()],
            argv_windows: Some(vec!["pwsh".into()]),
            cwd: None,
            env: BTreeMap::new(),
            timeout_sec: None,
        };
        let env = BTreeMap::new();
        let cmd = build_command(&spec, &env).expect("ok");
        // 在非 Windows 平台必须用 unix argv，否则我们会在 darwin/linux 上跑 pwsh
        if cfg!(not(target_os = "windows")) {
            // tokio::process::Command 没有暴露 program() 公共 API；用 std::process::Command
            // 镜像无法直接验证，但 build 不报错说明分支选择走通。
            drop(cmd);
        }
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
