//! Command hook handler — feeds the step envelope JSON to an external subprocess.
//! The IO protocol passes stdout through as verdict JSON.
//!
//! ## Shape
//!
//! - [`CommandSpec`]: handler configuration — either direct argv spawn or explicit shell.
//! - [`CommandHandler`]: implements [`StepHandler`]; spawn / kill_on_drop / timeout
//!   follow the semantics of §4.2.3.
//!
//! No shell dependency: direct argv spawn is the default; only the explicit `shell` field
//! uses a shell.
//!
//! Platform fallback: on `cfg(unix)` and `cfg(windows)`, spawns the child process via
//! `tokio::process::Command`.

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

/// Configuration for a command handler.
///
/// See module-level docs.
///
/// Conceptually equivalent to `defect_config::HookCommandSpec`, but lives in the agent
/// crate. During CLI assembly, the config shape is translated into this form — the agent
/// crate does not depend on the config crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandSpec {
    /// Spawn `argv` directly, without any shell.
    Argv {
        argv: Vec<String>,
        /// Windows override; `None` falls back to `argv`.
        argv_windows: Option<Vec<String>>,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
    /// Explicit shell. The engine no longer auto-selects `sh`; an invalid shell kind is
    /// reported as a configuration error.
    Shell {
        shell: ShellKind,
        command: String,
        cwd: Option<PathBuf>,
        env: BTreeMap<String, String>,
        timeout_sec: Option<u64>,
    },
}

/// Explicit shell kind. The engine uses this tag to select the executable and its flag.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellKind {
    /// `sh -c <command>`.
    Sh,
    /// `bash -c <command>`.
    Bash,
    /// `pwsh -NoProfile -NonInteractive -Command <command>`.
    Pwsh,
    /// `cmd /C <command>`.
    Cmd,
    /// A user-supplied program with passthrough args (excluding the command itself).
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

/// `Command` handler implementation.
///
/// IO protocol:
/// - stdin = JSON serialization of the step envelope, one line
/// - stdout = verdict JSON object (empty = no intervention), passed through to the engine
///   as-is
/// - stderr = forwarded to tracing
/// - exit 0 = determined by stdout; non-zero = `HookError::HandlerFailed`
pub struct CommandHandler {
    spec: CommandSpec,
}

impl CommandHandler {
    #[must_use]
    pub fn new(spec: CommandSpec) -> Self {
        Self { spec }
    }

    /// The timeout configured on this handler. The CLI assembly forwards it into
    /// [`StepHandlerEntry::with_timeout`](super::StepHandlerEntry::with_timeout); the
    /// engine's default fallback is described in §8.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.spec.timeout()
    }
}

impl StepHandler for CommandHandler {
    /// Feeds the step envelope as JSON to the child process's stdin; stdout is the
    /// verdict JSON (empty stdout means no intervention).
    ///
    /// Simpler than the old `handle` — the envelope is already a `Value`, so no
    /// `CommandEventEnvelope` conversion is needed. stdout is passed directly as the
    /// verdict to the engine's `apply_verdict`, and the IO protocol is reduced from
    /// "parse into `HookOutcome`" to "pass JSON through as-is".
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
                // Writing to stdin may race with the child process exiting before reading
                // it (e.g. a script like `exit 2`). In that case the pipe is closed by
                // the peer and `write` returns `BrokenPipe`. This is legitimate: the
                // script is allowed to ignore stdin; its exit code is the output. Treat
                // `BrokenPipe` as "done feeding" and silently continue, letting the exit
                // code decide the outcome. Other write errors are considered handler
                // failures.
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

            // Exit code convention (aligned with Claude exit code 2):
            // - 0 → decision based on stdout (empty or non-JSON stdout = no intervention)
            // - 2 → veto this step (exact semantics interpreted by the step's
            //   `apply_verdict`: turn-end → continue,
            //         tool/turn/session → break, compact → skip); stderr is injected as
            //         feedback
            // - other non-zero / signal → handler error (engine degrades and skips)
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

// Command construction

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

/// Environment variables for the step model: common headers plus the tool name extracted
/// from the envelope (if any). Script authors can read both env and stdin JSON.
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

// Helpers

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

    /// Empty stdout (exit 0) → no intervention (`Ok(None)`).
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

    /// Non-JSON stdout → no intervention (audit scripts may simply echo logs).
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

    /// JSON stdout is passed through as the verdict verbatim.
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

    /// Exit code 2 → veto verdict (stderr used as feedback injection).
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

    /// Script exits (exit 2) without reading stdin, and the envelope exceeds the pipe
    /// buffer → writing stdin hits `BrokenPipe`, but the verdict must be based on the
    /// exit code (veto), not treating `BrokenPipe` as a handler failure.
    /// Regression test: previously, `BrokenPipe` was directly propagated as
    /// `HandlerFailed`, causing intermittent CI failures.
    /// Use an envelope far larger than the 64 KiB pipe buffer so that `write_all`
    /// necessarily blocks before the child exits, reliably reproducing the race (small
    /// payloads can fit in the buffer and miss this path).
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
        // 1 MiB padding, far exceeding the typical 64 KiB pipe buffer.
        let env = serde_json::json!({"tool": "bash", "pad": "x".repeat(1024 * 1024)});
        let v = h
            .handle_step(&env, ctx(&session_id, cwd))
            .await
            .expect("ok")
            .expect("verdict");
        assert_eq!(v["control"], "veto");
        assert_eq!(v["additional_context"][0], "tests failed\n");
    }

    /// Other non-zero exit (not 2) → HandlerFailed.
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

    /// Cancellation → Timeout.
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
