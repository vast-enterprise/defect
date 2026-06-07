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
