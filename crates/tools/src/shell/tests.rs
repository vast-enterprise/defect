//! LocalShellBackend unit tests covering the semantic contract of key operations: create
//! / output / wait_for_exit / release / kill.

use std::time::Duration;

use defect_agent::shell::{ShellBackend, ShellError};
use tempfile::tempdir;

use super::LocalShellBackend;

#[tokio::test]
async fn create_and_wait_for_exit_returns_zero_for_true() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    let id = backend
        .create("true".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    let status = backend.wait_for_exit(&id).await.expect("wait");
    assert_eq!(status.exit_code, Some(0));
    assert!(status.signal.is_none());
}

#[tokio::test]
async fn output_collects_stdout_and_stderr() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    let id = backend
        .create(
            "echo hello; echo world >&2".to_string(),
            dir.path().to_path_buf(),
        )
        .await
        .expect("create");
    let _ = backend.wait_for_exit(&id).await.expect("wait");
    let out = backend.output(&id).await.expect("output");
    assert!(out.text.contains("hello"), "missing stdout: {:?}", out.text);
    assert!(out.text.contains("world"), "missing stderr: {:?}", out.text);
    assert!(!out.truncated);
    assert_eq!(out.exit_status.as_ref().and_then(|s| s.exit_code), Some(0));
}

#[tokio::test]
async fn output_is_idempotent() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    let id = backend
        .create("echo once".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    let _ = backend.wait_for_exit(&id).await.expect("wait");
    let first = backend.output(&id).await.expect("output1");
    let second = backend.output(&id).await.expect("output2");
    assert_eq!(first.text, second.text);
}

#[tokio::test]
async fn nonzero_exit_propagates_exit_code() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    let id = backend
        .create("exit 7".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    let status = backend.wait_for_exit(&id).await.expect("wait");
    assert_eq!(status.exit_code, Some(7));
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill_terminates_long_running_command() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    // `exec` replaces `sh` with `sleep` directly; otherwise SIGKILL would only kill `sh`,
    // leaving `sleep` as an orphan that still holds the pipe, so stdout/stderr won't EOF
    // until `sleep` exits on its own.
    let id = backend
        .create("exec sleep 30".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    // Give sleep time to start before killing it.
    tokio::time::sleep(Duration::from_millis(50)).await;
    backend.kill(&id).await.expect("kill");
    let status = tokio::time::timeout(Duration::from_secs(3), backend.wait_for_exit(&id))
        .await
        .expect("wait_for_exit timed out")
        .expect("wait");
    // SIGKILL sets exit_code to None and signal to SIGKILL
    assert!(
        status.exit_code.is_none() && status.signal.as_deref() == Some("SIGKILL"),
        "expected SIGKILL, got {:?}",
        status
    );
}

#[tokio::test]
async fn release_removes_terminal_and_subsequent_lookups_fail() {
    let dir = tempdir().unwrap();
    let backend = LocalShellBackend::new();
    let id = backend
        .create("true".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    let _ = backend.wait_for_exit(&id).await.expect("wait");
    backend.release(&id).await.expect("release");
    let err = backend.output(&id).await.expect_err("output after release");
    assert!(matches!(err, ShellError::NotFound(_)));
}

#[tokio::test]
async fn wait_for_exit_unknown_id_is_not_found() {
    let backend = LocalShellBackend::new();
    let err = backend
        .wait_for_exit(&defect_agent::shell::TerminalId::new("does-not-exist"))
        .await
        .expect_err("should fail");
    assert!(matches!(err, ShellError::NotFound(_)));
}
