//! LocalShellBackend unit tests covering key
//! 行为：create / output / wait_for_exit / release / kill 的语义合同。

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
    // exec 让 sh 直接被 sleep 替换；否则 SIGKILL 只杀 sh，sleep 成为孤儿
    // 仍持有 pipe，stdout/stderr 不 EOF 直到 sleep 自己结束。
    let id = backend
        .create("exec sleep 30".to_string(), dir.path().to_path_buf())
        .await
        .expect("create");
    // 给 sleep 起跑的时间再 kill
    tokio::time::sleep(Duration::from_millis(50)).await;
    backend.kill(&id).await.expect("kill");
    let status = tokio::time::timeout(Duration::from_secs(3), backend.wait_for_exit(&id))
        .await
        .expect("wait_for_exit timed out")
        .expect("wait");
    // SIGKILL 让 exit_code = None, signal = SIGKILL
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
