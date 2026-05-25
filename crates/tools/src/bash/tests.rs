//! BashTool 单元测试。覆盖 docs/internal/tools-bash.md §7 的 #1–#9。
//! #10（真 LLM e2e）不在这里跑。

use std::time::Duration;

use agent_client_protocol::schema::{ContentBlock, ToolCallContent};
use defect_agent::tool::{SafetyClass, Tool, ToolContext, ToolError, ToolEvent};
use futures::StreamExt;
use serde_json::json;
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

use super::BashTool;

/// 把 ToolStream 跑到尽头收集成 Vec。
async fn drive(stream: defect_agent::tool::ToolStream) -> Vec<ToolEvent> {
    stream.collect().await
}

fn ctx_with(cwd: &std::path::Path, cancel: CancellationToken) -> ToolContext<'_> {
    ToolContext::new(cwd, cancel)
}

fn extract_text(event: &ToolEvent) -> String {
    let fields = match event {
        ToolEvent::Completed(f) => f,
        _ => panic!("expected Completed, got {event:?}"),
    };
    let content = fields.content.as_ref().expect("content");
    let mut out = String::new();
    for c in content {
        if let ToolCallContent::Content(inner) = c
            && let ContentBlock::Text(t) = &inner.content
        {
            out.push_str(&t.text);
        }
    }
    out
}

fn extract_raw(event: &ToolEvent) -> &serde_json::Value {
    let fields = match event {
        ToolEvent::Completed(f) => f,
        _ => panic!("expected Completed, got {event:?}"),
    };
    fields.raw_output.as_ref().expect("raw_output")
}

#[test]
fn schema_smoke() {
    let tool = BashTool::new();
    assert_eq!(tool.schema().name, "bash");
    assert!(tool.schema().description.contains("shell command"));
    let safety = tool.safety_hint(&json!({"command": "ls"}));
    assert!(matches!(safety, SafetyClass::Destructive));
}

#[test]
fn describe_renders_command_in_title() {
    let tool = BashTool::new();
    let desc = tool.describe(&json!({"command": "echo hello"}));
    let title = desc.fields.title.as_deref().unwrap_or("");
    assert!(title.starts_with("$ "));
    assert!(title.contains("echo hello"));
}

#[test]
fn describe_truncates_long_command() {
    let tool = BashTool::new();
    let long = "x".repeat(200);
    let desc = tool.describe(&json!({"command": long}));
    let title = desc.fields.title.as_deref().unwrap_or("");
    assert!(title.chars().count() <= 100, "title was {title:?}");
    assert!(title.ends_with('…'));
}

#[tokio::test]
async fn case1_echo_hello() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(json!({"command": "echo hello"}), ctx)).await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    assert!(text.contains("hello"), "text was {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["exit_code"], json!(0));
    assert_eq!(raw["timed_out"], json!(false));
}

#[tokio::test]
async fn case2_nonzero_exit_is_completed_with_exit_code_marker() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(
        json!({"command": "echo err >&2; exit 3"}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let text = extract_text(&events[0]);
    assert!(text.contains("err"), "stderr missing: {text:?}");
    assert!(text.contains("[exit code: 3]"), "marker missing: {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["exit_code"], json!(3));
}

#[tokio::test]
async fn case3_timeout_marks_timed_out() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(
        json!({"command": "sleep 5", "timeout_ms": 100}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let text = extract_text(&events[0]);
    assert!(text.contains("timed out"), "marker missing: {text:?}");
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["timed_out"], json!(true));
}

#[tokio::test]
async fn case4_cancel_yields_failed_canceled_quickly() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        cancel_clone.cancel();
    });
    let started = std::time::Instant::now();
    let ctx = ctx_with(dir.path(), cancel);
    let events = drive(tool.execute(json!({"command": "sleep 5"}), ctx)).await;
    let elapsed = started.elapsed();
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::Canceled)),
        "expected Failed(Canceled), got {:?}",
        events[0]
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "cancel took too long: {elapsed:?}"
    );
}

#[tokio::test]
async fn case5_huge_output_is_truncated() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    // 写 ~2 MiB 数据，cap 1 MiB
    let cmd = "yes a | head -c 2097152";
    let events = drive(tool.execute(
        json!({"command": cmd, "timeout_ms": 30000}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    assert!(
        text.contains("[output truncated"),
        "truncation marker missing"
    );
    let raw = extract_raw(&events[0]);
    let truncated = raw["truncated_bytes"].as_u64().unwrap_or(0);
    assert!(truncated > 0, "truncated_bytes should be > 0");
}

#[tokio::test]
async fn case6_workdir_escape_is_invalid_args() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    // /tmp 是 dir 的祖先节点的姐妹（dir 在 /tmp/xxxxx 下），".." 跳出 = escape
    let events = drive(tool.execute(
        json!({"command": "pwd", "workdir": "../../../etc"}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(
        matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))),
        "expected InvalidArgs, got {:?}",
        events[0]
    );
}

#[tokio::test]
async fn case7_workdir_subdir_resolves() {
    let dir = tempdir().unwrap();
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(
        json!({"command": "pwd", "workdir": "sub"}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    let text = extract_text(&events[0]);
    // canonicalize 后应包含 sub 目录路径
    assert!(text.contains("sub"), "pwd output: {text:?}");
}

#[tokio::test]
async fn case9_stdin_null_does_not_hang() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    // cat 从 stdin 读；stdin = null 应当立刻 EOF，cat 退出 0
    let events = drive(tool.execute(
        json!({"command": "cat", "timeout_ms": 5000}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Completed(_)));
    let raw = extract_raw(&events[0]);
    assert_eq!(raw["exit_code"], json!(0));
    assert_eq!(raw["timed_out"], json!(false));
}

#[tokio::test]
async fn invalid_args_missing_command() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(json!({}), ctx)).await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))));
}

#[tokio::test]
async fn invalid_args_zero_timeout() {
    let dir = tempdir().unwrap();
    let tool = BashTool::new();
    let ctx = ctx_with(dir.path(), CancellationToken::new());
    let events = drive(tool.execute(
        json!({"command": "echo x", "timeout_ms": 0}),
        ctx,
    ))
    .await;
    assert_eq!(events.len(), 1);
    assert!(matches!(events[0], ToolEvent::Failed(ToolError::InvalidArgs(_))));
}
