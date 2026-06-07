//! Unit tests for `inspect_background_task` / `cancel_background_task`.
//!
//! Constructs a [`BackgroundTasks`] directly, spawns a few controllable tasks, then runs
//! the tools' `execute` to produce tool events and asserts the rendered text and control
//! effects. The sub-agent progress path (`spawn_agent` → `ProgressSink`) is covered by
//! `spawn_agent`'s own tests; here we only verify the tools' read/control over the task
//! table.

use super::*;

use std::path::Path;
use std::sync::Arc;

use futures::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::fs::{FsBackend, NoopFsBackend};
use crate::http::NoopHttpClient;
use crate::session::{BackgroundResult, BackgroundTasks};
use crate::shell::ShellBackend;
use crate::tool::ToolContext;

/// Runs a tool, injecting the given background handle (or `None`), and collects all tool
/// events.
fn run(
    tool: &dyn Tool,
    args: serde_json::Value,
    background: Option<BackgroundTasks>,
) -> Vec<ToolEvent> {
    let fs: Arc<dyn FsBackend> = Arc::new(NoopFsBackend);
    let shell: Arc<dyn ShellBackend> = Arc::new(crate::shell::NoopShellBackend);
    let http = Arc::new(NoopHttpClient);
    let cwd = Path::new("/tmp");
    let mut ctx = ToolContext::new(cwd, CancellationToken::new(), fs, shell, http, "fake-1");
    if let Some(bg) = background {
        ctx = ctx.with_background(bg);
    }
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        let mut stream = tool.execute(args, ctx);
        let mut out = Vec::new();
        while let Some(ev) = stream.next().await {
            out.push(ev);
        }
        out
    })
}

/// Extract the text from a `Completed` event's `raw_output` string. Panics if the event
/// is not `Completed`.
fn completed_text_of(ev: &ToolEvent) -> &str {
    match ev {
        ToolEvent::Completed(fields) => fields
            .raw_output
            .as_ref()
            .and_then(|v| v.as_str())
            .expect("raw_output string"),
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[test]
fn inspect_without_background_fails_loud() {
    let tool = InspectBackgroundTaskTool::new();
    let out = run(&tool, serde_json::json!({}), None);
    assert!(matches!(out.as_slice(), [ToolEvent::Failed(_)]));
}

#[test]
fn cancel_without_background_fails_loud() {
    let tool = CancelBackgroundTaskTool::new();
    let out = run(&tool, serde_json::json!({ "task_id": "bg-0" }), None);
    assert!(matches!(out.as_slice(), [ToolEvent::Failed(_)]));
}

#[test]
fn inspect_empty_lists_nothing() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let tool = InspectBackgroundTaskTool::new();
    let out = run(&tool, serde_json::json!({}), Some(bg));
    assert_eq!(out.len(), 1);
    assert!(completed_text_of(&out[0]).contains("No background tasks"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inspect_lists_running_and_finished() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    // A task that completes immediately.
    let done_id = bg.spawn("reviewer".to_string(), |_c, _p| async {
        BackgroundResult::Completed("ok".to_string())
    });
    // A blocking task (keeps running).
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let running_id = bg.spawn("builder".to_string(), |_c, _p| async move {
        let _ = rx.await;
        BackgroundResult::Completed("late".to_string())
    });

    // Wait for the task to reach its final state in the table.
    for _ in 0..200 {
        if bg.peek(&done_id, Some(1)).map(|s| s.status)
            == Some(crate::session::TaskStatus::Completed)
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let tool = InspectBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let out = tokio::task::spawn_blocking(move || run(&tool, serde_json::json!({}), Some(bg2)))
        .await
        .unwrap();
    let text = completed_text_of(&out[0]);
    assert!(text.contains(&done_id), "lists completed task");
    assert!(text.contains(&running_id), "lists running task");
    assert!(text.contains("completed"));
    assert!(text.contains("running"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inspect_unknown_task_id_fails() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let tool = InspectBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let out = tokio::task::spawn_blocking(move || {
        run(&tool, serde_json::json!({ "task_id": "bg-99" }), Some(bg2))
    })
    .await
    .unwrap();
    assert!(matches!(out.as_slice(), [ToolEvent::Failed(_)]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_running_task_then_it_ends_canceled() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let id = bg.spawn("cancellable".to_string(), |cancel, _p| async move {
        cancel.cancelled().await;
        BackgroundResult::Failed("stopped".to_string())
    });

    let tool = CancelBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let id2 = id.clone();
    let out = tokio::task::spawn_blocking(move || {
        run(&tool, serde_json::json!({ "task_id": id2 }), Some(bg2))
    })
    .await
    .unwrap();
    assert!(completed_text_of(&out[0]).contains("cancellation"));

    // After the task actually finishes, its status should be `Canceled`.
    let mut status = None;
    for _ in 0..200 {
        status = bg.peek(&id, Some(1)).map(|s| s.status);
        if status == Some(crate::session::TaskStatus::Canceled) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(status, Some(crate::session::TaskStatus::Canceled));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_unknown_task_id_fails() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let tool = CancelBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let out = tokio::task::spawn_blocking(move || {
        run(&tool, serde_json::json!({ "task_id": "bg-99" }), Some(bg2))
    })
    .await
    .unwrap();
    assert!(matches!(out.as_slice(), [ToolEvent::Failed(_)]));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_finished_task_is_noop() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let id = bg.spawn("quick".to_string(), |_c, _p| async {
        BackgroundResult::Completed("done".to_string())
    });
    for _ in 0..200 {
        if bg.peek(&id, Some(1)).map(|s| s.status) == Some(crate::session::TaskStatus::Completed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let tool = CancelBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let id2 = id.clone();
    let out = tokio::task::spawn_blocking(move || {
        run(&tool, serde_json::json!({ "task_id": id2 }), Some(bg2))
    })
    .await
    .unwrap();
    assert!(completed_text_of(&out[0]).contains("already finished"));
}
