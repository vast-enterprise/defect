//! `inspect_background_task` / `cancel_background_task` 工具单测。
//!
//! 直接构造 [`BackgroundTasks`]、spawn 几个可控任务，再用工具的 `execute` 跑出 tool 事件
//! 断言渲染文本与控制效果。子 agent 那条进度链路（spawn_agent → ProgressSink）由
//! spawn_agent 自己的测试覆盖；这里只验工具对任务表的读/控。

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

/// 跑一个工具，注入指定的 background 句柄（或 `None`），收集全部 tool 事件。
fn run(tool: &dyn Tool, args: serde_json::Value, background: Option<BackgroundTasks>) -> Vec<ToolEvent> {
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

/// 取一个 `Completed` 事件里的文本（raw_output 字符串）。非 Completed 直接 panic。
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
    // 一个立刻完成的任务。
    let done_id = bg.spawn("reviewer".to_string(), |_c, _p| async {
        BackgroundResult::Completed("ok".to_string())
    });
    // 一个阻塞着的任务（保持 running）。
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let running_id = bg.spawn("builder".to_string(), |_c, _p| async move {
        let _ = rx.await;
        BackgroundResult::Completed("late".to_string())
    });

    // 等完成任务入表终态。
    for _ in 0..200 {
        if bg.peek(&done_id, 1).map(|s| s.status) == Some(crate::session::TaskStatus::Completed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    let tool = InspectBackgroundTaskTool::new();
    let bg2 = bg.clone();
    let out =
        tokio::task::spawn_blocking(move || run(&tool, serde_json::json!({}), Some(bg2)))
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

    // 任务实际结束后状态应为 Canceled。
    let mut status = None;
    for _ in 0..200 {
        status = bg.peek(&id, 1).map(|s| s.status);
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
        if bg.peek(&id, 1).map(|s| s.status) == Some(crate::session::TaskStatus::Completed) {
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
