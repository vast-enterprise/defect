//! Tests for [`BackgroundTasks`]: tasks survive spawn, completion flows back,
//! cancellation, and unique IDs.

use super::*;

#[tokio::test]
async fn completed_task_flows_into_queue() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let id = bg.spawn("reviewer".to_string(), |_cancel, _progress| async move {
        BackgroundResult::Completed("done".to_string())
    });
    assert_eq!(id, "bg-0");

    // Poll until the task finishes and is enqueued (spawn is asynchronous).
    let outcome = wait_for_one(&bg).await;
    assert_eq!(outcome.task_id, "bg-0");
    assert_eq!(outcome.label, "reviewer");
    assert_eq!(
        outcome.result,
        BackgroundResult::Completed("done".to_string())
    );

    // Draining again after draining is empty.
    assert!(bg.drain_completed().is_empty());
}

#[tokio::test]
async fn task_outlives_spawn_call() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    // The task blocks on `rx` — it is still running after `spawn` returns.
    bg.spawn("slow".to_string(), |_cancel, _progress| async move {
        let _ = rx.await;
        BackgroundResult::Completed("late".to_string())
    });
    assert_eq!(
        bg.running_count(),
        1,
        "task should still be running after spawn returned"
    );
    assert!(bg.drain_completed().is_empty(), "not done yet");

    // Allow the task to complete, enqueue the result, and reset running to zero.
    tx.send(()).unwrap();
    let outcome = wait_for_one(&bg).await;
    assert_eq!(
        outcome.result,
        BackgroundResult::Completed("late".to_string())
    );
    assert_eq!(bg.running_count(), 0);
}

#[tokio::test]
async fn cancel_all_propagates_to_task_token() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    bg.spawn("cancellable".to_string(), |cancel, _progress| async move {
        cancel.cancelled().await;
        BackgroundResult::Failed("cancelled".to_string())
    });
    bg.cancel_all();
    let outcome = wait_for_one(&bg).await;
    assert_eq!(
        outcome.result,
        BackgroundResult::Failed("cancelled".to_string())
    );
}

#[tokio::test]
async fn ids_are_unique_and_monotonic() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let id0 = bg.spawn("a".to_string(), |_c, _p| async {
        BackgroundResult::Completed(String::new())
    });
    let id1 = bg.spawn("b".to_string(), |_c, _p| async {
        BackgroundResult::Completed(String::new())
    });
    assert_eq!(id0, "bg-0");
    assert_eq!(id1, "bg-1");
}

#[test]
fn format_outcome_labels_source_and_status() {
    let ok = BackgroundOutcome {
        task_id: "bg-3".to_string(),
        label: "reviewer".to_string(),
        result: BackgroundResult::Completed("looks good".to_string()),
    };
    let s = format_background_outcome(&ok);
    assert!(s.contains("bg-3"));
    assert!(s.contains("reviewer"));
    assert!(s.contains("completed"));
    assert!(s.contains("looks good"));

    let err = BackgroundOutcome {
        task_id: "bg-4".to_string(),
        label: "builder".to_string(),
        result: BackgroundResult::Failed("boom".to_string()),
    };
    assert!(format_background_outcome(&err).contains("failed"));
}

#[test]
fn truncate_body_respects_limit() {
    // limit 0 ⇒ empty string (metadata only).
    assert_eq!(truncate_body("hello world", 0), "");
    // Not exceeding limit → unchanged.
    assert_eq!(truncate_body("hello", 10), "hello");
    assert_eq!(truncate_body("hello", 5), "hello");
    // Exceeds limit → truncate + marker, counted in characters (not bytes).
    let out = truncate_body("hello world", 5);
    assert!(out.starts_with("hello "), "kept prefix: {out}");
    assert!(out.contains("+6 more chars"), "marker: {out}");
    // Multi-byte characters are split on scalar boundaries; no panics or broken
    // characters.
    let cjk = "你好世界啊"; // 5 scalar values
    assert_eq!(truncate_body(cjk, 5), cjk);
    let cut = truncate_body(cjk, 2);
    assert!(cut.starts_with("你好"));
    assert!(cut.contains("+3 more chars"));
}

// Build an assistant message containing text and one tool_use.
fn assistant_msg(text: &str, tool: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![
            MessageContent::Text {
                text: text.to_string(),
            },
            MessageContent::ToolUse {
                id: "tu-1".to_string(),
                name: tool.to_string(),
                args: serde_json::json!({}),
            },
        ]
        .into(),
    }
}

fn user_msg(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![MessageContent::Text {
            text: text.to_string(),
        }]
        .into(),
    }
}

#[test]
fn recent_blocks_flattens_messages_and_orders() {
    // Two messages flatten into user + (assistant text + tool_use) = 3 blocks.
    let msgs = vec![
        user_msg("do the thing"),
        assistant_msg("working on it", "read_file"),
    ];
    // Large limit; full content retained.
    let (total, recent) = recent_blocks_of(&msgs, 10, 1000);
    assert_eq!(total, 3);
    let kinds: Vec<BlockKind> = recent.iter().map(|b| b.kind).collect();
    assert_eq!(
        kinds,
        vec![
            BlockKind::User,
            BlockKind::AssistantText,
            BlockKind::ToolUse
        ]
    );
    // The text of a `ToolUse` block is the tool name.
    assert_eq!(recent[2].text, "read_file");
    // `recent(n)` takes the last `n` items.
    let (_total, last2) = recent_blocks_of(&msgs, 2, 1000);
    assert_eq!(last2.len(), 2);
    assert_eq!(last2[0].kind, BlockKind::AssistantText);
    assert_eq!(last2[1].kind, BlockKind::ToolUse);
}

#[test]
fn recent_blocks_default_limit_drops_free_form_body_keeps_tool_name() {
    let msgs = vec![assistant_msg("a long assistant reply", "read_file")];
    // With limit 0, free-form assistant text is cleared but the tool name (non-text) is
    // preserved.
    let (_total, recent) = recent_blocks_of(&msgs, 10, 0);
    let text_block = recent
        .iter()
        .find(|b| b.kind == BlockKind::AssistantText)
        .unwrap();
    assert_eq!(text_block.text, "", "free-form body dropped at limit 0");
    let tool_block = recent
        .iter()
        .find(|b| b.kind == BlockKind::ToolUse)
        .unwrap();
    assert_eq!(
        tool_block.text, "read_file",
        "tool name kept (not a free-form body)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peek_reads_attached_history_committed_blocks() {
    use crate::session::{History, VecHistory};

    let bg = BackgroundTasks::new(
        CancellationToken::new(),
        BackgroundProgressConfig::default(),
    );
    // Use an externally owned `history` Arc: the task body attaches it, and we append to
    // it from outside to simulate child-turn submissions.
    let history: Arc<dyn History> = Arc::new(VecHistory::new());
    let history_for_task = history.clone();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let id = bg.spawn("worker".to_string(), move |_c, handle| async move {
        handle.attach_history(history_for_task);
        let _ = rx.await; // Blocks, keeping the task running so it can be peeked mid-execution.
        BackgroundResult::Completed("done".to_string())
    });

    // Simulate a child turn submitting message blocks into the history (the same `Arc`
    // that the task body attached).
    history.append(user_msg("task instructions"));
    history.append(assistant_msg("on it", "read_file"));

    // Poll until the attach takes effect (the task body attaches as soon as it enters;
    // `status == Running` flips at spawn time, before the future body executes, so we
    // must wait until the history is actually attached and `block_count` reflects it).
    let mut snap = bg.peek(&id, None).expect("task exists");
    for _ in 0..200 {
        if snap.block_count == 3 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        snap = bg.peek(&id, None).expect("task exists");
    }
    assert_eq!(snap.status, TaskStatus::Running);
    // With default limit=0, the assistant body is empty but user/tool name structures are
    // visible; 3 blocks total.
    assert_eq!(snap.block_count, 3);
    let kinds: Vec<BlockKind> = snap.recent.iter().map(|b| b.kind).collect();
    assert_eq!(
        kinds,
        vec![
            BlockKind::User,
            BlockKind::AssistantText,
            BlockKind::ToolUse
        ]
    );
    assert_eq!(snap.recent[2].text, "read_file");

    // Allow the task to finish.
    let _ = tx.send(());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peek_without_attached_history_returns_empty_blocks() {
    let bg = BackgroundTasks::new(
        CancellationToken::new(),
        BackgroundProgressConfig::default(),
    );
    // The task does not attach history.
    let id = bg.spawn("worker".to_string(), |_c, _handle| async {
        BackgroundResult::Completed("done".to_string())
    });
    for _ in 0..200 {
        if bg.peek(&id, None).map(|s| s.status) == Some(TaskStatus::Completed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let snap = bg.peek(&id, None).expect("task exists");
    assert_eq!(snap.block_count, 0);
    assert!(snap.recent.is_empty());
}

/// Poll `drain_completed` until exactly one result is available (with a timeout to
/// prevent hanging).
async fn wait_for_one(bg: &BackgroundTasks) -> BackgroundOutcome {
    for _ in 0..200 {
        let mut done = bg.drain_completed();
        if let Some(o) = done.pop() {
            return o;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("background task did not complete in time");
}
