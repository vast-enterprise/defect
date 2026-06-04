//! [`BackgroundTasks`] 单测：任务活过 spawn、完成回流、取消、id 唯一。

use super::*;

#[tokio::test]
async fn completed_task_flows_into_queue() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let id = bg.spawn("reviewer".to_string(), |_cancel, _progress| async move {
        BackgroundResult::Completed("done".to_string())
    });
    assert_eq!(id, "bg-0");

    // 轮询直到任务跑完并入队（spawn 是异步的）。
    let outcome = wait_for_one(&bg).await;
    assert_eq!(outcome.task_id, "bg-0");
    assert_eq!(outcome.label, "reviewer");
    assert_eq!(
        outcome.result,
        BackgroundResult::Completed("done".to_string())
    );

    // drain 取空后再 drain 为空。
    assert!(bg.drain_completed().is_empty());
}

#[tokio::test]
async fn task_outlives_spawn_call() {
    let bg = BackgroundTasks::new(CancellationToken::new(), Default::default());
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    // 任务阻塞在 rx 上——spawn 返回后它仍在跑。
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

    // 放行 → 任务完成 → 入队、running 清零。
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
    // limit 0 ⇒ 空串（只留元信息）。
    assert_eq!(truncate_body("hello world", 0), "");
    // 未超限 ⇒ 原样。
    assert_eq!(truncate_body("hello", 10), "hello");
    assert_eq!(truncate_body("hello", 5), "hello");
    // 超限 ⇒ 截断 + 标记，按字符数（非字节）。
    let out = truncate_body("hello world", 5);
    assert!(out.starts_with("hello "), "kept prefix: {out}");
    assert!(out.contains("+6 more chars"), "marker: {out}");
    // 多字节字符按标量切，不 panic、不切坏。
    let cjk = "你好世界啊"; // 5 个标量
    assert_eq!(truncate_body(cjk, 5), cjk);
    let cut = truncate_body(cjk, 2);
    assert!(cut.starts_with("你好"));
    assert!(cut.contains("+3 more chars"));
}

#[test]
fn progress_ring_honors_cap_and_recent_order() {
    let mut ring = ProgressRing::with_cap(3);
    for i in 0..5 {
        ring.push(ProgressBlock {
            kind: ProgressKind::AssistantText,
            text: i.to_string(),
        });
    }
    // 容量 3：最旧的 0/1 被淘汰，保留 2/3/4，顺序旧→新。
    let recent = ring.recent(10);
    let texts: Vec<&str> = recent.iter().map(|b| b.text.as_str()).collect();
    assert_eq!(texts, vec!["2", "3", "4"]);
    // recent(n) 取尾部 n 个。
    let last2 = ring.recent(2);
    assert_eq!(
        last2.iter().map(|b| b.text.as_str()).collect::<Vec<_>>(),
        vec!["3", "4"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_truncates_bodies_but_keeps_tool_titles() {
    // 默认配置：block_text_limit = 0 ⇒ 正文清空，工具标题保留。
    let bg = BackgroundTasks::new(CancellationToken::new(), BackgroundProgressConfig::default());
    let id = bg.spawn("p".to_string(), |_c, sink| async move {
        sink.push(ProgressKind::AssistantText, "a long assistant reply".to_string());
        sink.push(ProgressKind::Thought, "deep thoughts".to_string());
        sink.push(ProgressKind::ToolStart, "Read foo.rs".to_string());
        sink.push(ProgressKind::ToolFinish, "Read foo.rs".to_string());
        BackgroundResult::Completed(String::new())
    });
    // sink.push 同步，但任务体在 tokio task 里跑——轮询等它把 4 个 block 写进去。
    for _ in 0..200 {
        if bg.peek(&id, 10).map(|s| s.block_count) == Some(4) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let snap = bg.peek(&id, 10).expect("task exists");
    let by_kind = |k: ProgressKind| {
        snap.recent
            .iter()
            .find(|b| b.kind == k)
            .map(|b| b.text.clone())
            .unwrap()
    };
    // 自由正文被清空（鸟瞰：只剩"发生了 assistant/thought"这一元信息）。
    assert_eq!(by_kind(ProgressKind::AssistantText), "");
    assert_eq!(by_kind(ProgressKind::Thought), "");
    // 工具调用标题不受正文上限约束，原样保留。
    assert_eq!(by_kind(ProgressKind::ToolStart), "Read foo.rs");
    assert_eq!(by_kind(ProgressKind::ToolFinish), "Read foo.rs");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sink_keeps_bodies_when_limit_set() {
    let cfg = BackgroundProgressConfig {
        ring_cap: 64,
        block_text_limit: 8,
    };
    let bg = BackgroundTasks::new(CancellationToken::new(), cfg);
    let id = bg.spawn("p".to_string(), |_c, sink| async move {
        sink.push(ProgressKind::AssistantText, "short".to_string());
        sink.push(ProgressKind::Thought, "way too long to keep".to_string());
        BackgroundResult::Completed(String::new())
    });
    for _ in 0..200 {
        if bg.peek(&id, 10).map(|s| s.block_count) == Some(2) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let snap = bg.peek(&id, 10).expect("task exists");
    let texts: Vec<&str> = snap.recent.iter().map(|b| b.text.as_str()).collect();
    assert!(texts.contains(&"short"), "under-limit kept whole: {texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("more chars")),
        "over-limit truncated with marker: {texts:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ring_cap_zero_is_treated_as_one() {
    let cfg = BackgroundProgressConfig {
        ring_cap: 0,
        block_text_limit: 0,
    };
    let bg = BackgroundTasks::new(CancellationToken::new(), cfg);
    let id = bg.spawn("p".to_string(), |_c, sink| async move {
        sink.push(ProgressKind::ToolStart, "a".to_string());
        sink.push(ProgressKind::ToolStart, "b".to_string());
        BackgroundResult::Completed(String::new())
    });
    // 至少能写进东西（cap 被规整成 1），且不溢出 panic。
    for _ in 0..200 {
        if bg.peek(&id, 10).map(|s| s.status) == Some(TaskStatus::Completed) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let snap = bg.peek(&id, 10).expect("task exists");
    // cap=1：只剩最后一个 block。
    assert_eq!(snap.block_count, 1);
    assert_eq!(snap.recent.last().map(|b| b.text.as_str()), Some("b"));
}

/// 轮询 `drain_completed` 直到拿到恰好一个结果（带超时上限，避免挂死）。
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
