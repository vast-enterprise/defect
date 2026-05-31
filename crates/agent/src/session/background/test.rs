//! [`BackgroundTasks`] 单测：任务活过 spawn、完成回流、取消、id 唯一。

use super::*;

#[tokio::test]
async fn completed_task_flows_into_queue() {
    let bg = BackgroundTasks::new(CancellationToken::new());
    let id = bg.spawn("reviewer".to_string(), |_cancel| async move {
        BackgroundResult::Completed("done".to_string())
    });
    assert_eq!(id, "bg-0");

    // 轮询直到任务跑完并入队（spawn 是异步的）。
    let outcome = wait_for_one(&bg).await;
    assert_eq!(outcome.task_id, "bg-0");
    assert_eq!(outcome.label, "reviewer");
    assert_eq!(outcome.result, BackgroundResult::Completed("done".to_string()));

    // drain 取空后再 drain 为空。
    assert!(bg.drain_completed().is_empty());
}

#[tokio::test]
async fn task_outlives_spawn_call() {
    let bg = BackgroundTasks::new(CancellationToken::new());
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    // 任务阻塞在 rx 上——spawn 返回后它仍在跑。
    bg.spawn("slow".to_string(), |_cancel| async move {
        let _ = rx.await;
        BackgroundResult::Completed("late".to_string())
    });
    assert_eq!(bg.running_count(), 1, "task should still be running after spawn returned");
    assert!(bg.drain_completed().is_empty(), "not done yet");

    // 放行 → 任务完成 → 入队、running 清零。
    tx.send(()).unwrap();
    let outcome = wait_for_one(&bg).await;
    assert_eq!(outcome.result, BackgroundResult::Completed("late".to_string()));
    assert_eq!(bg.running_count(), 0);
}

#[tokio::test]
async fn cancel_all_propagates_to_task_token() {
    let bg = BackgroundTasks::new(CancellationToken::new());
    bg.spawn("cancellable".to_string(), |cancel| async move {
        cancel.cancelled().await;
        BackgroundResult::Failed("cancelled".to_string())
    });
    bg.cancel_all();
    let outcome = wait_for_one(&bg).await;
    assert_eq!(outcome.result, BackgroundResult::Failed("cancelled".to_string()));
}

#[tokio::test]
async fn ids_are_unique_and_monotonic() {
    let bg = BackgroundTasks::new(CancellationToken::new());
    let id0 = bg.spawn("a".to_string(), |_c| async { BackgroundResult::Completed(String::new()) });
    let id1 = bg.spawn("b".to_string(), |_c| async { BackgroundResult::Completed(String::new()) });
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
