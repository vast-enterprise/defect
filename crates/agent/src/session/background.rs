//! Session 级后台任务表。
//!
//! 设计详见 `docs/proposals/task-arrange.md` §3.1。
//!
//! ## 解决什么
//!
//! 工具（首要场景 `spawn_agent { run_in_background: true }`）想 fire-and-forget 地跑一个
//! 任务，让发起它的 turn **不阻塞**。但 turn 主循环的 `run_tools_concurrently` 用一个**函数
//! 局部** `JoinSet` 持有工具 task——函数返回时 `JoinSet` drop、task 被 abort。所以没有任何
//! 任务能活过发起它的 turn。
//!
//! [`BackgroundTasks`] 把任务的 `JoinHandle` 挪到 **session 级**持有（与 `events` / `history`
//! 同档生命周期），任务因此活过发起它的 turn；并用一个 **session 级 [`CancellationToken`]**
//! （不是 turn 的子 token）给每个任务 mint 子 token，使后台任务的取消生命周期独立于发起它
//! 的 turn。
//!
//! ## 回流（阶段一：被动）
//!
//! 任务完成后把 [`BackgroundOutcome`] push 进 `completed` 队列。`DefaultSession::run_turn`
//! 在每次起 turn 之前 [`drain_completed`](BackgroundTasks::drain_completed)，把待回流结果作为
//! **本轮 user prompt 的前缀块**带入 history——结果搭着用户下一次输入一起被 LLM 看到。
//! 阶段二（主动续转）改由 session input loop 在后台完成时立即竞争一个新 turn，见
//! `docs/proposals/task-arrange.md` §3.2。

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// 一个后台任务完成后的产物。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackgroundOutcome {
    /// 任务 id（`spawn` 返回的同一个），用于回流消息标注与外部诊断。
    pub task_id: String,
    /// 任务标签（首要来源：`spawn_agent` 的 profile 名），进回流消息让模型/用户辨识来源。
    pub label: String,
    /// 任务结果。
    pub result: BackgroundResult,
}

/// 后台任务的最终结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackgroundResult {
    /// 正常完成，携带任务的最终文本输出。
    Completed(String),
    /// 失败（含被取消），携带错误描述。
    Failed(String),
}

impl BackgroundResult {
    fn is_error(&self) -> bool {
        matches!(self, BackgroundResult::Failed(_))
    }

    fn text(&self) -> &str {
        match self {
            BackgroundResult::Completed(t) | BackgroundResult::Failed(t) => t,
        }
    }
}

struct BackgroundInner {
    /// 单调递增的 id 计数器。
    next_id: u64,
    /// 运行中的任务。持有 `JoinHandle` 使任务活过发起它的 turn。完成时 task 自删本项。
    running: HashMap<String, JoinHandle<()>>,
    /// 已完成、待回流的结果（FIFO）。`drain_completed` 取空。
    completed: Vec<BackgroundOutcome>,
}

/// Session 级后台任务表。`Clone` 廉价（内部 `Arc`）——`DefaultSession` 持有一份，
/// 经 `ToolContext` clone 给工具。
#[derive(Clone)]
pub struct BackgroundTasks {
    /// session 级取消令牌。每个任务从它 `child_token()`，故 `cancel_all` 一次性掐掉全部，
    /// 且任意单个任务的取消不影响其他任务。
    cancel: CancellationToken,
    /// 任务完成通知。每当一个任务结果入队就 `notify_waiters`——session driver 等在它上面，
    /// 被唤醒后起一个自主 turn 主动续转（阶段二）。被动回流不依赖它。
    completed_notify: Arc<Notify>,
    inner: Arc<Mutex<BackgroundInner>>,
}

impl BackgroundTasks {
    /// 用一个 session 级取消令牌构造。`session_cancel` 由 session 持有、随 session 终结而取消。
    #[must_use]
    pub fn new(session_cancel: CancellationToken) -> Self {
        Self {
            cancel: session_cancel,
            completed_notify: Arc::new(Notify::new()),
            inner: Arc::new(Mutex::new(BackgroundInner {
                next_id: 0,
                running: HashMap::new(),
                completed: Vec::new(),
            })),
        }
    }

    /// 等到"有任务完成入队"事件。session driver 用它驱动主动续转。
    ///
    /// 用 `Notify`：driver 先 `notified()` 拿到 future、再检查队列、然后 await——避免漏掉
    /// 在两次检查之间到达的通知（`Notify` 的 permit 语义保证已发生的 notify 不丢）。
    pub async fn wait_for_completion(&self) {
        self.completed_notify.notified().await;
    }

    /// 当前是否有已完成、待回流的结果。driver 唤醒后先查它再决定起不起 turn。
    #[must_use]
    pub fn has_completed(&self) -> bool {
        !self
            .inner
            .lock()
            .expect("BackgroundTasks mutex poisoned")
            .completed
            .is_empty()
    }

    /// Spawn 一个后台任务，**立即**返回它的 id。
    ///
    /// `make_fut` 收到本任务专属的 [`CancellationToken`]（session 级 token 的子 token），
    /// 任务体应在其上感知取消。任务完成时结果进 `completed` 队列、并从 `running` 删除自身。
    ///
    /// 取这个"收 token 再造 future"的闭包形态，是因为 token 要在 spawn 内部 mint，而 future
    /// 需要捕获它——直接收 future 就拿不到这个生命周期独立于 turn 的 token。
    pub fn spawn<F, Fut>(&self, label: String, make_fut: F) -> String
    where
        F: FnOnce(CancellationToken) -> Fut,
        Fut: Future<Output = BackgroundResult> + Send + 'static,
    {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        let id = format!("bg-{}", inner.next_id);
        inner.next_id += 1;

        let task_cancel = self.cancel.child_token();
        let fut = make_fut(task_cancel);

        let inner_arc = self.inner.clone();
        let notify = self.completed_notify.clone();
        let id_for_task = id.clone();
        let label_for_task = label;
        let handle = tokio::spawn(async move {
            let result = fut.await;
            if let Ok(mut inner) = inner_arc.lock() {
                inner.running.remove(&id_for_task);
                inner.completed.push(BackgroundOutcome {
                    task_id: id_for_task,
                    label: label_for_task,
                    result,
                });
            }
            // 唤醒等在 wait_for_completion 上的 session driver（主动续转）。
            // 用 notify_one 而非 notify_waiters：前者在无等待者时**保留一个 permit**，
            // 下次 notified().await 立即返回——避免"任务在 driver park 之前就完成"导致的
            // 丢唤醒。单消费者（恰好一个 driver），notify_one 语义正合适。在锁外 notify。
            notify.notify_one();
        });

        inner.running.insert(id.clone(), handle);
        id
    }

    /// 取出全部已完成结果（清空队列）。`run_turn` 在起 turn 前调用做被动回流。
    pub fn drain_completed(&self) -> Vec<BackgroundOutcome> {
        let mut inner = self.inner.lock().expect("BackgroundTasks mutex poisoned");
        std::mem::take(&mut inner.completed)
    }

    /// 当前运行中的任务数。供诊断 / 未来的 list 控制面用。
    #[must_use]
    pub fn running_count(&self) -> usize {
        self.inner
            .lock()
            .expect("BackgroundTasks mutex poisoned")
            .running
            .len()
    }

    /// 取消所有后台任务（session 终结时调用）。幂等。
    pub fn cancel_all(&self) {
        self.cancel.cancel();
    }
}

/// 把一个后台任务结果格式化成回流到对话里的文本块内容。
///
/// 措辞按"延迟工具结果回流"组织，明确标注来源（task id + label）与成败，避免模型误判为用户发言。
/// 阶段二会换成 `IngestSource::Background` 的正派 ingest 路径（§5.1），届时此函数被相应载荷取代。
#[must_use]
pub fn format_background_outcome(outcome: &BackgroundOutcome) -> String {
    let status = if outcome.result.is_error() {
        "failed"
    } else {
        "completed"
    };
    format!(
        "⟨background task {} ({}) {}⟩\n{}",
        outcome.task_id,
        outcome.label,
        status,
        outcome.result.text()
    )
}

#[cfg(test)]
#[path = "background/test.rs"]
mod test;
