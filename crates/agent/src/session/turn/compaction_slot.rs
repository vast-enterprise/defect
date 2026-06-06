//! 后台压缩单槽（single-flight）。
//!
//! 全量压缩（摘要）要调 LLM，几秒级延迟。放在 turn 关键路径上同步跑会让越线那一轮
//! 的用户干等。后台压缩把这次摘要调用挪到 `tokio::spawn` 的任务里，turn 不阻塞——
//! 趁还没撞到 hard 水位，悄悄把历史压下去。
//!
//! ## 为什么是 session 级、单槽
//!
//! - **单槽（single-flight）**：同时至多一个压缩在飞。这正是 `History::splice_prefix`
//!   并发不变式的前提——「飞行期间不增删中段消息」，而唯一会删中段的操作就是压缩
//!   本身。两个并发压缩会互相作废对方算出的 `drop_count`。
//! - **session 级**：压缩任务要活过发起它的 turn（摘要还没回来 turn 可能就结束了），
//!   所以 `JoinHandle` 挂在 session 上，与 [`BackgroundTasks`](crate::session::BackgroundTasks) 同档生命周期。
//!
//! ## 回写
//!
//! 任务完成后直接 `history.splice_prefix(drop_count, summary)` **静默改写历史**——
//! 注意这与 [`BackgroundTasks`](crate::session::BackgroundTasks) 不同：那条路径把结果当 user 消息回灌对话，
//! 而压缩必须无声。完成时通过回调发 `ContextCompressed` 事件供 observability 消费。

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use tokio::task::JoinHandle;

use super::compact::{self, CompactionCtx};
use crate::session::{CompactionReport, History};

/// 完成回调：拿到压缩报告后发事件。返回 `BoxFuture` 使回调能 `await`（`emit` 是
/// async）——压缩任务体直接 `.await` 它，事件发送因此在压缩任务内完成、受同一个
/// cancel/track 约束，不另起游离任务（对齐 workspace `BoxFuture` 惯例）。
type OnDone = Arc<dyn Fn(CompactionReport) -> BoxFuture<'static, ()> + Send + Sync>;

struct SlotInner {
    /// 在飞的压缩任务。`Some` = 有一个在跑。完成时任务自清此项。
    flight: Option<JoinHandle<()>>,
}

/// Session 级后台压缩槽。`Clone` 廉价（内部 `Arc`）。
#[derive(Clone)]
pub struct CompactionSlot {
    inner: Arc<Mutex<SlotInner>>,
}

impl Default for CompactionSlot {
    fn default() -> Self {
        Self::new()
    }
}

impl CompactionSlot {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SlotInner { flight: None })),
        }
    }

    /// 当前是否有压缩在飞。
    #[must_use]
    pub(crate) fn is_in_flight(&self) -> bool {
        let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
        // 顺手回收已结束的 handle（任务体已自行回写，这里只清账）。
        if let Some(h) = &inner.flight
            && h.is_finished()
        {
            inner.flight = None;
        }
        inner.flight.is_some()
    }

    /// 尝试起一个后台压缩。已有在飞 → 返回 `false`（单飞，不重复起）。
    ///
    /// `history` 取 `Arc<dyn History>` 以便任务 `'static` 持有它跨 turn。`ctx` 携
    /// provider / model / tools / 取消令牌。`on_done` 在压缩成功后被调用（发事件）。
    pub(crate) fn try_spawn(
        &self,
        history: Arc<dyn History>,
        ctx: CompactionCtx,
        threshold: u64,
        on_done: OnDone,
    ) -> bool {
        let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
        // 先清掉已结束的 handle，避免「上一个已完成但 handle 还挂着」挡住新的。
        if let Some(h) = &inner.flight
            && h.is_finished()
        {
            inner.flight = None;
        }
        if inner.flight.is_some() {
            return false;
        }

        let slot = self.inner.clone();
        let handle = tokio::spawn(async move {
            run_once(history.as_ref(), &ctx, threshold, &on_done).await;
            // 任务收尾：清掉自己的 handle 占位（若仍是自己——已结束）。
            if let Ok(mut inner) = slot.lock()
                && let Some(h) = &inner.flight
                && h.is_finished()
            {
                inner.flight = None;
            }
        });
        inner.flight = Some(handle);
        true
    }

    /// 等待在飞的压缩落地（若有）。hard 水位兜底用：构造请求前若已有后台压缩在飞，
    /// 与其同步重压，不如等它完成。无在飞任务则立即返回。
    pub(crate) async fn await_in_flight(&self) {
        let handle = {
            let mut inner = self.inner.lock().expect("CompactionSlot mutex poisoned");
            inner.flight.take()
        };
        if let Some(handle) = handle {
            // 任务体已自行回写历史；这里只等它结束。忽略 JoinError（panic/abort）。
            let _ = handle.await;
        }
    }
}

/// 跑一次后台压缩：snapshot → plan → summarize → `splice_prefix` 回写 → 发事件。
/// 任何最佳努力跳过（无边界 / 摘要失败）都静默返回。
async fn run_once(history: &dyn History, ctx: &CompactionCtx, threshold: u64, on_done: &OnDone) {
    let messages = history.snapshot();
    let Some(plan) = compact::plan(&messages, threshold) else {
        return;
    };
    let Some(summary) = compact::summarize(ctx, &plan.head, plan.prev_summary.as_deref()).await
    else {
        return;
    };
    let summary_msg = compact::summary_message(&summary);

    // 关键：splice_prefix 只丢当前列表前 drop_count 条、保留其后全部（含摘要
    // 飞行期间前台 append 的尾部消息）。单飞保证了 drop_count 在当前列表仍合法。
    history.splice_prefix(plan.drop_count, summary_msg);
    let tokens_after = history.token_estimate().unwrap_or(plan.tokens_before);

    tracing::info!(
        drop_count = plan.drop_count,
        tokens_before = plan.tokens_before,
        tokens_after,
        "context compacted (background)"
    );
    on_done(CompactionReport {
        tokens_before: plan.tokens_before,
        tokens_after,
    })
    .await;
}
