//! `LangfuseObserver`：把每个 session 的 [`AgentEvent`](defect_agent::event::AgentEvent) 流上报到 Langfuse。
//!
//! 形状照抄 `defect-storage::StorageObserver`（`crates/storage/src/lib.rs`）：
//! [`SessionObserver::on_session_created`] 里 `session.subscribe()` 拿一条独立
//! mpsc 流，`tokio::spawn` 一个消费任务，逐事件喂 [`TraceProjector`] 翻译、
//! 经 [`LangfuseIngest`] 上报；流结束（session drop）后 `flush` 残留。
//!
//! 与 storage 的关键区别：**可丢弃降级**。storage 慢消费会 backpressure 主循环
//! （“不丢”语义），langfuse 不行——所以消费循环里只做 `enqueue`（非阻塞）+
//! 轻量翻译，真正的网络 IO 全在 [`LangfuseIngest`] 的后台任务里，且满了丢弃。
//! Any Langfuse failure must NOT affect the agent.

use std::sync::Arc;

use defect_agent::error::BoxError;
use defect_agent::session::{Session, SessionCreateInfo, SessionObserver};
use futures::StreamExt;

use super::ingest::LangfuseIngest;
use super::projector::TraceProjector;

/// Langfuse 上报观察器。`Clone` 廉价（内部 `Arc`）。
#[derive(Clone)]
pub struct LangfuseObserver {
    ingest: LangfuseIngest,
}

impl LangfuseObserver {
    /// 用一个已启动的上报器构造。上报器的后台任务在 [`LangfuseIngest::spawn`]
    /// 时已拉起，本观察器只负责把 per-session 事件流接进去。
    #[must_use]
    pub fn new(ingest: LangfuseIngest) -> Self {
        Self { ingest }
    }
}

impl SessionObserver for LangfuseObserver {
    fn on_session_created(
        &self,
        session: Arc<dyn Session>,
        info: SessionCreateInfo,
    ) -> Result<(), BoxError> {
        let mut events = session.subscribe();
        let ingest = self.ingest.clone();
        let session_id = info.id.0.to_string();

        tokio::spawn(async move {
            let mut projector = TraceProjector::new(session_id);
            // 每个 ingestion 事件的信封 id / trace id：随机 UUID。
            let mut new_id = || uuid::Uuid::new_v4().to_string();

            while let Some(event) = events.next().await {
                // 用接收时刻近似事件发生时刻（AgentEvent 不带时间戳，见设计文档 §3.4）。
                let now = chrono::Utc::now().to_rfc3339();
                for ev in projector.project(event, &now, &mut new_id) {
                    ingest.enqueue(ev);
                }
            }

            // 流结束（session drop / 进程退出前）：尽力冲刷残留遥测。
            ingest.flush().await;
        });

        Ok(())
    }
}
