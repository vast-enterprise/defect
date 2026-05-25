//! 事件发布：mpsc bounded + fan-out。
//!
//! 设计详见 [`crate::session`] 文档与 `docs/internal/session.md` §5。
//!
//! ## 形状
//!
//! 主循环只 [`EventEmitter::emit`]；订阅者通过 [`EventEmitter::subscribe`]
//! 拿一个独立的 mpsc receiver。emit 内部串行 send 到所有 receiver，
//! **慢消费者会让 emit 阻塞**（backpressure）——这正是我们要的"不丢事件"。
//!
//! ## 不用 broadcast 的理由
//!
//! [`tokio::sync::broadcast`] 在 receiver 跟不上时标 `Lagged` 并跳过事件，
//! 直接违反 [`AgentEvent`](crate::event::AgentEvent) "不丢"约束（见
//! `docs/internal/event-model.md` §5）。

use std::sync::Mutex;

use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::event::AgentEvent;

/// 默认 mpsc 容量。订阅者跟不上时主循环阻塞在第 257 条事件。
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// 单个订阅者的 sender 句柄。`Mutex` 包它仅是因为 [`EventEmitter::emit`]
/// 是 `&self` + `async`——dashmap / RwLock 都可以；这里选 std Mutex 是因为
/// 我们只在 emit 时短暂持锁取列表快照、send 在锁外。
type SubscriberHandle = mpsc::Sender<AgentEvent>;

/// 事件总线。每 session 一个实例。
pub struct EventEmitter {
    capacity: usize,
    /// 注册中的订阅者。drop receiver 后下次 emit 会自动清理。
    senders: Mutex<Vec<SubscriberHandle>>,
}

impl EventEmitter {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            senders: Mutex::new(Vec::new()),
        }
    }

    /// 新增订阅者。返回的 stream 在 [`Self`] 被 drop 后自然结束。
    pub fn subscribe(&self) -> BoxStream<'static, AgentEvent> {
        let (tx, rx) = mpsc::channel(self.capacity);
        self.senders
            .lock()
            .expect("EventEmitter senders mutex poisoned")
            .push(tx);
        ReceiverStream::new(rx).boxed()
    }

    /// 把事件投递给所有订阅者。
    ///
    /// 串行 await 每个 sender。任一订阅者填满自己的 channel 时，本调用会
    /// 阻塞直到对方消费——这是有意为之的 backpressure。
    pub async fn emit(&self, event: AgentEvent) {
        // 取快照，避免在 await 期间持锁。
        let snapshot: Vec<SubscriberHandle> = {
            let guard = self
                .senders
                .lock()
                .expect("EventEmitter senders mutex poisoned");
            guard.clone()
        };

        let mut dead_indices: Vec<usize> = Vec::new();
        for (idx, tx) in snapshot.iter().enumerate() {
            if tx.send(event.clone()).await.is_err() {
                dead_indices.push(idx);
            }
        }

        if !dead_indices.is_empty() {
            self.prune(&snapshot, &dead_indices);
        }
    }

    /// 清理已经 drop 的 receiver 对应的 sender。
    fn prune(&self, snapshot: &[SubscriberHandle], dead_indices: &[usize]) {
        let mut guard = self
            .senders
            .lock()
            .expect("EventEmitter senders mutex poisoned");
        // snapshot 与 *guard 可能由于其他 subscribe 调用而长度不一致；
        // 我们按"指针相等"判断，避免删错。
        guard.retain(|tx| {
            !dead_indices.iter().any(|&i| {
                snapshot
                    .get(i)
                    .map(|dead| dead.same_channel(tx))
                    .unwrap_or(false)
            })
        });
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::AgentEvent;

    #[tokio::test]
    async fn emits_to_all_subscribers() {
        let bus = EventEmitter::new();
        let mut a = bus.subscribe();
        let mut b = bus.subscribe();

        bus.emit(AgentEvent::TurnStarted).await;

        let ea = a.next().await.expect("subscriber a closed");
        let eb = b.next().await.expect("subscriber b closed");
        assert!(matches!(ea, AgentEvent::TurnStarted));
        assert!(matches!(eb, AgentEvent::TurnStarted));
    }

    #[tokio::test]
    async fn slow_consumer_backpressures_emit() {
        // capacity = 1：填满后下次 emit 必须阻塞到消费者读取
        let bus = EventEmitter::with_capacity(1);
        let mut sub = bus.subscribe();

        bus.emit(AgentEvent::TurnStarted).await; // 填满
        let emit_fut = bus.emit(AgentEvent::TurnStarted);
        tokio::pin!(emit_fut);

        // 不消费时 emit 应当 pending
        tokio::select! {
            biased;
            () = &mut emit_fut => panic!("emit must block when channel full"),
            () = tokio::task::yield_now() => {}
        }

        // 一旦消费，emit 完成
        let _ = sub.next().await;
        emit_fut.await;
    }

    #[tokio::test]
    async fn dropped_subscriber_is_pruned() {
        let bus = EventEmitter::new();
        {
            let _sub = bus.subscribe();
        } // dropped here
        let mut alive = bus.subscribe();

        bus.emit(AgentEvent::TurnStarted).await;
        let count = bus.senders.lock().expect("mutex poisoned").len();
        // 第一次 emit 后死链已被清掉，只剩 alive
        assert_eq!(count, 1);
        let _ = alive.next().await;
    }
}
