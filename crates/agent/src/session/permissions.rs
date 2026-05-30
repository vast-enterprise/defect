//! 权限请求的回写通道。
//!
//! `Session::resolve_permission` 把 ACP 反向 request `session/request_permission`
//! 的客户端响应送回主循环；主循环用 [`PermissionGate::wait`] 等待。
//!
//! 设计详见 `docs/internal/session.md` §3.4 与 `docs/internal/turn-loop.md` §3.3 / §5。

use agent_client_protocol_schema::ToolCallId;
use dashmap::DashMap;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::event::PermissionResolution;

/// 等待中的权限请求登记表。
///
/// 每个进行中的 turn 持有一个共享 `Arc<PermissionGate>`：
/// - 主循环 [`Self::wait`] 注册等待并 await
/// - acp 桥接层在拿到客户端响应后调用 [`Self::resolve`]
#[derive(Default)]
pub struct PermissionGate {
    waiters: DashMap<ToolCallId, oneshot::Sender<PermissionResolution>>,
}

impl PermissionGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个等待，并 await 直到 [`Self::resolve`] 被调用或 cancel 触发。
    ///
    /// cancel 触发时返回 [`PermissionResolution::Cancelled`]——主循环按
    /// "用户取消"处理（`docs/internal/turn-loop.md` §5）。
    ///
    /// 如果同一 `id` 已经有等待者，旧 sender 被丢弃（旧 wait 会收到
    /// [`PermissionResolution::Cancelled`]，避免悬挂）。这条路径理论上
    /// 不应触发——主循环对每个 tool_use 只 wait 一次。
    pub async fn wait(&self, id: ToolCallId, cancel: CancellationToken) -> PermissionResolution {
        let (tx, rx) = oneshot::channel();
        if let Some(prev) = self.waiters.insert(id.clone(), tx) {
            // 不应该发生：同一 id 多次 wait。把旧 waiter 唤醒为 Cancelled，
            // 避免旧 future 永远挂着。
            tracing::warn!(
                tool_call_id = %id,
                "PermissionGate::wait called twice for same id; cancelling previous waiter"
            );
            let _ = prev.send(PermissionResolution::Cancelled);
        }

        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                // 摘掉自己的登记（如果还在）；resolve 可能正好与 cancel 竞速
                self.waiters.remove(&id);
                PermissionResolution::Cancelled
            }
            recv = rx => match recv {
                Ok(outcome) => outcome,
                // sender 被替换或 gate 被 drop；走取消语义
                Err(_) => PermissionResolution::Cancelled,
            }
        }
    }

    /// 把 outcome 投递给等待者。如果 `id` 没有等待者（已被 cancel 摘走、
    /// 或主循环还没来得及 wait），静默 no-op——acp 桥接层不感知主循环
    /// 时序，重复 / 迟到的 resolve 不应破坏 turn。
    pub fn resolve(&self, id: &ToolCallId, outcome: PermissionResolution) {
        if let Some((_, tx)) = self.waiters.remove(id) {
            // receiver 已 drop 的话忽略——主循环可能已 cancel 路径返回
            let _ = tx.send(outcome);
        }
    }
}

#[cfg(test)]
mod test;
