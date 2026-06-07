//! Write-back channel for permission requests.
//!
//! `Session::resolve_permission` sends the client response to the ACP reverse request
//! `session/request_permission` back to the main loop, which waits using
//! [`PermissionGate::wait`].
//!
//! Permission management — see session and turn-loop designs.

use agent_client_protocol_schema::ToolCallId;
use dashmap::DashMap;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use crate::event::PermissionResolution;

/// A registry of pending permission requests.
///
/// Each in-flight turn holds a shared `Arc<PermissionGate>`:
/// - The main loop registers a waiter and awaits via [`Self::wait`]
/// - The ACP bridge layer calls [`Self::resolve`] after receiving the client response
#[derive(Default)]
pub struct PermissionGate {
    waiters: DashMap<ToolCallId, oneshot::Sender<PermissionResolution>>,
}

impl PermissionGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a waiter and await until [`Self::resolve`] is called or `cancel` fires.
    ///
    /// When `cancel` fires, returns [`PermissionResolution::Cancelled`] — the main loop
    /// handles this as "User cancelled".
    ///
    /// If a waiter already exists for the same `id`, the old sender is dropped (the old
    /// wait receives [`PermissionResolution::Cancelled`], avoiding a hang). This path
    /// should theoretically never be hit — the main loop only calls `wait` once per
    /// tool_use.
    pub async fn wait(&self, id: ToolCallId, cancel: CancellationToken) -> PermissionResolution {
        let (tx, rx) = oneshot::channel();
        if let Some(prev) = self.waiters.insert(id.clone(), tx) {
            // This should not happen: `wait` called twice for the same `id`. Wake the old
            // waiter with `Cancelled` to prevent it from hanging forever.
            tracing::warn!(
                tool_call_id = %id,
                "PermissionGate::wait called twice for same id; cancelling previous waiter"
            );
            let _ = prev.send(PermissionResolution::Cancelled);
        }

        tokio::select! {
            biased;
            () = cancel.cancelled() => {
                // Remove our registration if it is still present; resolve may race with
                // cancel.
                self.waiters.remove(&id);
                PermissionResolution::Cancelled
            }
            recv = rx => match recv {
                Ok(outcome) => outcome,
                // Sender was replaced or gate was dropped; use cancellation semantics.
                Err(_) => PermissionResolution::Cancelled,
            }
        }
    }

    /// Deliver the outcome to the waiter. If `id` has no waiter (already removed by
    /// cancel, or the main loop hasn't called wait yet), silently no-op — the ACP bridge
    /// layer is unaware of main-loop timing, and duplicate or late resolves must not
    /// corrupt the turn.
    pub fn resolve(&self, id: &ToolCallId, outcome: PermissionResolution) {
        if let Some((_, tx)) = self.waiters.remove(id) {
            // Ignore if the receiver has been dropped — the main loop may have already
            // returned via the cancel path.
            let _ = tx.send(outcome);
        }
    }
}

#[cfg(test)]
mod tests;
