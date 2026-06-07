//! Event publishing: mpsc bounded + fan-out.
//!
//! ## Shape
//!
//! The main loop only calls [`EventEmitter::emit`]; subscribers get an independent mpsc
//! receiver via [`EventEmitter::subscribe`]. `emit` sends to all receivers serially.
//! **Slow consumers block `emit`** (backpressure) — this is the desired "no event loss"
//! behavior.
//!
//! ## Why not broadcast
//!
//! [`tokio::sync::broadcast`] returns `Lagged` and skips events when receivers fall
//! behind, violating the "no drop" invariant of [`AgentEvent`].

use std::sync::Mutex;

use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::event::AgentEvent;

/// Default mpsc capacity. The main loop blocks on the 257th event when subscribers fall
/// behind.
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// A single subscriber's sender handle. It is wrapped in `Mutex` only because
/// [`EventEmitter::emit`] is `&self` + `async` — either `DashMap` or `RwLock` would work;
/// `std::Mutex` is chosen here because we only briefly hold the lock during `emit` to
/// snapshot the list, and `send` happens outside the lock.
type SubscriberHandle = mpsc::Sender<AgentEvent>;

/// Event bus. One instance per session.
pub struct EventEmitter {
    capacity: usize,
    /// Subscribers currently registered. Dropping a receiver will be cleaned up
    /// automatically on the next `emit`.
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

    /// Subscribes a new listener. The returned stream ends naturally when [`Self`] is
    /// dropped.
    pub fn subscribe(&self) -> BoxStream<'static, AgentEvent> {
        let (tx, rx) = mpsc::channel(self.capacity);
        self.senders
            .lock()
            .expect("EventEmitter senders mutex poisoned")
            .push(tx);
        ReceiverStream::new(rx).boxed()
    }

    /// Delivers the event to every subscriber.
    ///
    /// Awaits each sender serially. If a subscriber's channel is full, this call blocks
    /// until the subscriber consumes — this is intentional backpressure.
    pub async fn emit(&self, event: AgentEvent) {
        // Take a snapshot to avoid holding the lock across an await point.
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

    /// Remove senders whose receivers have been dropped.
    fn prune(&self, snapshot: &[SubscriberHandle], dead_indices: &[usize]) {
        let mut guard = self
            .senders
            .lock()
            .expect("EventEmitter senders mutex poisoned");
        // The snapshot and `*guard` may have different lengths due to concurrent
        // `subscribe` calls; we compare by pointer equality to avoid removing the wrong
        // sender.
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
mod tests;
